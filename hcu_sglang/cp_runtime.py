# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.layers.cp import utils as C
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.cp_token_counts import (
    enable_cp_v2, normalize_dp_cp_token_counts, get_cp_v2_physical_token_count,
)


def is_glm_dsa_cache_layer_split_enabled(model_runner: "ModelRunner") -> bool:
    """Whether DSA GPU KV/indexer cache layers are sharded across CP ranks.

    Layer split is a prefill-CP-only optimization for DSA (DeepSeek Sparse
    Attention) MLA models (e.g. GLM-5.2). Single-layer NextN drafts can opt in
    after the EAGLE worker supplies a compatible target Main-KV scratch pool.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa

    return (
        (
            not model_runner.is_draft_worker
            or getattr(model_runner, "dsa_layer_split_scratch_source", None) is not None
        )
        and C.get_parallel().enable_dsa_cache_layer_split
        and model_runner.use_mla_backend
        and is_deepseek_dsa(model_runner.model_config.hf_config)
    )



def get_glm_dsa_layer_split_effective_num_layers(
    model_runner: "ModelRunner", num_layers: int
) -> int:
    """Per-rank owned layer count used when sizing the DSA KV cell.

    Under layer split each CP rank only stores ``ceil(num_layers / shard_size)``
    layers, plus one extra layer for the remote scratch buffer used when reading
    a layer owned by another CP rank.
    """
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return num_layers
    shard_size = C.get_parallel().attn_cp_size
    if shard_size <= 1:
        return num_layers
    owned_layers_upper_bound = (num_layers + shard_size - 1) // shard_size
    if getattr(model_runner, "dsa_layer_split_scratch_source", None) is not None:
        # The draft aliases target Main-KV scratch, so count only its owner
        # storage here. Index-K has its own independent scratch budget.
        return owned_layers_upper_bound
    return max(1, owned_layers_upper_bound + 1)



def cp_v2_dp_token_counts_for_strategy(
    global_cp_num_tokens: Optional[list[int]],
) -> Optional[list[int]]:
    """Expose synchronized CP shard counts only for the Zigzag DP layout.

    The scheduler computes these counts only for Zigzag. Interleave's zero
    placeholders must not disable its existing CP-v2 path or remap its DP
    buffer as though it had Zigzag-local shards.
    """
    if isinstance(C.get_cp_strategy(), C.ZigzagCPStrategy):
        return global_cp_num_tokens
    return None



def is_cp_active(forward_batch) -> bool:
    """Return whether the current forward batch is running through CP."""
    # HCU defaults to platform CP unless explicitly opted in; NPU/MUSA retain
    # their platform CP implementations. Treating those
    # batches as strategy-CP here shards the model inputs a second time while
    # their attention backends still use the legacy layout.
    if not enable_cp_v2():
        return False

    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False

    strategy = C.get_cp_strategy()
    if strategy is None:
        return False

    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return False

    # DP attention synchronizes the CP layout before ForwardBatch reaches the
    # model.  An all-zero list means that another non-idle DP replica cannot
    # enter CP for this global forward (for example, a 1-token batch paired
    # with a 16K Prefill).  Every replica must then use the ordinary full-TP
    # layout; allowing only the locally eligible replica to shard would make
    # the model-body collectives disagree.
    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is not None and not any(global_cp_tokens):
        return False

    return strategy.can_apply(len(input_ids), forward_batch)



def is_mla_prefill_cp_enabled() -> bool:
    """Compatibility name used by the CUDA-graph runner."""
    return C.is_mla_cp_enabled()



def mla_use_prefill_cp(forward_batch) -> bool:
    """Compatibility predicate for platform-specific MLA prefill CP."""
    if enable_cp_v2():
        return is_mla_prefill_cp_enabled() and is_cp_active(forward_batch)
    return (
        getattr(forward_batch, "attn_cp_metadata", None) is not None
        and is_mla_prefill_cp_enabled()
        and forward_batch.forward_mode.is_context_parallel_extend()
    )



def prepare_cp_forward(forward_batch) -> None:
    """Build CP metadata for an active context-parallel prefill batch."""
    assert is_cp_active(forward_batch)
    strategy = C.get_cp_strategy()
    assert strategy is not None

    seq_lens_cpu = C._to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = C._to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    num_tokens = (
        sum(extend_lens_cpu)
        if extend_lens_cpu is not None
        else len(forward_batch.input_ids)
    )
    if forward_batch.attn_cp_metadata is None:
        forward_batch.attn_cp_metadata = strategy.build_metadata(
            num_tokens=num_tokens,
            seqs_len=seq_lens_cpu,
            extend_seqs_len=extend_lens_cpu,
        )
        C.pad_logical_token_to_physical(forward_batch.attn_cp_metadata)

    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is not None:
        dp_rank = C.get_parallel().attn_dp_rank
        expected = int(global_cp_tokens[dp_rank])
        actual = int(
            forward_batch.attn_cp_metadata.per_rank_actual_token[
                C.get_parallel().attn_cp_rank
            ]
        )
        if actual != expected:
            raise RuntimeError(
                "CP-v2 scheduler/model physical-token mismatch: "
                f"dp_rank={dp_rank}, cp_rank={C.get_parallel().attn_cp_rank}, "
                f"metadata={actual}, scheduled={expected}, "
                f"global_cp_tokens={global_cp_tokens}"
            )
    elif getattr(forward_batch, "global_num_tokens_cpu", None) is not None:
        # Interleave CP-v2 retains the original full-TP DP collective layout.
        # Its model body runs on a padded CP shard, while the shared DP buffer
        # must be large enough for all CP ranks before the MoE/MLA exchange.
        from sglang.srt.layers.dp_attention import set_local_dp_buffer_len

        set_local_dp_buffer_len(
            sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
        )

    elif (
        not C.get_parallel().enable_cp_tp_group_sharing
        and getattr(forward_batch, "global_num_tokens_cpu", None) is not None
    ):
        from sglang.srt.layers.dp_attention import set_local_dp_buffer_len

        set_local_dp_buffer_len(
            sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
        )

    if getattr(forward_batch, "out_cache_loc", None) is not None:
        forward_batch.out_cache_loc = forward_batch.out_cache_loc[:num_tokens]



def cp_materialize_global_token_order(
    x: Any, forward_batch, stream: Optional[Any] = None
):
    """Materialize a CP tensor in the global logical token order."""
    if not enable_cp_v2():
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_cp_utils import (
            cp_all_gather_rerange_output,
        )

        return cp_all_gather_rerange_output(
            x, C.get_parallel().attn_cp_size, forward_batch, stream
        )
    assert is_cp_active(forward_batch)
    strategy = C.get_cp_strategy()
    assert strategy is not None
    return strategy.gather_kv_cache(x, forward_batch, stream)



@contextmanager
def cp_shard_model_inputs(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
    complete_input_ids: Optional[Any] = None,
):
    """Restore the shared batch so logits processing keeps full-batch metadata."""
    assert is_cp_active(forward_batch)
    sharded_hidden_states = (
        C.cp_shard_hidden_states(complete_hidden_states, forward_batch)
        if complete_hidden_states is not None
        else None
    )
    sharded_positions = C.cp_shard_position_ids(complete_position_ids, forward_batch)
    model_input_ids = (
        C.cp_shard_hidden_states(complete_input_ids, forward_batch)
        if complete_input_ids is not None
        else None
    )

    had_input_ids_global = hasattr(forward_batch, "input_ids_global")
    input_ids_global_backup = getattr(forward_batch, "input_ids_global", None)
    if complete_input_ids is not None:
        forward_batch.input_ids_global = C.cp_interleave_input_ids(
            complete_input_ids, forward_batch
        )

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_hidden_states = getattr(spec_info, "hidden_states", None)
    spec_hidden_states_backup = None
    if (
        spec_hidden_states is not None
        and complete_hidden_states is not None
        and spec_hidden_states.shape[0] == complete_hidden_states.shape[0]
    ):
        spec_hidden_states_backup = spec_hidden_states
        spec_info.hidden_states = C.cp_shard_hidden_states(
            spec_hidden_states, forward_batch
        )

    # ``global_cp_num_tokens_cpu`` is populated only when attention-DP peers
    # must agree on a CP-local model-body buffer.  Plain DP1/CP runs retain
    # the legacy full-TP collective contract and do not need that remapping.
    cp_dp_context = (
        cp_local_dp_state(forward_batch)
        if getattr(forward_batch, "global_cp_num_tokens_cpu", None) is not None
        else nullcontext()
    )
    try:
        with cp_dp_context:
            yield sharded_hidden_states, sharded_positions, model_input_ids
    finally:
        if spec_hidden_states_backup is not None:
            spec_info.hidden_states = spec_hidden_states_backup
        if had_input_ids_global:
            forward_batch.input_ids_global = input_ids_global_backup
        elif hasattr(forward_batch, "input_ids_global"):
            delattr(forward_batch, "input_ids_global")



@contextmanager
def cp_local_dp_state(forward_batch):
    """Use the per-CP DP layout while a model body participates in CP prefill.

    Active ranks enter through ``cp_shard_model_inputs``.  Idle DP peers must
    enter the same state explicitly so their MLP collectives use the identical
    per-CP communication group and buffer layout.
    """
    state = _enter_cp_local_dp_state(forward_batch)
    try:
        yield
    finally:
        _restore_cp_local_dp_state(forward_batch, state)



def _enter_cp_local_dp_state(forward_batch):
    """Expose CP-local token slots to DP-attention during the model body.

    Each CP rank gathers only the corresponding shard across attention-DP
    replicas.  MoE then gathers those DP-complete shards across CP.  After the
    model body returns, logits processing sees the original full-sequence
    per-DP layout again.
    """
    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is None:
        raise RuntimeError(
            "CP-v2 with DP attention requires scheduler-provided "
            "global_cp_num_tokens_cpu"
        )

    import torch

    from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len

    parallel = C.get_parallel()
    cp_local_tokens = [int(tokens) for tokens in global_cp_tokens]
    local_index = parallel.attn_dp_rank if len(cp_local_tokens) > 1 else 0
    cp_local_gpu = torch.tensor(
        cp_local_tokens,
        dtype=forward_batch.global_num_tokens_gpu.dtype,
        device=forward_batch.global_num_tokens_gpu.device,
    )
    state = (
        forward_batch.global_num_tokens_cpu,
        forward_batch.global_num_tokens_gpu,
        forward_batch.global_dp_buffer_len,
        forward_batch.dp_padding_mode,
        forward_batch.dp_local_start_pos,
        forward_batch.dp_local_num_tokens,
        getattr(forward_batch, "dp_local_token_index", None),
        getattr(forward_batch, "cp_local_dp_layout", False),
    )

    forward_batch.global_num_tokens_cpu = cp_local_tokens
    forward_batch.global_num_tokens_gpu = cp_local_gpu
    forward_batch.global_dp_buffer_len = sum(cp_local_tokens)
    # Variable CP shards require SUM_LEN.  The full-TP all-reduce gathers the
    # distinct slots; MAX_LEN reduce-scatter assumes one slot per attention DP.
    forward_batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    forward_batch.dp_local_start_pos = None
    forward_batch.dp_local_num_tokens = None
    forward_batch.dp_local_token_index = local_index
    forward_batch.cp_local_dp_layout = True
    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        cp_local_tokens[local_index],
        False,
        cp_local_tokens,
        cp_local_gpu,
    )
    return state



def _restore_cp_local_dp_state(forward_batch, state) -> None:
    from sglang.srt.layers.dp_attention import set_dp_buffer_len

    (
        global_num_tokens_cpu,
        global_num_tokens_gpu,
        global_dp_buffer_len,
        dp_padding_mode,
        dp_local_start_pos,
        dp_local_num_tokens,
        dp_local_token_index,
        cp_local_dp_layout,
    ) = state
    forward_batch.global_num_tokens_cpu = global_num_tokens_cpu
    forward_batch.global_num_tokens_gpu = global_num_tokens_gpu
    forward_batch.global_dp_buffer_len = global_dp_buffer_len
    forward_batch.dp_padding_mode = dp_padding_mode
    forward_batch.dp_local_start_pos = dp_local_start_pos
    forward_batch.dp_local_num_tokens = dp_local_num_tokens
    forward_batch.dp_local_token_index = dp_local_token_index
    forward_batch.cp_local_dp_layout = cp_local_dp_layout
    parallel = C.get_parallel()
    local_index = parallel.attn_dp_rank if len(global_num_tokens_cpu) > 1 else 0
    set_dp_buffer_len(
        global_dp_buffer_len,
        int(global_num_tokens_cpu[local_index]),
        dp_padding_mode.is_max_len(),
        global_num_tokens_cpu,
        global_num_tokens_gpu,
    )



def _dispatch(implementation):
    def wrapper(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return wrapper


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name in ('_enter_cp_local_dp_state', '_restore_cp_local_dp_state', 'cp_local_dp_state', 'cp_materialize_global_token_order', 'cp_shard_model_inputs', 'cp_v2_dp_token_counts_for_strategy', 'enable_cp_v2', 'get_cp_v2_physical_token_count', 'get_glm_dsa_layer_split_effective_num_layers', 'is_cp_active', 'is_glm_dsa_cache_layer_split_enabled', 'is_mla_prefill_cp_enabled', 'mla_use_prefill_cp', 'normalize_dp_cp_token_counts', 'prepare_cp_forward'):
        implementation = globals()[name]
        if hasattr(C, name):
            HookRegistry.register('sglang.srt.layers.cp.utils.' + name,
                                  _dispatch(implementation), HookType.AROUND)
        else:
            setattr(C, name, implementation)
    C.is_cp_v2_active = is_cp_active
