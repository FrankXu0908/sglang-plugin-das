"""HCU extensions for SGLang server argument choices."""

from __future__ import annotations

from typing import get_args, get_type_hints

from sglang.srt.arg_groups.arg_utils import field_names, is_record
from hcu_sglang.config import minimax_opt_enabled, hy3_sp_enabled, register_parallel_options
from hcu_sglang.layers.quantization.registry import (
    QUANTIZATION_NAME,
    SLIMQUANT_NAME,
    SLIMQUANT_MARLIN_NAME,
)

HCU_ATTENTION_BACKEND = "hcu_mla"
HCU_QUANTIZATION_METHODS = (QUANTIZATION_NAME, SLIMQUANT_NAME, SLIMQUANT_MARLIN_NAME, "slimquant_marlin")
HCU_SERVER_ARG_DEFAULTS = {
    "minimax_opt": False,
    "hy3_sp": False,
}


def _extend_unique(values: list[str], additions: tuple[str, ...]) -> None:
    values.extend(value for value in additions if value not in values)


def _extend_moe_a2a_choices(server_args) -> None:
    legacy_choice_attrs = (
        "MOE_A2A_BACKEND_CHOICES",
        "SPECULATIVE_MOE_A2A_BACKEND_CHOICES",
    )
    legacy_choices_found = False
    for attr_name in legacy_choice_attrs:
        choices = getattr(server_args, attr_name, None)
        if choices is not None:
            _extend_unique(choices, ("mori-das",))
            legacy_choices_found = True
    if legacy_choices_found:
        return
    try:
        from sglang.srt.arg_groups import arg_utils
        from sglang.srt.arg_groups.arg_utils import Arg
        from sglang.srt.arg_groups.fields.exec_ import ExecMoe
    except ImportError:
        return

    if getattr(arg_utils.get_type_hints, "_hcu_moridas_choices", False):
        return

    original_get_type_hints = arg_utils.get_type_hints

    def _get_type_hints_with_moridas(obj, *args, **kwargs):
        hints = original_get_type_hints(obj, *args, **kwargs)
        # SGLang builds the parser from the aggregate ServerArgs record rather
        # than ExecMoe directly.  Match the field metadata so this works for
        # both the leaf record and the generated aggregate record.
        # Target and speculative MoE use separate parser fields.  Extending
        # only moe_a2a_backend makes MoriDAS work for normal decode but causes
        # DSPARK to fail during argument parsing before workers are created.
        for field_name in (
            "moe_a2a_backend",
            "speculative_moe_a2a_backend",
        ):
            hint = hints.get(field_name)
            for metadata in get_args(hint)[1:]:
                if isinstance(metadata, Arg) and metadata.choices is not None:
                    _extend_unique(metadata.choices, ("mori-das",))
                    break
        return hints

    _get_type_hints_with_moridas._hcu_moridas_choices = True
    arg_utils.get_type_hints = _get_type_hints_with_moridas


def ensure_choices(server_args) -> bool:
    _extend_moe_a2a_choices(server_args)
    required = (
        "ATTENTION_BACKEND_CHOICES",
        "DRAFT_ATTENTION_BACKEND_CHOICES",
        "QUANTIZATION_CHOICES",
        "CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS",
    )
    if not all(hasattr(server_args, name) for name in required):
        return False
    declared = field_names(server_args.ServerArgs)
    for name, default in HCU_SERVER_ARG_DEFAULTS.items():
        # Native HCU bases already declare writable constructor fields.
        if name in declared:
            continue
        if name == "minimax_opt" and is_record(server_args.ServerArgs):
            # A compatibility read view, not an undeclared write to frozen inputs.
            setattr(server_args.ServerArgs, name, property(lambda _record: minimax_opt_enabled()))
        elif name == "hy3_sp" and is_record(server_args.ServerArgs):
            setattr(server_args.ServerArgs, name, property(lambda _record: hy3_sp_enabled()))
        elif not hasattr(server_args.ServerArgs, name):
            setattr(server_args.ServerArgs, name, default)

    _extend_unique(
        server_args.ATTENTION_BACKEND_CHOICES,
        (HCU_ATTENTION_BACKEND,),
    )
    _extend_unique(
        server_args.DRAFT_ATTENTION_BACKEND_CHOICES,
        (HCU_ATTENTION_BACKEND,),
    )
    _extend_unique(
        server_args.QUANTIZATION_CHOICES,
        HCU_QUANTIZATION_METHODS,
    )

    if (
        HCU_ATTENTION_BACKEND
        not in server_args.CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        server_args.add_chunked_prefix_cache_attention_backend(HCU_ATTENTION_BACKEND)
    return True


def register() -> None:
    from sglang.srt import server_args

    register_parallel_options()
    ensure_choices(server_args)
