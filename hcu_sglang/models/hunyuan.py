"""Install plugin-owned Hunyuan V3 model implementations on HCU."""

from __future__ import annotations

from importlib import import_module


_METHODS = {
    "HYV3FeedForward": (("__init__", "forward"), ()),
    "HYV3MoEFused": (
        ("__init__", "forward", "_forward_single_stream", "_forward_dual_stream"),
        ("get_moe_weights", "_combine_routed_and_shared", "_forward_deepep", "forward_normal"),
    ),
    "HYV3Attention": (("__init__", "forward"), ("forward_sp",)),
    "HYV3DecoderLayer": (("__init__", "forward"), ("forward_sp",)),
    "HYV3Model": (("__init__", "forward"), ()),
    "HYV3ForCausalLM": (
        ("__init__", "forward", "load_weights"),
        ("routed_experts_weights_of_layer", "get_model_config_for_expert_location"),
    ),
}

_NEXTN_METHODS = {
    "HYV3ModelNextN": (("__init__",), ()),
    "HYV3ForCausalLMNextN": (
        ("__init__", "load_weights"),
        ("get_model_config_for_expert_location",),
    ),
}


def _register_methods(base_module: str, source_module: str, methods: dict) -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    base = import_module(base_module)
    source = import_module(source_module)
    for class_name, (replaced, added) in methods.items():
        base_class = getattr(base, class_name)
        source_class = getattr(source, class_name)
        for name in added:
            if not hasattr(base_class, name):
                setattr(base_class, name, source_class.__dict__[name])
        for name in replaced:
            HookRegistry.register(
                f"{base_module}.{class_name}.{name}",
                getattr(source_class, name),
                HookType.REPLACE,
            )


def register_hooks() -> None:
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return
    _register_methods(
        "sglang.srt.models.hunyuan_v3",
        "hcu_sglang.models.hunyuan_v3",
        _METHODS,
    )
    _register_methods(
        "sglang.srt.models.hunyuan_v3_nextn",
        "hcu_sglang.models.hunyuan_v3_nextn",
        _NEXTN_METHODS,
    )
