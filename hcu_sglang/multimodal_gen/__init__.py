"""HCU adaptations for SGLang multimodal generation."""


from importlib import import_module
from importlib.util import find_spec


def _register(registry=None):
    for name in (
        "wan_sla", "loader.integration", "platform", "config_compat",
        "profiling", "attention", "minimax_h3", "qknorm_rope", "runtime.layers.quantization.fp8",
        "runtime.layers.attention.local_attention",
    ):
        import_module(f"hcu_sglang.multimodal_gen.{name}").register_hooks(registry)


def register():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

    _register(HookRegistry)


def register_legacy():
    if find_spec("sglang.multimodal_gen.runtime.platforms.plugins") is None:
        _register()
