from copy import copy

from hcu_sglang.utils.detection import is_hcu


def around_convert_numpy_to_tensor(original, numpy_array, device=None):
    if not is_hcu() or device is None:
        return original(numpy_array, device)

    import numpy as np
    import torch

    if torch.device(device).type != "cuda":
        return original(numpy_array, device)
    if isinstance(numpy_array, list):
        numpy_array = np.stack(numpy_array, axis=0)
    if (
        not isinstance(numpy_array, np.ndarray)
        or numpy_array.dtype != np.uint8
        or numpy_array.ndim != 4
    ):
        return original(numpy_array, device)

    frames = torch.from_numpy(numpy_array).to(device).permute(0, 3, 1, 2)
    return frames.to(torch.float32, memory_format=torch.contiguous_format).div_(255.0)


def around_setup_forward(original, model, **kwargs):
    if not is_hcu():
        return original(model, **kwargs)

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
        AutoencoderKLLegacy,
    )

    if type(model) is not AutoencoderKLLegacy:
        return original(model, **kwargs)

    first_setup = not hasattr(model, "processor")
    result = original(model, **kwargs)
    if first_setup:
        default = model.transform
        inplace = copy(default)
        inplace.inplace = True
        model._hcu_default_transforms = (default, inplace)
    default, inplace = getattr(model, "_hcu_default_transforms", (None, None))
    model.processor._runtime_owned_transform = (
        inplace if model.transform is default else None
    )
    return result
