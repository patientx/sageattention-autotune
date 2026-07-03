import functools

import torch
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning

from . import autotune_utils
from .utils import _padded_head_dim

_AUTOTUNE_CONFIGS = (
    (128, 64, 32, 64),
    (128, 32, 32, 32),
    (64, 64, 32, 64),
    (128, 64, 16, 64),
)
_AUTOTUNE_CACHE: dict[object, tuple[int, int, int, int]] = {}


@functools.cache
def _config_is_valid(
    config: tuple[int, int, int, int],
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> bool:
    blk_q, blk_k, _, _ = config
    if is_causal and blk_q // blk_k > 2:
        return False

    head_dim = _padded_head_dim(head_dim)
    # See smem_max in launch_sm80_qk_kernel
    smem_bytes = head_dim * max(blk_q + 3 * blk_k, 2 * blk_q)
    return smem_bytes <= autotune_utils._shared_memory_limit(device_index)


@functools.cache
def _valid_configs(
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> tuple[tuple[int, int, int, int], ...]:
    return autotune_utils._valid_configs(_AUTOTUNE_CONFIGS, _config_is_valid, head_dim, is_causal, device_index)


def _eager_autotune_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
) -> tuple[int, int, int, int]:
    from .cuda_attn import _sageattn_configured

    configs = _valid_configs(q.size(-1), is_causal, q.device.index)
    key = autotune_utils._tensor_autotune_cache_key(
        q, k, v, tensor_layout, is_causal, pv_accum_dtype, smooth_k, smooth_v, return_lse
    )
    return autotune_utils._eager_autotune_select(
        configs,
        _AUTOTUNE_CACHE,
        key,
        lambda config: _sageattn_configured(
            q,
            k,
            v,
            tensor_layout,
            is_causal,
            pv_accum_dtype,
            smooth_k,
            smooth_v,
            return_lse,
            config,
        ),
    )


@torch.library.custom_op("sageattention_internal::sageattn_autotuned", mutates_args=())
def _sageattn_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    blk_q: int = 0,
    blk_k: int = 0,
    warp_q: int = 0,
    warp_k: int = 0,
) -> torch.Tensor:
    from .cuda_attn import _sageattn_configured

    qk_config = (blk_q, blk_k, warp_q, warp_k)
    configs = _valid_configs(q.size(-1), is_causal, q.device.index)
    if min(qk_config) <= 0 or qk_config not in configs:
        qk_config = configs[0]

    return _sageattn_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        False,
        qk_config,
    )


@_sageattn_autotuned.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    blk_q: int = 0,
    blk_k: int = 0,
    warp_q: int = 0,
    warp_k: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


register_custom_op_autotuning(
    _sageattn_autotuned,
    config_generator=lambda fake_tensors: [
        CustomOpConfig(
            blk_q=config[0],
            blk_k=config[1],
            warp_q=config[2],
            warp_k=config[3],
        )
        for config in _valid_configs(
            fake_tensors["q"].size(-1),
            False,  # For now we hardcode is_causal=False and we assume it allows more configs than is_causal=True
            fake_tensors["q"].device.index,
        )
    ],
)
