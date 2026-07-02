import functools

import torch
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning

from . import autotune_utils
from .triton.attn_autotune import _valid_attn_configs

_TRITON_BLOCK_CONFIGS = (
    (256, 64),
    (128, 128),
    (128, 64),
    (128, 32),
    (64, 128),
    (64, 64),
    (64, 32),
    (32, 128),
    (32, 64),
    (32, 32),
)
_TRITON_AUTOTUNE_CACHE: dict[object, tuple[int, int]] = {}


@functools.cache
def _config_is_valid(
    block_config: tuple[int, int],
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> bool:
    return bool(_valid_attn_configs(block_config, head_dim, is_causal, device_index))


@functools.cache
def _valid_configs(
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        block_config
        for block_config in _TRITON_BLOCK_CONFIGS
        if _config_is_valid(block_config, head_dim, is_causal, device_index)
    )


def _eager_autotune_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    return_lse: bool,
) -> tuple[int, int]:
    from .triton_attn import _sageattn_triton_configured

    block_configs = _valid_configs(q.size(-1), is_causal, q.device.index)
    key = autotune_utils._tensor_autotune_cache_key(
        q, k, v, tensor_layout, is_causal, pv_accum_dtype, smooth_k, return_lse
    )
    return autotune_utils._eager_autotune_select(
        block_configs,
        _TRITON_AUTOTUNE_CACHE,
        key,
        lambda block_config: _sageattn_triton_configured(
            q,
            k,
            v,
            tensor_layout,
            is_causal,
            pv_accum_dtype,
            smooth_k,
            return_lse,
            block_config,
        ),
    )


@torch.library.custom_op("sageattention_internal::sageattn_triton_autotuned", mutates_args=())
def _sageattn_triton_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> torch.Tensor:
    from .triton_attn import _sageattn_triton_configured

    block_config = (block_m, block_n)
    block_configs = _valid_configs(q.size(-1), is_causal, q.device.index)
    if min(block_config) <= 0 or block_config not in block_configs:
        block_config = block_configs[0]

    return _sageattn_triton_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        False,
        block_config,
    )


@_sageattn_triton_autotuned.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    block_m: int = 0,
    block_n: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


register_custom_op_autotuning(
    _sageattn_triton_autotuned,
    config_generator=lambda fake_tensors: [
        CustomOpConfig(
            block_m=block_config[0],
            block_n=block_config[1],
        )
        for block_config in _valid_configs(
            fake_tensors["q"].size(-1),
            False,  # For now we hardcode is_causal=False and we assume it allows more configs than is_causal=True
            fake_tensors["q"].device.index,
        )
    ],
)
