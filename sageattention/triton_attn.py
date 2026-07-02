from typing import Literal, Union, overload

import torch

from .triton.attn_qk_int8_per_block import forward as _attn_forward
from .triton.quant_per_block import per_block_int8
from .triton_autotune import _eager_autotune_select, _sageattn_triton_autotuned
from .utils import DEFAULT_PV_ACCUM_DTYPE, LOG2_E, _lse_correction, _pad_qkv

SageAttnResult = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


@overload
def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    return_lse: Literal[False] = False,
    attn_mask: object = None,
) -> torch.Tensor: ...


@overload
def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    return_lse: Literal[True] = True,
    attn_mask: object = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    return_lse: bool = False,
    attn_mask: object = None,  # For ComfyUI compatibility. Not implemented yet.
) -> SageAttnResult:
    assert attn_mask is None

    if torch.compiler.is_compiling() and not return_lse:
        return _sageattn_triton_autotuned(
            q,
            k,
            v,
            tensor_layout,
            is_causal,
            pv_accum_dtype,
            smooth_k,
        )

    config = _eager_autotune_select(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        return_lse,
    )

    return _sageattn_triton_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        return_lse,
        config,
    )


@overload
def _sageattn_triton_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    return_lse: Literal[False],
    block_config: tuple[int, int],
) -> torch.Tensor: ...


@overload
def _sageattn_triton_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    return_lse: Literal[True],
    block_config: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def _sageattn_triton_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    return_lse: bool,
    block_config: tuple[int, int],
) -> SageAttnResult:
    dtype = q.dtype
    if not q.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype: {dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("All tensors must have the same dtype.")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape.")

    head_dim, q, k, v = _pad_qkv(q, k, v)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")

    sm_scale = head_dim**-0.5

    if tensor_layout == "NHD":
        seq_dim_index = 1
        head_dim_index = 2
    elif tensor_layout == "HND":
        seq_dim_index = 2
        head_dim_index = 1
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")

    if smooth_k:
        km = k.mean(dim=seq_dim_index, keepdim=True)
    else:
        km = None

    if pv_accum_dtype not in ("fp32", "fp16"):
        raise ValueError("pv_accum_dtype must be 'fp32' or 'fp16'.")

    block_m, block_n = block_config

    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=km,
        BLKQ=block_m,
        BLKK=block_n,
        tensor_layout=tensor_layout,
    )

    output, lse = _attn_forward(
        q_int8,
        k_int8,
        v.to(torch.float16),
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        pv_accum_dtype=pv_accum_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        output_dtype=dtype,
        return_lse=return_lse,
    )

    output = output[..., :head_dim]
    if not return_lse:
        return output

    lse /= LOG2_E
    if smooth_k:
        assert km is not None
        lse += _lse_correction(q, km, tensor_layout, head_dim_index) * sm_scale
    return output, lse
