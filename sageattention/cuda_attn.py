import warnings
from typing import Literal, Union, overload

import torch

from .cuda_autotune import _eager_autotune_select, _sageattn_autotuned
from .cuda_compile import _qattn_sm80
from .triton.quant_per_thread import per_thread_int8
from .utils import DEFAULT_PV_ACCUM_DTYPE, LOG2_E, _lse_correction, _pad_qkv

SageAttnResult = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


@overload
def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: Literal[False] = False,
    attn_mask: object = None,
) -> torch.Tensor: ...


@overload
def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: Literal[True] = True,
    attn_mask: object = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    attn_mask: object = None,  # For ComfyUI compatibility. Not implemented yet.
) -> SageAttnResult:
    assert attn_mask is None

    if torch.compiler.is_compiling() and not return_lse:
        return _sageattn_autotuned(
            q,
            k,
            v,
            tensor_layout,
            is_causal,
            pv_accum_dtype,
            smooth_k,
            smooth_v,
        )

    qk_config = _eager_autotune_select(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        return_lse,
    )

    return _sageattn_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        return_lse,
        qk_config,
    )


@overload
def _sageattn_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: Literal[False],
    qk_config: tuple[int, int, int, int],
) -> torch.Tensor: ...


@overload
def _sageattn_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: Literal[True],
    qk_config: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def _sageattn_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
    qk_config: tuple[int, int, int, int],
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
        layout_i = 0
        seq_dim_index = 1
        head_dim_index = 2
    elif tensor_layout == "HND":
        layout_i = 1
        seq_dim_index = 2
        head_dim_index = 1
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")

    if smooth_k:
        km = k.mean(dim=seq_dim_index, keepdim=True)
    else:
        km = None

    if pv_accum_dtype in ("fp32", "fp16+fp32") and smooth_v:
        warnings.warn("pv_accum_dtype is fp32 or fp16+fp32, smooth_v will be ignored.", stacklevel=2)
        smooth_v = False

    blk_q, blk_k, warp_q, warp_k = qk_config

    q_int8, q_scale, k_int8, k_scale = per_thread_int8(
        q,
        k,
        km=km,
        BLKQ=blk_q,
        WARPQ=warp_q,
        BLKK=blk_k,
        WARPK=warp_k,
        tensor_layout=tensor_layout,
    )

    output = torch.empty(q.size(), device=q.device, dtype=dtype)

    if pv_accum_dtype == "fp32":
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
            q_int8,
            k_int8,
            v.to(torch.float16),
            output,
            q_scale,
            k_scale,
            layout_i,
            is_causal,
            sm_scale,
            blk_q,
            blk_k,
            warp_q,
            warp_k,
            return_lse,
        )
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            vm = v.mean(dim=seq_dim_index)
            smoothed_v = (v - vm.unsqueeze(seq_dim_index)).to(torch.float16)
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
                q_int8,
                k_int8,
                smoothed_v,
                output,
                q_scale,
                k_scale,
                vm,
                layout_i,
                is_causal,
                sm_scale,
                blk_q,
                blk_k,
                warp_q,
                warp_k,
                return_lse,
            )
        else:
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn(
                q_int8,
                k_int8,
                v.to(torch.float16),
                output,
                q_scale,
                k_scale,
                layout_i,
                is_causal,
                sm_scale,
                blk_q,
                blk_k,
                warp_q,
                warp_k,
                return_lse,
            )
    elif pv_accum_dtype == "fp16+fp32":
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(
            q_int8,
            k_int8,
            v.to(torch.float16),
            output,
            q_scale,
            k_scale,
            layout_i,
            is_causal,
            sm_scale,
            blk_q,
            blk_k,
            warp_q,
            warp_k,
            return_lse,
        )
    else:
        raise ValueError("pv_accum_dtype must be 'fp32', 'fp16', or 'fp16+fp32'.")

    output = output[..., :head_dim]
    if not return_lse:
        return output

    lse /= LOG2_E
    if smooth_k:
        assert km is not None
        lse += _lse_correction(q, km, tensor_layout, head_dim_index) * sm_scale
    return output, lse
