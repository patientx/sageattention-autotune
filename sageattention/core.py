import importlib

from .triton_attn import sageattn_qk_int8_pv_fp16_triton
from .utils import _env_flag_enabled

try:
    importlib.import_module(f"{__package__}._qattn_sm80")
except (ImportError, OSError):
    sageattn_qk_int8_pv_fp16_cuda = None
else:
    from .cuda_attn import sageattn_qk_int8_pv_fp16_cuda

if _env_flag_enabled("SAGEATTN_TRITON_BACKEND") or sageattn_qk_int8_pv_fp16_cuda is None:
    _sageattn_impl = sageattn_qk_int8_pv_fp16_triton
else:
    _sageattn_impl = sageattn_qk_int8_pv_fp16_cuda


def sageattn(
    q,
    k,
    v,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale=None,
    **kwargs,
):
    """
    Compat shim over the selected backend.

    Callers like ComfyUI-WanVideoWrapper or SDNEXT's sageattn_func call
    sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal),
    but our triton/cuda kernels don't accept dropout_p. That raised a bare
    TypeError, which upstream swallows and reports as "sage not found,
    falling back to pytorch attention." We don't implement dropout or a
    custom softmax scale, so just accept and drop them here instead of
    exploding the whole sage detection.
    """
    if dropout_p:
        raise ValueError("sageattn: dropout_p is not supported (must be 0.0)")
    if scale is not None:
        raise ValueError("sageattn: custom scale is not supported")
    return _sageattn_impl(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, **kwargs)


# Placeholders for compatibility with libraries such as diffusers. Not implemented yet.
sageattn_qk_int8_pv_fp8_cuda = None
sageattn_qk_int8_pv_fp8_cuda_sm90 = None
sageattn_varlen = None
