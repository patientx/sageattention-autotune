import importlib
import os

from .triton_attn import sageattn_qk_int8_pv_fp16_triton

try:
    importlib.import_module(f"{__package__}._qattn_sm80")
except (ImportError, OSError):
    sageattn_qk_int8_pv_fp16_cuda = None
else:
    from .cuda_attn import sageattn_qk_int8_pv_fp16_cuda

SAGEATTN_BACKEND = os.getenv("SAGEATTN_BACKEND", "").lower()
if SAGEATTN_BACKEND == "triton":
    sageattn = sageattn_qk_int8_pv_fp16_triton
else:
    sageattn = sageattn_qk_int8_pv_fp16_cuda

# Placeholders for compatibility with libraries such as diffusers. Not implemented yet.
sageattn_qk_int8_pv_fp8_cuda = None
sageattn_qk_int8_pv_fp8_cuda_sm90 = None
sageattn_varlen = None
