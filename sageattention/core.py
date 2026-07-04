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
    sageattn = sageattn_qk_int8_pv_fp16_triton
else:
    sageattn = sageattn_qk_int8_pv_fp16_cuda

# Placeholders for compatibility with libraries such as diffusers. Not implemented yet.
sageattn_qk_int8_pv_fp8_cuda = None
sageattn_qk_int8_pv_fp8_cuda_sm90 = None
sageattn_varlen = None
