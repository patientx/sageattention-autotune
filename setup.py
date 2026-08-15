import os
import shlex
import shutil
import subprocess
from typing import Any, cast

import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


def _quote_command_arg(arg: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([arg])
    return shlex.quote(arg)


def _prepend_ccache(command: str, ccache_path: str) -> str:
    if "ccache" in command.lower():
        return command
    return f"{_quote_command_arg(ccache_path)} {command}"


def _enable_ccache() -> None:
    ccache_path = shutil.which("ccache")
    if not ccache_path:
        return

    assert CUDA_HOME is not None
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
    os.environ["PYTORCH_NVCC"] = _prepend_ccache(os.getenv("PYTORCH_NVCC", _quote_command_arg(nvcc)), ccache_path)

    original_write_ninja_file = cpp_extension._write_ninja_file
    if original_write_ninja_file is None:
        return

    if getattr(original_write_ninja_file, "_sageattention_ccache", False):
        return

    ccache_prefix = "" if "ccache" in os.getenv("CXX", "").lower() else f"{_quote_command_arg(ccache_path)} "

    def write_ninja_file_with_ccache(*args: Any, **kwargs: Any) -> None:
        original_write_ninja_file(*args, **kwargs)
        path = kwargs.get("path", args[0] if args else None)
        if path is None or not os.path.exists(path):
            return

        with open(path, encoding="utf-8") as f:
            content = f.read()

        patched = content.replace(
            "  command = cl /showIncludes",
            f"  command = {ccache_prefix}$cxx /showIncludes",
        ).replace(
            "  command = $cxx -MMD",
            f"  command = {ccache_prefix}$cxx -MMD",
        )

        if patched != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)

    cast(Any, write_ninja_file_with_ccache)._sageattention_ccache = True
    cast(Any, cpp_extension)._write_ninja_file = write_ninja_file_with_ccache


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


build_triton_only = _env_flag_enabled("SAGEATTN_BUILD_TRITON_ONLY")

if not build_triton_only:
    if CUDA_HOME is None:
        raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available to build SageAttention.")
    _enable_ccache()

if os.name == "nt":
    cxx_flags = ["/O2", "/permissive-", "-DENABLE_BF16"]
else:
    cxx_flags = ["-O3", "-DENABLE_BF16"]

cxx_flags += ["-DTORCH_STABLE_ONLY"]

nvcc_flags = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    f"--threads={os.cpu_count()}",
    "-diag-suppress=174",
    "-diag-suppress=177",
    "-diag-suppress=221",
    "-DTORCH_STABLE_ONLY",
    "-gencode",
    "arch=compute_86,code=sm_86",
]

if os.name == "nt":
    nvcc_flags += ["-Xcompiler=/Zc:preprocessor", "-D_WIN32=1", "-DUSE_CUDA=1"]
else:
    abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    cxx_flags += [f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
    nvcc_flags += [f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]

cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
if cxx_append:
    cxx_flags += cxx_append.split()

nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
if nvcc_append:
    nvcc_flags += nvcc_append.split()

if build_triton_only:
    ext_modules = []
else:
    ext_modules = [
        CUDAExtension(
            name="sageattention._qattn_sm80",
            sources=[
                "csrc/qattn/pybind_sm80.cpp",
                "csrc/qattn/qk_int8_sv_f16_accum_f16_attn.cu",
                "csrc/qattn/qk_int8_sv_f16_accum_f16_attn_inst_buf.cu",
                "csrc/qattn/qk_int8_sv_f16_accum_f16_fuse_v_mean_attn.cu",
                "csrc/qattn/qk_int8_sv_f16_accum_f32_attn.cu",
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            py_limited_api=True,
        ),
    ]

setup(
    name="sageattention",
    version="2.2.0" + os.environ.get("SAGEATTENTION_WHEEL_VERSION_SUFFIX", ""),
    author="SageAttention team",
    license="Apache 2.0 License",
    packages=find_packages(),
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
