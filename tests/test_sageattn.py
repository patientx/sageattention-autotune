from itertools import product

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattention.cuda_autotune import _AUTOTUNE_CONFIGS, _valid_configs

_MODES = tuple(
    product(
        (64, 128, 256),
        (torch.float16, torch.bfloat16),
        ("HND", "NHD"),
        (False, True),
        ("fp32", "fp16", "fp16+fp32"),
        (False, True),
    )
)


def _make_qkv(
    batch_size: int = 2,
    num_heads: int = 16,
    seq_len: int = 1024,
    head_dim: int = 64,
    tensor_layout: str = "HND",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor_layout == "HND":
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _expected(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str, is_causal: bool) -> torch.Tensor:
    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    return expected.transpose(1, 2) if tensor_layout == "NHD" else expected


def _error_report(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    fro_rel_err = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()

    passed = fro_rel_err <= 0.014 and max_abs_err <= 0.1
    msg = f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    return passed, msg


def _mode_id(mode: tuple[int, torch.dtype, str, bool, str, bool]) -> str:
    head_dim, dtype, tensor_layout, is_causal, pv_accum_dtype, smooth_k = mode
    return (
        f"head_dim={head_dim}-dtype={dtype}-layout={tensor_layout}-causal={is_causal}-"
        f"pv={pv_accum_dtype}-smooth_k={smooth_k}"
    )


def _valid_cases():
    device_index = torch.cuda.current_device()
    cases = []
    for config in _AUTOTUNE_CONFIGS:
        for mode in _MODES:
            head_dim, _, _, is_causal, _, _ = mode
            if config in _valid_configs(head_dim, is_causal, device_index):
                cases.append(pytest.param(config, mode, id=f"config={config}-{_mode_id(mode)}"))
    return tuple(cases)


def _run_case(
    config: tuple[int, int, int, int],
    head_dim: int,
    dtype: torch.dtype,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
) -> tuple[bool, str]:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    actual = cuda_attn._sageattn_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        False,
        False,
        config,
    )

    return _error_report(actual, expected)


@pytest.mark.parametrize(("config", "mode"), _valid_cases())
def test_sageattn_cuda_autotune_config(
    config: tuple[int, int, int, int],
    mode: tuple[int, torch.dtype, str, bool, str, bool],
) -> None:
    passed, msg = _run_case(config, *mode)
    assert passed, msg
