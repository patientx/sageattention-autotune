from itertools import product

import pytest
import torch
from test_sageattn import _attention_report, _expected, _make_qkv, _mode_id

from sageattention.triton_attn import _sageattn_triton_configured
from sageattention.triton_autotune import _valid_triton_configs_for_head_dim

_MODES = tuple(
    product(
        (64, 128, 256),
        (torch.float16, torch.bfloat16),
        ("HND", "NHD"),
        (False, True),
        ("fp32", "fp16"),
        (False, True),
        (False, True),
    )
)


def _valid_cases():
    device = torch.device("cuda", torch.cuda.current_device())
    cases = []
    for mode in _MODES:
        head_dim, _, _, is_causal, _, _, _ = mode
        for config in _valid_triton_configs_for_head_dim(head_dim, is_causal, device):
            cases.append(pytest.param(config, mode, id=f"config={config}-{_mode_id(mode)}"))
    return tuple(cases)


def _run_case(
    triton_config: tuple[int, int, int, int],
    head_dim: int,
    dtype: torch.dtype,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
    return_lse: bool,
) -> tuple[bool, str]:
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal, return_lse)

    actual = _sageattn_triton_configured(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        pv_accum_dtype,
        smooth_k,
        return_lse,
        triton_config,
    )
    return _attention_report(actual, expected, rtol=0.014, atol=0.1, lse_rtol=0.0004, lse_atol=0.06)


@pytest.mark.parametrize(("triton_config", "mode"), _valid_cases())
def test_sageattn_triton_config(
    triton_config: tuple[int, int, int, int],
    mode: tuple[int, torch.dtype, str, bool, str, bool, bool],
) -> None:
    passed, msg = _run_case(triton_config, *mode)
    assert passed, msg
