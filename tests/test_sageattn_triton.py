from itertools import product

import pytest
import torch
from test_sageattn import _error_report, _expected, _make_qkv, _mode_id

from sageattention.triton_attn import _sageattn_triton_configured
from sageattention.triton_autotune import _TRITON_BLOCK_CONFIGS, _valid_configs

_MODES = tuple(
    product(
        (64, 128, 256),
        (torch.float16, torch.bfloat16),
        ("HND", "NHD"),
        (False, True),
        ("fp32", "fp16"),
        (False, True),
    )
)


def _valid_cases():
    device_index = torch.cuda.current_device()
    cases = []
    for block_config in _TRITON_BLOCK_CONFIGS:
        for mode in _MODES:
            head_dim, _, _, is_causal, _, _ = mode
            if block_config in _valid_configs(head_dim, is_causal, device_index):
                cases.append(pytest.param(block_config, mode, id=f"block={block_config}-{_mode_id(mode)}"))
    return tuple(cases)


def _run_case(
    block_config: tuple[int, int],
    head_dim: int,
    dtype: torch.dtype,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    smooth_k: bool,
) -> tuple[bool, str]:
    q, k, v = _make_qkv(head_dim=head_dim, tensor_layout=tensor_layout, dtype=dtype)
    expected = _expected(q, k, v, tensor_layout, is_causal)

    actual = _sageattn_triton_configured(
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

    return _error_report(actual, expected)


@pytest.mark.parametrize(("block_config", "mode"), _valid_cases())
def test_sageattn_triton_block_config(
    block_config: tuple[int, int],
    mode: tuple[int, torch.dtype, str, bool, str, bool],
) -> None:
    passed, msg = _run_case(block_config, *mode)
    assert passed, msg
