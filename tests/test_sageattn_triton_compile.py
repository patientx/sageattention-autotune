import pytest
import torch
from test_sageattn import _expected, _make_qkv
from test_sageattn_compile import _check

from sageattention import sageattn_qk_int8_pv_fp16_triton


@pytest.mark.parametrize("return_lse", (False, True))
def test_eager_autotuned(return_lse: bool) -> None:
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False, return_lse)

    actual = sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False, return_lse=return_lse)
    _check(actual, expected, f"eager autotuned return_lse={return_lse}")


def test_compile_autotuned() -> None:
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)

    fn = torch.compile(sageattn_qk_int8_pv_fp16_triton, fullgraph=True, mode="max-autotune")
    actual = fn(q, k, v, tensor_layout="HND", is_causal=False)
    _check(actual, expected, "compile autotuned")
