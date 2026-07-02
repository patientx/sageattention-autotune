import torch
from test_sageattn import _expected, _make_qkv
from test_sageattn_compile import _check

from sageattention import sageattn_qk_int8_pv_fp16_triton


def test_eager_autotuned() -> None:
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)
    actual = sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)
    _check(actual, expected, "eager autotuned")


def test_compile_autotuned() -> None:
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)

    actual = fn(q, k, v)
    _check(actual, expected, "compile autotuned")
