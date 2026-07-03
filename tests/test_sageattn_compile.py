import pytest
import torch
from test_sageattn import _attention_report, _expected, _make_qkv


def _check(
    actual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    expected: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    label: str,
) -> None:
    passed, msg = _attention_report(actual, expected)
    msg = f"{label}: {msg}"
    assert passed, msg
    print(msg)


@pytest.mark.parametrize("return_lse", (False, True))
def test_eager_autotuned(return_lse: bool) -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False, return_lse)

    actual = cuda_attn.sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, tensor_layout="HND", is_causal=False, return_lse=return_lse
    )
    _check(actual, expected, f"eager autotuned return_lse={return_lse}")


def test_compile_autotuned() -> None:
    cuda_attn = pytest.importorskip("sageattention.cuda_attn", reason="sageattention CUDA kernel is not installed")
    q, k, v = _make_qkv()
    expected = _expected(q, k, v, "HND", False)

    fn = torch.compile(cuda_attn.sageattn_qk_int8_pv_fp16_cuda, fullgraph=True, mode="max-autotune")
    actual = fn(q, k, v, tensor_layout="HND", is_causal=False)
    _check(actual, expected, "compile autotuned")
