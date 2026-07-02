import argparse
import logging

import torch
from flash_attn import flash_attn_func
from flash_attn.utils.benchmark import benchmark_forward
from torch.nn.functional import scaled_dot_product_attention as sdpa

from sageattention import sageattn

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="sage", choices=["sdpa", "flash", "sage"])
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--head_dim", type=int, default=64)
parser.add_argument("--seq_lens", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()

num_heads = args.num_heads
batch_size = args.batch_size
head_dim = args.head_dim

print(f"method: {args.method}")
print(f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}")


def run_benchmark(is_causal: bool) -> None:
    print(f"is_causal: {is_causal}")
    for seq_len in args.seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len // (2 if is_causal else 1)

        # Generate inputs in BHSD format (Batch, Head, Seq, Dim)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        if args.method in ("flash", "sage"):
            # flash_attn_func and SageAttention NHD expect (Batch, Seq, Head, Dim)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        if args.method == "flash":

            def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return flash_attn_func(q, k, v, causal=is_causal)
        elif args.method == "sage":

            def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                assert sageattn is not None
                return sageattn(q, k, v, tensor_layout="NHD", is_causal=is_causal)
        else:

            def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return sdpa(q, k, v, is_causal=is_causal)

        for _ in range(args.warmup):
            fn(q, k, v)
        torch.cuda.synchronize()

        _, time = benchmark_forward(fn, q, k, v, repeats=args.repeats, verbose=False, desc=args.method)
        print(f"{seq_len} ms:{time.mean * 1e3:.3f} flops:{flops / time.mean * 1e-12:.3f}")


run_benchmark(is_causal=False)
run_benchmark(is_causal=True)
