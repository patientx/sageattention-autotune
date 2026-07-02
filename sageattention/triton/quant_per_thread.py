"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

from .quant_per_block import _quantize_tile_to_int8, _round_to_int8


@triton.jit
def quant_query_per_thread_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    mask = offs_n[:, None] < L
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_int8, scale = _quantize_tile_to_int8(x)

    tl.store(output_ptrs, x_int8, mask=mask)
    tl.store(scale_ptrs, scale)


@triton.jit
def quant_key_per_thread_int8_kernel(
    Input,
    Mean,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_mz,
    stride_mh,
    stride_mk,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    C: tl.constexpr,
    BLK: tl.constexpr,
    HAS_MEAN: tl.constexpr,
):
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
    offs_n1 = offs_n0 + 1
    offs_k = tl.arange(0, C)

    input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + offs_k[None, :]
    input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + offs_k[None, :]
    output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None] * stride_on + offs_k[None, :]
    output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    mask0 = offs_n0[:, None] < L
    mask1 = offs_n1[:, None] < L
    x0 = tl.load(input_ptrs0, mask=mask0, other=0.0).to(tl.float32)
    x1 = tl.load(input_ptrs1, mask=mask1, other=0.0).to(tl.float32)

    if HAS_MEAN:
        mean_ptrs = Mean + off_b * stride_mz + off_h * stride_mh + offs_k * stride_mk
        mean = tl.load(mean_ptrs).to(tl.float32)
        x0 = tl.where(mask0, x0 - mean[None, :], 0.0)
        x1 = tl.where(mask1, x1 - mean[None, :], 0.0)

    scale = tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127.5 + 1e-7
    x0_int8 = _round_to_int8(x0 / scale)
    x1_int8 = _round_to_int8(x1 / scale)

    tl.store(output_ptrs0, x0_int8, mask=mask0)
    tl.store(output_ptrs1, x1_int8, mask=mask1)
    tl.store(scale_ptrs, scale)


def per_thread_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: torch.Tensor | None,
    BLKQ: int,
    WARPQ: int,
    BLKK: int,
    WARPK: int,
    tensor_layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    k_int8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
        if km is not None:
            km = km.squeeze(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
        if km is not None:
            km = km.squeeze(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    has_mean = km is not None
    mean = km if has_mean else k
    stride_bz_m, stride_h_m, stride_k_m = (mean.stride(0), mean.stride(1), mean.stride(2)) if has_mean else (0, 0, 0)

    q_blocks = triton.cdiv(qo_len, BLKQ)
    k_blocks = triton.cdiv(kv_len, BLKK)

    q_scale = torch.empty((b, h_qo, q_blocks * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, k_blocks * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    grid = (q_blocks * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        C=head_dim,
        BLK=WARPQ,
    )

    grid = (k_blocks * (BLKK // WARPK) * 4, h_kv, b)
    quant_key_per_thread_int8_kernel[grid](
        k,
        mean,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_m,
        stride_h_m,
        stride_k_m,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        C=head_dim,
        BLK=WARPK,
        HAS_MEAN=has_mean,
    )

    return q_int8, q_scale, k_int8, k_scale
