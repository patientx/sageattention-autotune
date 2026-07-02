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

from ..autotune_utils import _autotune_seq_len_bucket
from .attn_autotune import _TRITON_ATTN_CONFIGS, _prune_attn_configs

LOG2_E = 1.44269504088896340736


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    sm_scale,
    kv_len,
    K_ptrs,
    K_scale_ptr,
    V_ptrs,
    stride_kn,
    stride_vn,
    start_m,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    PV_ACCUM_FP32: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    offs_m,
    offs_n,
):
    if IS_CAUSAL:
        if STAGE == 1:
            lo, hi = 0, start_m * BLOCK_M
        else:
            lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
            lo = tl.multiple_of(lo, BLOCK_M)
            K_scale_ptr += lo // BLOCK_N
            K_ptrs += stride_kn * lo
            V_ptrs += stride_vn * lo
    else:
        lo, hi = 0, kv_len

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_EVEN_N:
            k = tl.load(K_ptrs)
        else:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale * sm_scale)

        if IS_CAUSAL:
            if STAGE == 2:
                qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        elif not IS_EVEN_N:
            qk = tl.where(k_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if IS_EVEN_N:
            v = tl.load(V_ptrs)
        else:
            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n), other=0.0)
        p = p.to(tl.float16)
        if PV_ACCUM_FP32:
            acc += tl.dot(p, v, out_dtype=tl.float32)
        else:
            acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij

        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps, num_stages in _TRITON_ATTN_CONFIGS
    ],
    key=[
        "H",
        "num_kv_groups",
        "HEAD_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "RETURN_LSE",
        "IS_CAUSAL",
        "PV_ACCUM_FP32",
        "IS_EVEN_M",
        "IS_EVEN_N",
        "Q_BUCKET",
        "K_BUCKET",
    ],
    prune_configs_by={"early_config_prune": _prune_attn_configs},
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    Lse,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    qo_len,
    kv_len,
    sm_scale,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    PV_ACCUM_FP32: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    Q_BUCKET: tl.constexpr,
    K_BUCKET: tl.constexpr,
):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = (
        K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    )
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = (
        V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    )
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_EVEN_M:
        q = tl.load(Q_ptrs)
    else:
        q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)

    if IS_CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            sm_scale,
            kv_len,
            K_ptrs,
            K_scale_ptr,
            V_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            1,
            IS_CAUSAL,
            PV_ACCUM_FP32,
            IS_EVEN_M,
            IS_EVEN_N,
            offs_m,
            offs_n,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            sm_scale,
            kv_len,
            K_ptrs,
            K_scale_ptr,
            V_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            IS_CAUSAL,
            PV_ACCUM_FP32,
            IS_EVEN_M,
            IS_EVEN_N,
            offs_m,
            offs_n,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            sm_scale,
            kv_len,
            K_ptrs,
            K_scale_ptr,
            V_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            0,
            IS_CAUSAL,
            PV_ACCUM_FP32,
            IS_EVEN_M,
            IS_EVEN_N,
            offs_m,
            offs_n,
        )

    acc = acc / l_i[:, None]
    if IS_EVEN_M:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        if IS_EVEN_M:
            tl.store(lse_ptrs, l_i)
        else:
            tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    pv_accum_dtype: str,
    BLOCK_M: int,
    BLOCK_N: int,
    output_dtype: torch.dtype,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(q.shape, device=q.device, dtype=output_dtype)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    sm_scale = head_dim**-0.5

    if is_causal and qo_len != kv_len:
        raise ValueError("qo_len and kv_len must be equal for causal attention")

    if h_qo % h_kv != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], device=q.device, dtype=torch.float32)
    else:
        lse = torch.empty([0], device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        o,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        qo_len,
        kv_len,
        sm_scale * LOG2_E,
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        RETURN_LSE=return_lse,
        IS_CAUSAL=is_causal,
        PV_ACCUM_FP32=(pv_accum_dtype == "fp32"),
        IS_EVEN_M=(qo_len % BLOCK_M == 0),
        IS_EVEN_N=(kv_len % BLOCK_N == 0),
        Q_BUCKET=_autotune_seq_len_bucket(qo_len),
        K_BUCKET=_autotune_seq_len_bucket(kv_len),
    )

    return o, lse
