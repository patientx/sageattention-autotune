#pragma once

#include "qk_int8_sv_f16_kernel_sm80.cuh"
#include "../dispatch_utils.h"

#include <stdexcept>

enum class TensorLayout {
  kNHD,
  kHND,
};

inline TensorLayout parse_tensor_layout(const int64_t tensor_layout)
{
  if (tensor_layout == 0)
  {
    return TensorLayout::kNHD;
  }
  if (tensor_layout == 1)
  {
    return TensorLayout::kHND;
  }
  throw std::invalid_argument("tensor_layout must be 0 (NHD) or 1 (HND)");
}

struct Sm80QkLaunchParams {
  int head_dim;
  int batch_size;
  int qo_len;
  int kv_len;
  int num_qo_heads;
  int num_kv_heads;
  int num_kv_groups;
  int stride_bz_q;
  int stride_seq_q;
  int stride_h_q;
  int stride_bz_k;
  int stride_seq_k;
  int stride_h_k;
  int stride_bz_v;
  int stride_seq_v;
  int stride_h_v;
  int stride_bz_o;
  int stride_seq_o;
  int stride_h_o;
  Tensor lse;
};

inline Sm80QkLaunchParams prepare_sm80_qk_launch_params(const Tensor &query,
                                                        const Tensor &key,
                                                        const Tensor &value,
                                                        const Tensor &output,
                                                        const Tensor &query_scale,
                                                        const Tensor &key_scale,
                                                        const TensorLayout tensor_layout,
                                                        const bool return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(key, torch::headeronly::ScalarType::Char);
  CHECK_DTYPE(value, torch::headeronly::ScalarType::Half);
  CHECK_DTYPE(query_scale, torch::headeronly::ScalarType::Float);
  CHECK_DTYPE(key_scale, torch::headeronly::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  Sm80QkLaunchParams params = {
    static_cast<int>(query.size(3)),
    static_cast<int>(query.size(0)),
    0,
    0,
    0,
    0,
    0,
    static_cast<int>(query.stride(0)),
    0,
    0,
    static_cast<int>(key.stride(0)),
    0,
    0,
    static_cast<int>(value.stride(0)),
    0,
    0,
    static_cast<int>(output.stride(0)),
    0,
    0,
    torch::stable::new_empty(query, {0}, std::make_optional(torch::headeronly::ScalarType::Float)),
  };

  if (tensor_layout == TensorLayout::kNHD)
  {
    params.qo_len = static_cast<int>(query.size(1));
    params.kv_len = static_cast<int>(key.size(1));
    params.num_qo_heads = static_cast<int>(query.size(2));
    params.num_kv_heads = static_cast<int>(key.size(2));
    CHECK_SHAPE(key, params.batch_size, params.kv_len, params.num_kv_heads, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.kv_len, params.num_kv_heads, params.head_dim);

    params.stride_seq_q = static_cast<int>(query.stride(1));
    params.stride_seq_k = static_cast<int>(key.stride(1));
    params.stride_seq_v = static_cast<int>(value.stride(1));
    params.stride_seq_o = static_cast<int>(output.stride(1));

    params.stride_h_q = static_cast<int>(query.stride(2));
    params.stride_h_k = static_cast<int>(key.stride(2));
    params.stride_h_v = static_cast<int>(value.stride(2));
    params.stride_h_o = static_cast<int>(output.stride(2));
  }
  else if (tensor_layout == TensorLayout::kHND)
  {
    params.qo_len = static_cast<int>(query.size(2));
    params.kv_len = static_cast<int>(key.size(2));
    params.num_qo_heads = static_cast<int>(query.size(1));
    params.num_kv_heads = static_cast<int>(key.size(1));
    CHECK_SHAPE(key, params.batch_size, params.num_kv_heads, params.kv_len, params.head_dim);
    CHECK_SHAPE(value, params.batch_size, params.num_kv_heads, params.kv_len, params.head_dim);

    params.stride_seq_q = static_cast<int>(query.stride(2));
    params.stride_seq_k = static_cast<int>(key.stride(2));
    params.stride_seq_v = static_cast<int>(value.stride(2));
    params.stride_seq_o = static_cast<int>(output.stride(2));

    params.stride_h_q = static_cast<int>(query.stride(1));
    params.stride_h_k = static_cast<int>(key.stride(1));
    params.stride_h_v = static_cast<int>(value.stride(1));
    params.stride_h_o = static_cast<int>(output.stride(1));
  }

  if (params.num_qo_heads % params.num_kv_heads != 0) {
    STD_TORCH_CHECK(false, "num_qo_heads (", params.num_qo_heads, ") must be divisible by num_kv_heads (", params.num_kv_heads, ")");
  }

  params.num_kv_groups = params.num_qo_heads / params.num_kv_heads;
  params.lse = return_lse
    ? torch::stable::new_empty(query, {params.batch_size, params.num_qo_heads, params.qo_len}, std::make_optional(torch::headeronly::ScalarType::Float))
    : torch::stable::new_empty(query, {0}, std::make_optional(torch::headeronly::ScalarType::Float));

  return params;
}

struct Sm80QkLaunchContext {
  const Tensor &query;
  const Tensor &key;
  const Tensor &value;
  const Tensor &output;
  const Tensor &query_scale;
  const Tensor &key_scale;
  const Sm80QkLaunchParams &params;
  const void *value_mean;
  double sm_scale;
  int64_t blk_q;
  int64_t blk_k;
  int64_t warp_q;
  int64_t warp_k;
  bool is_causal;
  bool return_lse;
};

template <int HeadDim, bool IsCausal, bool ReturnLse, typename DTypeOut, int CtaQ, int CtaK, int WarpQ, int WarpK, typename DTypeSVAccum, bool UseInstBuffer, ComputeUnit DenominatorAccumUnit, bool FuseVMean>
void launch_sm80_qk_kernel(const Sm80QkLaunchContext &ctx)
{
  CHECK_SHAPE(ctx.query_scale, ctx.params.batch_size, ctx.params.num_qo_heads, div_ceil(ctx.params.qo_len, CtaQ) * (CtaQ / WarpQ) * 8);
  CHECK_SHAPE(ctx.key_scale, ctx.params.batch_size, ctx.params.num_kv_heads, div_ceil(ctx.params.kv_len, CtaK) * (CtaK / WarpK) * 4);

  constexpr MaskMode mask_mode = IsCausal ? MaskMode::kCausal : MaskMode::kNone;
  const size_t smem_max = std::max(CtaQ * HeadDim * sizeof(int8_t) + CtaK * HeadDim * sizeof(int8_t) + CtaK * HeadDim * sizeof(half), CtaQ * HeadDim * sizeof(half));
  auto kernel_func = qk_int8_sv_f16_attn_kernel<CtaQ, CtaK, WarpQ, WarpK, HeadDim, DTypeSVAccum, UseInstBuffer, DTypeOut, DenominatorAccumUnit, mask_mode, ReturnLse, FuseVMean>;
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  const dim3 grid(div_ceil(ctx.params.qo_len, CtaQ), ctx.params.num_qo_heads, ctx.params.batch_size);
  const dim3 block(32, (CtaQ / WarpQ) * (CtaK / WarpK));
  const auto device_guard = make_device_guard(ctx.query);
  const auto stream = get_current_cuda_stream(ctx.query);

  kernel_func<<<grid, block, smem_max, stream>>>(
    const_ptr<int8_t>(ctx.query),
    const_ptr<int8_t>(ctx.key),
    const_ptr<half>(ctx.value),
    mutable_ptr<DTypeOut>(ctx.output),
    ReturnLse ? mutable_ptr<float>(ctx.params.lse) : nullptr,
    const_ptr<float>(ctx.query_scale),
    const_ptr<float>(ctx.key_scale),
    static_cast<const DTypeOut*>(ctx.value_mean),
    ctx.params.qo_len,
    ctx.params.kv_len,
    ctx.params.num_kv_groups,
    ctx.params.stride_bz_q, ctx.params.stride_seq_q, ctx.params.stride_h_q,
    ctx.params.stride_bz_k, ctx.params.stride_seq_k, ctx.params.stride_h_k,
    ctx.params.stride_bz_v, ctx.params.stride_seq_v, ctx.params.stride_h_v,
    ctx.params.stride_bz_o, ctx.params.stride_seq_o, ctx.params.stride_h_o,
    ctx.sm_scale);
}

template <int HeadDim, bool IsCausal, bool ReturnLse, typename DTypeOut, typename DTypeSVAccum, bool UseInstBuffer, ComputeUnit DenominatorAccumUnit, bool FuseVMean>
void launch_configured_sm80_qk_kernel(const Sm80QkLaunchContext &ctx)
{
  if (ctx.blk_q == 128 && ctx.blk_k == 64 && ctx.warp_q == 32 && ctx.warp_k == 64)
  {
    launch_sm80_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 128, 64, 32, 64, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVMean>(ctx);
  }
  else if (ctx.blk_q == 128 && ctx.blk_k == 32 && ctx.warp_q == 32 && ctx.warp_k == 32)
  {
    if constexpr (!IsCausal)
    {
      launch_sm80_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 128, 32, 32, 32, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVMean>(ctx);
    }
    else
    {
      throw std::invalid_argument("blk_q=128 blk_k=32 is not supported for causal attention");
    }
  }
  else if (ctx.blk_q == 64 && ctx.blk_k == 64 && ctx.warp_q == 32 && ctx.warp_k == 64)
  {
    launch_sm80_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 64, 64, 32, 64, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVMean>(ctx);
  }
  else if (ctx.blk_q == 128 && ctx.blk_k == 64 && ctx.warp_q == 16 && ctx.warp_k == 64)
  {
    launch_sm80_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, 128, 64, 16, 64, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVMean>(ctx);
  }
  else
  {
    throw std::invalid_argument("Unsupported blk_q/blk_k/warp_q/warp_k configuration");
  }
}

template <typename DTypeSVAccum, bool UseInstBuffer, ComputeUnit DenominatorAccumUnit, bool FuseVMean>
Tensor run_sm80_qk_attn(const Tensor &query,
                        const Tensor &key,
                        const Tensor &value,
                        const Tensor &output,
                        const Tensor &query_scale,
                        const Tensor &key_scale,
                        const Tensor *value_mean,
                        const int64_t tensor_layout,
                        const bool is_causal,
                        const double sm_scale,
                        const int64_t blk_q,
                        const int64_t blk_k,
                        const int64_t warp_q,
                        const int64_t warp_k,
                        const bool return_lse)
{
  const auto layout = parse_tensor_layout(tensor_layout);
  const auto params = prepare_sm80_qk_launch_params(query, key, value, output, query_scale, key_scale, layout, return_lse);
  const void *value_mean_ptr = nullptr;

  if constexpr (FuseVMean)
  {
    STD_TORCH_CHECK(value_mean != nullptr, "value_mean is required when fusing V mean");
    const Tensor &value_mean_tensor = *value_mean;
    CHECK_CUDA(value_mean_tensor);
    CHECK_CONTIGUOUS(value_mean_tensor);
    CHECK_DIMS(value_mean_tensor, 3);
    STD_TORCH_CHECK(value_mean_tensor.scalar_type() == output.scalar_type(), "value_mean and output must have the same dtype");
    CHECK_SHAPE(value_mean_tensor, params.batch_size, params.num_kv_heads, params.head_dim);
    value_mean_ptr = value_mean_tensor.const_data_ptr();
  }

  const Sm80QkLaunchContext ctx{
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    params,
    value_mean_ptr,
    sm_scale,
    blk_q,
    blk_k,
    warp_q,
    warp_k,
    is_causal,
    return_lse,
  };

  sageattention::dispatch::fp16_dtype(ctx.output.scalar_type(), [&]<typename DTypeOut>() {
    sageattention::dispatch::head_dim(ctx.params.head_dim, [&]<int HeadDim>() {
      sageattention::dispatch::boolean(ctx.is_causal, [&]<bool IsCausal>() {
        sageattention::dispatch::boolean(ctx.return_lse, [&]<bool ReturnLse>() {
          launch_configured_sm80_qk_kernel<HeadDim, IsCausal, ReturnLse, DTypeOut, DTypeSVAccum, UseInstBuffer, DenominatorAccumUnit, FuseVMean>(ctx);
        });
      });
    });
  });
  return params.lse;
}
