# sageattention-autotune

Port of [SageAttention](https://github.com/thu-ml/SageAttention) with autotuned block sizes and other quality-of-life improvements.

In the master branch I've ported the SageAttention 2 CUDA sm80 kernels and the Triton kernels, and the autotune configs are mostly optimized for RTX 30xx. The CUDA sm89 kernels for RTX 40xx/50xx are in `sm89` branch.

This repo also serves as an example of how to do autotune when multiple kernels (like quant kernel and attn kernel) need consistent parameters (like block sizes).

## Installation

This is experimental. You should know what you're doing. If not, maybe this is not for you and you can install my [SageAttention Windows wheel](https://github.com/woct0rdho/SageAttention) instead.

This repo builds a package with pip package name `sageattention` and import name `sageattention`. It's a drop-in replacement of the official one.

PyTorch >= 2.12 is required for the latest fixes with `torch.compile`.

## Usage

The APIs `sageattn_qk_int8_pv_fp16_cuda` and `sageattn_qk_int8_pv_fp16_triton` are provided. `sageattn` is aliased to CUDA by default, and Triton if the env var `SAGEATTN_BACKEND=triton`.

The official SageAttention's Triton kernel behave like `pv_accum_dtype="fp16"`. I've added `pv_accum_dtype="fp32"` to the Triton kernel. `pv_accum_dtype="fp16+fp32"` is still not supported in the Triton kernel.

You can set the default `pv_accum_dtype` using the env var like `SAGEATTN_DEFAULT_PV_ACCUM_DTYPE=fp16`. As always, `pv_accum_dtype="fp16+fp32"` is faster than `pv_accum_dtype="fp32"`, and `pv_accum_dtype="fp16"` is even faster, but more likely to cause black/noise/degraded output. Maybe the time has come that we have to tune it for each model.

## Development

After each change, run `pre-commit` and `pytest tests/`.
