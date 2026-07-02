import functools

import torch
import triton

from ..autotune_utils import _shared_memory_limit
from ..utils import _padded_head_dim

_TRITON_ATTN_CONFIGS = (
    # (num_warps, num_stages)
    (4, 2),
    (4, 3),
    (4, 4),
    # num_warps = 8 is currently disabled because it does not help much
    # (8, 2),
    # (8, 3),
    # (8, 4),
)


def _estimated_smem_bytes(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_stages: int,
    is_causal: bool,
) -> int:
    int8_bytes = 1
    fp16_bytes = 2
    pipeline_prologue_stages = 1
    stage_bookkeeping_bytes = 4
    min_smem_bytes = 8 * 1024
    causal_mask_smem_slack_bytes = 16

    head_dim = _padded_head_dim(head_dim)

    # Triton reports shared memory as a linear function of pipeline stages for this kernel.
    # Each extra live K/V pipeline stage materializes one K int8 tile and one V fp16 tile.
    # PV_ACCUM_FP32 changes registers/runtime, but not the reported shared memory.
    operand_tile_bytes = head_dim * (block_m + block_n)
    kv_pipeline_stage_bytes = head_dim * block_n * (int8_bytes + fp16_bytes)
    live_kv_pipeline_stages = max(num_stages - pipeline_prologue_stages, 1)

    estimated = operand_tile_bytes + live_kv_pipeline_stages * (kv_pipeline_stage_bytes + stage_bookkeeping_bytes)
    estimated = max(estimated, min_smem_bytes)
    if is_causal:
        estimated += causal_mask_smem_slack_bytes
    return estimated


@functools.cache
def _attn_config_is_valid(
    block_m: int,
    block_n: int,
    num_stages: int,
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> bool:
    if is_causal and block_m % block_n != 0:
        return False
    return _estimated_smem_bytes(block_m, block_n, head_dim, num_stages, is_causal) <= _shared_memory_limit(
        device_index
    )


def _prune_attn_configs(
    configs: list[triton.Config], named_args: dict[str, object], **meta: object
) -> list[triton.Config]:
    q = named_args["Q"]
    assert isinstance(q, torch.Tensor)
    block_m = meta["BLOCK_M"]
    block_n = meta["BLOCK_N"]
    head_dim = meta["HEAD_DIM"]
    is_causal = meta["IS_CAUSAL"]
    assert isinstance(block_m, int)
    assert isinstance(block_n, int)
    assert isinstance(head_dim, int)
    assert isinstance(is_causal, bool)
    return [
        config
        for config in configs
        if _attn_config_is_valid(block_m, block_n, config.num_stages, head_dim, is_causal, q.device.index)
    ]


@functools.cache
def _valid_attn_configs(
    block_config: tuple[int, int],
    head_dim: int,
    is_causal: bool,
    device_index: int,
) -> tuple[tuple[int, int], ...]:
    block_m, block_n = block_config
    return tuple(
        attn_config
        for attn_config in _TRITON_ATTN_CONFIGS
        if _attn_config_is_valid(block_m, block_n, attn_config[1], head_dim, is_causal, device_index)
    )
