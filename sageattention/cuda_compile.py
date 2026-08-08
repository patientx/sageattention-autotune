import importlib

import torch

importlib.import_module(f"{__package__}._qattn_sm80")
_qattn_sm80 = torch.ops.sageattention_qattn_sm80


def _empty_lse(query: torch.Tensor, tensor_layout: int, return_lse: bool) -> torch.Tensor:
    batch_size = query.size(0)

    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    else:
        lse = torch.empty((0,), dtype=torch.float32, device=query.device)
    return lse


def _fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: bool,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
    return_lse: bool,
) -> torch.Tensor:
    return _empty_lse(query, tensor_layout, return_lse)


torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f32_attn")(_fake_impl)
torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn")(_fake_impl)
torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn_inst_buf")(_fake_impl)


@torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_fuse_v_mean_attn")
def _qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_mean: torch.Tensor,
    tensor_layout: int,
    is_causal: bool,
    sm_scale: float,
    blk_q: int,
    blk_k: int,
    warp_q: int,
    warp_k: int,
    return_lse: bool,
) -> torch.Tensor:
    return _empty_lse(query, tensor_layout, return_lse)
