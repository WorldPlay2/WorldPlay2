"""FlashAttention 4 for WorldPlay2's non-causal attention."""

from __future__ import annotations

import torch
from flash_attn.cute import flash_attn_func

from ..inference import model as wp2_model

_ORIGINAL: dict[str, object] = {}


def fa4(q, k, v, softmax_scale=None):
    """Non-causal attention on ``[B, S, H, D]`` bf16/fp16 tensors."""
    out = flash_attn_func(q, k, v, softmax_scale=softmax_scale)
    return out[0] if isinstance(out, tuple) else out


def _dense(q, k, v, *, softmax_scale=None, causal=False):
    if causal:
        return _ORIGINAL["flash_attention_dense"](q, k, v, softmax_scale=softmax_scale, causal=causal)
    return fa4(q, k, v, softmax_scale)


def _varlen(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
            causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    if (q_lens is None and k_lens is None and not causal and q_scale is None and dropout_p == 0
            and tuple(window_size) == (-1, -1) and q.dtype in (torch.bfloat16, torch.float16)):
        return fa4(q, k, v, softmax_scale).type(q.dtype)
    return _ORIGINAL["flash_attention"](
        q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
        q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic,
        dtype=dtype, version=version)


def install(pipeline) -> None:
    if not _ORIGINAL:
        _ORIGINAL["flash_attention_dense"] = wp2_model.flash_attention_dense
        _ORIGINAL["flash_attention"] = wp2_model.flash_attention
        wp2_model.flash_attention_dense = _dense
        wp2_model.flash_attention = _varlen
    for expert in (pipeline.high_noise_model, pipeline.low_noise_model):
        for block in pipeline._find_wan_model(expert).blocks:
            # The cache write runs on FA4 only where it is split into unmasked parts.
            if block.self_attn.prefill_attention == "split":
                _ORIGINAL.setdefault("prefill_attention", {})[block.self_attn] = "split"
                block.self_attn.prefill_attention = "flash"


def uninstall() -> None:
    if not _ORIGINAL:
        return
    wp2_model.flash_attention_dense = _ORIGINAL.pop("flash_attention_dense")
    wp2_model.flash_attention = _ORIGINAL.pop("flash_attention")
    for module, mode in _ORIGINAL.pop("prefill_attention", {}).items():
        module.prefill_attention = mode
