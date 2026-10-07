# Modified for the Reactor runtime.
"""Preserve zero-stride views when sharding; collectives come from upstream."""
from __future__ import annotations

import torch
from worldplay2.distributed.ulysses import _balanced_splits


def shard_sequence(
    tensor: torch.Tensor | None,
    *,
    global_sequence: int,
    rank: int,
    world_size: int,
    dim: int,
) -> torch.Tensor | None:
    if tensor is None or world_size <= 1:
        return tensor
    splits = _balanced_splits(global_sequence, world_size)
    start = sum(splits[:rank])
    local = tensor.narrow(dim, start, splits[rank])
    # An expanded (zero-stride) tensor is returned as a view.
    return local if tensor.stride(dim) == 0 else local.contiguous()
