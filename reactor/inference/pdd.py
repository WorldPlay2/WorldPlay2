# Modified for the Reactor runtime.
"""Upstream four-step schedule with device-aware checkpoint loading."""
from __future__ import annotations

import torch
from worldplay2.pdd import FixedPDD4Scheduler as NativeFixedPDD4Scheduler

from .checkpoint import load_state_dict


class FixedPDD4Scheduler(NativeFixedPDD4Scheduler):
    def prepare_checkpoint(
        self,
        path: str,
        *,
        expert: str,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load either interval-head or already compact fixed-PDD weights.

        Tensors are read onto ``device``. An interval head bank is compacted
        on the host, whatever ``device`` is, and the result moved to ``device``.
        """
        state = load_state_dict(path, device=device)
        compact_weight = "head.block_weight"
        compact_bias = "head.block_bias"
        if compact_weight not in state or compact_bias not in state:
            weight_key = next((
                key for key in (
                    "head.weight",
                    "head.pdd_weight",
                    "head.head.weight",
                )
                if key in state and state[key].ndim == 3
            ), None)
            bias_key = next((
                key for key in (
                    "head.bias",
                    "head.pdd_bias",
                    "head.head.bias",
                )
                if key in state and state[key].ndim == 2
            ), None)
            if weight_key is None or bias_key is None:
                raise KeyError(
                    f"{path} has neither compact fixed-PDD heads nor an "
                    "interval PDD head bank"
                )
            block_weight, block_bias = self.compact_head(
                expert,
                state.pop(weight_key).cpu(),
                state.pop(bias_key).cpu(),
            )
            state[compact_weight] = block_weight.to(device)
            state[compact_bias] = block_bias.to(device)

        # A compact resume can retain a now-unused standard projection.
        state.pop("head.head.weight", None)
        state.pop("head.head.bias", None)
        for key, tensor in state.items():
            if tensor.is_floating_point() and tensor.dtype != dtype:
                state[key] = tensor.to(dtype=dtype)
        return state
