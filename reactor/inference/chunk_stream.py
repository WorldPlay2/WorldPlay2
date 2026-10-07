"""Per-chunk streaming helpers for ``WorldPlay2Pipeline.stream_world``."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .vae import _same_causal_state, amp


class FrameEncoder:
    """One step of the causal VAE encoder: pixel frames -> one y latent ``[16, 1, h, w]``
    in the VAE's dtype, carrying the causal state passed in."""

    def __init__(self, vae):
        self.vae = vae

    def new_state(self):
        from .vae import count_conv3d

        return [None] * count_conv3d(self.vae.model.encoder)

    def step(self, x, state):
        model = self.vae.model
        model._enc_conv_idx = [0]
        out = model.encoder(x.to(self.vae.dtype), feat_cache=state, feat_idx=model._enc_conv_idx)
        mu, _ = model.conv1(out).chunk(2, dim=1)
        scale = self.vae.scale
        mu = (mu - scale[0].view(1, model.z_dim, 1, 1, 1)) * scale[1].view(1, model.z_dim, 1, 1, 1)
        return mu.to(self.vae.dtype).squeeze(0)

    def agree(self, flag):
        return flag


def _y_row(latent, first, dtype):
    mask = torch.zeros(4, 1, *latent.shape[2:], device=latent.device, dtype=latent.dtype)
    if first:
        mask.fill_(1)
    return torch.cat([mask, latent], dim=0).to(dtype)


class _CausalY:
    """``y = [4-ch I2V mask | VAE latent of the clip [image, zeros...]]`` produced latent by latent.

    Indexed like the full-horizon volume (``y[:, a:b]``).
    """

    def __init__(self, encoder, head, tail, dtype):
        self.encoder, self.head, self.dtype = encoder, head.to(encoder.vae.dtype), dtype
        self.fixed_state, self.tail = tail
        self.state = None
        self.rows = []
        self.converged = False

    def _extend(self, upto):
        vae = self.encoder.vae
        with amp.autocast(dtype=vae.dtype):
            while not self.converged and len(self.rows) < upto:
                k = len(self.rows)
                if k == 0:
                    self.state = self.encoder.new_state()
                    latent = self.encoder.step(self.head, self.state)
                else:
                    zeros = self.head.new_zeros(*self.head.shape[:2], 4, *self.head.shape[3:])
                    latent = self.encoder.step(zeros, self.state)
                self.rows.append(_y_row(latent, k == 0, self.dtype))
                if k > 1 and self.encoder.agree(_same_causal_state(self.fixed_state, self.state)):
                    self.converged = True
                    self.state = None

    def __getitem__(self, index):
        _, sl = index
        start, stop = sl.start or 0, sl.stop
        self._extend(stop)
        known = len(self.rows)
        parts = [self.rows[k] for k in range(start, min(stop, known))]
        if stop > known:
            parts.append(self.tail.expand(-1, stop - max(start, known), -1, -1))
        return torch.cat(parts, dim=1)


def zero_tail(encoder, height, width, dtype, steps_cap=256):
    """Encoder state and y latent for the zero frames of clips of ``height x width``."""
    vae = encoder.vae
    device = vae.mean.device
    head = torch.linspace(-1, 1, height * width * 3, device=device,
                          dtype=vae.dtype).view(1, 3, 1, height, width)
    with amp.autocast(dtype=vae.dtype):
        state = encoder.new_state()
        zeros = head.new_zeros(1, 3, 4, height, width)
        encoder.step(head, state)
        for step in range(1, steps_cap):
            before = [v.clone() if isinstance(v, torch.Tensor) else v for v in state]
            latent = encoder.step(zeros, state)
            if step > 1 and encoder.agree(_same_causal_state(before, state)):
                logging.info("I2V zero tail for %dx%d: fixed point after %d encoder steps", height, width, step)
                return ([v.clone() if isinstance(v, torch.Tensor) else v for v in state],
                        _y_row(latent, False, dtype))
    raise RuntimeError("the VAE encoder did not reach a zero-input fixed point")


class _LatentWindow:
    """The trailing ``capacity`` latent frames of a world, indexed by absolute latent frame."""

    def __init__(self, buffers, total):
        self.buffers = buffers                      # two [16, capacity, h, w] fp32
        # Shape of the full-horizon latent volume this window stands in for.
        self.shape = (buffers[0].shape[0], total, *buffers[0].shape[2:])
        self.which = 0
        self.base = 0                               # absolute index of buffer frame 0
        self.length = 0

    def write(self, start, latent):
        n = latent.shape[1]
        cap = self.buffers[0].shape[1]
        if start + n - self.base > cap:
            keep = cap - n
            src = self.buffers[self.which]
            dst = self.buffers[1 - self.which]
            first = start - keep
            dst[:, :keep].copy_(src[:, first - self.base:start - self.base])
            self.which, self.base = 1 - self.which, first
        self.buffers[self.which][:, start - self.base:start - self.base + n].copy_(latent)
        self.length = start + n

    def __getitem__(self, index):
        _, sl = index
        start, stop = sl.start or 0, sl.stop
        if start < self.base:
            raise IndexError(f"latent {start} is outside the kept window (from {self.base})")
        buf = self.buffers[self.which]
        return buf[:, start - self.base:stop - self.base]
