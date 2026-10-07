"""Multi-GPU VAE: every GPU works on one horizontal band of each frame.

Decode: the decoder's low-resolution stages (up to and including the second temporal upsample) run
on the whole frame on every GPU. From there on each GPU runs the remaining layers on its own band of
rows, plus HALO rows of context on either side, which covers the receptive field of those layers.
Rank 0 gathers the bands, re-encodes the low-resolution memory as before, and broadcasts it.

Conditioning encode (the image clip encoded when a world is set up): the mirror image. Each GPU
runs the full-resolution stage of the encoder on its band plus ENC_HALO rows, the bands are
gathered on every GPU, and every GPU runs the rest of the encoder on the whole frame.
"""

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

from ..inference.vae import CACHE_T, CausalConv3d, ResidualBlock

SPLIT = 8   # first decoder.upsamples layer that runs per band (a quarter of the output resolution)
HALO = 12   # context rows on either side of a band, at that resolution
ENC_SPLIT = 3     # encoder.downsamples layers that run per band: the full-resolution stage
ENC_HALO = 16     # context rows on either side of an encoder band, in input pixels (even)


def _cached_conv(conv, x, feat_cache, feat_idx):
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
    x = conv(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


def _front(dec, x, feat_cache, feat_idx):
    x = _cached_conv(dec.conv1, x, feat_cache, feat_idx)
    for layer in dec.middle:
        x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) else layer(x)
    for layer in dec.upsamples[:SPLIT]:
        x = layer(x, feat_cache, feat_idx)
    return x


def _tail(dec, x, feat_cache, feat_idx):
    for layer in dec.upsamples[SPLIT:]:
        x = layer(x, feat_cache, feat_idx)
    for layer in dec.head:
        x = _cached_conv(layer, x, feat_cache, feat_idx) if isinstance(layer, CausalConv3d) else layer(x)
    return x


class BandDecoder:
    def __init__(self, vae):
        self.vae = vae
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

    def decode(self, latent, is_first_chunk):
        """Decode a ``[C, T, H, W]`` latent chunk. Returns the pixels on rank 0, None elsewhere."""
        vae, model = self.vae, self.vae.model
        dec = model.decoder
        with amp.autocast(dtype=vae.dtype):
            if is_first_chunk:
                model.clear_decode_cache()
            mean, inv_std = vae.scale
            z = latent.unsqueeze(0).to(vae.dtype) / inv_std.view(1, model.z_dim, 1, 1, 1) + mean.view(1, model.z_dim, 1, 1, 1)
            x = model.conv2(z)
            frames = []
            for i in range(x.shape[2]):
                feat_idx = [0]
                h = _front(dec, x[:, :, i:i + 1], model._feat_map, feat_idx)
                height = h.shape[3]
                if height % self.world:
                    raise ValueError(f"{height} decoder rows do not split over {self.world} GPUs")
                rows = height // self.world
                lo, hi = self.rank * rows, (self.rank + 1) * rows
                elo, ehi = max(lo - HALO, 0), min(hi + HALO, height)
                out = _tail(dec, h[:, :, :, elo:ehi].contiguous(), model._feat_map, feat_idx)
                scale = out.shape[3] // (ehi - elo)
                frames.append(out[:, :, :, (lo - elo) * scale:(hi - elo) * scale])
            band = torch.cat(frames, 2).to(vae.dtype).clamp_(-1, 1).squeeze(0).contiguous()
        bands = [torch.empty_like(band) for _ in range(self.world)] if self.rank == 0 else None
        dist.gather(band, bands, dst=0)
        return torch.cat(bands, dim=2) if self.rank == 0 else None

    def reset(self):
        self.vae.model.clear_decode_cache()


class BandEncoder:
    """One causal encoder step, split by rows: same interface as ``chunk_stream.FrameEncoder``."""

    def __init__(self, vae):
        self.vae = vae
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

    def new_state(self):
        from ..inference.vae import count_conv3d

        return [None] * count_conv3d(self.vae.model.encoder)

    def step(self, x, state):
        model = self.vae.model
        enc = model.encoder
        x = x.to(self.vae.dtype)
        height = x.shape[3]
        rows = height // 2 // self.world            # output rows of the full-resolution stage
        lo, hi = self.rank * rows, (self.rank + 1) * rows
        elo, ehi = max(2 * lo - ENC_HALO, 0), min(2 * hi + ENC_HALO, height)
        feat_idx = [0]
        h = _cached_conv(enc.conv1, x[:, :, :, elo:ehi].contiguous(), state, feat_idx)
        for layer in enc.downsamples[:ENC_SPLIT]:
            h = layer(h, state, feat_idx)
        band = h[:, :, :, lo - elo // 2:hi - elo // 2].contiguous()
        parts = [torch.empty_like(band) for _ in range(self.world)]
        dist.all_gather(parts, band)
        h = torch.cat(parts, dim=3)
        for layer in enc.downsamples[ENC_SPLIT:]:
            h = layer(h, state, feat_idx)
        for layer in enc.middle:
            h = layer(h, state, feat_idx) if isinstance(layer, ResidualBlock) else layer(h)
        for layer in enc.head:
            h = _cached_conv(layer, h, state, feat_idx) if isinstance(layer, CausalConv3d) else layer(h)
        mu, _ = model.conv1(h).chunk(2, dim=1)
        mean, inv_std = self.vae.scale
        mu = (mu - mean.view(1, model.z_dim, 1, 1, 1)) * inv_std.view(1, model.z_dim, 1, 1, 1)
        return mu.to(self.vae.dtype).squeeze(0)

    def agree(self, flag):
        t = torch.tensor([1 if flag else 0], device=self.vae.mean.device, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())


def _y_encoder(pipeline, world):
    from ..inference.chunk_stream import FrameEncoder

    def make(latent_rows):
        # Split the full-resolution clip only; the low-resolution one is too small to be worth it.
        out_rows = 8 * latent_rows // 2
        if out_rows % world == 0 and out_rows // world >= 2 * ENC_HALO:
            return BandEncoder(pipeline.vae)
        return FrameEncoder(pipeline.vae)

    return make


def install(pipeline, set_attr):
    if not (pipeline.use_sp and pipeline.exact_optimizations and dist.is_initialized()):
        raise RuntimeError("vae_split needs the sequence-parallel pipeline on more than one GPU")
    decoder = BandDecoder(pipeline.vae)

    def decode_chunk_and_lr_encode(hr_latent_chunk, is_first_vae_chunk):
        rgb = decoder.decode(hr_latent_chunk, is_first_vae_chunk)
        if rgb is None:
            channels, frames, height, width = hr_latent_chunk.shape
            lr_lat = torch.empty((channels, frames // 2, height // 4, width // 4),
                                 device=pipeline.device, dtype=pipeline.vae.dtype)
            dist.broadcast(lr_lat, src=0)
            return lr_lat, None
        lr_pix = rgb[:, ::2]
        lr_pix = F.interpolate(lr_pix.permute(1, 0, 2, 3), size=(lr_pix.shape[2] // 4, lr_pix.shape[3] // 4),
                               mode='bilinear', align_corners=False).permute(1, 0, 2, 3).contiguous()
        lr_lat = pipeline.vae.encode_chunk([lr_pix], is_first_chunk=is_first_vae_chunk)[0].contiguous()
        dist.broadcast(lr_lat, src=0)
        return lr_lat, rgb

    set_attr(pipeline, "_decode_chunk_and_lr_encode", decode_chunk_and_lr_encode)
    set_attr(pipeline, "_decode_last_chunk", decoder.decode)
    set_attr(pipeline, "y_band_encoder", _y_encoder(pipeline, decoder.world))
    return decoder
