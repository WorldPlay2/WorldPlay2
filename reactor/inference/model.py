# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for the Reactor runtime.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from worldplay2.models.attention import (
    flash_attention,
    flash_attention_dense,
    resolve_attention_backend,
    sage_attention,
)
from worldplay2.models.kernels import (
    action_mlp as run_compiled_action_mlp,
    ffn as run_compiled_ffn,
    gated_residual as run_compiled_gated_residual,
    layer_norm_modulation as run_fused_layer_norm_modulation,
    qk_norm_rope as run_qk_norm_rope,
)
from worldplay2.models.fp8 import fp8_autocast
from worldplay2.distributed.ops import get_rank, get_world_size
from worldplay2.distributed.ulysses import (
    gather_sequence,
    output_all_to_all,
    packed_qkv_all_to_all,
)

from .sequence import shard_sequence

__all__ = ['WorldPlay2Model']

# Precision of the sinusoidal timestep embedding and of the eager RoPE products.
TABLE_DTYPE = torch.float64


def _compact_identical_rows(value):
    first = value[:, :1]
    if value.stride(-1) == 1 and torch.equal(
        value.view(torch.uint8), first.expand_as(value).view(torch.uint8)
    ):
        return first.clone().expand_as(value)
    return value


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(TABLE_DTYPE)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(TABLE_DTYPE).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_flat(x, freqs_flat):
    """Apply RoPE given a per-token freqs tensor.

    Used by the memory-compress path where the sequence is a heterogeneous
    concat of context/generate tokens, so a single (F, H, W) grid_sizes
    description is no longer sufficient.

    Args:
        x (Tensor): Shape [B, L, Nh, C_head] (real).
        freqs_flat (Tensor): complex64, shape [L, C_head / 2].

    Returns:
        Tensor: Same shape as x, with RoPE applied (float).
    """
    b, l, n = x.shape[:3]
    x_c = torch.view_as_complex(x.to(TABLE_DTYPE).reshape(b, l, n, -1, 2))
    freqs_i = freqs_flat.view(1, l, 1, -1)
    x_out = torch.view_as_real(x_c * freqs_i).flatten(2)
    return x_out.float().view(b, l, n, -1)


def build_freqs_flat(freqs, grid_size):
    """Build per-token RoPE freqs for a single grid (F, H, W).

    Returns a complex tensor of shape [F*H*W, C_head/2].
    """
    f, h, w = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    c = freqs.size(1)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_i = torch.cat([
        freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, -1)
    return freqs_i


def build_freqs_rows(freqs, frames, h, w):
    """``build_freqs_flat`` restricted to the temporal positions ``frames`` (a slice)."""
    c = freqs.size(1)
    f0, f1, f2 = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    t = f0[frames]
    n = t.shape[0]
    return torch.cat([
        t[:, None, None, :].expand(n, h, w, -1),
        f1[:h][None, :, None, :].expand(n, h, w, -1),
        f2[:w][None, None, :, :].expand(n, h, w, -1),
    ], dim=-1).reshape(n * h * w, -1)


class WorldPlay2RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WorldPlay2LayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WorldPlay2SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_backend = 'flash'

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WorldPlay2RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WorldPlay2RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.compile_qk_rope = True
        self.qk_rope_compile_failed = False
        self.qkv = None
        # Cache-write attention: 'mask', 'bias', 'split' or 'flash'.
        self.prefill_attention = 'mask'
        self.prefill_bias = None

    @staticmethod
    def _kv_slot(kv_cache_entry, like, length):
        """The entry's preallocated K/V storage when ``length`` tokens fit, else None.

        ``kv_cache_entry['_buffers']`` (from ``WorldPlay2Model.init_kv_cache``)
        holds one ``[1, capacity, heads, head_dim]`` K and V per layer.
        """
        slot = kv_cache_entry.get('_buffers')
        if slot is None or like.shape[0] != 1 or length > slot['capacity']:
            return None
        if slot['k'] is None:
            shape = (1, slot['capacity'], *like.shape[2:])
            slot['k'] = torch.empty(shape, device=like.device, dtype=like.dtype)
            slot['v'] = torch.empty(shape, device=like.device, dtype=like.dtype)
        if slot['k'].shape[2:] != like.shape[2:] or slot['k'].dtype != like.dtype:
            return None
        return slot

    @staticmethod
    def _holds(slot, cached):
        return cached is not None and cached.data_ptr() == slot['k'].data_ptr()

    def fuse_qkv_projections(self) -> None:
        """Replace the three self-attention projections with one exact GEMM."""
        if self.qkv is not None:
            return
        if not all(hasattr(self, name) for name in ("q", "k", "v")):
            raise RuntimeError("Q/K/V projections are unavailable for fusion")

        projections = (self.q, self.k, self.v)
        reference = projections[0]
        if any(
            projection.in_features != reference.in_features
            or projection.out_features != reference.out_features
            or (projection.bias is None) != (reference.bias is None)
            for projection in projections[1:]
        ):
            raise RuntimeError("Q/K/V projection shapes are incompatible")

        qkv = nn.Linear(
            reference.in_features,
            3 * reference.out_features,
            bias=reference.bias is not None,
            device=reference.weight.device,
            dtype=reference.weight.dtype,
        )
        with torch.no_grad():
            qkv.weight.copy_(torch.cat(
                [projection.weight for projection in projections], dim=0))
            if qkv.bias is not None:
                qkv.bias.copy_(torch.cat(
                    [projection.bias for projection in projections], dim=0))

        self.qkv = qkv
        # Do not retain duplicate parameters: fusion must not increase FSDP
        # storage or checkpoint memory after installation.
        del self.q
        del self.k
        del self.v

    def forward(self, x, seq_lens, grid_sizes, freqs, freqs_flat=None,
                ar=False, cmp_token_count=0, cmp_chunk_token_count=0,
                tmp_token_count=0,
                kv_cache_entry=None, cache_write=False,
                sink_token_count=0, cache_update=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W).
                Ignored when ``freqs_flat`` is provided.
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            freqs_flat(Tensor, optional): Per-token complex RoPE tensor of shape
                ``[L, C / num_heads / 2]``.  When provided, RoPE is applied via
                ``rope_apply_flat`` (used by the memory-compress path where
                the sequence is a heterogeneous concat of context/generate
                tokens).  When ``None``, falls back to the original grid-based
                ``rope_apply``.
            ar(bool): when True, run the chunk-AR (KV-cache) self-attention.
                Cache behavior is selected by ``cache_write``:
                  * prefill (cache_write=True): ``x`` = ctx tokens only
                    (``[cmp | tmp]``).  A chunk-causal mask is applied (cmp
                    chunk-causal to cmp; tmp -> all cmp+tmp; no pred rows).
                    The post-RoPE ctx K/V are OVERWRITE-stored into
                    ``kv_cache_entry['k_vision'/'v_vision']``.  Returns the
                    ctx output (threaded layer-to-layer by the caller).
                  * cache-hit (cache_write=False): ``x`` = pred tokens only.
                    K/V = ``[cached_ctx | pred]`` (no mask -- pred sees all),
                    via the selected backend. ``cached_*`` may be None on the
                    first chunk (no compressor ctx) -> pred-only attention.
            kv_cache_entry(dict, optional): per-layer ``{'k_vision','v_vision'}``
                slot (see ``WorldPlay2Model.init_kv_cache``).
            cache_write(bool): write/update context cache vs read it for denoise.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if self.qkv is not None:
            q_projection, k_projection, v_projection = self.qkv(x).chunk(
                3, dim=-1)
        else:
            q_projection = self.q(x)
            k_projection = self.k(x)
            v_projection = self.v(x)
        v = v_projection.view(b, s, n, d)

        if freqs_flat is not None:
            if self.qk_norm:
                q, k, compiled = run_qk_norm_rope(
                    q_projection,
                    k_projection,
                    self.norm_q.weight,
                    self.norm_k.weight,
                    freqs_flat,
                    eps=self.eps,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    compile_enabled=(
                        self.compile_qk_rope
                        and not self.qk_rope_compile_failed
                    ),
                )
                if self.compile_qk_rope and not compiled:
                    self.qk_rope_compile_failed = True
            else:
                q = q_projection.view(b, s, n, d)
                k = k_projection.view(b, s, n, d)
                q = rope_apply_flat(q, freqs_flat)
                k = rope_apply_flat(k, freqs_flat)
        else:
            q = self.norm_q(q_projection).view(b, s, n, d)
            k = self.norm_k(k_projection).view(b, s, n, d)
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

        # RoPE is evaluated in FP32 for stability, but attention and packed
        # QKV communication use the value/projection dtype (BF16 in inference).
        # This also prevents torch.stack in packed QKV A2A from promoting V
        # and tripling its communication precision.
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        use_sp = getattr(self, "use_sp", False)
        if use_sp:
            if not torch.all(seq_lens == seq_lens[0]):
                raise ValueError(
                    "sequence parallel requires equal sequence lengths "
                    "within the batch")
            global_sequence = int(seq_lens[0].item())
            q, k, v = packed_qkv_all_to_all(
                q, k, v, global_sequence=global_sequence)
        else:
            global_sequence = q.shape[1]

        def finish_attention(out):
            if use_sp:
                out = output_all_to_all(
                    out,
                    global_sequence=global_sequence,
                    num_heads=self.num_heads,
                )
            return self.o(out.flatten(2))

        if not ar:
            attention_fn = (
                sage_attention
                if self.attention_backend == 'sage' else flash_attention
            )
            x = attention_fn(
                q=q, k=k, v=v, k_lens=seq_lens,
                window_size=self.window_size)
            return finish_attention(x)

        assert kv_cache_entry is not None, (
            "self_attn(ar=True) requires kv_cache_entry "
            "(use WorldPlay2Model.init_kv_cache()).")

        cache_initialized = kv_cache_entry.get('k_vision') is not None
        if not cache_write:
            # Case 1: regular AR denoise. Read the persistent context cache;
            # an empty cache on the very first generated chunk is valid.
            out = self._attend_with_ar_cache(q, k, v, kv_cache_entry)
        elif not cache_initialized:
            # Case 2: first history prefill. Bootstrap the cache as
            # [sink | cmp | tmp], including the sink-specific attention mask.
            out = self._initialize_ar_cache(
                q=q,
                k=k,
                v=v,
                kv_cache_entry=kv_cache_entry,
                sink_token_count=sink_token_count,
                cmp_token_count=cmp_token_count,
                cmp_chunk_token_count=cmp_chunk_token_count,
                tmp_token_count=tmp_token_count,
            )
        else:
            # Case 3: later history prefill. Keep [sink | old_cmp], discard
            # the previous tmp, then append [new_cmp | new_tmp].
            if cache_update is None:
                raise ValueError("AR cache update metadata is required")
            out = self._update_ar_cache(
                q, k, v, kv_cache_entry, cache_update,
                cmp_chunk_token_count)
        return finish_attention(out)

    def _initialize_ar_cache(
        self,
        q,
        k,
        v,
        kv_cache_entry,
        sink_token_count,
        cmp_token_count,
        cmp_chunk_token_count,
        tmp_token_count,
    ):
        """Bootstrap ``[sink | cmp | tmp]`` on the first history prefill."""
        sequence_length = q.shape[1]
        expected_length = (
            sink_token_count + cmp_token_count + tmp_token_count
        )
        if sequence_length != expected_length:
            raise ValueError(
                f"initial AR cache expects {expected_length} tokens, "
                f"got {sequence_length}"
            )

        q_attn = q.transpose(1, 2)
        k_attn = k.transpose(1, 2)
        v_attn = v.transpose(1, 2)
        cmp_end = sink_token_count + cmp_token_count
        if (self.prefill_attention == 'flash' and sequence_length
                and 0 < cmp_token_count <= cmp_chunk_token_count and tmp_token_count):
            # Sink rows see the sink, cmp rows see [sink | cmp], tmp rows see everything.
            parts = []
            if sink_token_count:
                parts.append(flash_attention_dense(
                    q[:, :sink_token_count], k[:, :sink_token_count], v[:, :sink_token_count]))
            parts.append(flash_attention_dense(q[:, sink_token_count:cmp_end], k[:, :cmp_end], v[:, :cmp_end]))
            parts.append(flash_attention_dense(q[:, cmp_end:], k, v))
            out = torch.cat(parts, dim=1).transpose(1, 2)
        elif (self.prefill_attention == 'split' and sequence_length
                and 0 < cmp_token_count <= cmp_chunk_token_count and tmp_token_count):
            # Sink rows see the sink, cmp rows see [sink | cmp], tmp rows see everything.
            backend = SDPBackend.EFFICIENT_ATTENTION
            parts = []
            with sdpa_kernel(backend):
                if sink_token_count:
                    parts.append(F.scaled_dot_product_attention(
                        q_attn[:, :, :sink_token_count], k_attn[:, :, :sink_token_count],
                        v_attn[:, :, :sink_token_count]))
                parts.append(F.scaled_dot_product_attention(
                    q_attn[:, :, sink_token_count:cmp_end], k_attn[:, :, :cmp_end], v_attn[:, :, :cmp_end]))
                parts.append(F.scaled_dot_product_attention(
                    q_attn[:, :, cmp_end:], k_attn, v_attn))
            out = torch.cat(parts, dim=2)
        elif sequence_length == 0:
            out = torch.zeros_like(q_attn)
        elif cmp_chunk_token_count <= 0:
            out = F.scaled_dot_product_attention(
                q_attn, k_attn, v_attn, is_causal=False)
        else:
            mask = torch.zeros(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=q.device,
            )
            if sink_token_count:
                mask[:sink_token_count, :sink_token_count] = True
            for chunk_start in range(
                0, cmp_token_count, cmp_chunk_token_count
            ):
                query_start = sink_token_count + chunk_start
                query_end = min(
                    query_start + cmp_chunk_token_count,
                    sink_token_count + cmp_token_count,
                )
                mask[query_start:query_end, :query_end] = True
            tmp_start = sink_token_count + cmp_token_count
            if tmp_token_count:
                mask[tmp_start:expected_length, :expected_length] = True
            out = F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v_attn,
                attn_mask=mask.unsqueeze(0).unsqueeze(0),
                is_causal=False,
            )

        slot = self._kv_slot(kv_cache_entry, k, sequence_length)
        if slot is not None:
            slot['k'][:, :sequence_length].copy_(k)
            slot['v'][:, :sequence_length].copy_(v)
            kv_cache_entry['k_vision'] = slot['k'][:, :sequence_length]
            kv_cache_entry['v_vision'] = slot['v'][:, :sequence_length]
        else:
            kv_cache_entry['k_vision'] = k.detach()
            kv_cache_entry['v_vision'] = v.detach()
        return out.transpose(1, 2)

    def _update_ar_cache(self, q, k, v, kv_cache_entry, cache_update,
                         cmp_chunk_token_count):
        r"""Replace old tmp and append ``[new_cmp | new_tmp]``.

        ``q/k/v`` contain only ``[new_cmp | new_tmp]``. The persistent prefix
        ``[sink | old_cmp]`` is sliced directly from ``k_vision/v_vision``;
        the previous tmp suffix is discarded. After attention, the cache is
        replaced by ``[cached_prefix | new_cmp | new_tmp]``.
        """
        n_new_cmp = int(cache_update['n_new_cmp'])
        n_tmp = int(cache_update['n_tmp'])
        cached_prefix_tokens = int(cache_update['cached_prefix_tokens'])
        Sq = q.shape[1]
        assert Sq == n_new_cmp + n_tmp, (
            f"ctx prefill expects q = [new_cmp | tmp] "
            f"({n_new_cmp}+{n_tmp}), got {Sq}")

        cached_k = kv_cache_entry.get('k_vision', None)
        cached_v = kv_cache_entry.get('v_vision', None)
        assert cached_k is not None and cached_v is not None, (
            "Incremental prefill requires an existing k_vision/v_vision.")
        assert cached_prefix_tokens <= cached_k.shape[1], (
            f"cached_prefix_tokens ({cached_prefix_tokens}) exceeds cached "
            f"length ({cached_k.shape[1]}).")
        L_pre = cached_prefix_tokens
        L_tot = L_pre + Sq
        slot = self._kv_slot(kv_cache_entry, k, L_tot)
        if slot is not None and self._holds(slot, cached_k):
            slot['k'][:, L_pre:L_tot].copy_(k)
            slot['v'][:, L_pre:L_tot].copy_(v)
            k_full = slot['k'][:, :L_tot]
            v_full = slot['v'][:, :L_tot]
        else:
            pre_k = cached_k[:, :L_pre].to(k.device).to(k.dtype)
            pre_v = cached_v[:, :L_pre].to(v.device).to(v.dtype)
            k_full = torch.cat([pre_k, k], dim=1)
            v_full = torch.cat([pre_v, v], dim=1)

        # Column index landmarks within k_full = [sink | old_cmp | new_cmp | tmp]
        cmp_new_c0 = L_pre                       # first new_cmp column
        cmp_new_c1 = L_pre + n_new_cmp           # end of new_cmp / start of tmp
        tmp_c1 = L_tot                           # end of tmp
        ccc = cmp_chunk_token_count if cmp_chunk_token_count > 0 else n_new_cmp

        if self.prefill_attention == 'flash' and 0 < n_new_cmp <= ccc and n_tmp > 0:
            # new_cmp rows see [:cmp_new_c1], tmp rows see everything.
            out = torch.cat([
                flash_attention_dense(q[:, :n_new_cmp], k_full[:, :cmp_new_c1], v_full[:, :cmp_new_c1]),
                flash_attention_dense(q[:, n_new_cmp:], k_full, v_full),
            ], dim=1)
        elif self.prefill_attention == 'split' and 0 < n_new_cmp <= ccc and n_tmp > 0:
            # new_cmp rows see [:cmp_new_c1], tmp rows see everything.
            backend = SDPBackend.EFFICIENT_ATTENTION
            with sdpa_kernel(backend):
                out_cmp = F.scaled_dot_product_attention(
                    q[:, :n_new_cmp].transpose(1, 2),
                    k_full[:, :cmp_new_c1].transpose(1, 2),
                    v_full[:, :cmp_new_c1].transpose(1, 2))
                out_tmp = F.scaled_dot_product_attention(
                    q[:, n_new_cmp:].transpose(1, 2),
                    k_full.transpose(1, 2), v_full.transpose(1, 2))
            out = torch.cat([out_cmp, out_tmp], dim=2).transpose(1, 2)
        else:
            if self.prefill_attention == 'bias' and self.prefill_bias is not None:
                attn_mask = self.prefill_bias(
                    Sq, L_tot, L_pre, n_new_cmp, n_tmp,
                    cmp_chunk_token_count, q.dtype, q.device)
            else:
                # Build mask over QUERY rows (Sq) x all columns (L_tot).
                attn_mask = torch.zeros(Sq, L_tot, dtype=torch.bool, device=q.device)
                if n_new_cmp > 0:
                    # new_cmp is chunk-causal *within itself* and sees all prefix
                    # (sink + old_cmp), but NOT tmp.  Split new_cmp into cmp-chunks.
                    for rs in range(0, n_new_cmp, ccc):
                        re = min(rs + ccc, n_new_cmp)
                        # columns: everything up to and including this cmp sub-chunk
                        ke = cmp_new_c0 + re
                        ke = min(ke, cmp_new_c1)
                        attn_mask[rs:re, :ke] = True
                if n_tmp > 0:
                    # tmp rows (after new_cmp in q) see EVERYTHING (sink+cmp+tmp).
                    attn_mask[n_new_cmp:Sq, :tmp_c1] = True
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2),
                attn_mask=attn_mask, is_causal=False)
            out = out.transpose(1, 2)  # [B, Sq, H, D]

        kv_cache_entry['k_vision'] = k_full.detach()
        kv_cache_entry['v_vision'] = v_full.detach()
        return out

    def _attend_with_ar_cache(self, q, k, v, kv_cache_entry):
        """Attend over ``[cached ctx | pred]`` with the selected backend."""
        cached_k = kv_cache_entry.get('k_vision')
        cached_v = kv_cache_entry.get('v_vision')
        if cached_k is not None:
            n_ctx = cached_k.shape[1]
            total = n_ctx + k.shape[1]
            slot = self._kv_slot(kv_cache_entry, k, total)
            if slot is not None and self._holds(slot, cached_k):
                slot['k'][:, n_ctx:total].copy_(k)
                slot['v'][:, n_ctx:total].copy_(v)
                k = slot['k'][:, :total]
                v = slot['v'][:, :total]
            else:
                k = torch.cat([cached_k.to(k.device, k.dtype), k], dim=1)
                v = torch.cat([cached_v.to(v.device, v.dtype), v], dim=1)
        if self.attention_backend == 'sage':
            return sage_attention(q, k, v, causal=False)
        return flash_attention_dense(q, k, v, causal=False)



class WorldPlay2CrossAttention(WorldPlay2SelfAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditioning_cache_enabled = True
        # Keyed by the context tensor itself: torch.Tensor hashes and compares
        # by identity, so the key doubles as the strong reference that keeps a
        # stale entry from ever matching a different tensor.
        self._context_kv_cache: dict[torch.Tensor, tuple] = {}

    def clear_conditioning_cache(self) -> None:
        self._context_kv_cache.clear()

    def _get_context_kv(self, context):
        if not self.conditioning_cache_enabled:
            return self.norm_k(self.k(context)), self.v(context)

        entry = self._context_kv_cache.get(context)
        if entry is not None:
            return entry

        key_tensor = self.norm_k(self.k(context)).detach()
        value_tensor = self.v(context).detach()
        self._context_kv_cache[context] = (key_tensor, value_tensor)
        return key_tensor, value_tensor

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k, v = self._get_context_kv(context)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WorldPlay2AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WorldPlay2LayerNorm(dim, eps)
        self.self_attn = WorldPlay2SelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WorldPlay2LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WorldPlay2CrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WorldPlay2LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.compile_block_glue = True
        self.compile_ffn = True

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        vec_action,
        freqs_flat=None,
        ar=False,
        cmp_token_count=0,
        cmp_chunk_token_count=0,
        tmp_token_count=0,
        kv_cache_entry=None,
        cache_write=False,
        sink_token_count=0,
        cache_update=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            freqs_flat(Tensor, optional): Per-token RoPE freqs for the
                memory-compress path; forwarded to ``self_attn``.
            ar(bool): when True, run the chunk-AR (KV-cache) self-attention.
            kv_cache_entry/cache_write: forwarded to
                ``self_attn`` for the chunk-AR prefill / cache-hit paths.
            cache_update(dict, optional): metadata for replacing the old tmp
                suffix and appending the new cmp/tmp tokens.
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if e.stride(1) == 0:
                e = (self.modulation.unsqueeze(0) + e[:, :1]).expand_as(e).chunk(
                    6, dim=2)
            else:
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention (BI when ar=False; chunk-AR KV-cache when ar=True)
        attn_in, _ = run_fused_layer_norm_modulation(
            x,
            e[1].squeeze(2),
            e[0].squeeze(2),
            eps=self.eps,
            enabled=self.compile_block_glue,
        )
        y = self.self_attn(
            attn_in, seq_lens, grid_sizes, freqs, freqs_flat=freqs_flat,
            ar=ar,
            cmp_token_count=cmp_token_count,
            cmp_chunk_token_count=cmp_chunk_token_count,
            tmp_token_count=tmp_token_count,
            kv_cache_entry=kv_cache_entry,
            cache_write=cache_write,
            sink_token_count=sink_token_count,
            cache_update=cache_update)
        x, _ = run_compiled_gated_residual(
            x,
            y,
            e[2].squeeze(2),
            enabled=self.compile_block_glue,
        )

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            cross_input = self.norm3(x)
            cross_output = self.cross_attn(
                cross_input, context, context_lens)
            x = x + cross_output

            # mlp inject action instead of attention
            if vec_action is not None:
                x, _ = run_compiled_action_mlp(
                    x,
                    vec_action,
                    self.img_action_mlp[0].weight,
                    self.img_action_mlp[0].bias,
                    self.img_action_mlp[2].weight,
                    self.img_action_mlp[2].bias,
                    enabled=self.compile_block_glue,
                )

            ffn_in, _ = run_fused_layer_norm_modulation(
                x,
                e[4].squeeze(2),
                e[3].squeeze(2),
                eps=self.eps,
                enabled=self.compile_block_glue,
            )
            if getattr(self.ffn, "uses_fp8", False):
                y = self.ffn(ffn_in.to(x.dtype))
            else:
                y, _ = run_compiled_ffn(
                    ffn_in,
                    self.ffn[0].weight,
                    self.ffn[0].bias,
                    self.ffn[2].weight,
                    self.ffn[2].bias,
                    enabled=self.compile_ffn,
                )
            x, _ = run_compiled_gated_residual(
                x,
                y,
                e[5].squeeze(2),
                enabled=self.compile_block_glue,
            )
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WorldPlay2LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self._active_block_index = None
        self._last_endpoint_output = None

    def make_fixed_pdd(self, num_blocks: int = 2) -> None:
        num_blocks = int(num_blocks)
        if num_blocks <= 0:
            raise ValueError("fixed PDD block count must be positive")
        if hasattr(self, "block_weight"):
            if self._num_fixed_pdd_blocks != num_blocks:
                raise ValueError("fixed PDD block count mismatch")
            return
        self._num_fixed_pdd_blocks = num_blocks
        output_dim = self.head.out_features
        self.block_weight = nn.Parameter(torch.empty(
            num_blocks,
            2 * output_dim,
            self.dim,
            device=self.head.weight.device,
            dtype=self.head.weight.dtype,
        ))
        self.block_bias = nn.Parameter(torch.empty(
            num_blocks,
            2 * output_dim,
            device=self.head.bias.device,
            dtype=self.head.bias.dtype,
        ))
        del self.head

    def set_active_block(self, block_index: int) -> None:
        if not hasattr(self, "block_weight"):
            raise RuntimeError("set_active_block requires a fixed PDD head")
        block_index = int(block_index)
        if not 0 <= block_index < self._num_fixed_pdd_blocks:
            raise IndexError(
                f"fixed PDD block index {block_index} is out of range "
                f"[0, {self._num_fixed_pdd_blocks})"
            )
        self._active_block_index = block_index

    def pop_endpoint_output(self):
        output = self._last_endpoint_output
        self._last_endpoint_output = None
        return output

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            shift, scale = (
                self.modulation.unsqueeze(0) + e.unsqueeze(2)
            ).chunk(2, dim=2)
            hidden = (
                self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)
            )
            if hasattr(self, "block_weight"):
                if self._active_block_index is None:
                    raise RuntimeError(
                        "fixed PDD inference block was not selected"
                    )
                projected = F.linear(
                    hidden,
                    self.block_weight[self._active_block_index],
                    self.block_bias[self._active_block_index],
                )
                displacement, endpoint = projected.chunk(2, dim=-1)
                self._last_endpoint_output = endpoint
                return displacement
            self._last_endpoint_output = None
            return self.head(hidden)


class WorldPlay2Model(ModelMixin, ConfigMixin):
    r"""
    WorldPlay2 diffusion backbone based on Wan2.2.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WorldPlay2AttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WorldPlay2AttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        # Always computed on the host (also when the module is constructed
        # under another device context, e.g. meta): it is not a parameter, so
        # no checkpoint overwrites it, and it is moved to the device on first use.
        with torch.device("cpu"):
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
                                   dim=1)

        # initialize weights
        self.init_weights()

        self.use_sp = False
        self.attention_backend = 'flash'
        self.conditioning_cache_enabled = True
        self._text_projection_cache: dict[tuple, torch.Tensor] = {}
        self._time_conditioning_cache = OrderedDict()
        self._time_conditioning_signature = None
        self.self_qkv_fused = False
        self.fp8_enabled = False
        self.fp8_ffn_enabled = False
        self.fp8_qkv_enabled = False
        self.fp8_recipe = None
        self.fp8_block_range = None
        self.fp8_qkv_block_range = None

    def add_embedding_action_parameters(self):
        """Continuous (first 2 dims) + per-dim discrete embeddings (last 4).

        Action layout (last dim = 6):
          [:, 0:2]  continuous → ``action_in_cont`` MLP → (N, dim)
          [:, 2]    discrete in {-1, 0, 1} → Embedding(3, dim)   (idx +1)
          [:, 3]    discrete in {-1, 0, 1} → Embedding(3, dim)   (idx +1)
          [:, 4]    discrete in {0, 1}     → Embedding(2, dim)
          [:, 5]    discrete in {0, 1}     → Embedding(2, dim)
        Final ``vec_action = cont + Σ discrete`` so the output shape and
        downstream usage stay identical to the MLP path.
        """
        # continuous branch: (N, 2) -> (N, dim)
        self.action_in_cont = nn.Sequential(
            nn.Linear(2, 256),
            nn.SiLU(),
            nn.Linear(256, 1024),
            nn.SiLU(),
            nn.Linear(1024, self.dim),
        )
        # one independent embedding per discrete dim
        self.action_in_disc = nn.ModuleList([
            nn.Embedding(3, self.dim),  # dim2: {-1,0,1} (+1 offset → {0,1,2})
            nn.Embedding(3, self.dim),  # dim3: {-1,0,1} (+1 offset → {0,1,2})
            nn.Embedding(2, self.dim),  # dim4: {0,1}
            nn.Embedding(2, self.dim),  # dim5: {0,1}
        ])
        # Match the per-dim contribution of the continuous branch so all 6
        # action dims start training with comparable signal strength.
        #
        # The continuous branch's last layer (Linear(1024, dim)) under Xavier
        # has weight std ≈ sqrt(2/(fan_in+fan_out)); its output std ≈ that
        # value, spread over 2 input dims, so each continuous dim contributes
        # std ≈ cont_std / sqrt(2).  Initialising the discrete embeddings to
        # the same per-dim std keeps the 6 dims roughly equal-strength.
        #
        # Doing this with explicit Xavier on the cont MLP (rather than relying
        # on Linear's default Kaiming) also keeps it consistent with how
        # init_weights() initialises every other Linear in this model.
        for m in self.action_in_cont.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        cont_last = self.action_in_cont[-1]  # Linear(1024, dim)
        cont_std = math.sqrt(
            2.0 / (cont_last.in_features + cont_last.out_features))
        per_dim_std = cont_std / math.sqrt(2)
        for emb in self.action_in_disc:
            nn.init.normal_(emb.weight, std=per_dim_std)

        # img_action_mlp identical to the MLP path (zero-init last layer)
        for block in self.blocks:
            block.img_action_mlp = nn.Sequential(
                nn.Linear(block.dim + block.dim, block.dim),
                nn.SiLU(),
                nn.Linear(block.dim, block.dim),
            )
            nn.init.zeros_(block.img_action_mlp[-1].weight)
            if block.img_action_mlp[-1].bias is not None:
                nn.init.zeros_(block.img_action_mlp[-1].bias)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Encode ``[pitch,yaw,forward,lateral,perspective,space]``."""
        a = action.reshape(-1, 6)
        cont = self.action_in_cont(a[:, :2])
        # discrete indices; +1 offset for the {-1,0,1} dims, clamp as a
        # cheap guard against stray out-of-range values.
        idx2 = (a[:, 2].long() + 1).clamp_(0, 2)
        idx3 = (a[:, 3].long() + 1).clamp_(0, 2)
        idx4 = a[:, 4].long().clamp_(0, 1)
        idx5 = a[:, 5].long().clamp_(0, 1)
        disc = (self.action_in_disc[0](idx2)
                + self.action_in_disc[1](idx3)
                + self.action_in_disc[2](idx4)
                + self.action_in_disc[3](idx5))
        return cont + disc

    def add_memory_compress_model(self):
        """Attach the hierarchical HR -> compressed-memory feature extractor
        used by the stage-2 / stage-3 chunked inference recipe.

        Wan2.2 5B contract (lr is 1/2 in T and 1/4 in H/W):
          - hr_latent : [B, 16, T_hr,   H_hr,    W_hr]
          - lr_latent : [B, 16, T_hr/2, H_hr/4,  W_hr/4]
          - y         : 20ch (4 mask + 16 first-frame image latent)
          - compressor input  : 36 = 16 + 20 channels at hr resolution.
          - compressor output : spatial_down=[1,1,1,0,0,0] + temporal_down=[1,0,...]
            reduces hr ``[T_hr, H_hr, W_hr]`` -> ``[T_hr/2, H_hr/8, W_hr/8]``.
            ``patch_embedding(lr)`` with patch_size=(1,2,2) also produces
            ``[T_hr/2, H_hr/8, W_hr/8]``, so ``lr_tokens + hr_tokens`` lines up.

        Only the causal-in-time compressor is supported.
        """
        from worldplay2.models.causal_compressor import CausalMemCompressModel
        self.memory_compress = CausalMemCompressModel(
            output_dim=self.dim,
            input_dim=self.in_dim,
            spatial_down=[1, 1, 1, 0, 0, 0],
            temporal_down=[1, 0, 0, 0, 0, 0],
        )

    def set_sequence_parallel(self, enabled: bool) -> None:
        """Select SP/non-SP paths without runtime monkey-patching."""
        self.use_sp = bool(enabled)
        for block in self.blocks:
            block.self_attn.use_sp = self.use_sp

    def set_compiled_qk_rope(self, enabled: bool) -> None:
        for block in self.blocks:
            block.self_attn.compile_qk_rope = bool(enabled)
            block.self_attn.qk_rope_compile_failed = False

    def set_self_attention_backend(self, backend: str) -> None:
        """Select visual denoising attention; preserve masked prefill and text."""
        backend = resolve_attention_backend(backend)
        if backend == 'sage' and any(
            tuple(block.self_attn.window_size) != (-1, -1)
            for block in self.blocks
        ):
            raise NotImplementedError(
                "SageAttention requires window_size=(-1, -1); "
                "use attention_backend='flash' for sliding-window attention."
            )
        self.attention_backend = backend
        for block in self.blocks:
            block.self_attn.attention_backend = backend

    def set_compiled_regions(self, *, block_glue: bool, ffn: bool) -> None:
        for block in self.blocks:
            block.compile_block_glue = bool(block_glue)
            block.compile_ffn = bool(ffn)

    def fuse_self_attention_qkv(self) -> None:
        """Fuse Q/K/V for every visual self-attention block."""
        if self.self_qkv_fused:
            return
        for block in self.blocks:
            block.self_attn.fuse_qkv_projections()
        self.self_qkv_fused = True

    def set_conditioning_cache(self, enabled: bool) -> None:
        self.conditioning_cache_enabled = bool(enabled)
        for block in self.blocks:
            block.cross_attn.conditioning_cache_enabled = bool(enabled)
        if not enabled:
            self.clear_conditioning_cache()

    def clear_conditioning_cache(self) -> None:
        self._text_projection_cache.clear()
        for block in self.blocks:
            block.cross_attn.clear_conditioning_cache()

    def prepare_time_conditioning(self, timestep, seq_len):
        """Timestep conditioning for a 1D CPU ``timestep`` at ``seq_len`` tokens."""
        if timestep.device.type != 'cpu' or timestep.ndim != 1:
            raise ValueError("Fixed conditioning requires a 1D CPU timestep")
        parameters = (
            tuple(self.time_embedding.parameters())
            + tuple(self.time_projection.parameters())
        )
        signature = (
            tuple((id(p), p.data_ptr(), p._version, p.dtype, p.device)
                  for p in parameters),
            torch.get_float32_matmul_precision(),
        )
        if signature != self._time_conditioning_signature:
            self._time_conditioning_cache.clear()
            self._time_conditioning_signature = signature
        key = (tuple(timestep.tolist()), timestep.dtype, seq_len)
        cacheable = not self.training and not torch.is_grad_enabled()
        if cacheable and key in self._time_conditioning_cache:
            self._time_conditioning_cache.move_to_end(key)
            return self._time_conditioning_cache[key]

        t = timestep.to(parameters[0].device)
        expanded = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32, enabled=t.is_cuda):
            e = self.time_embedding(sinusoidal_embedding_1d(
                self.freq_dim, expanded.flatten()
            ).unflatten(0, (t.size(0), seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        if cacheable:
            e, e0 = _compact_identical_rows(e), _compact_identical_rows(e0)
        result = t, (e, e0)
        if cacheable:
            self._time_conditioning_cache[key] = result
            if len(self._time_conditioning_cache) > 2:
                self._time_conditioning_cache.popitem(last=False)
        return result

    def _project_text_context(self, context):
        # Keyed by the source tensors themselves: torch.Tensor hashes and
        # compares by identity, so the key both distinguishes cond/uncond and
        # keeps a strong reference to the tensors it was derived from. Model
        # movement/offload explicitly clears this cache.
        key = tuple(context)
        if self.conditioning_cache_enabled:
            cached = self._text_projection_cache.get(key)
            if cached is not None:
                return cached

        padded = torch.stack([
            torch.cat([
                tensor,
                tensor.new_zeros(
                    self.text_len - tensor.size(0), tensor.size(1)),
            ])
            for tensor in key
        ])
        projected = self.text_embedding(padded).detach()
        if self.conditioning_cache_enabled:
            self._text_projection_cache[key] = projected
        return projected

    def _forward_compress(
        self,
        hr_latents_input,
        lr_latents_input,
        lr_action,
        memory_t,
        hr_temporal_latents_input=None,
        hr_temporal_action=None,
        sink_action=None,
    ):
        """Build ``[optional sink | compressed history | temporal anchor]``."""
        if hr_temporal_latents_input is None or hr_temporal_action is None:
            raise ValueError("Stage-Three compressor requires a temporal anchor")

        bs = hr_latents_input.shape[0]
        hr_pf = hr_latents_input.shape[2] // self.patch_size[0]
        device = hr_latents_input.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        memory_e = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim,
                memory_t.reshape(-1).float(),
            ).float()
        )

        # Compressed LR+HR history at the LR temporal cadence.
        _, _, lr_f, lr_h, lr_w = lr_latents_input.shape
        cmp_pf = lr_f // self.patch_size[0]
        cmp_ph = lr_h // self.patch_size[1]
        cmp_pw = lr_w // self.patch_size[2]
        cmp_spatial = cmp_ph * cmp_pw
        cmp_tokens = (
            self.patch_embedding(lr_latents_input).flatten(2).transpose(1, 2)
            + self.memory_compress(hr_latents_input)
        )
        cmp_freqs = build_freqs_flat(self.freqs,(hr_pf, cmp_ph, cmp_pw)).view(hr_pf, cmp_spatial, -1)
        cmp_freqs = cmp_freqs[::2][:cmp_pf].reshape(cmp_pf * cmp_spatial, -1)
        cmp_e = repeat(
            memory_e,
            "1 D -> B (T HW) D",
            B=bs,
            T=cmp_pf,
            HW=cmp_spatial,
        )
        cmp_e0 = self.time_projection(cmp_e).unflatten(2, (6, self.dim))
        cmp_action = repeat(
            self._encode_action(lr_action),
            "(B T) D -> B (T HW) D",
            B=bs,
            HW=cmp_spatial,
        )

        # Temporal anchor occupies the final HR temporal positions.
        _, _, tmp_f, tmp_h, tmp_w = hr_temporal_latents_input.shape
        tmp_pf = tmp_f // self.patch_size[0]
        tmp_ph = tmp_h // self.patch_size[1]
        tmp_pw = tmp_w // self.patch_size[2]
        tmp_spatial = tmp_ph * tmp_pw

        tmp_tokens = self.patch_embedding(hr_temporal_latents_input)
        tmp_tokens = tmp_tokens.flatten(2).transpose(1, 2)
        tmp_freqs = build_freqs_flat(self.freqs,(hr_pf, tmp_ph, tmp_pw)).view(hr_pf, tmp_spatial, -1)
        tmp_freqs = tmp_freqs[-tmp_pf:].reshape(tmp_pf * tmp_spatial, -1)
        tmp_e = repeat(
            memory_e,
            "1 D -> B (T HW) D",
            B=bs,
            T=tmp_pf,
            HW=tmp_spatial,
        )
        tmp_e0 = self.time_projection(tmp_e).unflatten(2, (6, self.dim))
        tmp_action = repeat(
            self._encode_action(hr_temporal_action),
            "(B T) D -> B (T HW) D",
            B=bs,
            HW=tmp_spatial,
        )

        sink_token_count = 0
        if sink_action is not None and sink_action.shape[0] > 0:
            sink_size = sink_action.shape[0]
            sink_latents = hr_latents_input[:, :, :sink_size]
            _, _, sink_f, sink_h, sink_w = sink_latents.shape
            sink_pf = sink_f // self.patch_size[0]
            sink_ph = sink_h // self.patch_size[1]
            sink_pw = sink_w // self.patch_size[2]
            sink_spatial = sink_ph * sink_pw

            sink_tokens = self.patch_embedding(sink_latents)
            sink_tokens = sink_tokens.flatten(2).transpose(1, 2)
            sink_freqs = build_freqs_flat(self.freqs,(hr_pf, sink_ph, sink_pw)).view(hr_pf, sink_spatial, -1)
            sink_freqs = sink_freqs[:sink_pf].reshape(sink_pf * sink_spatial, -1)
            sink_e = repeat(
                memory_e,
                "1 D -> B (T HW) D",
                B=bs,
                T=sink_pf,
                HW=sink_spatial,
            )
            sink_e0 = self.time_projection(sink_e).unflatten(2, (6, self.dim))
            sink_action_embedding = repeat(
                self._encode_action(sink_action),
                "(B T) D -> B (T HW) D",
                B=bs,
                HW=sink_spatial,
            )
            sink_token_count = sink_tokens.shape[1]

            tokens = torch.cat([sink_tokens, cmp_tokens, tmp_tokens], dim=1)
            freqs_flat = torch.cat([sink_freqs, cmp_freqs, tmp_freqs], dim=0)
            e0 = torch.cat([sink_e0, cmp_e0, tmp_e0], dim=1)
            e = torch.cat([sink_e, cmp_e, tmp_e], dim=1)
            vec_action = torch.cat([sink_action_embedding, cmp_action, tmp_action], dim=1)
        else:
            tokens = torch.cat([cmp_tokens, tmp_tokens], dim=1)
            freqs_flat = torch.cat([cmp_freqs, tmp_freqs], dim=0)
            e0 = torch.cat([cmp_e0, tmp_e0], dim=1)
            e = torch.cat([cmp_e, tmp_e], dim=1)
            vec_action = torch.cat([cmp_action, tmp_action], dim=1)

        return {
            "tokens": tokens,
            "freqs_flat": freqs_flat,
            "e0": e0,
            "e": e,
            "vec_action": vec_action,
            "rope_shift_idx": hr_pf,
            "sink_token_count": sink_token_count,
            "cmp_token_count": cmp_tokens.shape[1],
            "cmp_chunk_token_count": 2 * cmp_spatial,
            "tmp_token_count": tmp_tokens.shape[1],
        }

    def _forward_compress_window(
        self,
        hr_latents_input,
        lr_latents_input,
        lr_action,
        memory_t,
        hr_temporal_latents_input,
        hr_temporal_action,
        hr_frames,
        sink_frames,
    ):
        """The compressor tokens an incremental prefill reads: ``[new cmp | tmp]``.

        ``hr_latents_input`` is a trailing HR window; ``token_offset`` tells the
        prefill where this window starts in the full ``[sink | cmp | tmp]`` layout.
        """
        bs = hr_latents_input.shape[0]
        device = hr_latents_input.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        hr_pf = hr_frames // self.patch_size[0]

        memory_e = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim,
                memory_t.reshape(-1).float(),
            ).float()
        )

        _, _, new_f, lr_h, lr_w = lr_latents_input.shape
        new_pf = new_f // self.patch_size[0]
        cmp_pf = hr_pf // 2
        cmp_ph = lr_h // self.patch_size[1]
        cmp_pw = lr_w // self.patch_size[2]
        cmp_spatial = cmp_ph * cmp_pw
        new_tokens = new_pf * cmp_spatial
        cmp_tokens = (
            self.patch_embedding(lr_latents_input).flatten(2).transpose(1, 2)
            + self.memory_compress(hr_latents_input)[:, -new_tokens:]
        )
        # Temporal positions 2j of the newest LR frames j (cmp_freqs[::2]).
        cmp_freqs = build_freqs_rows(
            self.freqs, slice(2 * (cmp_pf - new_pf), 2 * cmp_pf, 2), cmp_ph, cmp_pw)
        cmp_e = repeat(memory_e, "1 D -> B (T HW) D", B=bs, T=new_pf, HW=cmp_spatial)
        cmp_e0 = self.time_projection(cmp_e).unflatten(2, (6, self.dim))
        # The action MLP runs over the full LR action sequence, as upstream.
        cmp_action = repeat(
            self._encode_action(lr_action)[-new_pf * bs:],
            "(B T) D -> B (T HW) D",
            B=bs,
            HW=cmp_spatial,
        )

        _, _, tmp_f, tmp_h, tmp_w = hr_temporal_latents_input.shape
        tmp_pf = tmp_f // self.patch_size[0]
        tmp_ph = tmp_h // self.patch_size[1]
        tmp_pw = tmp_w // self.patch_size[2]
        tmp_spatial = tmp_ph * tmp_pw
        tmp_tokens = self.patch_embedding(hr_temporal_latents_input)
        tmp_tokens = tmp_tokens.flatten(2).transpose(1, 2)
        tmp_freqs = build_freqs_rows(self.freqs, slice(hr_pf - tmp_pf, hr_pf), tmp_ph, tmp_pw)
        tmp_e = repeat(memory_e, "1 D -> B (T HW) D", B=bs, T=tmp_pf, HW=tmp_spatial)
        tmp_e0 = self.time_projection(tmp_e).unflatten(2, (6, self.dim))
        tmp_action = repeat(
            self._encode_action(hr_temporal_action),
            "(B T) D -> B (T HW) D",
            B=bs,
            HW=tmp_spatial,
        )

        sink_token_count = (sink_frames // self.patch_size[0]) * tmp_spatial
        cmp_token_count = cmp_pf * cmp_spatial
        return {
            "tokens": torch.cat([cmp_tokens, tmp_tokens], dim=1),
            "freqs_flat": torch.cat([cmp_freqs, tmp_freqs], dim=0),
            "e0": torch.cat([cmp_e0, tmp_e0], dim=1),
            "vec_action": torch.cat([cmp_action, tmp_action], dim=1),
            "rope_shift_idx": hr_pf,
            "sink_token_count": sink_token_count,
            "cmp_token_count": cmp_token_count,
            "cmp_chunk_token_count": 2 * cmp_spatial,
            "tmp_token_count": tmp_tokens.shape[1],
            "token_offset": sink_token_count + cmp_token_count - new_tokens,
        }

    def forward(
        self,
        x=None,
        t=None,
        context=None,
        seq_len=None,
        y=None,
        hr_action=None,
        memory_dict=None,
        forward_mode="bi_denoise",
        kv_cache=None,
        ar_chunk_start=0,
        hr_latents_input=None,
        lr_latents_input=None,
        lr_action=None,
        memory_t=None,
        hr_temporal_latents_input=None,
        hr_temporal_action=None,
        sink_action=None,
        pdd_block_index=None,
        time_conditioning=None,
        hr_frames=None,
        sink_frames=0,
    ):
        """Route all model operations through one explicit entry point."""
        with fp8_autocast(
            self.fp8_recipe,
            enabled=self.fp8_enabled,
        ):
            if forward_mode == "compress":
                return self._forward_compress(
                    hr_latents_input=hr_latents_input,
                    lr_latents_input=lr_latents_input,
                    lr_action=lr_action,
                    memory_t=memory_t,
                    hr_temporal_latents_input=hr_temporal_latents_input,
                    hr_temporal_action=hr_temporal_action,
                    sink_action=sink_action,
                )
            if forward_mode == "compress_window":
                return self._forward_compress_window(
                    hr_latents_input=hr_latents_input,
                    lr_latents_input=lr_latents_input,
                    lr_action=lr_action,
                    memory_t=memory_t,
                    hr_temporal_latents_input=hr_temporal_latents_input,
                    hr_temporal_action=hr_temporal_action,
                    hr_frames=hr_frames,
                    sink_frames=sink_frames,
                )
            if forward_mode == "bi_denoise":
                return self._forward_bi_denoise(
                    x=x, t=t, context=context, seq_len=seq_len, y=y,
                    hr_action=hr_action, memory_dict=memory_dict)
            if forward_mode == "ar_prefill":
                return self._forward_ar_prefill(
                    context=context, memory_dict=memory_dict,
                    kv_cache=kv_cache)
            if forward_mode == "ar_denoise":
                return self._forward_ar_denoise(
                    x=x, t=t, context=context, seq_len=seq_len, y=y,
                    hr_action=hr_action, memory_dict=memory_dict,
                    kv_cache=kv_cache, ar_chunk_start=ar_chunk_start)
            if forward_mode == "pdd_denoise":
                self.head.set_active_block(pdd_block_index)
                return self._forward_ar_denoise(
                    x=x, t=t, context=context, seq_len=seq_len, y=y,
                    hr_action=hr_action, memory_dict=memory_dict,
                    kv_cache=kv_cache, ar_chunk_start=ar_chunk_start,
                    time_conditioning=time_conditioning)
            raise ValueError(
                f"unsupported forward_mode={forward_mode!r}; expected "
                "'compress', 'bi_denoise', 'ar_prefill', 'ar_denoise', "
                "'pdd_denoise'"
            )

    def _forward_bi_denoise(
        self, x, t, context, seq_len, y, hr_action, memory_dict
    ):
        """Bidirectional denoise with optional compressed history context."""
        # BI uses the same body for SP and non-SP. SP only shards the token
        # tensors before the blocks and gathers the head output afterwards.
        ar = False

        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        h, w = x[0].shape[2:]
        tw = w // self.patch_size[2]
        th = h // self.patch_size[1]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat(
                [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                dim=1,
            )
            for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(
                    self.freq_dim, t
                ).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self._project_text_context(context)

        vec_action = None
        if hr_action is not None:
            vec_action = self._encode_action(hr_action)
            vec_action = repeat(
                vec_action, "(B T) D -> B (T H W) D",
                B=x.shape[0], H=th, W=tw)

        # Build per-token RoPE for the generate segment.  The generate
        # segment ALWAYS uses per-token ``freqs_flat`` (via ``rope_apply_flat``)
        # -- both BI and AR, with or without a compressor context -- so the
        # RoPE handling is uniform and never depends on the grid-based
        # ``rope_apply`` fallback.  When fused with a compressor context the
        # generate segment's temporal rope indices must START AFTER the
        # context's time range (= ``rope_shift_idx``); otherwise the generate
        # and context tokens would share the same temporal phases and the
        # model could not distinguish "past vs future".
        #
        # NB: for ``rope_shift_idx == 0`` (no context) the per-token freqs are
        # numerically identical to what grid-based ``rope_apply`` would build,
        # so this does not change the BI first-chunk result.
        gen_pf = int(grid_sizes[0, 0].item())
        gen_ph = int(grid_sizes[0, 1].item())
        gen_pw = int(grid_sizes[0, 2].item())
        rope_shift_idx = (int(memory_dict.get("rope_shift_idx", 0))
                          if memory_dict is not None else 0)
        full_freqs = build_freqs_flat(
            self.freqs,
            (rope_shift_idx + gen_pf, gen_ph, gen_pw)).to(x.device)
        full_freqs = full_freqs.view(
            rope_shift_idx + gen_pf, gen_ph * gen_pw, -1)
        gen_freqs_flat = full_freqs[-gen_pf:].reshape(
            gen_pf * gen_ph * gen_pw, -1)
        # Pad to seq_len (pad tokens are masked out in flash_attention
        # via ``seq_lens``).
        if gen_freqs_flat.shape[0] < seq_len:
            pad_rows = seq_len - gen_freqs_flat.shape[0]
            pad = gen_freqs_flat.new_zeros(
                pad_rows, gen_freqs_flat.shape[1])
            gen_freqs_flat = torch.cat(
                [gen_freqs_flat, pad], dim=0)

        # -- fuse context (memory_compress output) before generate segment --
        generate_seq_length = x.shape[1]
        cmp_token_count = 0
        cmp_chunk_token_count = 0
        tmp_token_count = 0
        if memory_dict is not None:
            ctx_tokens = memory_dict["tokens"]
            ctx_freqs_flat = memory_dict["freqs_flat"].to(x.device)
            ctx_e0 = memory_dict["e0"].float()
            ctx_e = memory_dict["e"].float()
            ctx_vec_action = memory_dict["vec_action"]

            x = torch.cat([ctx_tokens, x], dim=1)
            freqs_flat = torch.cat(
                [ctx_freqs_flat, gen_freqs_flat], dim=0)
            e0 = torch.cat([ctx_e0, e0], dim=1)
            e = torch.cat([ctx_e, e], dim=1)
            vec_action = torch.cat(
                [ctx_vec_action, vec_action], dim=1)
            # seq_lens now counts the full (ctx + generate) length per batch
            seq_lens = torch.tensor([x.shape[1]] * x.shape[0],
                                    dtype=torch.long)
            cmp_token_count = int(memory_dict.get("cmp_token_count", 0))
            cmp_chunk_token_count = int(
                memory_dict.get("cmp_chunk_token_count", 0))
            tmp_token_count = int(memory_dict.get("tmp_token_count", 0))
        else:
            # No compressor ctx (first chunk): generate-only, per-token RoPE.
            freqs_flat = gen_freqs_flat

        if self.use_sp:
            sp_size = get_world_size()
            sp_rank = get_rank()
            global_sequence = x.shape[1]
            x = shard_sequence(x, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            e = shard_sequence(e, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            e0 = shard_sequence(e0, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            vec_action = shard_sequence(vec_action, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            freqs_flat = shard_sequence(freqs_flat, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=0)
        else:
            global_sequence = x.shape[1]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            vec_action=vec_action,
            freqs_flat=freqs_flat,
            ar=ar,
            cmp_token_count=cmp_token_count,
            cmp_chunk_token_count=cmp_chunk_token_count,
            tmp_token_count=tmp_token_count,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)
        if self.use_sp:
            x = gather_sequence(x, global_sequence=global_sequence)

        # drop the prepended context tokens before unpatchify
        if memory_dict is not None:
            x = x[:, -generate_seq_length:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def init_kv_cache(self, slot=0):
        r"""Allocate contiguous per-layer ``[sink | cmp | tmp]`` K/V.

        With ``kv_capacity`` set (tokens), each layer of cache ``slot`` writes
        into K/V storage of that capacity.
        """
        entries = [{'k_vision': None, 'v_vision': None}
                   for _ in range(self.num_layers)]
        capacity = getattr(self, 'kv_capacity', 0)
        if capacity:
            pools = self.__dict__.setdefault('_kv_pools', {})
            pool = pools.get(slot)
            if pool is None or pool[0]['capacity'] < capacity:
                pool = [{'capacity': capacity, 'k': None, 'v': None}
                        for _ in range(self.num_layers)]
                pools[slot] = pool
            for entry, buffers in zip(entries, pool):
                entry['_buffers'] = buffers
        return entries

    def _forward_ar_prefill(self, context, memory_dict, kv_cache):
        r"""Chunk-AR KV-cache incremental prefill for SP and non-SP.

        Each layer stores only contiguous ``k_vision/v_vision`` with layout
        ``[sink | accumulated cmp | latest tmp]``. The first context is
        processed in one masked pass. Later calls infer the old cmp length
        from the cache shape, discard the previous tmp suffix, process only
        ``[new cmp | new tmp]``, and replace the cache with the new layout.

        Args:
            context (List[Tensor]): text embeddings (cond OR uncond).
            memory_dict (dict | None): compressor output; ``None`` -> no-op.
            kv_cache (list[dict]): from ``init_kv_cache()``; mutated in place.

        Returns:
            list[dict]: the same ``kv_cache`` (mutated in place).
        """
        assert kv_cache is not None and len(kv_cache) == self.num_layers, (
            "ar_prefill requires a kv_cache of length num_layers "
            "(call init_kv_cache()).")

        # First chunk: no compressor ctx -> nothing to cache.
        if memory_dict is None:
            return kv_cache

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # text context (embed cond/uncond text the same way as the denoise path)
        context_lens = None
        context = self._project_text_context(context)

        ctx_tokens = memory_dict["tokens"]
        freqs_flat = memory_dict["freqs_flat"].to(ctx_tokens.device)
        # FP32 modulation (the compressor emits BF16 under autocast).
        e0c = memory_dict["e0"]
        if not getattr(self, 'e0_slice', False):
            e0c = e0c.float()
        # A windowed compressor output omits the first ``token_offset`` tokens.
        token_offset = int(memory_dict.get("token_offset", 0))
        vec_action_c = memory_dict["vec_action"]
        sink_token_count = int(memory_dict.get("sink_token_count", 0))
        cmp_token_count = int(memory_dict.get("cmp_token_count", 0))
        cmp_chunk_token_count = int(
            memory_dict.get("cmp_chunk_token_count", 0))
        tmp_token_count = int(memory_dict.get("tmp_token_count", 0))

        cache_is_empty = kv_cache[0].get('k_vision') is None
        if any(
            (entry.get('k_vision') is None) != cache_is_empty
            for entry in kv_cache
        ):
            raise ValueError("all AR cache layers must have the same state")

        if cache_is_empty:
            if token_offset:
                raise ValueError("the first AR prefill needs the full context")
            self._prefill_initial_ar_context(
                tokens=ctx_tokens,
                modulation=e0c.float(),
                actions=vec_action_c,
                freqs_flat=freqs_flat,
                text_context=context,
                context_lens=context_lens,
                kv_cache=kv_cache,
                sink_token_count=sink_token_count,
                cmp_token_count=cmp_token_count,
                cmp_chunk_token_count=cmp_chunk_token_count,
                tmp_token_count=tmp_token_count,
            )
        else:
            self._prefill_ar_context_update(
                tokens=ctx_tokens,
                modulation=e0c,
                actions=vec_action_c,
                freqs_flat=freqs_flat,
                text_context=context,
                context_lens=context_lens,
                kv_cache=kv_cache,
                sink_token_count=sink_token_count,
                cmp_token_count=cmp_token_count,
                cmp_chunk_token_count=cmp_chunk_token_count,
                tmp_token_count=tmp_token_count,
                token_offset=token_offset,
            )
        return kv_cache

    def _prefill_initial_ar_context(
        self,
        *,
        tokens,
        modulation,
        actions,
        freqs_flat,
        text_context,
        context_lens,
        kv_cache,
        sink_token_count,
        cmp_token_count,
        cmp_chunk_token_count,
        tmp_token_count,
    ):
        """First prefill: write the complete ``[sink | cmp | tmp]`` cache."""
        self._run_ar_prefill_blocks(
            tokens=tokens,
            modulation=modulation,
            actions=actions,
            freqs_flat=freqs_flat,
            text_context=text_context,
            context_lens=context_lens,
            kv_cache=kv_cache,
            sink_token_count=sink_token_count,
            cmp_token_count=cmp_token_count,
            cmp_chunk_token_count=cmp_chunk_token_count,
            tmp_token_count=tmp_token_count,
            cache_update=None,
        )

    def _prefill_ar_context_update(
        self,
        *,
        tokens,
        modulation,
        actions,
        freqs_flat,
        text_context,
        context_lens,
        kv_cache,
        sink_token_count,
        cmp_token_count,
        cmp_chunk_token_count,
        tmp_token_count,
        token_offset=0,
    ):
        """Later prefill: append new cmp and replace the previous tmp.

        ``tokens``/``modulation``/``actions``/``freqs_flat`` start at context
        token ``token_offset`` (0 for a full compressor output).
        """
        cached_total = int(kv_cache[0]['k_vision'].shape[1])
        cached_cmp = cached_total - sink_token_count - tmp_token_count
        if not 0 <= cached_cmp <= cmp_token_count:
            raise ValueError(
                "invalid AR cache layout: "
                f"cached_total={cached_total}, sink={sink_token_count}, "
                f"tmp={tmp_token_count}, current_cmp={cmp_token_count}"
            )

        new_cmp_count = cmp_token_count - cached_cmp
        cmp_start = sink_token_count + cached_cmp
        cmp_end = sink_token_count + cmp_token_count
        tmp_start = cmp_end
        tmp_end = tmp_start + tmp_token_count

        token_parts = []
        modulation_parts = []
        action_parts = []
        freq_parts = []
        if token_offset > cmp_start:
            raise ValueError("windowed context does not cover the new tokens")
        cmp_start -= token_offset
        cmp_end -= token_offset
        tmp_start -= token_offset
        tmp_end -= token_offset
        if new_cmp_count:
            token_parts.append(tokens[:, cmp_start:cmp_end])
            modulation_parts.append(modulation[:, cmp_start:cmp_end])
            freq_parts.append(freqs_flat[cmp_start:cmp_end])
            if actions is not None:
                action_parts.append(actions[:, cmp_start:cmp_end])
        if tmp_token_count:
            token_parts.append(tokens[:, tmp_start:tmp_end])
            modulation_parts.append(modulation[:, tmp_start:tmp_end])
            freq_parts.append(freqs_flat[tmp_start:tmp_end])
            if actions is not None:
                action_parts.append(actions[:, tmp_start:tmp_end])
        if not token_parts:
            return

        self._run_ar_prefill_blocks(
            tokens=torch.cat(token_parts, dim=1),
            modulation=torch.cat(modulation_parts, dim=1).float(),
            actions=(
                torch.cat(action_parts, dim=1)
                if action_parts else None
            ),
            freqs_flat=torch.cat(freq_parts, dim=0),
            text_context=text_context,
            context_lens=context_lens,
            kv_cache=kv_cache,
            sink_token_count=sink_token_count,
            cmp_token_count=cmp_token_count,
            cmp_chunk_token_count=cmp_chunk_token_count,
            tmp_token_count=tmp_token_count,
            cache_update={
                'n_new_cmp': new_cmp_count,
                'n_tmp': tmp_token_count,
                'cached_prefix_tokens': sink_token_count + cached_cmp,
            },
        )

    def _run_ar_prefill_blocks(
        self,
        *,
        tokens,
        modulation,
        actions,
        freqs_flat,
        text_context,
        context_lens,
        kv_cache,
        sink_token_count,
        cmp_token_count,
        cmp_chunk_token_count,
        tmp_token_count,
        cache_update,
    ):
        """Run either initial or incremental prefill through all blocks."""
        global_length = tokens.shape[1]
        if self.use_sp:
            sp_size = get_world_size()
            sp_rank = get_rank()
            tokens = shard_sequence(tokens, global_sequence=global_length, rank=sp_rank, world_size=sp_size, dim=1)
            modulation = shard_sequence(modulation, global_sequence=global_length, rank=sp_rank, world_size=sp_size, dim=1)
            freqs_flat = shard_sequence(freqs_flat, global_sequence=global_length, rank=sp_rank, world_size=sp_size, dim=0)
            actions = shard_sequence(actions, global_sequence=global_length, rank=sp_rank, world_size=sp_size, dim=1)

        seq_lens = torch.tensor([global_length] * tokens.shape[0], dtype=torch.long)
        for index, block in enumerate(self.blocks):
            tokens = block(
                tokens,
                e=modulation,
                seq_lens=seq_lens,
                grid_sizes=None,
                freqs=self.freqs,
                context=text_context,
                context_lens=context_lens,
                vec_action=actions,
                freqs_flat=freqs_flat,
                ar=True,
                cmp_token_count=cmp_token_count,
                cmp_chunk_token_count=cmp_chunk_token_count,
                tmp_token_count=tmp_token_count,
                kv_cache_entry=kv_cache[index],
                cache_write=True,
                sink_token_count=sink_token_count,
                cache_update=cache_update,
            )

    def _forward_ar_denoise(
        self,
        x=None,
        t=None,
        context=None,
        seq_len=None,
        y=None,
        hr_action=None,
        memory_dict=None,
        kv_cache=None,
        ar_chunk_start=0,
        time_conditioning=None,
    ):
        r"""Chunk-AR KV-cache denoise for both SP and non-SP.

        Runs the gen (pred) tokens through every block with
        ``cache_write=False``; each block reads its cached ctx K/V (written
        by :meth:`_forward_ar_prefill`) and prepends it to the pred K/V.  No
        ctx tokens are concatenated to ``x`` here.  Returns ``list[Tensor]``.

        ``ar_chunk_start`` phase-shifts the generate segment's RoPE so K/V
        across chunks share a consistent temporal coordinate system.  When
        ``memory_dict`` is ``None`` / the cache is empty (first chunk) this
        degenerates to plain pred-only self-attention.
        """
        assert kv_cache is not None, (
            "AR denoise requires kv_cache (call init_kv_cache()).")
        assert len(kv_cache) == self.num_layers, (
            f"kv_cache length ({len(kv_cache)}) must equal num_layers "
            f"({self.num_layers}).")

        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        h, w = x[0].shape[2:]
        tw = w // self.patch_size[2]
        th = h // self.patch_size[1]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings (gen/pred tokens)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat(
                [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                dim=1,
            )
            for u in x
        ])

        if time_conditioning is None:
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        else:
            e, e0 = time_conditioning
            if (e.shape != (x.shape[0], seq_len, self.dim)
                    or e0.shape != (x.shape[0], seq_len, 6, self.dim)):
                raise ValueError("Cached timestep conditioning has the wrong shape")
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # text context
        context_lens = None
        context = self._project_text_context(context)

        vec_action = None
        if hr_action is not None:
            vec_action = self._encode_action(hr_action)
            vec_action = repeat(vec_action, "(B T) D -> B (T H W) D", B=x.shape[0], H=th, W=tw)

        # ---- per-token RoPE for the generate segment --------------------
        # The current chunk lives at hr-latent indices
        # [ar_chunk_start, ar_chunk_start + gen_pf).  When a compressor ctx is
        # present, the generate segment must start AFTER the ctx time range
        # (rope_shift_idx); resolve gen_start = max(ar_chunk_start, ctx_shift).
        gen_pf = int(grid_sizes[0, 0].item())
        gen_ph = int(grid_sizes[0, 1].item())
        gen_pw = int(grid_sizes[0, 2].item())
        ctx_shift = int(memory_dict.get("rope_shift_idx", 0)) if memory_dict is not None else 0
        gen_start = max(int(ar_chunk_start), ctx_shift)
        if getattr(self, 'rope_window', False):
            gen_freqs_flat = build_freqs_rows(
                self.freqs, slice(gen_start, gen_start + gen_pf), gen_ph, gen_pw)
        else:
            full_freqs = build_freqs_flat(self.freqs, (gen_start + gen_pf, gen_ph, gen_pw)).to(x.device)
            full_freqs = full_freqs.view(gen_start + gen_pf, gen_ph * gen_pw, -1)
            gen_freqs_flat = full_freqs[-gen_pf:].reshape(gen_pf * gen_ph * gen_pw, -1)

        # x = pred only; each block reads cached ctx K/V and prepends to pred.
        freqs_flat = gen_freqs_flat
        cmp_token_count = int(memory_dict.get("cmp_token_count", 0)) if memory_dict else 0
        cmp_chunk_token_count = int(memory_dict.get("cmp_chunk_token_count", 0)) if memory_dict else 0
        tmp_token_count = int(memory_dict.get("tmp_token_count", 0)) if memory_dict else 0
        seq_lens = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long)

        if self.use_sp:
            sp_size = get_world_size()
            sp_rank = get_rank()
            global_sequence = x.shape[1]
            x = shard_sequence(x, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            e = shard_sequence(e, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            e0 = shard_sequence(e0, global_sequence=global_sequence, rank=sp_rank, world_size=sp_size, dim=1)
            vec_action = shard_sequence(vec_action, global_sequence=global_sequence, rank=sp_rank,
                                        world_size=sp_size, dim=1)
            freqs_flat = shard_sequence(freqs_flat, global_sequence=global_sequence, rank=sp_rank,
                                        world_size=sp_size, dim=0)
        else:
            global_sequence = x.shape[1]

        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                vec_action=vec_action,
                freqs_flat=freqs_flat,
                ar=True,
                cmp_token_count=cmp_token_count,
                cmp_chunk_token_count=cmp_chunk_token_count,
                tmp_token_count=tmp_token_count,
                kv_cache_entry=kv_cache[idx],
                cache_write=False,
            )

        # head
        x = self.head(x, e)
        endpoint_x = self.head.pop_endpoint_output()
        if self.use_sp:
            x = gather_sequence(x, global_sequence=global_sequence)
            if endpoint_x is not None:
                endpoint_x = gather_sequence(endpoint_x, global_sequence=global_sequence)

        # unpatchify (no ctx prefix to strip -- x is pred-only throughout)
        x = self.unpatchify(x, grid_sizes)
        if endpoint_x is None:
            return [u.float() for u in x]
        endpoint_x = self.unpatchify(endpoint_x, grid_sizes)
        return [
            (displacement.float(), endpoint.float())
            for displacement, endpoint in zip(x, endpoint_x)
        ]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
