"""Runtime kernels for WorldPlay2."""

from __future__ import annotations


import numpy as np
import torch
import triton
import triton.language as tl

from ..inference import model as wp2_model

from . import vae_split

_INSTALLED: list[tuple[object, str, object]] = []
_MISSING = object()


def _set(obj, name, fn):
    previous = obj.__dict__.get(name, _MISSING) if not isinstance(obj, type(torch)) else getattr(obj, name)
    setattr(obj, name, fn)
    _INSTALLED.append((obj, name, previous))


def _set_value(obj, name, value):
    _INSTALLED.append((obj, name, getattr(obj, name)))
    setattr(obj, name, value)


def _install_fp32_tables(pipeline):
    _set_value(wp2_model, "TABLE_DTYPE", torch.float32)
    for model in _experts(pipeline):
        _set_value(model, "freqs", model.freqs.to(torch.complex64))


def _install_tf32():
    _set_value(torch.backends.cuda.matmul, "allow_tf32", True)
    _set_value(torch.backends.cudnn, "allow_tf32", True)


def _experts(pipeline):
    return (pipeline._find_wan_model(pipeline.high_noise_model),
            pipeline._find_wan_model(pipeline.low_noise_model))


_PREFILL_MASKS = {}


def _prefill_bias(sq, l_tot, l_pre, n_new_cmp, n_tmp, ccc, dtype, device):
    key = (sq, l_tot, l_pre, n_new_cmp, n_tmp, ccc, dtype, device)
    bias = _PREFILL_MASKS.get(key)
    if bias is None:
        mask = torch.zeros(sq, l_tot, dtype=torch.bool, device=device)
        if n_new_cmp > 0:
            step = ccc if ccc > 0 else n_new_cmp
            for rs in range(0, n_new_cmp, step):
                re = min(rs + step, n_new_cmp)
                mask[rs:re, :min(l_pre + re, l_pre + n_new_cmp)] = True
        if n_tmp > 0:
            mask[n_new_cmp:sq, :l_tot] = True
        bias = torch.zeros(sq, l_tot, dtype=dtype, device=device).masked_fill_(~mask, float("-inf"))
        bias = bias.unsqueeze(0).unsqueeze(0)
        if len(_PREFILL_MASKS) > 8:
            _PREFILL_MASKS.clear()
        _PREFILL_MASKS[key] = bias
    return bias


def _install_prefill_attention(pipeline, mode):
    _PREFILL_MASKS.clear()
    for model in _experts(pipeline):
        for block in model.blocks:
            _set(block.self_attn, "prefill_attention", mode)
            _set(block.self_attn, "prefill_bias", _prefill_bias)


def _install_switches(pipeline, pipeline_flags=(), model_flags=()):
    for name in pipeline_flags:
        _set(pipeline, name, True)
    for model in _experts(pipeline):
        for name in model_flags:
            _set(model, name, True)


@triton.jit
def _qk_rms_rope_kernel(x_ptr, stride_b, stride_s, w_ptr, cos_ptr, sin_ptr, out_ptr,
                        seq, eps,
                        DIM: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // seq
    s = row - b * seq
    pair = tl.arange(0, BLOCK)
    mask = pair < DIM // 2
    base = x_ptr + b * stride_b + s * stride_s
    even = tl.load(base + 2 * pair, mask=mask, other=0.0).to(tl.float32)
    odd = tl.load(base + 2 * pair + 1, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(even * even + odd * odd, axis=0)
    inv = 1.0 / tl.sqrt(total / DIM + eps)
    w_even = tl.load(w_ptr + 2 * pair, mask=mask, other=0.0).to(tl.float32)
    w_odd = tl.load(w_ptr + 2 * pair + 1, mask=mask, other=0.0).to(tl.float32)
    even = ((even * inv).to(tl.bfloat16).to(tl.float32) * w_even).to(tl.bfloat16).to(tl.float32)
    odd = ((odd * inv).to(tl.bfloat16).to(tl.float32) * w_odd).to(tl.bfloat16).to(tl.float32)
    freq = s * HALF + pair % HALF
    cos = tl.load(cos_ptr + freq, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + freq, mask=mask, other=0.0)
    out = out_ptr + row * DIM
    tl.store(out + 2 * pair, (even * cos - odd * sin).to(tl.bfloat16), mask=mask)
    tl.store(out + 2 * pair + 1, (even * sin + odd * cos).to(tl.bfloat16), mask=mask)


def _rms_rope(x, weight, cos, sin, eps, num_heads, head_dim):
    batch, seq, dim = x.shape
    out = torch.empty((batch, seq, num_heads, head_dim), device=x.device, dtype=torch.bfloat16)
    if batch * seq:
        _qk_rms_rope_kernel[(batch * seq,)](
            x, x.stride(0), x.stride(1), weight, cos, sin, out, seq, eps,
            DIM=dim, HALF=head_dim // 2, BLOCK=triton.next_power_of_2(dim // 2), num_warps=8)
    return out


def fixed_qk_norm_rope(q, k, norm_q_weight, norm_k_weight, freqs_flat, *, eps, num_heads, head_dim,
                       compile_enabled):
    if (q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or q.stride(-1) != 1
            or k.stride(-1) != 1 or norm_q_weight.dtype != torch.bfloat16
            or q.shape[-1] != num_heads * head_dim):
        return _ORIGINAL_QK(q, k, norm_q_weight, norm_k_weight, freqs_flat, eps=eps,
                            num_heads=num_heads, head_dim=head_dim, compile_enabled=compile_enabled)
    cos = freqs_flat.real.float().contiguous()
    sin = freqs_flat.imag.float().contiguous()
    return (_rms_rope(q, norm_q_weight, cos, sin, eps, num_heads, head_dim),
            _rms_rope(k, norm_k_weight, cos, sin, eps, num_heads, head_dim), True)


_ORIGINAL_QK = wp2_model.run_qk_norm_rope


def _install_fixed_qk(pipeline):
    _set(wp2_model, "run_qk_norm_rope", fixed_qk_norm_rope)


def rgb8(rgb):
    return rgb.float().clamp(-1, 1).add(1).mul(127.5).to(dtype=torch.uint8).permute(1, 2, 3, 0).contiguous()


def _install_async_rgb(pipeline, engine):
    side = torch.cuda.Stream()
    decode = pipeline._decode_chunk_and_lr_encode
    original = engine._host_frames
    pending = {}

    def decode_and_start_copy(*args, **kwargs):
        out = decode(*args, **kwargs)
        rgb = out[1]
        pending.clear()
        if rgb is not None and rgb.is_cuda:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                u8 = rgb8(rgb)
                host = torch.empty(u8.shape, dtype=torch.uint8, pin_memory=True)
                host.copy_(u8, non_blocking=True)
                done = torch.cuda.Event()
                done.record(side)
            rgb.record_stream(side)
            pending.update(rgb=rgb, u8=u8, host=host, done=done)
        return out

    def host_frames(rgb) -> np.ndarray:
        slot = dict(pending)
        pending.clear()
        if slot.get("rgb") is rgb:
            slot["done"].synchronize()
            return slot["host"].numpy()
        return original(rgb)

    _set(pipeline, "_decode_chunk_and_lr_encode", decode_and_start_copy)
    _set(engine, "_host_frames", host_frames)


TECHNIQUES = {
    "fp32_tables": lambda p, e: _install_fp32_tables(p),
    "tf32": lambda p, e: _install_tf32(),
    "zero_tail_clip": lambda p, e: _install_switches(p, ("lazy_zero_clip",)),
    "kv_inplace": lambda p, e: _install_switches(p, ("kv_inplace",)),
    "rope_window": lambda p, e: _install_switches(p, model_flags=("rope_window",)),
    "e0_slice": lambda p, e: _install_switches(p, ("slim_memory",), ("e0_slice",)),
    "compress_window": lambda p, e: _install_switches(p, ("compress_window",)),
    "chunk_stream": lambda p, e: _install_switches(p, ("chunk_stream",)),
    "chunk_noise": lambda p, e: _set(p, "noise_mode", "chunk"),
    "prefill_split": lambda p, e: _install_prefill_attention(p, "split"),
    "fixed_qk_norm_rope": lambda p, e: _install_fixed_qk(p),
    "vae_split": lambda p, e: vae_split.install(p, _set),
    "prefill_mask_cache": lambda p, e: _install_prefill_attention(p, "bias"),
    "async_rgb": _install_async_rgb,
}

_GENERIC = ("zero_tail_clip", "kv_inplace", "rope_window", "e0_slice", "prefill_split")
_WINDOWED = ("compress_window", "chunk_stream")
LOSSLESS = {
    1: _GENERIC + ("async_rgb",),
    2: _GENERIC + ("vae_split", "async_rgb"),
    4: _GENERIC + ("vae_split", "async_rgb"),
}
NUM_EQUAL = {n: _WINDOWED + ("fp32_tables", "tf32", "chunk_noise") for n in (1, 2, 4)}
_ORDER = ("fp32_tables", "tf32", "fixed_qk_norm_rope") + _GENERIC + _WINDOWED + (
    "chunk_noise", "vae_split", "prefill_mask_cache", "async_rgb")


def install(pipeline, names, engine=None):
    unknown = [name for name in names if name not in TECHNIQUES]
    if unknown:
        raise ValueError(f"not a lossless technique: {unknown}")
    if "async_rgb" in names and engine is None:
        raise ValueError("async_rgb needs the WorldPlay2Model that owns the pipeline")
    return {name: TECHNIQUES[name](pipeline, engine) for name in names}


def _install_release_on_reset(pipeline, engine, split_decoders=()):
    reset = engine.reset

    def release_reset():
        reset()
        for decoder in split_decoders:
            decoder.reset()
        _PREFILL_MASKS.clear()

    _set(engine, "reset", release_reset)


def select(world_size, fidelity="lossless", disabled=()):
    if fidelity not in ("lossless", "num-equal"):
        raise ValueError(f"fidelity must be lossless or num-equal, got {fidelity!r}")
    names = set(LOSSLESS[world_size])
    if fidelity == "num-equal":
        names |= set(NUM_EQUAL[world_size])
    unknown = [name for name in disabled if name not in TECHNIQUES]
    if unknown:
        raise ValueError(f"not a technique: {unknown}")
    names -= set(disabled)
    return tuple(name for name in _ORDER if name in names)


def install_lossless(engine, fidelity="lossless", disabled=()):
    names = select(engine.world_size, fidelity, disabled)
    installed = install(engine._pipeline, names, engine)
    split = [installed["vae_split"]] if "vae_split" in installed else []
    _install_release_on_reset(engine._pipeline, engine, split)
    return names


def uninstall():
    while _INSTALLED:
        obj, name, previous = _INSTALLED.pop()
        if previous is _MISSING:
            obj.__dict__.pop(name, None)
        else:
            setattr(obj, name, previous)
