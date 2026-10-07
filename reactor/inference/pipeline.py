# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for the Reactor runtime.
"""WorldPlay2 chunked inference with compressed memory and temporal anchors,
built on Wan2.2 I2V-A14B.

Recipe (per HY WorldPlay reference, adapted to Wan2.2):
  * Generate ``total_latent_frames`` HR latent frames in groups of
    ``chunk_length`` frames (default 4).
  * For chunk 0, run a vanilla i2v denoising loop without context.
  * For chunks > 0:
      1. Streaming-VAE-decode the just-finished HR chunk.
      2. Take ``frames[:, ::2]`` (every other frame) and bilinear
         downsample to (H/4, W/4) in pixel space.
      3. Streaming-VAE-encode the downsampled clip back to LR latents.
      4. Accumulate (lr_latents_cache, hr_latents_cache, decoded_rgb_cache).
      5. Build ``hr_temporal`` = last 2 hr latents before the generate
         window with zero y, build hr_y / lr_y context (4-mask + 16-image
         latent, image-only at t=0), then call
         ``model(forward_mode="compress", ...)`` to
         produce ``memory_dict``.
      6. Run the chunk's denoising loop with ``memory_dict`` plumbed in.
  * After the final chunk, decode the last chunk (continuing the same
    streaming cache) and concatenate with the cached RGB.

The pipeline builds the T5 encoder, VAE, and two ``WorldPlay2Model`` experts.
Each expert loads its complete DiT, action encoder, and memory compressor
weights from a separate checkpoint.

Sequence-parallel behavior is owned by ``WorldPlay2Model`` and
``WorldPlay2SelfAttention`` themselves. The pipeline only enables it through
``model.set_sequence_parallel``; no runtime monkey-patching is used.
"""

import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from enum import Enum
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from .checkpoint import load_expert_checkpoint
from worldplay2.distributed.fsdp import shard_model
from worldplay2.distributed.ops import get_world_size
from worldplay2.models.attention import resolve_attention_backend
from .model import WorldPlay2Model
from worldplay2.models.fp8 import (
    convert_model_ffn_to_fp8,
    convert_model_qkv_to_fp8,
    fp8_model_stats,
)
from .pdd import FixedPDD4Scheduler
from worldplay2.schedulers import (FlowDPMSolverMultistepScheduler,
                         FlowUniPCMultistepScheduler, get_sampling_sigmas,
                         retrieve_timesteps)
from .text import T5EncoderModel
from .vae import Wan2_1_VAE


def _assert_materialised(model: torch.nn.Module, tag: str) -> None:
    """Fail if any parameter, buffer or tensor attribute is still on meta."""
    left = []
    for module_name, module in model.named_modules():
        for name, value in module.__dict__.items():
            if name in ("_parameters", "_buffers"):
                continue
            if isinstance(value, torch.Tensor) and value.is_meta:
                left.append(f"{module_name}.{name}")
        for name, value in list(module._parameters.items()) + list(
                module._buffers.items()):
            if value is not None and value.is_meta:
                left.append(f"{module_name}.{name}")
    if left:
        raise RuntimeError(
            f"{tag}: tensors not loaded from the checkpoint: {left[:10]}"
            f"{' ...' if len(left) > 10 else ''}")


# ---------------------------------------------------------------------------
# small utilities (mirror the dataset/training conventions verbatim)
# ---------------------------------------------------------------------------

def _build_i2v_mask_4ch(t_lat: int, h_lat: int, w_lat: int,
                        device, dtype) -> torch.Tensor:
    """Build the 4-channel "1,0,0,..." mask used as the first 4 channels
    of ``y``.  Mirrors ``bi_camera_wan22_w_mem_dataset._build_i2v_mask``.
    """
    m = torch.ones(1, t_lat * 4 - 3, h_lat, w_lat, device=device, dtype=dtype)
    m[:, 1:] = 0
    m = torch.cat([m[:, 0:1].repeat_interleave(4, dim=1), m[:, 1:]], dim=1)
    m = m.view(1, m.shape[1] // 4, 4, h_lat, w_lat).transpose(1, 2)[0]
    return m  # [4, t_lat, h_lat, w_lat]


def _encode_i2v_clip(vae, img_pixel: torch.Tensor, t_lat: int,
                     h_pix: int, w_pix: int, device,
                     reuse_zero_tail: bool = False,
                     lazy_zeros: bool = False) -> torch.Tensor:
    """VAE-encode the I2V pixel clip ``[image, zeros..., zeros]`` into a
    16-channel latent of temporal length ``t_lat``.

    Mirrors the upstream :meth:`WanI2V.generate` recipe verbatim::

        y = vae.encode([
            cat([
                F.interpolate(img[None], (h, w), mode='bicubic').transpose(0,1),
                zeros(3, F_pix - 1, h, w),
            ], dim=1)
        ])[0]

    where ``F_pix = t_lat * 4 - 3`` so the temporal-causal VAE produces
    exactly ``t_lat`` latent frames.  ``img_pixel`` is on ``device``,
    shape ``[3, H_in, W_in]``, value range [-1, 1].
    """
    F_pix = t_lat * 4 - 3
    if lazy_zeros and reuse_zero_tail and isinstance(vae, Wan2_1_VAE):
        head = torch.nn.functional.interpolate(
            img_pixel[None].cpu(), size=(h_pix, w_pix),
            mode='bilinear').transpose(0, 1).to(device)      # [3, 1, h, w]
        return vae.encode_zero_tail(head, F_pix)
    clip = torch.cat([
        torch.nn.functional.interpolate(
            img_pixel[None].cpu(), size=(h_pix, w_pix),
            mode='bilinear').transpose(0, 1),                 # [3, 1, h, w]
        torch.zeros(3, F_pix - 1, h_pix, w_pix),
    ], dim=1).to(device)                                     # [3, F_pix, h, w]
    if reuse_zero_tail and isinstance(vae, Wan2_1_VAE):
        return vae.encode([clip], reuse_zero_tail=True)[0]
    return vae.encode([clip])[0]                             # [16, t_lat, h, w]


def _build_y(vae, img_pixel: torch.Tensor, t_lat: int,
             h_pix: int, w_pix: int, device, dtype,
             reuse_zero_tail: bool = False,
             lazy_zeros: bool = False) -> torch.Tensor:
    """Construct the 20-channel y volume = [4-mask | 16-image-clip-latent]
    using the same recipe as upstream :meth:`WanI2V.generate`.
    """
    y_img = _encode_i2v_clip(vae, img_pixel, t_lat, h_pix, w_pix, device,
                             reuse_zero_tail, lazy_zeros)
    h_lat, w_lat = y_img.shape[2], y_img.shape[3]
    msk = _build_i2v_mask_4ch(t_lat, h_lat, w_lat, device, y_img.dtype)
    return torch.cat([msk, y_img], dim=0).to(dtype=dtype)    # [20, t_lat, h_lat, w_lat]


# Trailing HR frames the windowed memory compressor reads.
COMPRESS_WINDOW = 32


def _memory_counts(memory_dict):
    return {key: memory_dict[key] for key in (
        "rope_shift_idx", "sink_token_count", "cmp_token_count",
        "cmp_chunk_token_count", "tmp_token_count") if key in memory_dict}


@contextmanager
def _noop_no_sync():
    yield


def _parameter_dtype_summary(module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        name = str(parameter.dtype).replace("torch.", "")
        counts[name] = counts.get(name, 0) + parameter.numel()
    return counts


def _fsdp_unit_count(module) -> int:
    return sum(
        1 for child in module.modules() if isinstance(child, FSDP)
    )


class InferenceMode(str, Enum):
    BI = "bi"
    AR = "ar"
    PDD = "pdd"


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

class WorldPlay2Pipeline:
    """WorldPlay2 pipeline for chunked action-conditioned video generation.

    Owns: T5 encoder, Wan2.1 VAE, two ``WorldPlay2Model`` experts (low / high
    noise) with the action MLP + memory_compress submodule installed.
    """

    def __init__(
        self,
        config,
        checkpoint_dir: str,
        low_noise_mem_ckpt: str,
        high_noise_mem_ckpt: str,
        device_id: int = 0,
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        use_sp: bool = False,
        compile_qk_rope: bool = True,
        conditioning_cache: bool = True,
        fuse_self_qkv: bool = True,
        compile_block_glue: bool = True,
        compile_ffn: bool = True,
        fp8_ffn: bool = False,
        fp8_start_block: int = 4,
        fp8_end_block: int = 35,
        fp8_qkv: bool = False,
        fp8_qkv_start_block: int = 4,
        fp8_qkv_end_block: int = 35,
        init_on_cpu: bool = True,
        inference_mode: InferenceMode | str = InferenceMode.BI,
        vae_type: str = "wan2_1",
        tae_ckpt: str = None,
        attention_backend: str = "flash",
        exact_optimizations: bool = False,
    ):
        """
        Args:
            config: model settings from ``configs.I2V_A14B_CONFIG``.
            checkpoint_dir: directory holding the base T5 / VAE / DiT
                checkpoints (same layout :class:`WanI2V` expects).
            low_noise_mem_ckpt: path to a ``.safetensors``/``.pt`` file
                holding the LOW-noise expert's trained ``action_in.*``,
                ``img_action_mlp.*`` and ``memory_compress.*`` weights.
            high_noise_mem_ckpt: same, but for the HIGH-noise expert.
                The two experts are trained separately (see
                ``run_bi_wan22_action_mem_stage_three.sh`` with /
                without ``--low_noise``), so each carries its own
                memory_compress + action weights and at inference we
                must run TWO compressor passes per chunk and dispatch
                by timestep just like the denoise step does.
            device_id, rank, t5_fsdp, dit_fsdp, use_sp, init_on_cpu:
                runtime placement options.
            attention_backend: visual denoising attention backend, 'flash'
                or 'sage'.
        """
        assert low_noise_mem_ckpt is not None and high_noise_mem_ckpt is not None, (
            "WorldPlay2Pipeline requires BOTH --low_noise_mem_ckpt and "
            "--high_noise_mem_ckpt; the FastVideo training pipeline "
            "produces a separate checkpoint for each expert (each with "
            "its own memory_compress + action weights) and the inference "
            "loop dispatches between them by timestep.")

        self.inference_mode = InferenceMode(inference_mode)
        if exact_optimizations and (dit_fsdp or fp8_ffn or fp8_qkv):
            raise ValueError("Exact conditioning reuse requires unquantized, non-FSDP experts")
        self.exact_optimizations = exact_optimizations
        self.lazy_zero_clip = False
        self.kv_inplace = False
        self.compress_window = False
        self.slim_memory = False
        self.chunk_stream = False
        self.noise_mode = "horizon"   # "horizon" or "chunk"
        self._zero_tails = {}
        self._window_buffers = None
        self.attention_backend = resolve_attention_backend(attention_backend)
        logging.info("Visual self-attention backend: %s", self.attention_backend)
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.init_on_cpu = init_on_cpu
        self.denoise_forward_mode = (
            "bi_denoise"
            if self.inference_mode is InferenceMode.BI
            else (
                "pdd_denoise"
                if self.inference_mode is InferenceMode.PDD
                else "ar_denoise"
            )
        )
        self.pdd_scheduler = (
            FixedPDD4Scheduler()
            if self.inference_mode is InferenceMode.PDD
            else None
        )
        self.use_sp = use_sp
        self.compile_qk_rope = compile_qk_rope
        self.conditioning_cache_enabled = conditioning_cache
        self.fuse_self_qkv = fuse_self_qkv
        self.compile_block_glue = compile_block_glue
        self.compile_ffn = compile_ffn
        self.fp8_ffn_enabled = fp8_ffn
        self.fp8_start_block = fp8_start_block
        self.fp8_end_block = fp8_end_block
        self.fp8_qkv_enabled = fp8_qkv
        self.fp8_qkv_start_block = fp8_qkv_start_block
        self.fp8_qkv_end_block = fp8_qkv_end_block
        self.fp8_enabled = fp8_ffn or fp8_qkv
        if self.fp8_enabled and dit_fsdp:
            raise ValueError(
                "DiT FP8 currently requires --dit_fsdp to be disabled")
        if fp8_qkv and not fuse_self_qkv:
            raise ValueError(
                "QKV FP8 requires self-attention QKV fusion")

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        # VAE is pluggable: the default Wan2.1 VAE, or the causal TAE
        # (lightvae) which exposes the SAME encode/decode/encode_chunk/
        # decode_chunk list interface via CausalTAEVAE, so nothing downstream
        # changes.  Both are temporally causal (T_pix = 4*T_lat - 3).
        assert vae_type in ("wan2_1", "causal_tae"), (
            f"vae_type must be 'wan2_1' or 'causal_tae', got {vae_type!r}")
        if vae_type == "causal_tae":
            from worldplay2.vae.causal_tae import CausalTAEVAE
            assert tae_ckpt is not None, (
                "vae_type='causal_tae' requires --tae_ckpt "
                "(the CausalTAE encoder+decoder weights).")
            self.vae = CausalTAEVAE(
                vae_pth=tae_ckpt,
                z_dim=16,
                device=self.device)
        else:
            self.vae = Wan2_1_VAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device)

        # Build the architecture from the base model config only. The supplied
        # expert checkpoints contain the complete DiT + action + compressor
        # state, so loading base-model parameters through from_pretrained would
        # be redundant and would temporarily consume additional host memory.
        # Each expert is materialised directly where it will run: on its GPU
        # unless it is kept on the host (CPU offload) or sharded from the host
        # (FSDP).
        load_device = (
            torch.device("cpu") if (self.init_on_cpu or dit_fsdp)
            else self.device
        )
        self.low_noise_model = self._build_expert_from_config(
            checkpoint_dir=checkpoint_dir,
            config_subfolder=config.low_noise_checkpoint,
            expert_ckpt=low_noise_mem_ckpt,
            tag="low_noise_model",
            dtype=self.param_dtype,
            device=load_device,
            fuse_self_qkv=fuse_self_qkv,
            pdd_scheduler=self.pdd_scheduler,
            pdd_expert="low",
        )
        self.high_noise_model = self._build_expert_from_config(
            checkpoint_dir=checkpoint_dir,
            config_subfolder=config.high_noise_checkpoint,
            expert_ckpt=high_noise_mem_ckpt,
            tag="high_noise_model",
            dtype=self.param_dtype,
            device=load_device,
            fuse_self_qkv=fuse_self_qkv,
            pdd_scheduler=self.pdd_scheduler,
            pdd_expert="high",
        )

        # ---- configure (FSDP / SP / dtype / device) ----------------------
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            attention_backend=self.attention_backend,
            compile_qk_rope=compile_qk_rope,
            conditioning_cache=conditioning_cache,
            compile_block_glue=compile_block_glue,
            compile_ffn=compile_ffn,
            fp8_ffn=fp8_ffn,
            fp8_start_block=fp8_start_block,
            fp8_end_block=fp8_end_block,
            fp8_qkv=fp8_qkv,
            fp8_qkv_start_block=fp8_qkv_start_block,
            fp8_qkv_end_block=fp8_qkv_end_block,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            attention_backend=self.attention_backend,
            compile_qk_rope=compile_qk_rope,
            conditioning_cache=conditioning_cache,
            compile_block_glue=compile_block_glue,
            compile_ffn=compile_ffn,
            fp8_ffn=fp8_ffn,
            fp8_start_block=fp8_start_block,
            fp8_end_block=fp8_end_block,
            fp8_qkv=fp8_qkv,
            fp8_qkv_start_block=fp8_qkv_start_block,
            fp8_qkv_end_block=fp8_qkv_end_block,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn)

        self.sp_size = get_world_size() if use_sp else 1
        self.sample_neg_prompt = config.sample_neg_prompt
        self.dtype_report = {
            "attention_backend": self.attention_backend,
            "configured_dit_compute_dtype": str(
                self.param_dtype).replace("torch.", ""),
            "t5_fsdp_requested": bool(t5_fsdp),
            "dit_fsdp": bool(dit_fsdp),
            "t5_fsdp_units": _fsdp_unit_count(self.text_encoder.model),
            "low_noise_fsdp_units": _fsdp_unit_count(
                self.low_noise_model),
            "high_noise_fsdp_units": _fsdp_unit_count(
                self.high_noise_model),
            "compile_qk_rope": bool(compile_qk_rope),
            "conditioning_cache": bool(conditioning_cache),
            "self_qkv_fused": bool(fuse_self_qkv),
            "compile_block_glue": bool(compile_block_glue),
            "compile_ffn": bool(compile_ffn),
            "fp8_ffn": bool(fp8_ffn),
            "fp8_block_range": (
                [fp8_start_block, fp8_end_block] if fp8_ffn else None
            ),
            "fp8_qkv": bool(fp8_qkv),
            "fp8_qkv_block_range": (
                [fp8_qkv_start_block, fp8_qkv_end_block]
                if fp8_qkv else None
            ),
            "low_noise_parameter_storage": _parameter_dtype_summary(
                self.low_noise_model),
            "high_noise_parameter_storage": _parameter_dtype_summary(
                self.high_noise_model),
            "t5_configured_dtype": str(
                config.t5_dtype).replace("torch.", ""),
            "t5_parameter_storage": _parameter_dtype_summary(
                self.text_encoder.model),
            "vae_configured_dtype": str(
                self.vae.dtype).replace("torch.", ""),
            "vae_parameter_storage": _parameter_dtype_summary(
                self.vae.model),
        }
        logging.info("Dtype report: %s", self.dtype_report)

    # ------------------------------------------------------------------
    # ckpt loading + model configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _build_expert_from_config(
        checkpoint_dir: str,
        config_subfolder: str,
        expert_ckpt: str,
        tag: str,
        dtype: torch.dtype,
        fuse_self_qkv: bool,
        pdd_scheduler: FixedPDD4Scheduler | None = None,
        pdd_expert: str | None = None,
        device: torch.device | str = "cpu",
    ) -> WorldPlay2Model:
        """Initialize architecture from config and strictly load a full expert.

        The architecture is created on the meta device, so no parameter is
        allocated or randomly initialised; the checkpoint is then read straight
        onto ``device`` in ``dtype`` and its tensors become the parameters
        (``assign=True``). Every parameter must come from the checkpoint
        (strict loading), so the result does not depend on how the module
        would have been initialised. Tensors the model computes itself at
        construction (the RoPE table) are built on the host, as before.
        """
        config_dir = os.path.join(checkpoint_dir, config_subfolder)
        logging.info("Creating %s from config on the meta device: %s",
                     tag, config_dir)
        model_config = WorldPlay2Model.load_config(config_dir)
        with torch.device("meta"):
            model = WorldPlay2Model.from_config(model_config)
            model.add_embedding_action_parameters()
            model.add_memory_compress_model()
            if pdd_scheduler is not None:
                model.head.make_fixed_pdd(num_blocks=2)
        if pdd_scheduler is None:
            load_expert_checkpoint(
                model, expert_ckpt, tag=tag, strict=True,
                device=device, dtype=dtype, assign=True)
        else:
            state = pdd_scheduler.prepare_checkpoint(
                expert_ckpt, expert=pdd_expert, dtype=dtype, device=device
            )
            provided = set(state)
            required_prefixes = (
                "memory_compress.",
                "action_in_cont.",
                "action_in_disc.",
            )
            absent = [
                prefix for prefix in required_prefixes
                if not any(key.startswith(prefix) for key in provided)
            ]
            if absent:
                raise ValueError(
                    f"{tag} PDD checkpoint is missing required parameter "
                    f"groups: {absent}"
                )
            model.load_state_dict(state, strict=True, assign=True)
            del state
            logging.info(
                "%s fixed-PDD checkpoint: loaded=%d expert=%s",
                tag,
                len(provided),
                pdd_expert,
            )
        _assert_materialised(model, tag)
        # Install after strict checkpoint loading (which expects q/k/v keys)
        # and before FSDP wrapping (which should only see the fused weights).
        if fuse_self_qkv:
            model.fuse_self_attention_qkv()
        return model

    def _configure_model(
        self,
        model,
        use_sp,
        attention_backend,
        compile_qk_rope,
        conditioning_cache,
        compile_block_glue,
        compile_ffn,
        fp8_ffn,
        fp8_start_block,
        fp8_end_block,
        fp8_qkv,
        fp8_qkv_start_block,
        fp8_qkv_end_block,
        dit_fsdp,
        shard_fn,
    ):
        """Configure dtype/device/FSDP and enable the model-owned SP path."""
        model.eval().requires_grad_(False)
        model.to(dtype=self.param_dtype)

        model.set_sequence_parallel(use_sp)
        model.set_self_attention_backend(attention_backend)
        model.set_compiled_qk_rope(compile_qk_rope)
        model.set_conditioning_cache(conditioning_cache)
        model.set_compiled_regions(
            block_glue=compile_block_glue,
            ffn=compile_ffn,
        )

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        if fp8_ffn or fp8_qkv:
            # TE primary FP8 weights must remain resident. This path
            # intentionally does not support DiT FSDP or model offload.
            model.to(self.device)
        if fp8_qkv:
            report = convert_model_qkv_to_fp8(
                model,
                start_block=fp8_qkv_start_block,
                end_block=fp8_qkv_end_block,
            )
            logging.info("Installed self-attention QKV FP8: %s", report)
        if fp8_ffn:
            report = convert_model_ffn_to_fp8(
                model,
                start_block=fp8_start_block,
                end_block=fp8_end_block,
            )
            logging.info("Installed FFN FP8: %s", report)

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    @staticmethod
    def _find_wan_model(model):
        for module in model.modules():
            if isinstance(module, WorldPlay2Model):
                return module
        raise RuntimeError("FSDP/model wrapper does not contain WorldPlay2Model")

    def clear_conditioning_cache(self) -> None:
        self._find_wan_model(
            self.low_noise_model).clear_conditioning_cache()
        self._find_wan_model(
            self.high_noise_model).clear_conditioning_cache()

    def fp8_stats(self) -> dict:
        return {
            "low_noise": fp8_model_stats(
                self._find_wan_model(self.low_noise_model)),
            "high_noise": fp8_model_stats(
                self._find_wan_model(self.high_noise_model)),
        }

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        """Mirrors :meth:`WanI2V._prepare_model_for_timestep`."""
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            other = getattr(self, offload_model_name)
            active = getattr(self, required_model_name)
            if next(other.parameters()).device.type == 'cuda':
                # Cached tensors are plain inference state and are not moved by
                # Module.to(). Drop them before offloading the expert.
                self._find_wan_model(other).clear_conditioning_cache()
                other.to('cpu')
            if next(active.parameters()).device.type == 'cpu':
                active.to(self.device)
        return getattr(self, required_model_name)

    def _prepare_named_expert(self, expert: str, offload_model: bool):
        required = f"{expert}_noise_model"
        other = "low_noise_model" if expert == "high" else "high_noise_model"
        if offload_model or self.init_on_cpu:
            other_model = getattr(self, other)
            active_model = getattr(self, required)
            if next(other_model.parameters()).device.type == "cuda":
                self._find_wan_model(other_model).clear_conditioning_cache()
                other_model.to("cpu")
            if next(active_model.parameters()).device.type == "cpu":
                active_model.to(self.device)
        return getattr(self, required)

    def _run_pdd_chunk(
        self,
        *,
        latent,
        y_chunk,
        hr_action_chunk,
        context,
        memory_dict_high,
        memory_dict_low,
        kv_high,
        kv_low,
        chunk_start,
        seed_g,
    ):
        for block in self.pdd_scheduler.ordered_blocks:
            model = self._prepare_named_expert(
                block.expert,
                self.offload_model,
            )

            kv_cache = kv_high if block.expert == "high" else kv_low
            memory_dict = (
                memory_dict_high
                if block.expert == "high"
                else memory_dict_low
            )
            cpu_timestep = self.pdd_scheduler.timesteps[block.start:block.start + 1]
            if self.exact_optimizations:
                timestep, time_conditioning = model.prepare_time_conditioning(cpu_timestep, self.seq_len_chunk)
            else:
                timestep, time_conditioning = cpu_timestep.to(self.device), None
            args = dict(
                context=[context[0]],
                seq_len=self.seq_len_chunk,
                y=[y_chunk],
                hr_action=hr_action_chunk,
                forward_mode="pdd_denoise",
                pdd_block_index=block.local_index,
                kv_cache=kv_cache,
                ar_chunk_start=chunk_start,
                memory_dict=memory_dict,
                time_conditioning=time_conditioning,
            )
            output = model(
                [latent.to(self.device)],
                t=timestep,
                **args,
            )[0]
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(
                    "fixed PDD model must return "
                    "(displacement, endpoint_velocity)"
                )
            displacement, endpoint_velocity = output
            sigma_endpoint = self.pdd_scheduler.sigmas[block.end - 1].to(device=latent.device, dtype=torch.float32)
            state_endpoint = latent.float() + displacement.float()
            x0 = (state_endpoint - sigma_endpoint * endpoint_velocity.float()).to(latent.dtype)

            sigma_end = self.pdd_scheduler.sigmas[block.end].to(device=latent.device, dtype=torch.float32)
            if block.end == self.pdd_scheduler.num_intervals:
                latent = x0
            else:
                noise = torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype)
                latent = ((1.0 - sigma_end) * x0.float() + sigma_end * noise.float()).to(x0.dtype)
        return latent

    def _build_chunk_scheduler(
        self,
        sample_solver: str,
        sampling_steps: int,
        shift: float,
    ):
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(
                sampling_steps,
                device=self.device,
                shift=shift,
            )
            return scheduler, scheduler.timesteps
        if sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
            return scheduler, timesteps
        raise NotImplementedError(f"Unsupported solver {sample_solver!r}")

    def _predict_chunk_noise(
        self,
        *,
        latent,
        timestep_value,
        boundary,
        context,
        context_null,
        memory_dict_high,
        memory_dict_low,
        kv_high,
        kv_high_neg,
        kv_low,
        kv_low_neg,
        y_chunk,
        hr_action_chunk,
        ar_cache,
        chunk_start,
    ):
        latent_model_input = [latent.to(self.device)]
        timestep = torch.stack([timestep_value]).to(self.device)
        model = self._prepare_model_for_timestep(
            timestep_value,
            boundary,
            self.offload_model,
        )

        if timestep_value.item() >= boundary:
            memory_dict = memory_dict_high
            kv_cache, kv_cache_neg = kv_high, kv_high_neg
            sample_guide_scale = self.guide_scale[1]
            use_cfg = self.use_cfg_high
        else:
            memory_dict = memory_dict_low
            kv_cache, kv_cache_neg = kv_low, kv_low_neg
            sample_guide_scale = self.guide_scale[0]
            use_cfg = self.use_cfg_low

        model_args = {
            "context": [context[0]],
            "seq_len": self.seq_len_chunk,
            "y": [y_chunk],
            "hr_action": hr_action_chunk,
            "forward_mode": self.denoise_forward_mode,
        }
        if ar_cache:
            model_args.update(
                kv_cache=kv_cache,
                ar_chunk_start=chunk_start,
            )
        else:
            model_args["memory_dict"] = memory_dict

        noise_pred_cond = model(
            latent_model_input,
            t=timestep,
            **model_args,
        )[0]
        if use_cfg:
            null_args = {
                "context": context_null,
                "seq_len": self.seq_len_chunk,
                "y": [y_chunk],
                "hr_action": hr_action_chunk,
                "forward_mode": self.denoise_forward_mode,
            }
            if ar_cache:
                null_args.update(
                    kv_cache=kv_cache_neg,
                    ar_chunk_start=chunk_start,
                )
            else:
                null_args["memory_dict"] = memory_dict
            noise_pred_uncond = model(
                latent_model_input,
                t=timestep,
                **null_args,
            )[0]
            if self.offload_model:
                torch.cuda.empty_cache()
            noise_pred_cond = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond
            )
        return noise_pred_cond

    def _run_solver_chunk(
        self,
        *,
        latent,
        sample_solver,
        sampling_steps,
        shift,
        seed_g,
        boundary,
        context,
        context_null,
        memory_dict_high,
        memory_dict_low,
        kv_high,
        kv_high_neg,
        kv_low,
        kv_low_neg,
        y_chunk,
        hr_action_chunk,
        ar_cache,
        chunk_start,
        chunk_index,
    ):
        scheduler, timesteps = self._build_chunk_scheduler(
            sample_solver,
            sampling_steps,
            shift,
        )
        if self.offload_model:
            torch.cuda.empty_cache()

        for timestep_value in tqdm(timesteps, desc=f"chunk {chunk_index}"):
            noise_pred = self._predict_chunk_noise(
                latent=latent,
                timestep_value=timestep_value,
                boundary=boundary,
                context=context,
                context_null=context_null,
                memory_dict_high=memory_dict_high,
                memory_dict_low=memory_dict_low,
                kv_high=kv_high,
                kv_high_neg=kv_high_neg,
                kv_low=kv_low,
                kv_low_neg=kv_low_neg,
                y_chunk=y_chunk,
                hr_action_chunk=hr_action_chunk,
                ar_cache=ar_cache,
                chunk_start=chunk_start,
            )
            latent = scheduler.step(
                noise_pred.unsqueeze(0),
                timestep_value,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0].squeeze(0)
        return latent

    def _prefill_kv_cache(
        self,
        model,
        memory_dict,
        kv_cache,
        kv_cache_neg,
        context,
        context_null,
        use_cfg,
    ):
        """Prefill conditional cache and the optional CFG cache.

        The ctx self-attn K/V pass through cross-attention (text), so the
        cond and uncond ctx K/V differ -> two separate caches.  When
        ``memory_dict`` is None (first chunk) prefill is a no-op and the
        caches stay empty (denoise then runs pred-only attention).

        Returns the (possibly mutated) ``(kv_cache, kv_cache_neg)`` so the
        caller rebinds explicitly rather than relying on in-place mutation.
        """
        if memory_dict is None:
            return kv_cache, kv_cache_neg
        kv_cache = model(
            forward_mode="ar_prefill",
            context=[context[0]],
            memory_dict=memory_dict,
            kv_cache=kv_cache,
        )
        if use_cfg:
            if context_null is None or kv_cache_neg is None:
                raise ValueError("CFG prefill requires null context and cache")
            kv_cache_neg = model(
                forward_mode="ar_prefill",
                context=context_null,
                memory_dict=memory_dict,
                kv_cache=kv_cache_neg,
            )
        return kv_cache, kv_cache_neg

    # ------------------------------------------------------------------
    # streaming VAE helpers
    # ------------------------------------------------------------------

    def _decode_chunk_and_lr_encode(
        self,
        hr_latent_chunk: torch.Tensor,
        is_first_vae_chunk: bool,
    ):
        """Decode one HR latent chunk via streaming VAE, downsample
        (T::2, HW//4 in pixel space), then re-encode via streaming VAE.

        Returns (lr_latent, rgb_frames) where:
            lr_latent  : [16, T_lr_lat, h//4, w//4]
            rgb_frames : [3, T_pix, H_pix, W_pix] in [-1, 1]
        """
        leader_vae = self.use_sp and self.exact_optimizations
        if leader_vae and self.rank != 0:
            channels, frames, height, width = hr_latent_chunk.shape
            lr_lat = torch.empty(
                (channels, frames // 2, height // 4, width // 4),
                device=self.device, dtype=self.vae.dtype)
            dist.broadcast(lr_lat, src=0)
            return lr_lat, None
        rgb = self.vae.decode_chunk([hr_latent_chunk],
                                    is_first_chunk=is_first_vae_chunk)[0]

        lr_pix = rgb[:, ::2, :, :]                        # T::2
        h_pix, w_pix = lr_pix.shape[2:]
        target_size = (h_pix // 4, w_pix // 4)
        lr_pix = lr_pix.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
        lr_pix = F.interpolate(lr_pix, size=target_size, mode='bilinear',
                               align_corners=False)
        lr_pix = lr_pix.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]

        lr_lat = self.vae.encode_chunk([lr_pix],
                                       is_first_chunk=is_first_vae_chunk)[0]
        if leader_vae:
            # Rank 0 decodes; the LR latent is broadcast to the other ranks.
            lr_lat = lr_lat.contiguous()
            dist.broadcast(lr_lat, src=0)
        return lr_lat, rgb

    def _decode_last_chunk(self, latent: torch.Tensor, is_first_vae_chunk: bool):
        """Decode the stream's last chunk, which needs no LR memory: rank 0 only."""
        if self.rank != 0:
            return None
        return self.vae.decode_chunk([latent], is_first_chunk=is_first_vae_chunk)[0]

    def _build_lr_image_pixel(self, img_pixel_hr: torch.Tensor,
                              h_hr_pix: int, w_hr_pix: int) -> torch.Tensor:
        """Bicubic-downsample the HR pixel image to the LR resolution
        ``(h_hr_pix//4, w_hr_pix//4)`` -- matches the (H//4, W//4) target
        used by :meth:`_decode_chunk_and_lr_encode`.  Returns ``[3, H_lr, W_lr]``
        on ``self.device``.
        """
        return F.interpolate(
            img_pixel_hr[None].cpu(),
            size=(h_hr_pix // 4, w_hr_pix // 4),
            mode='bilinear')[0].to(self.device)

    def _prepare_chunk_memory(
        self,
        *,
        chunk_index,
        chunk_end,
        latents,
        hr_y_full,
        lr_y_full,
        hr_action,
        lr_action,
        context,
        context_null,
        kv_high,
        kv_high_neg,
        kv_low,
        kv_low_neg,
        return_cpu=True,
    ):
        """Decode a generated chunk and prepare memory for the next chunk."""
        chunk_start = chunk_end - self.chunk_length
        generated_latents = latents[:, chunk_start:chunk_end].contiguous()
        new_lr_latents, decoded_rgb = self._decode_chunk_and_lr_encode(
            generated_latents,
            is_first_vae_chunk=chunk_index == 0,
        )
        if self.lr_latents_cache is None:
            self.lr_latents_cache = new_lr_latents
        else:
            self.lr_latents_cache = torch.cat(
                [self.lr_latents_cache, new_lr_latents],
                dim=1,
            )

        window = self.compress_window and self.ar_cache and chunk_index > 0
        hr_start = max(0, chunk_end - COMPRESS_WINDOW) if window else 0
        lr_start = (chunk_end - self.chunk_length) // 2 if window else 0
        hr_context_latents = latents[:, hr_start:chunk_end].contiguous()
        hr_context_y = hr_y_full[:, hr_start:chunk_end]
        lr_context_y = lr_y_full[:, lr_start:chunk_end // 2]
        hr_input = torch.cat([hr_context_latents, hr_context_y], dim=0).to(dtype=self.param_dtype).unsqueeze(0)
        lr_input = torch.cat(
            [self.lr_latents_cache[:, lr_start:], lr_context_y],
            dim=0,
        ).to(dtype=self.param_dtype).unsqueeze(0)

        temporal_latents = hr_context_latents[:, -self.temporal_size:].to(dtype=self.param_dtype)
        temporal_y = hr_y_full[:, chunk_end - self.temporal_size:chunk_end].to(dtype=self.param_dtype)
        temporal_input = torch.cat([temporal_latents, temporal_y], dim=0).unsqueeze(0)

        compressor_args = {
            "forward_mode": "compress_window" if window else "compress",
            "hr_latents_input": hr_input,
            "lr_latents_input": lr_input,
            "lr_action": lr_action[:chunk_end // 2],
            "memory_t": torch.zeros(1, device=self.device, dtype=self.param_dtype),
            "hr_temporal_latents_input": temporal_input,
            "hr_temporal_action": hr_action[chunk_end - self.temporal_size:chunk_end],
        }
        if window:
            compressor_args.update(hr_frames=chunk_end, sink_frames=self.sink_size)
        else:
            compressor_args["sink_action"] = (
                hr_action[:self.sink_size]
                if self.sink_size > 0
                else None
            )

        if self.offload_model:
            if next(self.high_noise_model.parameters()).device.type != "cuda":
                self.high_noise_model.to(self.device)
            if next(self.low_noise_model.parameters()).device.type == "cuda":
                self.low_noise_model.to("cpu")
        memory_dict_high = self.high_noise_model(**compressor_args)
        if self.ar_cache:
            kv_high, kv_high_neg = self._prefill_kv_cache(
                self.high_noise_model,
                memory_dict_high,
                kv_high,
                kv_high_neg,
                context=context,
                context_null=context_null,
                use_cfg=self.use_cfg_high,
            )

        if self.offload_model:
            if next(self.low_noise_model.parameters()).device.type != "cuda":
                self.low_noise_model.to(self.device)
            if next(self.high_noise_model.parameters()).device.type == "cuda":
                self.high_noise_model.to("cpu")
        memory_dict_low = self.low_noise_model(**compressor_args)
        if self.ar_cache:
            kv_low, kv_low_neg = self._prefill_kv_cache(
                self.low_noise_model,
                memory_dict_low,
                kv_low,
                kv_low_neg,
                context=context,
                context_null=context_null,
                use_cfg=self.use_cfg_low,
            )

        if self.slim_memory and self.ar_cache:
            memory_dict_high = _memory_counts(memory_dict_high)
            memory_dict_low = _memory_counts(memory_dict_low)
        return (
            decoded_rgb.cpu() if return_cpu and decoded_rgb is not None else decoded_rgb,
            memory_dict_high,
            memory_dict_low,
            kv_high,
            kv_high_neg,
            kv_low,
            kv_low_neg,
        )

    # ------------------------------------------------------------------
    # public entry: chunked autoregressive generation
    # ------------------------------------------------------------------

    def _y_encoder(self, latent_rows):
        """The causal-encoder step used for y."""
        from .chunk_stream import FrameEncoder

        factory = getattr(self, "y_band_encoder", None)
        return factory(latent_rows) if factory is not None else FrameEncoder(self.vae)

    def _zero_tail(self, encoder, height, width):
        from .chunk_stream import zero_tail

        key = (height, width, self.vae.dtype, type(encoder).__name__)
        if key not in self._zero_tails:
            self._zero_tails[key] = zero_tail(encoder, height, width, self.param_dtype)
        return self._zero_tails[key]

    def stream_world(
        self,
        *,
        img,
        seed: int,
        num_chunks: int,
        chunk_input,
        height: int = 448,
        width: int = 832,
        chunk_length: int = 4,
        sink_size: int = 1,
        temporal_size: int = 1,
        return_cpu: bool = True,
        prepare: bool = False,
        prompt: str | None = None,
    ):
        """Return a generator yielding one RGB chunk at a time.

        ``chunk_input(index)`` returns ``(prompt_text, action_rows [chunk_length, 6])`` for
        chunk ``index``. Fixed-PDD, chunk-AR inference with guide scale 1 only.

        The world's setup runs when this is called, not on the first chunk. With ``prepare``
        it also encodes the image conditioning and ``prompt`` before the first chunk.
        """
        from .chunk_stream import _CausalY, _LatentWindow

        if self.inference_mode is not InferenceMode.PDD or not self.compress_window:
            raise RuntimeError("stream_world needs fixed-PDD inference and the windowed compressor")
        self.clear_conditioning_cache()
        self.chunk_length = chunk_length
        self.temporal_size = temporal_size
        self.sink_size = sink_size
        self.offload_model = False
        self.ar_cache = True
        self.guide_scale = (1.0, 1.0)
        self.use_cfg_high = self.use_cfg_low = False
        self.lr_latents_cache = None
        total_latent_frames = num_chunks * chunk_length

        # ---- this world's first frame (the only image-dependent input) ----
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        h, w = height, width
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]
        img_pixel = F.interpolate(img[None].cpu(), size=(h, w), mode='bicubic')[0].to(self.device)
        img_pixel_lr = self._build_lr_image_pixel(img_pixel, h, w)

        def first_frame(pixel, fh, fw):
            return F.interpolate(pixel[None].cpu(), size=(fh, fw), mode='bilinear').transpose(0, 1).to(
                self.device).unsqueeze(0)                         # [1, 3, 1, fh, fw]

        hr_enc = self._y_encoder(lat_h)
        lr_enc = self._y_encoder(lat_h // 4)
        hr_y = _CausalY(hr_enc, first_frame(img_pixel, h, w), self._zero_tail(hr_enc, h, w), self.param_dtype)
        lr_y = _CausalY(lr_enc, first_frame(img_pixel_lr, h // 4, w // 4),
                        self._zero_tail(lr_enc, h // 4, w // 4), self.param_dtype)

        # ---- noise ----
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise_full = None
        if self.noise_mode == "horizon":
            noise_full = torch.randn(16, total_latent_frames, lat_h, lat_w,
                                     dtype=torch.float32, generator=seed_g, device=self.device)

        # ---- persistent state ----
        self.seq_len_chunk = chunk_length * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        frame_tokens = self.seq_len_chunk // chunk_length
        cmp_frame_tokens = ((lat_h // 4) // self.patch_size[1]) * ((lat_w // 4) // self.patch_size[2])
        capacity = ((sink_size + temporal_size) * frame_tokens
                    + (num_chunks - 1) * (chunk_length // 2) * cmp_frame_tokens
                    + self.seq_len_chunk) if self.kv_inplace else 0
        for expert in (self.high_noise_model, self.low_noise_model):
            self._find_wan_model(expert).kv_capacity = capacity
        kv_high = self.high_noise_model.init_kv_cache()
        kv_low = self.low_noise_model.init_kv_cache()
        shape = (16, COMPRESS_WINDOW + chunk_length, lat_h, lat_w)
        if self._window_buffers is None or tuple(self._window_buffers[0].shape) != shape:
            self._window_buffers = [torch.empty(shape, device=self.device, dtype=torch.float32)
                                    for _ in range(2)]
        latents = _LatentWindow(self._window_buffers, total_latent_frames)
        if prepare:
            with torch.no_grad():
                hr_y._extend(total_latent_frames)
                lr_y._extend(total_latent_frames // 2)
                if prompt is not None:
                    self.encode_prompt(prompt)
            torch.cuda.current_stream().synchronize()
        return self._world_chunks(
            num_chunks=num_chunks, chunk_input=chunk_input, chunk_length=chunk_length, seed=seed,
            lat_h=lat_h, lat_w=lat_w, hr_y=hr_y, lr_y=lr_y, noise_full=noise_full, seed_g=seed_g,
            kv_high=kv_high, kv_low=kv_low, latents=latents, return_cpu=return_cpu)

    def encode_prompt(self, text):
        """T5 context of ``text``, encoded once and kept (a few recent prompts)."""
        contexts = self.__dict__.setdefault("_text_contexts", {})
        if text not in contexts:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                contexts[text] = self.text_encoder([text], self.device)
            while len(contexts) > 8:
                contexts.pop(next(iter(contexts)))
        return contexts[text]

    def _world_chunks(self, *, num_chunks, chunk_input, chunk_length, seed, lat_h, lat_w, hr_y, lr_y,
                      noise_full, seed_g, kv_high, kv_low, latents, return_cpu):
        hr_action = torch.empty(0, 6, device=self.device)
        memory_dict_high = memory_dict_low = None

        for chunk_i in range(num_chunks):
            with (torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad()):
                s = chunk_i * chunk_length
                e = s + chunk_length
                prompt_text, action_rows = chunk_input(chunk_i)
                if tuple(action_rows.shape) != (chunk_length, 6):
                    raise ValueError("chunk_input must supply [chunk_length, 6] actions")
                hr_action = torch.cat([hr_action, action_rows.to(self.device, hr_action.dtype)])
                lr_action = hr_action[::2]
                context = self.encode_prompt(prompt_text)
                if noise_full is not None:
                    latent = noise_full[:, s:e].contiguous()
                    chunk_g = seed_g
                else:
                    chunk_g = torch.Generator(device=self.device)
                    chunk_g.manual_seed((seed * 1_000_003 + chunk_i) % (1 << 63))
                    latent = torch.randn(16, chunk_length, lat_h, lat_w, dtype=torch.float32,
                                         generator=chunk_g, device=self.device)
                latent = self._run_pdd_chunk(
                    latent=latent,
                    y_chunk=hr_y[:, s:e],
                    hr_action_chunk=hr_action[s:e],
                    context=context,
                    memory_dict_high=memory_dict_high,
                    memory_dict_low=memory_dict_low,
                    kv_high=kv_high,
                    kv_low=kv_low,
                    chunk_start=s,
                    seed_g=chunk_g,
                )
                latents.write(s, latent.float())
                if chunk_i < num_chunks - 1:
                    (decoded_rgb, memory_dict_high, memory_dict_low,
                     kv_high, _, kv_low, _) = self._prepare_chunk_memory(
                        chunk_index=chunk_i, chunk_end=e, latents=latents,
                        hr_y_full=hr_y, lr_y_full=lr_y, hr_action=hr_action, lr_action=lr_action,
                        context=context, context_null=None, kv_high=kv_high, kv_high_neg=None,
                        kv_low=kv_low, kv_low_neg=None, return_cpu=return_cpu)
                else:
                    decoded_rgb = self._decode_last_chunk(latent, chunk_i == 0)
                    if return_cpu and decoded_rgb is not None:
                        decoded_rgb = decoded_rgb.cpu()
            yield decoded_rgb if self.rank == 0 else None
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

    def generate_chunked(self, *args, **kwargs):
        """Collect the native streaming chunks for the offline CLI."""
        chunks = list(self.stream_chunked(*args, **kwargs))
        return torch.cat(chunks, dim=1) if self.rank == 0 else None

    def stream_chunked(
        self,
        prompts: dict[str, str],
        chunk_prompts: tuple[str, ...],
        img,
        hr_action: torch.Tensor,
        total_latent_frames: int = 16,
        chunk_length: int = 4,
        shift: float = 5.0,
        sample_solver: str = 'unipc',
        sampling_steps: int = 40,
        guide_scale=5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        sink_size: int = 0,
        temporal_size: int = 2,
        height: int = 448,
        width: int = 832,
        chunk_input=None,
        return_cpu: bool = True,
    ):
        """Yield one native RGB chunk at a time, without resetting model memory.

        Optional ``chunk_input(index)`` returns the current prompt text and
        four CPU action rows. It is sampled only at a chunk boundary. Offline
        callers omit it and retain the original prompt/action schedule.

        ``sink_size`` (default 0): number of leading hr latent frames used as
        the context "sink" (``hr_latents_input[:, :, :sink_size]``, prepended
        to the front of the context).  0 disables the sink.

        ``temporal_size`` (default 2): number of trailing hr latent frames
        used as the temporal anchor (``hr_ctx_lat[:, -temporal_size:]``),
        appended after the compressed lr+hr context.
        """
        assert total_latent_frames % chunk_length == 0, (
            f"total_latent_frames ({total_latent_frames}) must be a multiple "
            f"of chunk_length ({chunk_length})")
        num_chunks = total_latent_frames // chunk_length
        if len(chunk_prompts) != num_chunks:
            raise ValueError(
                f"chunk_prompts must contain {num_chunks} entries, "
                f"got {len(chunk_prompts)}"
            )
        # Conditioning is sample-local. Never carry projected text or
        # cross-attention K/V into the next JSON item.
        self.clear_conditioning_cache()

        if isinstance(guide_scale, (int, float)):
            guide_scale = (float(guide_scale), float(guide_scale))
        else:
            guide_scale = tuple(float(scale) for scale in guide_scale)
            if len(guide_scale) != 2:
                raise ValueError(
                    "guide_scale must be a scalar or (low_noise, high_noise)"
                )
        use_cfg_low = not math.isclose(
            guide_scale[0], 1.0, rel_tol=0.0, abs_tol=1e-8)
        use_cfg_high = not math.isclose(
            guide_scale[1], 1.0, rel_tol=0.0, abs_tol=1e-8)
        use_any_cfg = use_cfg_low or use_cfg_high
        if self.inference_mode is InferenceMode.PDD and use_any_cfg:
            raise ValueError("PDD consistency inference requires guide_scale=1")
        self.chunk_length = chunk_length
        self.temporal_size = temporal_size
        self.sink_size = sink_size
        self.offload_model = offload_model
        self.ar_cache = (
            self.denoise_forward_mode in {"ar_denoise", "pdd_denoise"}
        )
        self.guide_scale = guide_scale
        self.use_cfg_high = use_cfg_high
        self.use_cfg_low = use_cfg_low
        self.lr_latents_cache = None
        logging.info(
            "CFG enabled: low_noise=%s high_noise=%s",
            use_cfg_low,
            use_cfg_high,
        )

        # ---- preprocess image ----
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        h, w = height, width
        lat_h = h // self.vae_stride[1]                   # 56
        lat_w = w // self.vae_stride[2]                   # 104
        img_pixel = F.interpolate(img[None].cpu(), size=(h, w),
                                  mode='bicubic')[0].to(self.device)

        # ---- text encode ----
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        prompt_names = tuple(dict.fromkeys(chunk_prompts))
        prompt_texts = [prompts[name] for name in prompt_names]
        encoded_prompts = self.text_encoder(prompt_texts, self.device)
        context_null = (
            self.text_encoder([n_prompt], self.device)
            if use_any_cfg else None
        )
        prompt_contexts = {
            name: [encoded]
            for name, encoded in zip(prompt_names, encoded_prompts)
        }

        # ---- pre-build hr_y_full / lr_y_full (upstream WanI2V recipe:
        # encode the full pixel clip [image, zeros..., zeros] in one shot
        # so the temporally-causal VAE produces the correct latent volume).
        with torch.no_grad():
            hr_y_full = _build_y(
                self.vae, img_pixel,
                t_lat=total_latent_frames,
                h_pix=h, w_pix=w,
                device=self.device, dtype=self.param_dtype,
                reuse_zero_tail=self.exact_optimizations,
                lazy_zeros=self.lazy_zero_clip)

            img_pixel_lr = self._build_lr_image_pixel(img_pixel, h, w)
            lr_y_full = _build_y(
                self.vae, img_pixel_lr,
                t_lat=total_latent_frames // 2,
                h_pix=h // 4, w_pix=w // 4,
                device=self.device, dtype=self.param_dtype,
                reuse_zero_tail=self.exact_optimizations,
                lazy_zeros=self.lazy_zero_clip)

        # ---- noise allocation for the whole HR latent volume ----
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise_full = torch.randn(
            16, total_latent_frames, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)
        latents = noise_full.clone()

        # ---- hr_action sanity ----
        hr_action = hr_action.to(self.device)
        assert hr_action.shape[0] >= total_latent_frames, (
            f"hr_action must have >= total_latent_frames ({total_latent_frames}) "
            f"rows; got {hr_action.shape[0]}.")
        lr_action = hr_action[::2][:total_latent_frames // 2]

        # Chunk-AR always keeps conditional caches. Negative caches are only
        # allocated for experts whose guide scale actually enables CFG.
        ar_cache = self.ar_cache
        self.seq_len_chunk = chunk_length * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        if ar_cache:
            # Largest context a world reaches, plus one chunk of denoise tokens.
            frame_tokens = self.seq_len_chunk // chunk_length
            cmp_frame_tokens = ((lat_h // 4) // self.patch_size[1]) * (
                (lat_w // 4) // self.patch_size[2])
            capacity = (
                (sink_size + temporal_size) * frame_tokens
                + (num_chunks - 1) * (chunk_length // 2) * cmp_frame_tokens
                + self.seq_len_chunk
            ) if self.kv_inplace else 0
            for expert in (self.high_noise_model, self.low_noise_model):
                self._find_wan_model(expert).kv_capacity = capacity
        # PERSISTENT KV caches, created ONCE and reused across all chunks.
        # The unified prefill path is incremental: each chunk only pushes the
        # new cmp block + recomputed tmp anchor. Each layer stores one
        # contiguous [sink | accumulated cmp | latest tmp] K/V pair instead
        # of re-prefilling the whole growing ctx. When AR but no KV cache,
        # ar_cache=False and the denoise falls back to the fused [ctx|pred] AR
        # path (memory_dict). BI leaves them None.
        if ar_cache:
            kv_high = self.high_noise_model.init_kv_cache()
            kv_low = self.low_noise_model.init_kv_cache()
            kv_high_neg = (
                self.high_noise_model.init_kv_cache(slot=1)
                if use_cfg_high else None
            )
            kv_low_neg = (
                self.low_noise_model.init_kv_cache(slot=1)
                if use_cfg_low else None
            )
        else:
            kv_high = kv_high_neg = kv_low = kv_low_neg = None
        memory_dict_high = None
        memory_dict_low = None
        boundary = self.boundary * self.num_train_timesteps
        no_sync_low = getattr(self.low_noise_model, 'no_sync', _noop_no_sync)
        no_sync_high = getattr(self.high_noise_model, 'no_sync', _noop_no_sync)

        # Keep context managers within each next() call: yielding must not
        # leave autocast/no_grad enabled on the runtime's worker thread.
        live_contexts = {prompts[name]: value for name, value in prompt_contexts.items()}
        for chunk_i in range(num_chunks):
            with (torch.amp.autocast('cuda', dtype=self.param_dtype),
                  torch.no_grad(), no_sync_low(), no_sync_high()):
                s = chunk_i * chunk_length
                e = (chunk_i + 1) * chunk_length
                prompt_name = chunk_prompts[chunk_i]
                context = prompt_contexts[prompt_name]
                if chunk_input is not None:
                    prompt_text, action_rows = chunk_input(chunk_i)
                    if tuple(action_rows.shape) != (chunk_length, 6):
                        raise ValueError("chunk_input must supply [chunk_length, 6] actions")
                    hr_action[s:e].copy_(action_rows)
                    # lr_action is a strided view of hr_action, as upstream.
                    if prompt_text not in live_contexts:
                        # Offline T5 encoding happens outside the DiT autocast
                        # scope. Keep that precision when a live prompt arrives.
                        with torch.amp.autocast('cuda', enabled=False):
                            live_contexts[prompt_text] = self.text_encoder([prompt_text], self.device)
                    context = live_contexts[prompt_text]

                logging.info(f"[stage3] chunk {chunk_i}/{num_chunks - 1} "
                             f"latent frames [{s}, {e}) prompt={prompt_name}")

                # Per-chunk denoise inputs (also needed by AR prefill below).
                latent = latents[:, s:e].contiguous()
                hr_action_chunk = hr_action[s:e]
                y_chunk = hr_y_full[:, s:e]

                chunk_denoise_args = dict(
                    latent=latent,
                    seed_g=seed_g,
                    boundary=boundary,
                    context=context,
                    context_null=context_null,
                    memory_dict_high=memory_dict_high,
                    memory_dict_low=memory_dict_low,
                    kv_high=kv_high,
                    kv_high_neg=kv_high_neg,
                    kv_low=kv_low,
                    kv_low_neg=kv_low_neg,
                    y_chunk=y_chunk,
                    hr_action_chunk=hr_action_chunk,
                    chunk_start=s,
                )

                if self.inference_mode is InferenceMode.PDD:
                    latent = self._run_pdd_chunk(
                        latent=latent,
                        y_chunk=y_chunk,
                        hr_action_chunk=hr_action_chunk,
                        context=context,
                        memory_dict_high=memory_dict_high,
                        memory_dict_low=memory_dict_low,
                        kv_high=kv_high,
                        kv_low=kv_low,
                        chunk_start=s,
                        seed_g=seed_g,
                    )
                elif self.inference_mode is InferenceMode.BI:
                    latent = self._run_solver_chunk(
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        shift=shift,
                        ar_cache=False,
                        chunk_index=chunk_i,
                        **chunk_denoise_args,
                    )
                elif self.inference_mode is InferenceMode.AR:
                    latent = self._run_solver_chunk(
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        shift=shift,
                        ar_cache=True,
                        chunk_index=chunk_i,
                        **chunk_denoise_args,
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported inference mode: {self.inference_mode}"
                    )
                latents[:, s:e] = latent.to(latents.dtype)

                if chunk_i < num_chunks - 1:
                    (
                        decoded_rgb,
                        memory_dict_high,
                        memory_dict_low,
                        kv_high,
                        kv_high_neg,
                        kv_low,
                        kv_low_neg,
                    ) = self._prepare_chunk_memory(
                        chunk_index=chunk_i,
                        chunk_end=e,
                        latents=latents,
                        hr_y_full=hr_y_full,
                        lr_y_full=lr_y_full,
                        hr_action=hr_action,
                        lr_action=lr_action,
                        context=context,
                        context_null=context_null,
                        kv_high=kv_high,
                        kv_high_neg=kv_high_neg,
                        kv_low=kv_low,
                        kv_low_neg=kv_low_neg,
                        return_cpu=return_cpu,
                    )
                else:
                    decoded_rgb = self._decode_last_chunk(latent, chunk_i == 0)
                    if return_cpu and decoded_rgb is not None:
                        decoded_rgb = decoded_rgb.cpu()
            yield decoded_rgb if self.rank == 0 else None

        if offload_model:
            self.clear_conditioning_cache()
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])
