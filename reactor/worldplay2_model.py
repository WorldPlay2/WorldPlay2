"""Plain model boundary: no Reactor imports, environment reads or worker launchers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from io import BytesIO
import logging
import os
import time
from math import isfinite
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# The DiT's temporal RoPE table has 1024 latent positions, indexed by absolute latent frame, and a
# chunk is 4 latents, so a world ends after 256 chunks. This is the only cap on a session.
MAX_CHUNKS = 1024 // 4


@dataclass(frozen=True)
class Settings:
    height: int = 448
    width: int = 832
    seed: int = 42
    world_size: int = 1
    call_timeout_seconds: float = 240.0
    # Chunks of a synthetic world run inside load() as a warm-up.
    warmup_chunks: int = 10
    # Persistent compile cache. Cache directories already set in the environment take
    # precedence.
    compile_cache_dir: str | None = "~/.cache/worldplay2/compile"
    # Playback rate of the video track, in frames per second. Every chunk plays out at this
    # rate whatever its generation time, so a world does not start quickly and slow down as
    # its memory grows. It must not exceed the rate of the slowest chunk the world will reach: on
    # 4x B200 that is about 18 for a full 256-chunk world.
    target_fps: float = 17.0


def load_settings(config_path: Path | None) -> Settings:
    """Read settings without loading weights or initializing a GPU."""
    values = dict(yaml.safe_load(config_path.read_text()) if config_path else {})
    unknown = sorted(set(values) - {field.name for field in fields(Settings)})
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(unknown)}")
    settings = Settings(**values)
    if type(settings.world_size) is not int or settings.world_size not in (1, 2, 4):
        raise ValueError("world_size must be 1, 2 or 4")
    if (
        type(settings.call_timeout_seconds) not in (int, float)
        or not isfinite(settings.call_timeout_seconds)
        or settings.call_timeout_seconds <= 0
    ):
        raise ValueError("call_timeout_seconds must be finite and positive")
    if type(settings.target_fps) not in (int, float) or not isfinite(settings.target_fps) or settings.target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    if type(settings.warmup_chunks) is not int or not 0 <= settings.warmup_chunks <= MAX_CHUNKS:
        raise ValueError(f"warmup_chunks must be an integer in [0, {MAX_CHUNKS}]")
    if (
        settings.height % 32
        or settings.width % 32
        or min(settings.height, settings.width) < 128
    ):
        raise ValueError(
            "Resolution must be a multiple of 32, at least 128"
        )
    return settings


@dataclass(frozen=True)
class Anchor:
    image: bytes
    seed: int


@dataclass(frozen=True)
class WorldPlay2Input:
    world_id: int
    anchor: Anchor | None
    prompt: str
    action: str
    action_rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PrepareWorld:
    """Set up world ``world_id`` from its image and prompt before its first chunk."""

    world_id: int
    anchor: Anchor
    prompt: str


@dataclass(frozen=True)
class EncodePrompt:
    """Encode a prompt when it is set, so the chunk that first uses it does not."""

    prompt: str


@dataclass(frozen=True)
class Prepared:
    world_id: int | None
    seconds: float


@dataclass(frozen=True)
class WorldPlay2Result:
    world_id: int
    chunk_index: int
    frames: np.ndarray
    prompt: str
    action: str


def plan_action(
    action: str, perspective: str, yaw: float, pitch: float, first: bool
) -> tuple[tuple[float, ...], ...]:
    """Use the upstream action parser; only the rollout's first latent is idle."""
    from worldplay2.inputs import parse_action_string

    if "," in action or "-" in action:
        raise ValueError("Supply one held action without a duration, e.g. w+right.")
    rows = parse_action_string(
        f"{action}-{4 if first else 5}",
        perspective=perspective,
        yaw_rotation_speed_deg=yaw,
        pitch_rotation_speed_deg=pitch,
    ).tensor
    if not first:
        rows = rows[1:]
    return tuple(tuple(float(x) for x in row) for row in rows)


class WorldPlay2Model:
    """Own the native two-expert pipeline and one persistent streaming rollout."""

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.device = "cuda:0"
        self._settings = Settings()
        self._pipeline = None
        self._stream = None
        self._world_id = None
        self._chunk_index = 0
        self._pending = None

    def load(self, settings: Settings, weights_root: Path) -> None:
        from configs import I2V_A14B_CONFIG
        from .inference.pipeline import WorldPlay2Pipeline

        self._settings = settings
        if settings.compile_cache_dir:
            root = Path(settings.compile_cache_dir).expanduser()
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
            os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))
        self._pipeline = WorldPlay2Pipeline(
            config=I2V_A14B_CONFIG,
            checkpoint_dir=str(weights_root / "base"),
            low_noise_mem_ckpt=str(
                weights_root
                / "fast/low_noise_model/diffusion_pytorch_model.safetensors"
            ),
            high_noise_mem_ckpt=str(
                weights_root
                / "fast/high_noise_model/diffusion_pytorch_model.safetensors"
            ),
            device_id=int(self.device.split(":")[-1]),
            rank=self.rank,
            init_on_cpu=False,
            inference_mode="pdd",
            use_sp=self.world_size > 1,
            dit_fsdp=False,
            t5_fsdp=False,
            attention_backend="flash",
            exact_optimizations=True,
        )
        self._use_bf16_vae()
        from .acceleration import install_lossless

        install_lossless(self, "num-equal")
        self._install_flash_attention4()
        if settings.warmup_chunks:
            self._warm_up(settings.warmup_chunks)

    def _warm_up(self, chunks: int) -> None:
        """Run a synthetic world through the real session path, then reset."""
        started = time.perf_counter()
        height, width = self._settings.height, self._settings.width
        rows = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
        cols = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
        pixels = np.concatenate(
            [np.broadcast_to(rows, (height, width, 1)),
             np.broadcast_to(cols, (height, width, 1)),
             np.broadcast_to((rows + cols) / 2, (height, width, 1))], axis=2)
        buffer = BytesIO()
        Image.fromarray(pixels.astype(np.uint8)).save(buffer, format="PNG")
        anchor = Anchor(buffer.getvalue(), 0)
        prompt = "A quiet street between low buildings on a clear day."
        # Two synthetic worlds, then reset.
        for world, count in ((-2, 2), (-1, chunks)):
            if world == -1:
                # The served path: conditions encoded by a prepare step, then the chunks.
                WorldPlay2Model.generate(self, PrepareWorld(world, anchor, prompt))
            for index in range(count):
                action = plan_action("w+right", "fps", 3.0, 1.0, index == 0)
                WorldPlay2Model.generate(self, WorldPlay2Input(
                    world, anchor if index == 0 else None, prompt, "w+right", action))
        self.reset()
        logging.info("warm-up: %d chunks in %.1f s; ready", chunks, time.perf_counter() - started)

    def _use_bf16_vae(self) -> None:
        import torch

        self._pipeline.vae.set_dtype(torch.bfloat16)

    def _install_flash_attention4(self) -> None:
        from .acceleration import flash_attention4

        flash_attention4.install(self._pipeline)

    def _chunk_input(self, index):
        import torch

        if index != self._chunk_index:
            raise RuntimeError("Native stream advanced beyond the requested chunk")
        return self._pending.prompt, torch.tensor(
            self._pending.action_rows, dtype=torch.float32
        )

    def _begin(self, input: WorldPlay2Input, prepare: bool = False):
        import torch

        if input.anchor is None:
            raise ValueError("A new world requires its uploaded image anchor")
        self.reset()
        anchor = input.anchor
        with Image.open(BytesIO(anchor.image)) as source:
            image = source.convert("RGB")
        if getattr(self._pipeline, "chunk_stream", False):
            self._stream = self._pipeline.stream_world(
                img=image,
                seed=anchor.seed,
                num_chunks=MAX_CHUNKS,
                chunk_input=self._chunk_input,
                height=self._settings.height,
                width=self._settings.width,
                return_cpu=False,
                prepare=prepare,
                prompt=input.prompt,
            )
            self._world_id = input.world_id
            return
        self._stream = self._pipeline.stream_chunked(
            prompts={"live": input.prompt},
            chunk_prompts=("live",) * MAX_CHUNKS,
            img=image,
            hr_action=torch.zeros(MAX_CHUNKS * 4, 6),
            total_latent_frames=MAX_CHUNKS * 4,
            chunk_length=4,
            sampling_steps=4,
            shift=7.0,
            guide_scale=1.0,
            seed=anchor.seed,
            offload_model=False,
            sink_size=1,
            temporal_size=1,
            height=self._settings.height,
            width=self._settings.width,
            chunk_input=self._chunk_input,
            return_cpu=False,
        )
        self._world_id = input.world_id

    def generate(self, input):
        """A chunk of the current world, or (``PrepareWorld`` / ``EncodePrompt``) condition
        encoding at the moment the condition is set."""
        if self._pipeline is None:
            raise RuntimeError("Load weights before generating")
        if isinstance(input, (PrepareWorld, EncodePrompt)):
            started = time.perf_counter()
            if isinstance(input, EncodePrompt):
                self._pipeline.encode_prompt(input.prompt)
                world = None
            else:
                self._begin(input, prepare=True)
                world = input.world_id
            return Prepared(world, time.perf_counter() - started)
        if input.world_id != self._world_id:
            self._begin(input)
        if self._stream is None or self._chunk_index >= MAX_CHUNKS:
            raise ValueError("Rollout complete; start a new world explicitly")
        self._pending = input
        try:
            rgb = next(self._stream)
            frames = (
                self._host_frames(rgb)
                if self.rank == 0
                else np.empty((0,), dtype=np.uint8)
            )
        except Exception:
            self.reset()
            raise
        finally:
            self._pending = None
        self._chunk_index += 1
        result = WorldPlay2Result(
            input.world_id, self._chunk_index, frames, input.prompt, input.action
        )
        if self._chunk_index == MAX_CHUNKS:
            self._stream.close()
            self._stream = None
        return result

    def _host_frames(self, rgb) -> np.ndarray:
        import torch

        return (
            rgb.float()
            .clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .to(dtype=torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

    def reset(self) -> None:
        """Drop rollout state and VAE/text caches, retaining loaded weights."""
        if self._stream is not None:
            self._stream.close()
        self._stream = self._pending = self._world_id = None
        self._chunk_index = 0
        if self._pipeline is not None:
            self._pipeline.clear_conditioning_cache()
            self._pipeline.vae.model.clear_cache()
            self._pipeline.lr_latents_cache = None
