"""Reactor application: client policy, immutable step inputs and output effects."""

from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    event,
    get_logger,
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.distributed import DistributedRunner

from .worldplay2_model import (
    MAX_CHUNKS,
    Settings,
    Anchor,
    EncodePrompt,
    PrepareWorld,
    WorldPlay2Input,
    WorldPlay2Model,
    WorldPlay2Result,
    load_settings,
    plan_action,
)
from .worldplay2_types import (
    ChunkGenerated,
    CommandApplied,
    Look,
    Movement,
    RolloutCompleted,
    StateUpdate,
    Turn,
    WorldPlay2Output,
    WorldPlay2State,
    compose_action,
    default_prompt,
)

logger = get_logger(__name__)


class WorldPlay2(ReactorApp):
    """Steer an uploaded scene using WorldPlay2's native streaming chunks."""

    state: WorldPlay2State
    buffer_size = 16
    # Pins playout to a fixed rate instead of each chunk's measured generation time; load()
    # replaces it with the configured target_fps.
    fps = 22.0

    def __init__(self):
        super().__init__()
        self._engine = None
        self._engine_error = None
        self._settings = Settings()
        self._weights = None
        self._image = None
        self._image_name = ""
        self._world_id = 0
        self._active_world = None
        self._completed = 0
        self._seed = 42
        # One engine call at a time: a chunk, a world preparation or a prompt encode.
        self._engine_lock = threading.Lock()
        self._preparing = 0

    def _engine_generate(self, input):
        with self._engine_lock:
            return self._engine.generate(input)

    async def _encode_now(self, input, code):
        """Run a condition-encoding call on the engine as the condition is set."""
        if self._engine is None:
            return None
        try:
            return await asyncio.to_thread(self._engine_generate, input)
        except Exception as exc:
            raise CommandError(code, f"Could not prepare the input: {exc}") from exc

    async def _prepare_world(self, command, image, prompt, seed):
        """Encode a new world's conditions before its first chunk; steps wait meanwhile."""
        self._preparing += 1
        try:
            world_id = self._world_id + 1
            await self._encode_now(PrepareWorld(world_id, Anchor(image, seed), prompt), "prepare_failed")
        finally:
            self._preparing -= 1
        self._world_id = world_id - 1
        return await self._request_world(command)

    def load(self, config_path: Path | None) -> None:
        self._settings = load_settings(config_path)
        self.output.fps = self._settings.target_fps
        self._weights = get_weights_path()
        self._start_engine()

    def _start_engine(self):
        if self._settings.world_size == 1:
            engine = WorldPlay2Model()
            engine.load(self._settings, self._weights)
        else:
            engine = DistributedRunner(
                WorldPlay2Model,
                world_size=self._settings.world_size,
                call_timeout=self._settings.call_timeout_seconds,
                load_kwargs={
                    "settings": self._settings,
                    "weights_root": self._weights,
                },
            )
            engine.start()
        self._engine = engine
        self._engine_error = None

    @session_started
    def on_session_started(self):
        self._image = None
        self._image_name = ""
        self._world_id += 1
        self._active_world = None
        self._completed = 0
        self._seed = self._settings.seed

    @connected
    async def on_connected(self, client: ClientInfo):
        await client.send(self._snapshot())

    # Publish a complete snapshot after accepted control changes.
    @event(
        name="set_prompt",
        description="Set scene text for the next chunk, preserving world memory. Valid at any time; an empty prompt waits for new input. Broadcasts `state_update`. Rejects with `invalid_prompt` if the text cannot be prepared.",
    )
    async def set_prompt(
        self, prompt: str = WorldPlay2State._public_fields["prompt"]
    ) -> None:
        prompt = prompt.strip()
        if prompt:
            await self._encode_now(EncodePrompt(prompt), "invalid_prompt")
        self.state.prompt = prompt
        await self._send_state_update()

    @event(
        name="set_action",
        description="Set movement, turn, look and space together for the next chunk. Selections are held until changed and combined into one native action. Broadcasts `state_update`. Rejects with `invalid_action` for an unsupported selection and with `rollout_complete` once the world has reached its chunk limit, without changing controls. After completion use reset or set_image.",
    )
    async def set_action(
        self,
        movement: Movement = InputField(  # noqa: B008 - Reactor schema declaration
            default="none",
            description="Movement dropdown, relative to the current view. Diagonal choices combine two movement keys. Applied next chunk; none releases movement.",
        ),
        turn: Turn = InputField(  # noqa: B008 - Reactor schema declaration
            default="none",
            description="Horizontal turn dropdown. Applied next chunk at yaw_degrees per latent frame; none releases turning.",
        ),
        look: Look = InputField(  # noqa: B008 - Reactor schema declaration
            default="none",
            description="Vertical look dropdown. Applied next chunk at pitch_degrees per latent frame; none releases looking.",
        ),
        space: bool = InputField(
            default=False,
            description="Hold the upstream space action when checked, release it when unchecked. Combined with the selected movement and rotation next chunk.",
        ),
    ) -> None:
        if self._completed >= MAX_CHUNKS:
            raise CommandError(
                "rollout_complete",
                f"The world reached its {MAX_CHUNKS}-chunk limit (the model's 1024 temporal positions). Use reset or set_image to start another world; controls have not changed.",
            )
        try:
            action = compose_action(movement, turn, look, space)
            plan_action(
                action,
                self.state.perspective,
                self.state.yaw_degrees,
                self.state.pitch_degrees,
                False,
            )
        except ValueError as exc:
            raise CommandError("invalid_action", str(exc)) from exc
        self.state._movement = movement
        self.state._turn = turn
        self.state._look = look
        self.state._space = space
        await self._send_state_update()

    @event(
        name="set_perspective",
        description="Select first- or third-person controls for the next chunk. Valid at any time; provide a matching image and prompt for a fresh world. Broadcasts `state_update`; an unknown value is rejected with `invalid_command`.",
    )
    async def set_perspective(
        self,
        perspective: Literal["fps", "tps"] = WorldPlay2State._public_fields[
            "perspective"
        ],
    ) -> None:
        self.state.perspective = perspective
        await self._send_state_update()

    @event(
        name="set_yaw_degrees",
        description="Set horizontal turn speed for the next chunk. Valid at any time. Broadcasts `state_update`; an out-of-range value is rejected with `invalid_command`.",
    )
    async def set_yaw_degrees(
        self, yaw_degrees: float = WorldPlay2State._public_fields["yaw_degrees"]
    ) -> None:
        self.state.yaw_degrees = yaw_degrees
        await self._send_state_update()

    @event(
        name="set_pitch_degrees",
        description="Set vertical turn speed for the next chunk. Valid at any time. Broadcasts `state_update`; an out-of-range value is rejected with `invalid_command`.",
    )
    async def set_pitch_degrees(
        self, pitch_degrees: float = WorldPlay2State._public_fields["pitch_degrees"]
    ) -> None:
        self.state.pitch_degrees = pitch_degrees
        await self._send_state_update()

    @session_ended
    def on_session_ended(self):
        with self._engine_lock:
            self._end_session()

    def _end_session(self):
        try:
            engine = self._engine
            if isinstance(engine, DistributedRunner):
                if engine.healthy:
                    try:
                        engine.reset()
                        return
                    except Exception:
                        logger.exception("Worker reset failed; replacing the worker group")
                # A failed reset can leave ranks in different worlds even when
                # all of them raised the same exception and healthy stays true.
                engine.shutdown()
                self._engine = None
                self._start_engine()
            elif engine is not None:
                engine.reset()
        except Exception as error:
            # Runtime logs lifecycle-hook errors. Retain the failure so the
            # next live step terminates instead of reusing an incomplete reset.
            self._engine_error = error
            raise
        finally:
            self._image = None
            self._image_name = ""
            self._active_world = None
            self._completed = 0

    @event(
        name="set_image",
        description="Start a fresh world from an uploaded image and optional prompt. Execute commits both together; blank text uses the upstream FPS/TPS template for the selected perspective. Generation starts on the next runtime playback step; resume if playback is paused. Emits `command_applied` and `state_update`. Rejects with `invalid_image` for an unsupported or undecodable upload and with `prepare_failed` if the world cannot be prepared, without changing the world.",
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - Reactor schema declaration
            moderate=True,
            description="Required uploaded PNG, JPEG or WebP, at most 25 MiB and 32 million pixels. No built-in image is selected.",
        ),
        prompt: str = InputField(
            default="",
            moderate=True,
            max_length=4096,
            description="Optional scene text for this new world. Blank text uses a reference-preserving default for the current fps/tps perspective, without an event. Replaces the previous prompt; set_prompt can change it during generation.",
        ),
    ) -> CommandApplied:
        try:
            if not image.data or len(image.data) > 25 * 1024 * 1024:
                raise ValueError("Image must contain 1 byte to 25 MiB")
            with Image.open(BytesIO(image.data)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Use a PNG, JPEG or WebP image")
                if source.width * source.height > 32_000_000:
                    raise ValueError("Image exceeds 32 million pixels")
                source.load()
        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        prompt = prompt.strip() or default_prompt(self.state.perspective)
        data = bytes(image.data)
        result = await self._prepare_world("set_image", data, prompt, self._seed)
        self._image = data
        self._image_name = image.name
        self.state.prompt = prompt
        return result

    @event(
        name="reset",
        description="Restart the selected image with a seed. Preserves prompt and held controls; starts a fresh world on the next step. Emits `command_applied` and `state_update`. Requires an uploaded image, otherwise rejects with `image_required`; rejects with `prepare_failed` if the world cannot be prepared.",
    )
    async def reset(
        self,
        seed: int = InputField(
            default=42,
            ge=0,
            le=2**63 - 1,
            description="Seed for the entire rollout noise volume.",
        ),
    ) -> CommandApplied:
        if self._image is None:
            raise CommandError("image_required", "Upload an image before resetting")
        result = await self._prepare_world("reset", self._image, self.state.prompt, seed)
        self._seed = seed
        return result

    async def _request_world(self, command):
        self._world_id += 1
        self._completed = 0
        self.output.flush()
        await self._send_state_update()
        return CommandApplied(command=command, world_id=self._world_id)

    async def process_input(self) -> WorldPlay2Input:
        if self._engine_error is not None:
            raise RuntimeError(
                "Model recovery failed; restart the service"
            ) from self._engine_error
        if self._preparing:
            raise ApplicationError("Preparing the world; generation starts when it is ready")
        if self._image is None:
            raise ApplicationError("Upload an image before generating")
        prompt = self.state.prompt
        if not prompt:
            raise ApplicationError("Set a non-empty scene prompt before generating")
        fresh = self._world_id != self._active_world
        if not fresh and self._completed >= MAX_CHUNKS:
            raise ApplicationError(
                "Rollout complete; reset explicitly to generate again"
            )
        try:
            rows = plan_action(
                self.state.action,
                self.state.perspective,
                self.state.yaw_degrees,
                self.state.pitch_degrees,
                fresh,
            )
        except ValueError as exc:
            raise ApplicationError(str(exc)) from exc
        anchor = Anchor(self._image, self._seed) if fresh else None
        return WorldPlay2Input(self._world_id, anchor, prompt, self.state.action, rows)

    def generate(self, input: WorldPlay2Input) -> WorldPlay2Result:
        return self._engine_generate(input)

    async def process_output(self, outcome: StepOutcome) -> WorldPlay2Output:
        if outcome.error is not None:
            # Input refusal happens before inference; remaining errors are fatal.
            raise outcome.error
        result: WorldPlay2Result = outcome.result
        if result.world_id != self._active_world:
            self.output.flush()
        self._active_world = result.world_id
        self._completed = result.chunk_index
        await self.send(
            ChunkGenerated(
                world_id=result.world_id,
                chunk_index=result.chunk_index,
                frames=len(result.frames),
                prompt=result.prompt,
                action=result.action,
                elapsed_seconds=outcome.elapsed,
            )
        )
        await self.send(self._snapshot())
        if self._completed == MAX_CHUNKS:
            await self.send(
                RolloutCompleted(
                    world_id=result.world_id,
                    completed_chunks=self._completed,
                    detail=f"The world reached its {MAX_CHUNKS}-chunk limit (the model's 1024 temporal positions). Use reset to restart this image, or set_image to start a new world.",
                )
            )
        return WorldPlay2Output(main_video=result.frames)

    def _snapshot(self):
        return StateUpdate.from_state(
            self.state,
            image_name=self._image_name,
            world_id=self._world_id,
            completed_chunks=self._completed,
            max_chunks=MAX_CHUNKS,
            seed=self._seed,
        )

    async def _send_state_update(self):
        await self.send(self._snapshot())
