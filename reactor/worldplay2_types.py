"""Public controls, tracks and messages for the WorldPlay2 application."""

from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

Movement = Literal[
    "none",
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
]
Turn = Literal["none", "left", "right"]
Look = Literal["none", "up", "down"]


def compose_action(movement: Movement, turn: Turn, look: Look, space: bool) -> str:
    """Combine mutually exclusive selectors into one upstream held action."""
    movements = {
        "none": "none",
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "forward_left": "wa",
        "forward_right": "wd",
        "backward_left": "sa",
        "backward_right": "sd",
    }
    if (
        movement not in movements
        or turn not in {"none", "left", "right"}
        or look not in {"none", "up", "down"}
        or not isinstance(space, bool)
    ):
        raise ValueError(
            "Select a supported movement, turn, look and boolean space control"
        )
    tokens = [movements[movement], turn, look, "space" if space else "none"]
    return "+".join(token for token in tokens if token != "none") or "none"


def default_prompt(perspective: str) -> str:
    """Reference-preserving fallback using the upstream FPS/TPS prompt template."""
    scene = "The scene in the video is: the environment shown in the reference image, with consistent layout, objects, and lighting."
    if perspective == "tps":
        return (
            scene
            + " The character is: the same main character shown in the reference image, with consistent appearance and clothing."
        )
    return scene


class WorldPlay2State(InputState):
    """Held controls sampled at the next native four-latent chunk boundary."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description="Scene text applied next chunk, preserving world memory. FPS: The scene in the video is: ...; TPS also includes The character is: ... . The event is: ... is optional. Empty text waits for a new prompt; set_image supplies a default when its prompt is blank.",
    )
    _movement: Movement = "none"
    _turn: Turn = "none"
    _look: Look = "none"
    _space: bool = False

    @property
    def action(self) -> str:
        return compose_action(self._movement, self._turn, self._look, self._space)

    perspective: Literal["fps", "tps"] = InputField(
        default="fps",
        description="Native perspective bit: fps is first-person, tps is third-person. Applied next chunk; use a matching image and prompt.",
    )
    yaw_degrees: float = InputField(
        default=3.0,
        ge=0.01,
        le=90,
        description="Positive yaw speed in degrees per latent frame for left/right. Applied next chunk; native launcher default is 3.",
    )
    pitch_degrees: float = InputField(
        default=1.0,
        ge=0.01,
        le=90,
        description="Positive pitch speed in degrees per latent frame for up/down. Applied next chunk; native launcher default is 1.",
    )


class WorldPlay2Output(Output):
    """Native RGB chunks: 13 frames initially, then 16; runtime-paced playback."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted on connection, accepted control changes and completed chunks; full shared state."""

    image_name: str | None = MessageField(
        description="Uploaded anchor filename; null until explicitly supplied."
    )
    prompt: str | None = MessageField(
        description="Current prompt, sampled on the next step; null when unset."
    )
    action: str = MessageField(
        description="Native action composed from the selectors, sampled on the next step."
    )
    movement: str = MessageField(description="Selected movement direction.")
    turn: str = MessageField(description="Selected horizontal turn direction.")
    look: str = MessageField(description="Selected vertical look direction.")
    space: bool = MessageField(description="Whether the native space action is held.")
    perspective: str = MessageField(description="Current fps/tps perspective control.")
    yaw_degrees: float = MessageField(description="Yaw degrees per latent frame.")
    pitch_degrees: float = MessageField(description="Pitch degrees per latent frame.")
    world_id: int = MessageField(
        description="Identity of the requested rollout, increased by image/reset commands."
    )
    completed_chunks: int = MessageField(
        description="Successfully completed chunks in this world."
    )
    max_chunks: int = MessageField(
        description="Chunks a world can reach: the model's 1024 temporal positions, 4 latents per chunk."
    )
    limit_reached: bool = MessageField(
        description="True after the final chunk; reset or set_image is required to generate again."
    )
    seed: int = MessageField(
        description="RNG seed for the full rollout, sampled when the world starts."
    )

    @classmethod
    def from_state(
        cls, state, *, image_name, world_id, completed_chunks, max_chunks, seed
    ):
        return cls(
            image_name=image_name or None,
            world_id=world_id,
            completed_chunks=completed_chunks,
            max_chunks=max_chunks,
            limit_reached=completed_chunks >= max_chunks,
            seed=seed,
            prompt=state.prompt or None,
            action=state.action,
            movement=state._movement,
            turn=state._turn,
            look=state._look,
            space=state._space,
            perspective=state.perspective,
            yaw_degrees=state.yaw_degrees,
            pitch_degrees=state.pitch_degrees,
        )


class CommandApplied(ModelMessage):
    """Emitted as the acknowledgement of a successful image/reset command."""

    command: str = MessageField(description="Accepted command name.")
    world_id: int = MessageField(
        description="Fresh world identity; its anchor is consumed on the next allowed step."
    )


class ChunkGenerated(ModelMessage):
    """Emitted after a successful step, before its state snapshot and video output."""

    world_id: int = MessageField(description="World identity echoed by the model.")
    chunk_index: int = MessageField(description="One-based completed chunk index.")
    frames: int = MessageField(description="Number of RGB frames actually produced.")
    prompt: str = MessageField(description="Prompt actually used for this chunk.")
    action: str = MessageField(description="Action actually used for this chunk.")
    elapsed_seconds: float = MessageField(
        description="Inference duration measured by the runtime, in seconds."
    )


class RolloutCompleted(ModelMessage):
    """Emitted after the final successful chunk; reset or upload an image to continue."""

    world_id: int = MessageField(description="Completed world identity.")
    completed_chunks: int = MessageField(
        description="Number of successfully generated chunks."
    )
    detail: str = MessageField(
        description="Why the world ended and how to start another."
    )
