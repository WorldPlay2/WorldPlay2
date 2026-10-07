"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ReactorView, useReactor, useReactorMessage } from "@reactor-team/js-sdk";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  AddExampleModal,
  EventsEditor,
  PerspectiveToggle,
  cleanEvents,
  type NewExample,
} from "@/components/AddExampleModal";
import { cn } from "@/lib/utils";
import {
  BUILT_IN_EXAMPLES,
  MAX_EVENTS,
  composeBasePrompt,
  stripEvent,
  withEvent,
  type Example,
  type ExampleEvent,
  type Perspective,
} from "@/lib/examples";
import {
  deleteSavedExample,
  listSavedExamples,
  loadEventOverrides,
  saveEventOverrides,
  newExampleId,
  putSavedExample,
  type SavedExample,
} from "@/lib/example-store";
import { dragHasImagePayload, imageFileFromDataTransfer } from "@/lib/image-drop";

// ---- Model controls (mirrors worldplay2_types.py) ----

type Movement =
  | "none"
  | "forward"
  | "backward"
  | "left"
  | "right"
  | "forward_left"
  | "forward_right"
  | "backward_left"
  | "backward_right";
type Turn = "none" | "left" | "right";
type Look = "none" | "up" | "down";
type Action = { movement: Movement; turn: Turn; look: Look; space: boolean };

const IDLE_ACTION: Action = { movement: "none", turn: "none", look: "none", space: false };

/** Fields of the model's `state_update` message. */
type ModelState = {
  imageName: string | null;
  prompt: string | null;
  action: string;
  movement: Movement;
  turn: Turn;
  look: Look;
  space: boolean;
  perspective: Perspective;
  yawDegrees: number;
  pitchDegrees: number;
  worldId: number;
  completedChunks: number;
  maxChunks: number;
  limitReached: boolean;
  seed: number;
};

type ChunkInfo = { index: number; frames: number; elapsedSeconds: number; action: string; prompt: string };

const DEFAULT_YAW = 3;
const DEFAULT_PITCH = 1;
const DEFAULT_SEED = 42;
// The model's 1024 temporal positions at 4 latents per chunk; the server reports the same value.
const DEFAULT_MAX_CHUNKS = 256;
const SPEED_DEBOUNCE_MS = 250;

const MOVE_KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD"]);
const ARROW_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);
const CONTROL_KEYS = new Set([...MOVE_KEYS, ...ARROW_KEYS, "Space"]);

/** Event slot for Digit1..Digit9, or -1. */
function eventSlotForKey(code: string): number {
  const match = /^Digit([1-9])$/.exec(code);
  return match ? Number(match[1]) - 1 : -1;
}

/** Hold-to-fire event chip, with an optional hotkey badge. */
function EventChip({
  slot,
  name,
  pressed,
  disabled,
  onPress,
  onRelease,
}: {
  slot: number;
  name: string;
  pressed: boolean;
  disabled: boolean;
  onPress: () => void;
  onRelease: () => void;
}) {
  const handlers = disabled
    ? {}
    : {
        onPointerDown: (e: React.PointerEvent) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          onPress();
        },
        onPointerUp: onRelease,
        onPointerCancel: onRelease,
        onPointerLeave: (e: React.PointerEvent) => {
          if (e.buttons !== 0) onRelease();
        },
      };
  return (
    <button
      type="button"
      disabled={disabled}
      {...handlers}
      title={`Hold to fire (key ${slot + 1})`}
      className={cn(
        "flex max-w-full select-none items-center gap-1.5 rounded-full border px-2.5 py-1 text-left font-mono text-[10px] transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-40",
        pressed
          ? "border-amber-300/80 bg-amber-300/25 text-amber-100"
          : "border-white/15 bg-white/5 text-white/70 hover:border-white/25 hover:bg-white/10",
      )}
    >
      <span
        className={cn(
          "inline-flex h-4 min-w-4 items-center justify-center rounded border px-0.5 text-[9px] font-bold",
          pressed ? "border-amber-300/80 bg-amber-300/30 text-amber-100" : "border-white/25 bg-white/10 text-white/80",
        )}
      >
        {slot + 1}
      </span>
      <span className="truncate">{name}</span>
    </button>
  );
}

function actionsEqual(a: Action, b: Action): boolean {
  return a.movement === b.movement && a.turn === b.turn && a.look === b.look && a.space === b.space;
}

/** Combines held keys into the model's four selectors. */
function composeAction(held: Set<string>): Action {
  const fwd = held.has("KeyW") && !held.has("KeyS");
  const back = held.has("KeyS") && !held.has("KeyW");
  const left = held.has("KeyA") && !held.has("KeyD");
  const right = held.has("KeyD") && !held.has("KeyA");
  const longitudinal = fwd ? "forward" : back ? "backward" : null;
  const lateral = left ? "left" : right ? "right" : null;
  const movement = (
    longitudinal && lateral ? `${longitudinal}_${lateral}` : (longitudinal ?? lateral ?? "none")
  ) as Movement;

  const arrowLeft = held.has("ArrowLeft") && !held.has("ArrowRight");
  const arrowRight = held.has("ArrowRight") && !held.has("ArrowLeft");
  const arrowUp = held.has("ArrowUp") && !held.has("ArrowDown");
  const arrowDown = held.has("ArrowDown") && !held.has("ArrowUp");
  const turn: Turn = arrowLeft ? "left" : arrowRight ? "right" : "none";
  const look: Look = arrowUp ? "up" : arrowDown ? "down" : "none";

  return { movement, turn, look, space: held.has("Space") };
}

function parseModelState(data: Record<string, unknown>): ModelState {
  return {
    imageName: (data.image_name as string | null) ?? null,
    prompt: (data.prompt as string | null) ?? null,
    action: (data.action as string) ?? "none",
    movement: ((data.movement as Movement) ?? "none"),
    turn: ((data.turn as Turn) ?? "none"),
    look: ((data.look as Look) ?? "none"),
    space: Boolean(data.space),
    perspective: data.perspective === "tps" ? "tps" : "fps",
    yawDegrees: Number(data.yaw_degrees ?? DEFAULT_YAW),
    pitchDegrees: Number(data.pitch_degrees ?? DEFAULT_PITCH),
    worldId: Number(data.world_id ?? 0),
    completedChunks: Number(data.completed_chunks ?? 0),
    maxChunks: Number(data.max_chunks ?? DEFAULT_MAX_CHUNKS),
    limitReached: Boolean(data.limit_reached),
    seed: Number(data.seed ?? DEFAULT_SEED),
  };
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

/** A saved example with an object URL for its image. */
type SavedExampleView = Example & { image: Blob; createdAt: number };

function toView(saved: SavedExample): SavedExampleView {
  return {
    id: saved.id,
    name: saved.name,
    perspective: saved.perspective,
    scene: saved.scene,
    character: saved.character,
    events: saved.events,
    image: saved.image,
    createdAt: saved.createdAt,
    imageSrc: URL.createObjectURL(saved.image),
  };
}

// ---- On-screen keys ----

const KEY_IDLE = "bg-white/5 border-white/15 text-white/80 hover:bg-white/10 active:scale-95";
const KEY_LIT = "bg-amber-300/20 border-amber-300/60 text-amber-200 scale-95";

/** Hold-to-press key: pressed while the pointer is down. */
function PadButton({
  label,
  pressed,
  disabled,
  onPress,
  onRelease,
  className,
}: {
  label: React.ReactNode;
  pressed: boolean;
  disabled?: boolean;
  onPress: () => void;
  onRelease: () => void;
  className?: string;
}) {
  const handlers = disabled
    ? {}
    : {
        onPointerDown: (e: React.PointerEvent) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          onPress();
        },
        onPointerUp: onRelease,
        onPointerCancel: onRelease,
        onPointerLeave: (e: React.PointerEvent) => {
          if (e.buttons !== 0) onRelease();
        },
      };

  return (
    <button
      type="button"
      disabled={disabled}
      {...handlers}
      className={cn(
        "h-10 w-10 rounded border font-mono text-xs select-none transition-all",
        "disabled:opacity-30 disabled:cursor-not-allowed",
        pressed ? KEY_LIT : KEY_IDLE,
        className,
      )}
    >
      {label}
    </button>
  );
}

function SpeedSlider({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-20 shrink-0 font-mono text-[10px] uppercase tracking-wider text-white/50">
        {label}
      </label>
      <input
        type="range"
        min={0.1}
        max={20}
        step={0.1}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="min-w-0 flex-1 accent-amber-300 disabled:opacity-40"
      />
      <span className="shrink-0 whitespace-nowrap text-right font-mono text-xs tabular-nums text-white/70">
        {value.toFixed(1)}
        <span className="text-white/40"> °/frame</span>
      </span>
    </div>
  );
}

/** A prompt with its "The event is: ..." sentence highlighted in amber. */
function PromptText({ text, small }: { text: string; small?: boolean }) {
  const at = text.search(/\s*The event is:/);
  const base = at >= 0 ? text.slice(0, at) : text;
  const event = at >= 0 ? text.slice(at).trim() : "";
  return (
    <p
      className={cn(
        "whitespace-pre-wrap break-words font-mono leading-relaxed",
        small ? "text-[11px] text-white/70" : "text-[12px] text-white/85",
      )}
    >
      {base}
      {event && (
        <>
          {" "}
          <span className="box-decoration-clone rounded bg-amber-300/20 px-1 py-0.5 text-amber-100">{event}</span>
        </>
      )}
    </p>
  );
}

// ---- Controller ----

export function WorldPlay2Controller() {
  const { status, sendCommand, uploadFile, pauseTrack, resumeTrack, lastError } = useReactor((s) => ({
    status: s.status,
    sendCommand: s.sendCommand,
    uploadFile: s.uploadFile,
    pauseTrack: s.pauseTrack,
    resumeTrack: s.resumeTrack,
    lastError: s.lastError,
  }));
  const isReady = status === "ready";

  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [lastChunk, setLastChunk] = useState<ChunkInfo | null>(null);
  const [completionDetail, setCompletionDetail] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [errorToast, setErrorToast] = useState<string | null>(null);

  const [savedExamples, setSavedExamples] = useState<SavedExampleView[]>([]);
  const [activeExampleId, setActiveExampleId] = useState<string | null>(null);
  const [loadingExampleId, setLoadingExampleId] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [droppedImage, setDroppedImage] = useState<File | null>(null);
  const [galleryDropOver, setGalleryDropOver] = useState(false);
  const galleryDropDepth = useRef(0);

  const [promptDraft, setPromptDraft] = useState("");

  // Events: the active world's base prompt plus at most one held event.
  const [eventOverrides, setEventOverrides] = useState<Record<string, ExampleEvent[]>>({});
  const [activeEvents, setActiveEvents] = useState<ExampleEvent[]>([]);
  const [heldEvents, setHeldEvents] = useState<number[]>([]);
  const [editingEvents, setEditingEvents] = useState<ExampleEvent[] | null>(null);
  const basePromptRef = useRef("");
  const activeEventsRef = useRef<ExampleEvent[]>([]);
  const heldEventsRef = useRef<number[]>([]);
  const lastSentPromptRef = useRef<string | null>(null);
  // Mirrors for the live readout: the base prompt in effect and the last full prompt sent.
  const [basePrompt, setBasePrompt] = useState("");
  const [sentPrompt, setSentPrompt] = useState<string | null>(null);
  const [showOnScreen, setShowOnScreen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [perspective, setPerspective] = useState<Perspective>("fps");
  const [yawDegrees, setYawDegrees] = useState(DEFAULT_YAW);
  const [pitchDegrees, setPitchDegrees] = useState(DEFAULT_PITCH);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const promptBoxRef = useRef<HTMLTextAreaElement>(null);

  // The prompt box grows to show its whole text; past its max height it scrolls.
  useLayoutEffect(() => {
    const box = promptBoxRef.current;
    if (!box) return;
    const fit = () => {
      box.style.height = "auto";
      box.style.height = `${box.scrollHeight + 2}px`;
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [promptDraft]);

  // Held inputs: keyboard and on-screen pads are tracked separately and unioned.
  const keysHeld = useRef(new Set<string>());
  const padsHeld = useRef(new Set<string>());
  const [heldView, setHeldView] = useState<Set<string>>(new Set());

  // Last action sent (or reported by the model); set_action goes out only on change.
  const lastActionRef = useRef<Action>(IDLE_ACTION);
  const lastWorldIdRef = useRef<number | null>(null);
  const isReadyRef = useRef(isReady);
  const modelStateRef = useRef<ModelState | null>(null);
  useEffect(() => {
    isReadyRef.current = isReady;
  }, [isReady]);
  useEffect(() => {
    modelStateRef.current = modelState;
  }, [modelState]);

  const hasWorld = Boolean(modelState?.imageName);
  const limitReached = Boolean(modelState?.limitReached);

  // ---- set_action ----

  const syncAction = useCallback(() => {
    const held = new Set([...keysHeld.current, ...padsHeld.current]);
    setHeldView(held);
    const next = composeAction(held);
    if (!isReadyRef.current) return;
    const state = modelStateRef.current;
    if (!state?.imageName || state.limitReached) return;
    if (actionsEqual(next, lastActionRef.current)) return;
    lastActionRef.current = next;
    void sendCommand("set_action", next);
  }, [sendCommand]);

  // ---- set_prompt: base prompt plus the most recently held event ----

  const syncPrompt = useCallback(() => {
    if (!isReadyRef.current || !modelStateRef.current?.imageName || !basePromptRef.current) return;
    const held = heldEventsRef.current;
    const slot = held.length > 0 ? held[held.length - 1] : -1;
    const next = withEvent(basePromptRef.current, activeEventsRef.current[slot] ?? null);
    if (next === lastSentPromptRef.current) return;
    lastSentPromptRef.current = next;
    setSentPrompt(next);
    void sendCommand("set_prompt", { prompt: next });
  }, [sendCommand]);

  const pressEvent = useCallback(
    (slot: number) => {
      if (slot < 0 || slot >= activeEventsRef.current.length) return;
      if (heldEventsRef.current.includes(slot)) return;
      heldEventsRef.current = [...heldEventsRef.current, slot];
      setHeldEvents(heldEventsRef.current);
      syncPrompt();
    },
    [syncPrompt],
  );

  const releaseEvent = useCallback(
    (slot: number) => {
      if (!heldEventsRef.current.includes(slot)) return;
      heldEventsRef.current = heldEventsRef.current.filter((s) => s !== slot);
      setHeldEvents(heldEventsRef.current);
      syncPrompt();
    },
    [syncPrompt],
  );

  const releaseAll = useCallback(() => {
    keysHeld.current.clear();
    padsHeld.current.clear();
    syncAction();
    if (heldEventsRef.current.length > 0) {
      heldEventsRef.current = [];
      setHeldEvents([]);
      syncPrompt();
    }
  }, [syncAction, syncPrompt]);

  const pressPad = useCallback(
    (code: string) => {
      padsHeld.current.add(code);
      syncAction();
    },
    [syncAction],
  );
  const releasePad = useCallback(
    (code: string) => {
      padsHeld.current.delete(code);
      syncAction();
    },
    [syncAction],
  );

  // ---- Messages from the model ----

  useReactorMessage((raw) => {
    const msg = raw as { type?: string; data?: Record<string, unknown> };
    if (!msg?.type) return;
    const data = msg.data ?? {};
    switch (msg.type) {
      case "state_update": {
        const next = parseModelState(data);
        modelStateRef.current = next;
        setModelState(next);
        setPerspective(next.perspective);
        setYawDegrees(next.yawDegrees);
        setPitchDegrees(next.pitchDegrees);
        if (!next.limitReached) setCompletionDetail(null);
        // A new session or a new world: adopt the model's held controls, then
        // re-send ours if they differ.
        if (lastWorldIdRef.current !== next.worldId) {
          lastWorldIdRef.current = next.worldId;
          lastActionRef.current = {
            movement: next.movement,
            turn: next.turn,
            look: next.look,
            space: next.space,
          };
          syncAction();
        }
        break;
      }
      case "chunk_generated":
        setLastChunk({
          index: Number(data.chunk_index ?? 0),
          frames: Number(data.frames ?? 0),
          elapsedSeconds: Number(data.elapsed_seconds ?? 0),
          action: String(data.action ?? ""),
          prompt: String(data.prompt ?? ""),
        });
        break;
      case "rollout_completed":
        setCompletionDetail(String(data.detail ?? "All chunks are complete."));
        break;
    }
  });

  // Session-scoped state is cleared whenever the connection drops.
  useEffect(() => {
    if (status !== "disconnected") return;
    setModelState(null);
    modelStateRef.current = null;
    setLastChunk(null);
    setCompletionDetail(null);
    setIsPaused(false);
    setActiveExampleId(null);
    setActiveEvents([]);
    activeEventsRef.current = [];
    heldEventsRef.current = [];
    setHeldEvents([]);
    basePromptRef.current = "";
    lastSentPromptRef.current = null;
    setBasePrompt("");
    setSentPrompt(null);
    lastWorldIdRef.current = null;
    lastActionRef.current = IDLE_ACTION;
  }, [status]);

  // The runtime starts outbound tracks paused and the SDK resumes them on
  // connect, but that first resume can leave the video sender negotiated and
  // sending nothing (no RTP ever arrives; the stage stays black). An explicit
  // pause/resume round-trip renegotiates the track and re-attaches it, so do
  // one as soon as the session is ready.
  useEffect(() => {
    if (!isReady) return;
    void (async () => {
      try {
        await pauseTrack("main_video");
        await resumeTrack("main_video");
      } catch {
        // The session ended in between; the next one does its own round-trip.
      }
    })();
  }, [isReady, pauseTrack, resumeTrack]);

  useEffect(() => {
    if (lastError) setErrorToast(lastError.message);
  }, [lastError]);

  useEffect(() => {
    if (!errorToast) return;
    const id = setTimeout(() => setErrorToast(null), 6000);
    return () => clearTimeout(id);
  }, [errorToast]);

  // ---- Keyboard ----

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const slot = eventSlotForKey(e.code);
      if (slot >= 0 && !e.metaKey && !e.ctrlKey && !e.altKey && !isTypingTarget(e.target)) {
        e.preventDefault();
        if (!e.repeat) pressEvent(slot);
        return;
      }
      if (!CONTROL_KEYS.has(e.code) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      if (e.repeat || keysHeld.current.has(e.code)) return;
      keysHeld.current.add(e.code);
      syncAction();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const slot = eventSlotForKey(e.code);
      if (slot >= 0) {
        releaseEvent(slot);
        return;
      }
      if (!keysHeld.current.has(e.code)) return;
      keysHeld.current.delete(e.code);
      syncAction();
    };
    const onBlur = () => releaseAll();
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [syncAction, releaseAll, pressEvent, releaseEvent]);

  // ---- Speeds and perspective ----

  const speedTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const sendSpeed = useCallback(
    (field: "yaw_degrees" | "pitch_degrees", value: number) => {
      clearTimeout(speedTimers.current[field]);
      speedTimers.current[field] = setTimeout(() => {
        if (isReadyRef.current) void sendCommand(`set_${field}`, { [field]: value });
      }, SPEED_DEBOUNCE_MS);
    },
    [sendCommand],
  );

  const changePerspective = useCallback(
    (next: Perspective) => {
      setPerspective(next);
      if (isReady) void sendCommand("set_perspective", { perspective: next });
    },
    [isReady, sendCommand],
  );

  // ---- Starting worlds ----

  const startWorld = useCallback(
    async (example: Example, image?: Blob) => {
      if (!isReady || loadingExampleId) return;
      setLoadingExampleId(example.id);
      setErrorToast(null);
      const base = composeBasePrompt(example);
      const events = eventOverrides[example.id] ?? example.events;
      try {
        if (modelStateRef.current?.perspective !== example.perspective) {
          await sendCommand("set_perspective", { perspective: example.perspective });
        }
        let blob = image;
        if (!blob) {
          const response = await fetch(example.imageSrc);
          if (!response.ok) throw new Error(`Could not load ${example.imageSrc}`);
          blob = await response.blob();
        }
        const extension = blob.type.split("/")[1] || "jpg";
        const ref = await uploadFile(blob, { name: `${example.id}.${extension}` });
        await sendCommand("set_image", {
          image: ref,
          prompt: base,
        });
        basePromptRef.current = base;
        lastSentPromptRef.current = base;
        setBasePrompt(base);
        setSentPrompt(base);
        activeEventsRef.current = events;
        heldEventsRef.current = [];
        setActiveEvents(events);
        setHeldEvents([]);
        setActiveExampleId(example.id);
        setPromptDraft(base);
        setPerspective(example.perspective);
        setLastChunk(null);
        if (isPaused) {
          await resumeTrack("main_video");
          setIsPaused(false);
        }
      } catch (err) {
        setErrorToast(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingExampleId(null);
      }
    },
    [isReady, loadingExampleId, sendCommand, uploadFile, isPaused, resumeTrack, eventOverrides],
  );

  const resetWorld = useCallback(() => {
    if (!isReady || !hasWorld) return;
    setLastChunk(null);
    void sendCommand("reset", { seed });
  }, [isReady, hasWorld, sendCommand, seed]);

  // The prompt box replaces the base prompt; a held event is re-applied on top of it.
  const sendPrompt = useCallback(() => {
    if (!isReady) return;
    const base = stripEvent(promptDraft);
    if (!base) return;
    basePromptRef.current = base;
    setBasePrompt(base);
    setPromptDraft(base);
    lastSentPromptRef.current = null;
    syncPrompt();
  }, [isReady, promptDraft, syncPrompt]);

  // ---- Editing the active world's events ----

  useEffect(() => {
    setEventOverrides(loadEventOverrides());
  }, []);

  const saveActiveEvents = useCallback(
    async (edited: ExampleEvent[]) => {
      const events = cleanEvents(edited);
      const id = activeExampleId;
      if (!id) return;
      const saved = savedExamples.find((ex) => ex.id === id);
      try {
        if (saved) {
          await putSavedExample({
            id: saved.id,
            name: saved.name,
            perspective: saved.perspective,
            scene: saved.scene,
            character: saved.character,
            events,
            image: saved.image,
            createdAt: saved.createdAt,
          });
          setSavedExamples((list) => list.map((ex) => (ex.id === id ? { ...ex, events } : ex)));
        } else {
          const next = { ...eventOverrides, [id]: events };
          setEventOverrides(next);
          saveEventOverrides(next);
        }
      } catch (err) {
        setErrorToast(err instanceof Error ? err.message : String(err));
        return;
      }
      activeEventsRef.current = events;
      heldEventsRef.current = [];
      setActiveEvents(events);
      setHeldEvents([]);
      setEditingEvents(null);
      syncPrompt();
    },
    [activeExampleId, savedExamples, eventOverrides, syncPrompt],
  );

  const restoreDefaultEvents = useCallback(() => {
    const id = activeExampleId;
    const builtIn = BUILT_IN_EXAMPLES.find((ex) => ex.id === id);
    if (!id || !builtIn) return;
    const next = { ...eventOverrides };
    delete next[id];
    setEventOverrides(next);
    saveEventOverrides(next);
    activeEventsRef.current = builtIn.events;
    heldEventsRef.current = [];
    setActiveEvents(builtIn.events);
    setHeldEvents([]);
    setEditingEvents(null);
    syncPrompt();
  }, [activeExampleId, eventOverrides, syncPrompt]);

  const togglePause = useCallback(async () => {
    try {
      if (isPaused) {
        await resumeTrack("main_video");
        setIsPaused(false);
      } else {
        await pauseTrack("main_video");
        setIsPaused(true);
      }
    } catch (err) {
      setErrorToast(err instanceof Error ? err.message : String(err));
    }
  }, [isPaused, pauseTrack, resumeTrack]);

  // ---- Saved examples ----

  useEffect(() => {
    let cancelled = false;
    listSavedExamples()
      .then((list) => {
        if (!cancelled) setSavedExamples(list.map(toView));
      })
      .catch((err) => setErrorToast(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  const savedRef = useRef(savedExamples);
  useEffect(() => {
    savedRef.current = savedExamples;
  }, [savedExamples]);
  useEffect(() => () => savedRef.current.forEach((ex) => URL.revokeObjectURL(ex.imageSrc)), []);

  const saveExample = useCallback(
    async (input: NewExample, play: boolean) => {
      const saved: SavedExample = {
        id: newExampleId(),
        name: input.name,
        perspective: input.perspective,
        scene: input.scene,
        character: input.character,
        events: input.events,
        image: input.image,
        createdAt: Date.now(),
      };
      await putSavedExample(saved);
      const view = toView(saved);
      setSavedExamples((list) => [view, ...list]);
      if (play) void startWorld(view, view.image);
    },
    [startWorld],
  );

  const removeExample = useCallback(async (example: SavedExampleView) => {
    if (!window.confirm(`Delete "${example.name}"?`)) return;
    try {
      await deleteSavedExample(example.id);
      URL.revokeObjectURL(example.imageSrc);
      setSavedExamples((list) => list.filter((ex) => ex.id !== example.id));
    } catch (err) {
      setErrorToast(err instanceof Error ? err.message : String(err));
    }
  }, []);

  // Dropping an image anywhere on the page must not navigate away from it.
  useEffect(() => {
    const prevent = (e: DragEvent) => {
      if (dragHasImagePayload(e.dataTransfer)) e.preventDefault();
    };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  }, []);

  const openAddWithDrop = useCallback((dt: DataTransfer) => {
    imageFileFromDataTransfer(dt)
      .then((file) => {
        setDroppedImage(file);
        setAddOpen(true);
      })
      .catch((err) => setErrorToast(err instanceof Error ? err.message : String(err)));
  }, []);

  const currentAction = useMemo(
    () => composeAction(heldView),
    [heldView],
  );
  const controlsDisabled = !isReady || !hasWorld || limitReached;

  // A disabled on-screen key gets no pointerup, so release any it was holding.
  useEffect(() => {
    if (!controlsDisabled || padsHeld.current.size === 0) return;
    padsHeld.current.clear();
    syncAction();
  }, [controlsDisabled, syncAction]);

  // ---- Render: example card ----

  const renderExampleCard = (example: Example, saved?: SavedExampleView) => {
    const isActive = activeExampleId === example.id;
    const isLoading = loadingExampleId === example.id;
    const disabled = !isReady || !!loadingExampleId;
    const accent = saved ? "emerald" : "amber";
    return (
      <div
        key={example.id}
        className={cn(
          "group flex items-stretch gap-1.5 rounded-lg border transition-all",
          isActive
            ? accent === "emerald"
              ? "border-emerald-300/55 bg-emerald-300/10"
              : "border-amber-300/60 bg-amber-300/10"
            : "border-white/10 bg-white/[0.03] hover:border-white/20 hover:bg-white/[0.06]",
        )}
      >
        <button
          type="button"
          onClick={() => startWorld(example, saved?.image)}
          disabled={disabled}
          title={isReady ? "Start a world from this example" : "Connect first"}
          className={cn(
            "relative flex min-w-0 flex-1 items-center gap-3 p-2 text-left",
            saved ? "rounded-l-lg" : "rounded-lg",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          <div className="relative h-14 w-24 shrink-0 overflow-hidden rounded-md border border-white/10 bg-black/30">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={example.imageSrc} alt={example.name} className="h-full w-full object-cover" />
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-amber-300 border-t-transparent" />
              </div>
            )}
          </div>
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="truncate font-mono text-sm font-medium text-white">{example.name}</span>
              <span className="shrink-0 rounded border border-white/15 bg-white/5 px-1.5 py-0.5 font-mono text-[8px] uppercase tracking-wider text-white/60">
                {example.perspective}
              </span>
              {saved && (
                <span className="shrink-0 rounded border border-emerald-300/35 bg-emerald-300/10 px-1.5 py-0.5 font-mono text-[8px] uppercase tracking-wider text-emerald-200">
                  yours
                </span>
              )}
            </div>
            <span className="line-clamp-2 font-mono text-[10px] leading-snug text-white/50">
              {example.scene || "Generic description of the image"}
            </span>
            <span className="font-mono text-[9px] text-white/35">
              {(eventOverrides[example.id] ?? example.events).length} event
              {(eventOverrides[example.id] ?? example.events).length === 1 ? "" : "s"}
            </span>
          </div>
          {isActive && (
            <div
              className={cn(
                "h-2 w-2 shrink-0 rounded-full",
                accent === "emerald" ? "bg-emerald-300" : "bg-amber-300",
              )}
            />
          )}
        </button>
        {saved && (
          <button
            type="button"
            onClick={() => removeExample(saved)}
            title={`Delete ${example.name}`}
            aria-label={`Delete ${example.name}`}
            className="flex w-9 shrink-0 items-center justify-center rounded-r-lg border-l border-white/5 font-mono text-lg leading-none text-white/45 transition-colors hover:bg-white/[0.06] hover:text-red-300"
          >
            ×
          </button>
        )}
      </div>
    );
  };

  // ---- Live prompt readout ----

  const heldEventName = (() => {
    const slot = heldEvents.length > 0 ? heldEvents[heldEvents.length - 1] : -1;
    const ev = activeEvents[slot];
    return ev ? ev.name || `event ${slot + 1}` : null;
  })();
  const draftDirty = promptDraft.trim() !== basePrompt.trim();
  const promptStatus: { label: string; tone: "green" | "amber" | "yellow" } | null = !sentPrompt
    ? null
    : (modelState?.prompt ?? "").trim() !== sentPrompt.trim()
      ? { label: "sending…", tone: "yellow" }
      : !lastChunk
        ? { label: "accepted · waiting for a chunk", tone: "yellow" }
        : lastChunk.prompt.trim() !== sentPrompt.trim()
          ? { label: "accepted · on screen from next chunk", tone: "amber" }
          : { label: "on screen", tone: "green" };
  const copyLivePrompt = async () => {
    if (!sentPrompt) return;
    try {
      await navigator.clipboard.writeText(sentPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable.
    }
  };

  // ---- Render: sidebar ----

  const sidebar = (
    <div className="flex flex-col gap-4">
      <div
        className={cn(
          "relative flex flex-col gap-2 rounded-lg transition-colors",
          galleryDropOver && "bg-emerald-300/[0.045] outline outline-1 outline-emerald-300/55",
        )}
        onDragEnter={(e) => {
          if (!dragHasImagePayload(e.dataTransfer)) return;
          e.preventDefault();
          galleryDropDepth.current += 1;
          setGalleryDropOver(true);
        }}
        onDragOver={(e) => {
          if (!dragHasImagePayload(e.dataTransfer)) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={() => {
          galleryDropDepth.current = Math.max(0, galleryDropDepth.current - 1);
          if (galleryDropDepth.current === 0) setGalleryDropOver(false);
        }}
        onDrop={(e) => {
          if (!dragHasImagePayload(e.dataTransfer)) return;
          e.preventDefault();
          galleryDropDepth.current = 0;
          setGalleryDropOver(false);
          openAddWithDrop(e.dataTransfer);
        }}
      >
        {galleryDropOver && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-start justify-center rounded-lg border border-emerald-300/35 bg-black/35 p-3">
            <span className="rounded border border-emerald-300/35 bg-emerald-300/15 px-3 py-2 font-mono text-[10px] uppercase tracking-wider text-emerald-100 shadow-lg">
              Drop image to add an example
            </span>
          </div>
        )}
        <span className="font-mono text-xs uppercase tracking-widest text-primary">Examples</span>
        <p className="text-[10px] leading-snug text-white/40">
          Click an example to start a world from its image and prompt. Examples you add are saved
          in this browser.
        </p>
        <div className="flex flex-col gap-2">
          {BUILT_IN_EXAMPLES.map((ex) => renderExampleCard(ex))}
          {savedExamples.map((ex) => renderExampleCard(ex, ex))}
        </div>
        <button
          type="button"
          onClick={() => {
            setDroppedImage(null);
            setAddOpen(true);
          }}
          className="group flex w-full items-center gap-3 rounded-lg border border-emerald-300/25 bg-emerald-300/[0.07] p-3 text-left transition-colors hover:border-emerald-300/45 hover:bg-emerald-300/[0.11]"
        >
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-emerald-300/35 bg-emerald-300/10 font-mono text-xl leading-none text-emerald-100 group-hover:bg-emerald-300/15">
            +
          </span>
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="font-mono text-sm font-medium text-white">Add your own example</span>
            <span className="font-mono text-[10px] leading-snug text-white/45">
              An image, a prompt and a perspective. Or drop an image here.
            </span>
          </span>
        </button>
      </div>

      <div className="border-t border-white/[0.06]" />

      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className={cn("h-1.5 w-1.5 rounded-full", hasWorld ? "bg-green-400" : "bg-white/20")} />
            <span className="font-mono text-[10px] text-white/50">
              Image{modelState?.imageName ? `: ${modelState.imageName}` : ""}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={cn("h-1.5 w-1.5 rounded-full", modelState?.prompt ? "bg-green-400" : "bg-white/20")}
            />
            <span className="font-mono text-[10px] text-white/50">Prompt</span>
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-wider text-white/50">Perspective</span>
          <PerspectiveToggle value={perspective} onChange={changePerspective} disabled={!isReady} />
          <span className="font-mono text-[10px] leading-snug text-white/40">
            Applies from the next chunk. Use an image and prompt that match it.
          </span>
        </div>

      </div>

      {errorToast && (
        <div className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2 font-mono text-xs text-red-300">
          {errorToast}
        </div>
      )}

      <div className="border-t border-white/[0.06]" />

      <div className="flex flex-col gap-3">
        <span className="font-mono text-xs uppercase tracking-widest text-primary">Prompt</span>

        {/* Live readout: the full prompt last sent (base + held event) and what the model used. */}
        <div className="flex flex-col gap-2" role="region" aria-label="Live prompt">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-white/55">Live prompt</span>
            {promptStatus && (
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px]",
                  promptStatus.tone === "green" && "border-emerald-400/40 bg-emerald-400/10 text-emerald-200",
                  promptStatus.tone === "amber" && "border-amber-300/50 bg-amber-300/10 text-amber-200",
                  promptStatus.tone === "yellow" && "border-yellow-300/40 bg-yellow-300/10 text-yellow-200",
                )}
              >
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    promptStatus.tone === "green" ? "bg-emerald-400" : promptStatus.tone === "amber" ? "bg-amber-300" : "bg-yellow-300",
                  )}
                />
                {promptStatus.label}
              </span>
            )}
            <div className="flex-1" />
            {sentPrompt && (
              <>
                <span className="font-mono text-[10px] text-white/35">{sentPrompt.length} chars</span>
                <button
                  type="button"
                  onClick={copyLivePrompt}
                  className="font-mono text-[10px] text-white/45 transition-colors hover:text-white/80"
                >
                  {copied ? "Copied" : "Copy"}
                </button>
              </>
            )}
          </div>
          {heldEventName && (
            <span className="self-start rounded-full border border-amber-300/60 bg-amber-300/15 px-2.5 py-0.5 font-mono text-[10px] text-amber-100">
              event held · {heldEventName}
            </span>
          )}
          <div
            className={cn(
              "rounded-lg border bg-black/30 p-3 transition-colors",
              heldEventName ? "border-amber-300/40" : "border-white/10",
            )}
          >
            {sentPrompt ? (
              <PromptText text={sentPrompt} />
            ) : (
              <p className="font-mono text-[11px] italic text-white/30">
                Nothing sent yet. Pick an example to start a world.
              </p>
            )}
          </div>
          {sentPrompt && lastChunk && (
            <div className="flex flex-col gap-1.5">
              <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] text-white/45">
                <span>On screen · chunk {lastChunk.index}</span>
                {lastChunk.prompt.trim() === sentPrompt.trim() ? (
                  <span className="text-emerald-300/80">same as live</span>
                ) : (
                  <button
                    type="button"
                    onClick={() => setShowOnScreen((v) => !v)}
                    className="text-amber-200/80 transition-colors hover:text-amber-100"
                  >
                    previous prompt {showOnScreen ? "▾ hide" : "▸ show"}
                  </button>
                )}
              </div>
              {showOnScreen && lastChunk.prompt.trim() !== sentPrompt.trim() && (
                <div className="rounded-md border border-white/10 bg-white/[0.02] p-2.5 opacity-80">
                  <PromptText text={lastChunk.prompt} small />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Editable base prompt; the live readout above never overwrites the draft. */}
        <div className="flex flex-col gap-2 border-t border-dashed border-white/[0.08] pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-white/55">Edit base prompt</span>
            {draftDirty && (
              <span className="rounded border border-white/20 bg-white/5 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-white/60">
                edited · not sent
              </span>
            )}
            <div className="ml-auto flex items-center gap-1">
            {draftDirty && basePrompt && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setPromptDraft(basePrompt)}
                title="Discard edits and show the base prompt in effect"
                className="h-7 px-2 font-mono text-[10px] text-white/50"
              >
                Revert
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              disabled={!promptDraft}
              onClick={() => setPromptDraft("")}
              className="h-7 px-2 font-mono text-[10px] text-white/50 hover:text-red-200"
            >
              Clear
            </Button>
            <Button
              size="sm"
              disabled={!isReady || !promptDraft.trim()}
              onClick={sendPrompt}
              title="Replace the base prompt for upcoming chunks, keeping the current world"
              className="h-7 px-3 font-mono text-[10px]"
            >
              Send
            </Button>
            </div>
          </div>
          <span className="font-mono text-[10px] text-white/30">
            Enter sends · Shift+Enter new line · held events are added on top
          </span>
          <textarea
            ref={promptBoxRef}
            value={promptDraft}
            onChange={(e) => setPromptDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
              }
            }}
            maxLength={4096}
            rows={3}
            placeholder="The scene in the video is: ..."
            className="max-h-[45vh] min-h-[4.5rem] w-full resize-none overflow-y-auto rounded-md border border-white/10 bg-black/30 px-2.5 py-2 font-mono text-[11px] leading-snug text-white placeholder-white/25 outline-none focus:border-white/30"
          />
        </div>
      </div>

      {/* Advanced: rarely-touched settings, collapsed by default */}
      <div className="flex flex-col gap-2 border-t border-white/[0.06] pt-3">
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          aria-expanded={advancedOpen}
          className="self-start font-mono text-[10px] uppercase tracking-wider text-white/40 transition-colors hover:text-white/70"
        >
          {advancedOpen ? "▾" : "▸"} Advanced
        </button>
        {advancedOpen && (
          <div className="flex flex-col gap-3 rounded border border-white/10 bg-white/[0.02] p-3">
            <div className="flex items-center gap-3">
              <label className="w-20 shrink-0 font-mono text-[10px] uppercase tracking-wider text-white/50">
                Seed
              </label>
              <Input
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
                className="h-7 w-24 px-2 font-mono text-xs md:text-xs"
              />
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSeed(Math.floor(Math.random() * 1_000_000))}
                className="h-7 font-mono text-[10px]"
              >
                Random
              </Button>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-20 shrink-0" />
              <Button
                size="sm"
                variant="secondary"
                disabled={!isReady || !hasWorld}
                onClick={resetWorld}
                title="Restart the current image with this seed"
                className="h-7 bg-red-500/15 font-mono text-[10px] text-red-300 hover:bg-red-500/25"
              >
                Reset world
              </Button>
              <span className="font-mono text-[10px] leading-snug text-white/40">Keeps prompt and held controls.</span>
            </div>
            <SpeedSlider
              label="Yaw speed"
              value={yawDegrees}
              disabled={!isReady}
              onChange={(v) => {
                setYawDegrees(v);
                sendSpeed("yaw_degrees", v);
              }}
            />
            <SpeedSlider
              label="Pitch speed"
              value={pitchDegrees}
              disabled={!isReady}
              onChange={(v) => {
                setPitchDegrees(v);
                sendSpeed("pitch_degrees", v);
              }}
            />
          </div>
        )}
      </div>

      {addOpen && (
        <AddExampleModal
          initialImage={droppedImage}
          canPlay={isReady && !loadingExampleId}
          onSave={saveExample}
          onClose={() => {
            setAddOpen(false);
            setDroppedImage(null);
          }}
        />
      )}

      {editingEvents && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setEditingEvents(null)}
        >
          <div
            className="flex max-h-[90vh] w-full max-w-lg flex-col gap-4 overflow-y-auto rounded-xl border border-white/15 bg-neutral-950 p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="font-mono text-sm text-white">Edit events</h3>
            <span className="font-mono text-[10px] leading-snug text-white/40">
              Saved in this browser for this example.
            </span>
            <EventsEditor events={editingEvents} onChange={setEditingEvents} />
            <div className="flex items-center justify-end gap-2">
              {activeExampleId && eventOverrides[activeExampleId] && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={restoreDefaultEvents}
                  className="mr-auto font-mono text-[11px] text-white/50"
                >
                  Restore defaults
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={() => setEditingEvents(null)} className="font-mono text-[11px]">
                Cancel
              </Button>
              <Button size="sm" onClick={() => saveActiveEvents(editingEvents)} className="font-mono text-[11px]">
                Save
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // ---- Render: controls ----

  const lit = (code: string) => heldView.has(code);
  const turnLit = (dir: Turn) => currentAction.turn === dir;
  const lookLit = (dir: Look) => currentAction.look === dir;

  const controls = (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        {modelState && (
          <span className="font-mono text-[10px] text-white/40">
            chunk {modelState.completedChunks}/{modelState.maxChunks}
          </span>
        )}
        {lastChunk && (
          <span className="font-mono text-[10px] text-white/40">
            last chunk {lastChunk.frames} frames in {lastChunk.elapsedSeconds.toFixed(2)} s
          </span>
        )}
        <span className={cn("font-mono text-[10px]", isReady ? "text-amber-300/70" : "text-white/35")}>
          action: {modelState?.action ?? "none"}
        </span>
        <div className="flex-1" />
        {isReady && (
          <Button size="sm" variant="secondary" onClick={togglePause} className="font-mono text-[10px]">
            {isPaused ? "Resume" : "Pause"}
          </Button>
        )}
      </div>

      {activeExampleId && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
              Events · hold a chip or key 1–{Math.max(1, Math.min(MAX_EVENTS, activeEvents.length))}; shows after a
              short delay, ends on release
            </span>
            <button
              type="button"
              onClick={() => setEditingEvents(activeEvents.map((ev) => ({ ...ev })))}
              className="shrink-0 whitespace-nowrap font-mono text-[10px] text-white/40 transition-colors hover:text-amber-200"
              title="Edit this example's events"
            >
              ✎ edit
            </button>
          </div>
          {activeEvents.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {activeEvents.slice(0, MAX_EVENTS).map((ev, slot) => (
                <EventChip
                  key={slot}
                  slot={slot}
                  name={ev.name}
                  pressed={heldEvents.length > 0 && heldEvents[heldEvents.length - 1] === slot}
                  disabled={!isReady || !hasWorld}
                  onPress={() => pressEvent(slot)}
                  onRelease={() => releaseEvent(slot)}
                />
              ))}
            </div>
          ) : (
            <span className="font-mono text-[10px] text-white/35">No events for this example yet.</span>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">Move (WASD)</span>
          <div className="flex flex-col items-center gap-0.5">
            <PadButton
              label="W"
              pressed={lit("KeyW")}
              disabled={controlsDisabled}
              onPress={() => pressPad("KeyW")}
              onRelease={() => releasePad("KeyW")}
            />
            <div className="flex gap-0.5">
              {(["KeyA", "KeyS", "KeyD"] as const).map((code) => (
                <PadButton
                  key={code}
                  label={code.slice(3)}
                  pressed={lit(code)}
                  disabled={controlsDisabled}
                  onPress={() => pressPad(code)}
                  onRelease={() => releasePad(code)}
                />
              ))}
            </div>
            <PadButton
              label="Space"
              pressed={lit("Space")}
              disabled={controlsDisabled}
              onPress={() => pressPad("Space")}
              onRelease={() => releasePad("Space")}
              className="h-8 w-[124px] text-[11px]"
            />
          </div>
          <span className="font-mono text-[10px] text-white/50">{currentAction.movement}</span>
        </div>

        <div className="flex flex-col items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Turn ← → · Look ↑ ↓
          </span>
          <div className="flex flex-col items-center gap-0.5">
            <PadButton
              label="↑"
              pressed={lookLit("up")}
              disabled={controlsDisabled}
              onPress={() => pressPad("ArrowUp")}
              onRelease={() => releasePad("ArrowUp")}
            />
            <div className="flex gap-0.5">
              <PadButton
                label="←"
                pressed={turnLit("left")}
                disabled={controlsDisabled}
                onPress={() => pressPad("ArrowLeft")}
                onRelease={() => releasePad("ArrowLeft")}
              />
              <PadButton
                label="↓"
                pressed={lookLit("down")}
                disabled={controlsDisabled}
                onPress={() => pressPad("ArrowDown")}
                onRelease={() => releasePad("ArrowDown")}
              />
              <PadButton
                label="→"
                pressed={turnLit("right")}
                disabled={controlsDisabled}
                onPress={() => pressPad("ArrowRight")}
                onRelease={() => releasePad("ArrowRight")}
              />
            </div>
          </div>
          <span className="font-mono text-[10px] text-white/50">
            turn {currentAction.turn} · look {currentAction.look}
          </span>
        </div>

      </div>

      <p className="font-mono text-[10px] leading-snug text-white/35">
        Controls are held while pressed and released when you let go.
      </p>
    </div>
  );

  // ---- Render: stage ----

  const stageOverlay = loadingExampleId ? (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/70">
      <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.06] px-4 py-2">
        <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-amber-300 border-t-transparent" />
        <span className="font-mono text-xs text-white/70">Loading example</span>
      </div>
    </div>
  ) : limitReached ? (
    <div className="pointer-events-none absolute inset-x-0 top-4 flex justify-center px-4">
      <div className="max-w-md rounded-lg border border-amber-300/30 bg-black/70 px-4 py-2 text-center font-mono text-xs text-amber-100">
        {completionDetail ?? "All chunks are complete."}
      </div>
    </div>
  ) : isPaused ? (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/35">
      <div className="rounded-lg border border-white/10 bg-black/55 px-4 py-2 font-mono text-xs text-white/70">
        Paused
      </div>
    </div>
  ) : !hasWorld ? (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      <span className="font-mono text-xs text-white/40">
        {isReady ? "Pick an example to start a world" : "Connect, then pick an example"}
      </span>
    </div>
  ) : null;

  const stage = (
    <div
      className="relative aspect-video w-full shrink-0 overflow-hidden rounded-xl border border-white/[0.08] bg-black lg:min-w-[320px] lg:max-w-full lg:resize"
    >
      <ReactorView
        track="main_video"
        videoObjectFit="contain"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      />
      {stageOverlay}
    </div>
  );

  return { sidebar, stage, controls };
}
