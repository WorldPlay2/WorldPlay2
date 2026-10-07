"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { MAX_EVENTS, type ExampleEvent, type Perspective } from "@/lib/examples";
import {
  assertAcceptedImage,
  dragHasImagePayload,
  imageFileFromDataTransfer,
} from "@/lib/image-drop";

export type NewExample = {
  name: string;
  perspective: Perspective;
  scene: string;
  character: string;
  events: ExampleEvent[];
  image: File;
};

const FIELD_CLASS =
  "w-full resize-y rounded-md border border-white/10 bg-black/30 px-3 py-2 font-mono text-[11px] leading-relaxed text-white placeholder-white/25 outline-none focus:border-white/30";

/** Removes events with no text. */
export function cleanEvents(events: ExampleEvent[]): ExampleEvent[] {
  return events
    .map((ev) => ({ name: ev.name.trim(), text: ev.text.trim() }))
    .filter((ev) => ev.text)
    .map((ev, i) => ({ ...ev, name: ev.name || `Event ${i + 1}` }));
}

/** Editable list of named events (name + one sentence each). */
export function EventsEditor({
  events,
  onChange,
}: {
  events: ExampleEvent[];
  onChange: (events: ExampleEvent[]) => void;
}) {
  const update = (index: number, patch: Partial<ExampleEvent>) =>
    onChange(events.map((ev, i) => (i === index ? { ...ev, ...patch } : ev)));
  return (
    <div className="flex flex-col gap-2">
      {events.map((ev, i) => (
        <div key={i} className="flex items-start gap-2">
          <span className="mt-1.5 inline-flex h-4 min-w-4 items-center justify-center rounded border border-white/25 bg-white/10 px-0.5 font-mono text-[9px] font-bold text-white/80">
            {i + 1}
          </span>
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <Input
              value={ev.name}
              onChange={(e) => update(i, { name: e.target.value })}
              placeholder="Name, e.g. Lightning"
              className="h-7 font-mono text-xs"
            />
            <input
              value={ev.text}
              onChange={(e) => update(i, { text: e.target.value })}
              placeholder="What happens, e.g. lightning flashes across the sky"
              className="h-7 rounded-md border border-white/10 bg-black/30 px-3 font-mono text-[11px] text-white placeholder-white/25 outline-none focus:border-white/30"
            />
          </div>
          <button
            type="button"
            onClick={() => onChange(events.filter((_, j) => j !== i))}
            aria-label={`Remove event ${i + 1}`}
            className="mt-1 h-6 w-6 rounded font-mono text-sm text-white/45 hover:bg-white/10 hover:text-red-300"
          >
            ×
          </button>
        </div>
      ))}
      {events.length < MAX_EVENTS && (
        <button
          type="button"
          onClick={() => onChange([...events, { name: "", text: "" }])}
          className="self-start font-mono text-[10px] text-white/50 transition-colors hover:text-amber-200"
        >
          + Add event
        </button>
      )}
    </div>
  );
}

/** Form for adding an example: image, name, perspective, prompt parts and events. */
export function AddExampleModal({
  initialImage,
  canPlay,
  onSave,
  onClose,
}: {
  initialImage?: File | null;
  canPlay: boolean;
  onSave: (example: NewExample, play: boolean) => Promise<void>;
  onClose: () => void;
}) {
  const [image, setImage] = useState<File | null>(initialImage ?? null);
  const [preview, setPreview] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [perspective, setPerspective] = useState<Perspective>("fps");
  const [scene, setScene] = useState("");
  const [character, setCharacter] = useState("");
  const [events, setEvents] = useState<ExampleEvent[]>([]);
  const [dropOver, setDropOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!image) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const acceptFile = (file: File) => {
    try {
      assertAcceptedImage(file);
      setImage(file);
      setError(null);
      if (!name.trim()) setName(file.name.replace(/\.[^.]+$/, ""));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const save = async (play: boolean) => {
    if (!image) {
      setError("Choose an image first.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await onSave(
        {
          name: name.trim() || "Untitled example",
          perspective,
          scene: scene.trim(),
          character: perspective === "tps" ? character.trim() : "",
          events: cleanEvents(events),
          image,
        },
        play,
      );
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="flex max-h-[90vh] w-full max-w-lg flex-col gap-4 overflow-y-auto rounded-xl border border-white/15 bg-neutral-950 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="font-mono text-sm text-white">Add an example</h3>
          <button
            type="button"
            onClick={onClose}
            className="h-7 w-7 rounded font-mono text-xs text-white/60 hover:bg-white/10"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
          onDragOver={(e) => {
            if (!dragHasImagePayload(e.dataTransfer)) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            setDropOver(true);
          }}
          onDragLeave={() => setDropOver(false)}
          onDrop={(e) => {
            if (!dragHasImagePayload(e.dataTransfer)) return;
            e.preventDefault();
            setDropOver(false);
            imageFileFromDataTransfer(e.dataTransfer)
              .then(acceptFile)
              .catch((err) => setError(err instanceof Error ? err.message : String(err)));
          }}
          className={cn(
            "relative flex aspect-video w-full cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-dashed transition-colors",
            dropOver
              ? "border-emerald-300/60 bg-emerald-300/[0.06]"
              : "border-white/20 bg-white/[0.03] hover:border-white/35",
          )}
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt="Selected image" className="h-full w-full object-cover" />
          ) : (
            <span className="px-4 text-center font-mono text-[11px] text-white/45">
              Drop an image here or click to choose one (PNG, JPEG or WebP, up to 25 MiB)
            </span>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) acceptFile(file);
              e.target.value = "";
            }}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="font-mono text-[10px] uppercase tracking-wider text-white/50">Name</label>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My scene"
            className="font-mono text-xs"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="font-mono text-[10px] uppercase tracking-wider text-white/50">
            Perspective
          </label>
          <PerspectiveToggle value={perspective} onChange={setPerspective} />
          <span className="font-mono text-[10px] leading-snug text-white/40">
            FPS for a view through someone&apos;s eyes; TPS when the image shows the character to
            follow.
          </span>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="font-mono text-[10px] uppercase tracking-wider text-white/50">
            The scene in the video is:
          </label>
          <textarea
            value={scene}
            onChange={(e) => setScene(e.target.value)}
            rows={3}
            maxLength={2000}
            placeholder="a narrow alley at night in heavy rain, lined with brick buildings and neon signs"
            className={FIELD_CLASS}
          />
        </div>

        {perspective === "tps" && (
          <div className="flex flex-col gap-1.5">
            <label className="font-mono text-[10px] uppercase tracking-wider text-white/50">
              The character is:
            </label>
            <textarea
              value={character}
              onChange={(e) => setCharacter(e.target.value)}
              rows={2}
              maxLength={1000}
              placeholder="a man in a dark raincoat, seen from behind"
              className={FIELD_CLASS}
            />
          </div>
        )}
        <span className="-mt-2 font-mono text-[10px] leading-snug text-white/40">
          Empty fields use a generic description of the image.
        </span>

        <div className="flex flex-col gap-1.5">
          <label className="font-mono text-[10px] uppercase tracking-wider text-white/50">
            Events (optional)
          </label>
          <span className="font-mono text-[10px] leading-snug text-white/40">
            Each event adds &quot;The event is: ...&quot; to the prompt while its chip or number key is held.
          </span>
          <EventsEditor events={events} onChange={setEvents} />
        </div>

        {error && (
          <div className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2 font-mono text-xs text-red-300">
            {error}
          </div>
        )}

        <div className="flex items-center justify-end gap-2">
          <Button size="sm" variant="ghost" onClick={onClose} className="font-mono text-[11px]">
            Cancel
          </Button>
          <Button
            size="sm"
            variant="secondary"
            disabled={saving || !image}
            onClick={() => save(false)}
            className="font-mono text-[11px]"
          >
            Save
          </Button>
          <Button
            size="sm"
            disabled={saving || !image || !canPlay}
            onClick={() => save(true)}
            title={canPlay ? undefined : "Connect first to play"}
            className="font-mono text-[11px]"
          >
            Save &amp; play
          </Button>
        </div>
      </div>
    </div>
  );
}

/** Two-way FPS / TPS switch. */
export function PerspectiveToggle({
  value,
  onChange,
  disabled,
}: {
  value: Perspective;
  onChange: (value: Perspective) => void;
  disabled?: boolean;
}) {
  return (
    <div className="inline-flex w-fit rounded-md border border-white/10 bg-black/30 p-0.5">
      {(["fps", "tps"] as const).map((option) => (
        <button
          key={option}
          type="button"
          disabled={disabled}
          onClick={() => onChange(option)}
          aria-pressed={value === option}
          className={cn(
            "rounded px-3 py-1 font-mono text-[11px] uppercase tracking-wider transition-colors disabled:cursor-not-allowed disabled:opacity-40",
            value === option ? "bg-amber-300/20 text-amber-100" : "text-white/50 hover:text-white/80",
          )}
        >
          {option === "fps" ? "FPS · first person" : "TPS · third person"}
        </button>
      ))}
    </div>
  );
}
