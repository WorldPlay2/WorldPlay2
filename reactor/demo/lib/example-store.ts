import { splitPrompt, type ExampleEvent, type Perspective } from "@/lib/examples";

// Examples added from the UI, kept in this browser's IndexedDB.

/** A saved example as stored: the image is kept as a Blob. */
export type SavedExample = {
  id: string;
  name: string;
  perspective: Perspective;
  scene: string;
  character: string;
  events: ExampleEvent[];
  image: Blob;
  createdAt: number;
};

/** Entries saved before events existed carry one `prompt` string instead of parts. */
type StoredExample = Partial<SavedExample> & { id: string; image: Blob; prompt?: string };

function migrate(entry: StoredExample): SavedExample {
  const parts =
    entry.scene === undefined ? splitPrompt(entry.prompt ?? "") : { scene: entry.scene, character: entry.character ?? "" };
  return {
    id: entry.id,
    name: entry.name ?? "Untitled example",
    perspective: entry.perspective === "tps" ? "tps" : "fps",
    scene: parts.scene,
    character: parts.character,
    events: Array.isArray(entry.events) ? entry.events : [],
    image: entry.image,
    createdAt: entry.createdAt ?? 0,
  };
}

const DB_NAME = "worldplay2";
const DB_VERSION = 1;
const STORE = "examples";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("This browser has no IndexedDB, so examples cannot be saved."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open IndexedDB."));
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const request = run(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed."));
      tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted."));
    });
  } finally {
    db.close();
  }
}

/** All saved examples, newest first. */
export async function listSavedExamples(): Promise<SavedExample[]> {
  const all = await withStore<StoredExample[]>("readonly", (store) => store.getAll());
  return all.map(migrate).sort((a, b) => b.createdAt - a.createdAt);
}

export async function putSavedExample(example: SavedExample): Promise<void> {
  await withStore("readwrite", (store) => store.put(example));
}

export async function deleteSavedExample(id: string): Promise<void> {
  await withStore("readwrite", (store) => store.delete(id));
}

export function newExampleId(): string {
  return `saved-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

// Edited events of built-in examples, kept in localStorage by example id.
const EVENT_OVERRIDES_KEY = "worldplay2:event-overrides:v5";

export function loadEventOverrides(): Record<string, ExampleEvent[]> {
  try {
    const parsed = JSON.parse(localStorage.getItem(EVENT_OVERRIDES_KEY) ?? "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function saveEventOverrides(overrides: Record<string, ExampleEvent[]>): void {
  try {
    localStorage.setItem(EVENT_OVERRIDES_KEY, JSON.stringify(overrides));
  } catch {
    // Storage unavailable: edits last for this page load only.
  }
}
