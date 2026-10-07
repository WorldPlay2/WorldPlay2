// Server-only: reads the endpoint list from files at request time, so editing
// them needs a page reload, not a rebuild. Import only from server components
// and route handlers.
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import type { Endpoint, EndpointKind } from "@/lib/endpoints";

export type EndpointConfig = Endpoint & {
  /** Name of the server-side env var holding this hosted endpoint's API key. */
  apiKeyEnv?: string;
};

const SHIPPED_FILE = "endpoints.json";
const LOCAL_FILE = "endpoints.local.json";

function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function parseFile(file: string): EndpointConfig[] {
  const full = path.join(process.cwd(), file);
  const raw = JSON.parse(readFileSync(full, "utf8")) as { endpoints?: unknown };
  if (!Array.isArray(raw.endpoints)) throw new Error(`${file}: expected {"endpoints": [...]}`);
  return raw.endpoints.map((value, i) => {
    const entry = value as Record<string, unknown>;
    const where = `${file} entry ${i + 1}`;
    const label = typeof entry.label === "string" ? entry.label.trim() : "";
    const url = typeof entry.url === "string" ? entry.url.trim().replace(/\/+$/, "") : "";
    const kind = entry.kind as EndpointKind;
    if (!label) throw new Error(`${where}: "label" is required`);
    if (!/^https?:\/\//.test(url)) throw new Error(`${where}: "url" must start with http:// or https://`);
    if (kind !== "local" && kind !== "hosted") throw new Error(`${where}: "kind" must be "local" or "hosted"`);
    const id = typeof entry.id === "string" && entry.id.trim() ? entry.id.trim() : slug(label);
    const apiKeyEnv = typeof entry.apiKeyEnv === "string" ? entry.apiKeyEnv.trim() : undefined;
    if (kind === "hosted" && !apiKeyEnv) throw new Error(`${where}: a hosted entry needs "apiKeyEnv"`);
    const modelName = typeof entry.modelName === "string" && entry.modelName.trim() ? entry.modelName.trim() : undefined;
    return { id, label, url, kind, apiKeyEnv, modelName };
  });
}

/**
 * The shipped list, then the optional local file: a local entry with the same
 * id replaces the shipped one, any other local entry is appended.
 */
export function loadEndpoints(): EndpointConfig[] {
  const list = parseFile(SHIPPED_FILE);
  if (existsSync(path.join(process.cwd(), LOCAL_FILE))) {
    for (const entry of parseFile(LOCAL_FILE)) {
      const at = list.findIndex((e) => e.id === entry.id);
      if (at >= 0) list[at] = entry;
      else list.push(entry);
    }
  }
  const ids = list.map((e) => e.id);
  const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
  if (dupes.length) throw new Error(`Duplicate endpoint id(s): ${[...new Set(dupes)].join(", ")}`);
  if (list.length === 0) throw new Error("No endpoints configured");
  return list;
}

/** The browser-safe view of the list. */
export function publicEndpoints(): Endpoint[] {
  return loadEndpoints().map(({ id, label, url, kind, modelName }) => ({ id, label, url, kind, ...(modelName ? { modelName } : {}) }));
}
