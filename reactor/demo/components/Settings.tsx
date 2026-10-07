"use client";

import type { Endpoint } from "@/lib/endpoints";

/**
 * Endpoint switcher, laid out inline for the session bar. The Local
 * endpoint's URL is editable; the tooltip says how each kind connects.
 */
export function Settings({
  endpoints,
  endpoint,
  onEndpointChange,
  localUrl,
  onLocalUrlChange,
}: {
  endpoints: Endpoint[];
  endpoint: Endpoint;
  onEndpointChange: (endpoint: Endpoint) => void;
  localUrl: string;
  onLocalUrlChange: (url: string) => void;
}) {
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2" title={
        endpoint.kind === "local"
          ? `${endpoint.url} · the model's own runtime; no API key`
          : `${endpoint.url} · hosted; the server mints a token with its API key`
      }>
      <label
        htmlFor="endpoint-select"
        className="hidden shrink-0 font-mono text-xs text-white/40 sm:inline"
      >
        Endpoint
      </label>
      <select
        id="endpoint-select"
        value={endpoint.id}
        onChange={(e) => onEndpointChange(endpoints.find((ep) => ep.id === e.target.value) ?? endpoints[0])}
        className="h-7 cursor-pointer rounded-md border border-white/15 bg-white/10 px-2 font-mono text-xs text-white hover:bg-white/15 focus:border-white/40 focus:outline-none"
      >
        {endpoints.map((ep) => (
          <option key={ep.id} value={ep.id} className="bg-black">
            {ep.label}
          </option>
        ))}
      </select>

      {endpoint.kind === "local" && (
        <input
          type="text"
          aria-label="Local URL"
          value={localUrl}
          onChange={(e) => onLocalUrlChange(e.target.value)}
          spellCheck={false}
          className="h-7 w-44 min-w-0 rounded-md border border-white/10 bg-black/30 px-2 font-mono text-xs text-white/80 placeholder-white/30 focus:border-white/40 focus:outline-none sm:w-56"
          placeholder="http://localhost:8080"
        />
      )}
    </div>
  );
}
