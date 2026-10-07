"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ReactorProvider, useReactor } from "@reactor-team/js-sdk";
import { AppShell, StatusDot, Workspace } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Settings } from "@/components/Settings";
import { MODEL_NAME, type Endpoint } from "@/lib/endpoints";

const ENDPOINT_STORAGE = "worldplay2:endpoint";
// Per-endpoint URL edits for local endpoints, keyed by endpoint id.
const URL_STORAGE_PREFIX = "worldplay2:url:";
// The model holds its GPUs for the whole session, so an idle tab disconnects.
const AUTO_DISCONNECT_MS = 10 * 60 * 1000;
const TOKEN_TTL_MS = 5 * 60 * 1000;

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage unavailable: the choice just isn't remembered.
  }
}

/**
 * The one session row under the nav: connection status and Connect on the
 * left, the endpoint picker on the right.
 */
function SessionBar({
  endpoints,
  endpoint,
  onEndpointChange,
  localUrl,
  onLocalUrlChange,
  tokenError,
}: {
  endpoints: Endpoint[];
  endpoint: Endpoint;
  onEndpointChange: (endpoint: Endpoint) => void;
  localUrl: string;
  onLocalUrlChange: (url: string) => void;
  tokenError: string | null;
}) {
  const { status, connect, disconnect } = useReactor((state) => ({
    status: state.status,
    connect: state.connect,
    disconnect: state.disconnect,
  }));
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (status === "ready") {
      timerRef.current = setTimeout(() => disconnect(), AUTO_DISCONNECT_MS);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [status, disconnect]);

  const busy = status === "connecting" || status === "waiting";

  return (
    <div
      className="flex shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-2"
      style={{ background: "rgba(255,255,255,0.04)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}
    >
      <div className="flex min-w-0 items-center gap-3">
        <StatusDot status={status} />
        <Button
          size="xs"
          variant="secondary"
          onClick={() => (status === "disconnected" ? connect() : disconnect())}
          className="h-7 border-white/15 bg-white/10 px-3 font-mono text-xs text-white hover:bg-white/15"
        >
          {status === "disconnected" ? "Connect" : busy ? "Cancel" : "Disconnect"}
        </Button>
        {tokenError && endpoint.kind === "hosted" && (
          <span className="truncate font-mono text-[10px] text-red-300" title={tokenError}>
            {tokenError}
          </span>
        )}
      </div>
      <Settings
        endpoints={endpoints}
        endpoint={endpoint}
        onEndpointChange={onEndpointChange}
        localUrl={localUrl}
        onLocalUrlChange={onLocalUrlChange}
      />
    </div>
  );
}

export default function PageClient({ endpoints }: { endpoints: Endpoint[] }) {
  const [endpoint, setEndpoint] = useState<Endpoint>(endpoints[0]);
  const [urlEdits, setUrlEdits] = useState<Record<string, string>>({});
  const [tokenError, setTokenError] = useState<string | null>(null);
  const tokenCache = useRef<Record<string, { jwt: string; at: number }>>({});

  useEffect(() => {
    const storedEndpoint = endpoints.find((ep) => ep.id === readStorage(ENDPOINT_STORAGE));
    if (storedEndpoint) setEndpoint(storedEndpoint);
    const edits: Record<string, string> = {};
    for (const ep of endpoints) {
      const stored = ep.kind === "local" ? readStorage(URL_STORAGE_PREFIX + ep.id) : null;
      if (stored) edits[ep.id] = stored;
    }
    setUrlEdits(edits);
  }, [endpoints]);

  const changeEndpoint = (next: Endpoint) => {
    setEndpoint(next);
    setTokenError(null);
    writeStorage(ENDPOINT_STORAGE, next.id);
  };

  const localUrl = urlEdits[endpoint.id] ?? endpoint.url;
  const changeLocalUrl = (url: string) => {
    setUrlEdits((edits) => ({ ...edits, [endpoint.id]: url }));
    writeStorage(URL_STORAGE_PREFIX + endpoint.id, url);
  };

  // Session token for a hosted endpoint, minted by /api/token (which looks the
  // endpoint up by id in the server's endpoint files) and reused for a few minutes.
  const endpointId = endpoint.id;
  const resolveJwt = useCallback(async (): Promise<string> => {
    const cached = tokenCache.current[endpointId];
    if (cached && Date.now() - cached.at < TOKEN_TTL_MS) return cached.jwt;
    const response = await fetch("/api/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpointId }),
    });
    const body = (await response.json().catch(() => ({}))) as { jwt?: string; error?: string };
    if (!response.ok || !body.jwt) {
      const message = body.error ?? `Token request failed: ${response.status}`;
      setTokenError(message);
      throw new Error(message);
    }
    setTokenError(null);
    tokenCache.current[endpointId] = { jwt: body.jwt, at: Date.now() };
    return body.jwt;
  }, [endpointId]);

  const isLocal = endpoint.kind === "local";
  const apiUrl = isLocal ? localUrl : endpoint.url;

  return (
    <AppShell>
      <ReactorProvider
        // Remount on an endpoint change so the next Connect opens a fresh session there.
        // The Local URL is deliberately not part of the key: the provider already
        // builds a new Reactor when `apiUrl` changes, and remounting the subtree on
        // every keystroke would take focus away from the URL box mid-typing.
        key={endpoint.id}
        modelName={endpoint.modelName ?? MODEL_NAME}
        apiUrl={apiUrl}
        local={isLocal}
        jwtToken={isLocal ? undefined : resolveJwt}
        connectOptions={{ autoConnect: false }}
      >
        <div className="relative z-10 shrink-0">
          <SessionBar
            endpoints={endpoints}
            endpoint={endpoint}
            onEndpointChange={changeEndpoint}
            localUrl={localUrl}
            onLocalUrlChange={changeLocalUrl}
            tokenError={tokenError}
          />
        </div>

        <Workspace />
      </ReactorProvider>
    </AppShell>
  );
}
