import { NextResponse } from "next/server";
import { loadEndpoints } from "@/lib/endpoints-config";

// Mints a session token for a hosted endpoint. The URL and the API key's env var
// both come from the endpoint files on the server; the browser sends only an id,
// and the key never reaches it.
export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as { endpointId?: unknown };
  let endpoints;
  try {
    endpoints = loadEndpoints();
  } catch (err) {
    return NextResponse.json({ error: err instanceof Error ? err.message : String(err) }, { status: 500 });
  }
  const endpoint = endpoints.find((e) => e.id === body.endpointId && e.kind === "hosted");
  if (!endpoint?.apiKeyEnv) {
    return NextResponse.json({ error: "Unknown hosted endpoint" }, { status: 400 });
  }

  const apiKey = process.env[endpoint.apiKeyEnv];
  if (!apiKey) {
    return NextResponse.json(
      { error: `${endpoint.apiKeyEnv} is not set in .env` },
      { status: 500 },
    );
  }

  const response = await fetch(`${endpoint.url}/tokens`, {
    method: "POST",
    headers: { "Reactor-API-Key": apiKey },
  });

  if (!response.ok) {
    const text = await response.text();
    return NextResponse.json(
      { error: `Token request failed: ${response.status} ${text}` },
      { status: response.status },
    );
  }

  const { jwt } = await response.json();
  return NextResponse.json({ jwt });
}
