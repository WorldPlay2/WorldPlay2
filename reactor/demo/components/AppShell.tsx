"use client";

import Link from "next/link";
import { Logo, Nav } from "@reactor-team/ui";
import { WorldPlay2Controller } from "@/components/WorldPlay2Controller";

/** Background, nav pill and the column the session bar and workspace sit in. */
export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <style>{`
        .dot-grid {
          background-image: radial-gradient(circle, rgba(255,255,255,0.07) 1px, transparent 1px);
          background-size: 28px 28px;
        }
        @keyframes statusPulse {
          0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
        }
      `}</style>

      <div className="relative flex h-screen flex-col overflow-hidden" style={{ backgroundColor: "#000" }}>
        <div className="dot-grid pointer-events-none absolute inset-0" />
        <div
          className="pointer-events-none absolute inset-0"
          style={{ background: "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(199,192,153,0.12), transparent)" }}
        />

        <div className="relative z-10 shrink-0 px-4 py-3 sm:px-6">
          <Nav
            className="demos-nav"
            logo={
              <Link href="/" className="flex items-center gap-2 text-white/50 transition-colors hover:text-white/80">
                <span className="font-mono text-xs uppercase tracking-widest">WorldPlay2</span>
              </Link>
            }
            action={<Logo variant="symbol" color="white" height={18} />}
          />
        </div>

        {children}
      </div>
    </>
  );
}

/** Connection status dot + label, shared by both session bars. */
export function StatusDot({ status }: { status: string }) {
  const busy = status === "connecting" || status === "waiting";
  const dotColor = status === "ready" ? "#4ade80" : busy ? "#facc15" : "rgba(255,255,255,0.3)";
  const statusLabel =
    status === "ready"
      ? "Connected"
      : status === "waiting"
        ? "Waiting for GPUs..."
        : status === "connecting"
          ? "Connecting..."
          : "Disconnected";
  return (
    <div className="flex items-center gap-2">
      <div
        className="h-2 w-2 rounded-full transition-colors"
        style={{ backgroundColor: dotColor, animation: busy ? "statusPulse 1.5s infinite" : "none" }}
      />
      <span className="font-mono text-xs text-white/50">{statusLabel}</span>
    </div>
  );
}

/** Sidebar, display and controls card; must sit inside a ReactorProvider. */
export function Workspace() {
  const { sidebar, stage, controls } = WorldPlay2Controller();

  return (
    <main className="relative z-10 flex min-h-0 flex-1 flex-col px-4 pb-4 pt-3 sm:px-6 sm:pb-6 max-lg:overflow-y-auto lg:overflow-hidden">
      <div className="mx-auto flex min-h-0 w-full min-w-0 max-w-7xl flex-1 flex-col gap-4">
        <div className="flex min-w-0 flex-col gap-4 max-lg:flex-none lg:min-h-0 lg:flex-1 lg:flex-row lg:gap-6">
          {/* Sidebar: left on desktop, below on mobile */}
          <div className="order-2 flex min-h-0 min-w-0 flex-col gap-4 lg:order-1 lg:w-[min(100%,380px)] lg:max-w-[380px] lg:shrink-0 lg:overflow-y-auto lg:pr-1">
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3 sm:p-4">{sidebar}</div>
          </div>

          {/* Video and controls */}
          <div className="order-1 flex min-w-0 flex-col gap-4 lg:order-2 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
            {stage}
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3 sm:p-4">{controls}</div>
          </div>
        </div>
      </div>
    </main>
  );
}
