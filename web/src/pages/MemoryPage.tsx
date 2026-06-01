import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import {
  Brain,
  Database,
  HardDrive,
  Network,
  RefreshCw,
  Sparkles,
  BarChart3,
  GitBranch,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  MemoryBackendInfo,
  MemoryBackendName,
  MemoryBackendsResponse,
  MemoryComponentStatus,
  MemoryCountersResponse,
  MemoryGraphFullResponse,
  MemoryGraphStatsResponse,
  MemoryHealthStatus,
  MemoryStatusResponse,
} from "@/lib/api";
import ForceGraph2D from "react-force-graph-2d";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";
import { cn } from "@/lib/utils";

/**
 * MemoryPage — M1 (memory dashboard).
 *
 * Surfaces a rolled-up health view of the hermes-local memory provider
 * (SQLite + Qdrant + LMS embed + LMS LLM + disk) plus a small set of live
 * counters. M2 will add the dreamer panel; M3 the search playground and
 * activity feed; M4 the knowledge-graph view (a static placeholder lives
 * below the fold today so the page doesn't read as truncated).
 *
 * Wire format: see web/src/lib/api.ts — routes are at /api/dashboard/memory/…
 * (the SPEC and API.md document them under /api/memory/… for readability;
 * that's a documentation alias only).
 */

const POLL_INTERVAL_MS = 30_000;

// The five backends the memory health endpoint reports on, in display order.
const BACKEND_ORDER: Array<{
  key: MemoryBackendName;
  title: string;
  icon: typeof Database;
}> = [
  { key: "sqlite", title: "SQLite", icon: Database },
  { key: "qdrant", title: "Qdrant", icon: Network },
  { key: "embedding", title: "LMS embedding", icon: Sparkles },
  { key: "llm", title: "Dreamer LLM", icon: Brain },
  { key: "disk", title: "Disk", icon: HardDrive },
];

type LoadState = "idle" | "loading" | "ready" | "error";

function formatRelative(iso: string | null | undefined): string {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const deltaMs = Date.now() - then;
  if (deltaMs < 0) return "just now";
  const s = Math.floor(deltaMs / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function formatBytes(n: number | undefined): string {
  if (n === undefined || n === null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function statusTone(
  status: MemoryHealthStatus | MemoryComponentStatus,
): "success" | "warning" | "destructive" | "outline" {
  switch (status) {
    case "ok":
      return "success";
    case "degraded":
      return "warning";
    case "error":
      return "destructive";
    case "inactive":
    default:
      return "outline";
  }
}

function statusDot(
  status: MemoryHealthStatus | MemoryComponentStatus,
): string {
  switch (status) {
    case "ok":
      return "bg-emerald-400";
    case "degraded":
      return "bg-amber-400";
    case "error":
      return "bg-red-500";
    case "inactive":
    default:
      return "bg-midground/40";
  }
}

function formatDelta(n: number | undefined): string | null {
  if (n === undefined || n === null) return null;
  if (n === 0) return "±0";
  return n > 0 ? `↑${n}` : `↓${Math.abs(n)}`;
}

export default function MemoryPage() {
  const [status, setStatus] = useState<MemoryStatusResponse | null>(null);
  const [backends, setBackends] = useState<MemoryBackendsResponse | null>(null);
  const [counters, setCounters] = useState<MemoryCountersResponse | null>(null);
  const [graphStats, setGraphStats] = useState<MemoryGraphStatsResponse | null>(null);
  const [graphFull, setGraphFull] = useState<MemoryGraphFullResponse | null>(null);

  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [lastError, setLastError] = useState<string | null>(null);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<number | null>(null);
  const [refreshBusy, setRefreshBusy] = useState(false);
  const [pingBusy, setPingBusy] = useState<MemoryBackendName | null>(null);

  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  /** Pull /status, /backends, /counters in parallel. Each is allowed to fail
   *  independently — we degrade per-panel rather than blanking the page. */
  const refresh = useCallback(
    async (opts: { manual?: boolean } = {}) => {
      const isFirst = loadState === "idle";
      if (isFirst) setLoadState("loading");

      const [statusR, backendsR, countersR, graphR, graphFullR] = await Promise.allSettled([
        api.getMemoryStatus(),
        api.getMemoryBackends(),
        api.getMemoryCounters(),
        api.getMemoryGraphStats(),
        api.getMemoryGraphFull(),
      ]);

      if (statusR.status === "fulfilled") setStatus(statusR.value);
      if (backendsR.status === "fulfilled") setBackends(backendsR.value);
      if (countersR.status === "fulfilled") setCounters(countersR.value);
      if (graphR.status === "fulfilled") setGraphStats(graphR.value);
      if (graphFullR.status === "fulfilled") setGraphFull(graphFullR.value);

      const errs = [statusR, backendsR, countersR, graphR, graphFullR].filter(
        (r): r is PromiseRejectedResult => r.status === "rejected",
      );

      if (errs.length === 3) {
        const msg =
          errs[0].reason instanceof Error
            ? errs[0].reason.message
            : "Failed to load memory dashboard";
        setLastError(msg);
        setLoadState("error");
        if (opts.manual) showToast(msg, "error");
        return;
      }

      setLastError(null);
      setLastRefreshedAt(Date.now());
      setLoadState("ready");
      if (opts.manual) showToast("Refreshed", "success");
    },
    [loadState, showToast],
  );

  // Initial load + 30s poll.
  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      void refresh();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
    // refresh has stable deps after the first fetch; we don't re-create the
    // interval on every state change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onRefreshClick = useCallback(async () => {
    setRefreshBusy(true);
    try {
      await refresh({ manual: true });
    } finally {
      setRefreshBusy(false);
    }
  }, [refresh]);

  // Inject the [Refresh] button into the page header (top-right).
  useEffect(() => {
    setEnd(
      <div className="flex w-full min-w-0 justify-start sm:justify-end">
        <Button
          ghost
          size="sm"
          className="w-max max-w-full shrink-0 gap-2"
          disabled={refreshBusy}
          onClick={() => void onRefreshClick()}
        >
          {refreshBusy ? <Spinner /> : <RefreshCw className="h-3.5 w-3.5" />}
          Refresh
        </Button>
      </div>,
    );
    return () => setEnd(null);
  }, [refreshBusy, onRefreshClick, setEnd]);

  const onPing = useCallback(
    async (name: MemoryBackendName) => {
      setPingBusy(name);
      try {
        const result = await api.postMemoryBackendPing(name);
        // Patch the result into the current backends snapshot.
        setBackends((prev) =>
          prev ? { ...prev, [name]: result } : prev,
        );
        showToast(
          `${name}: ${result.status}${result.message ? ` — ${result.message}` : ""}`,
          result.status === "ok" ? "success" : "error",
        );
      } catch (e) {
        showToast(
          e instanceof Error ? e.message : `Ping ${name} failed`,
          "error",
        );
      } finally {
        setPingBusy(null);
      }
    },
    [showToast],
  );

  return (
    <div className="flex flex-col gap-4">
      <PluginSlot name="memory:top" />

      <div className={cn("flex w-full flex-col gap-8")}>
        {/* Tier 1 — sticky health banner */}
        <HealthBanner
          status={status}
          loadState={loadState}
          lastError={lastError}
          lastRefreshedAt={lastRefreshedAt}
        />

        {/* Tier 2 — counters strip */}
        <CountersStrip counters={counters} loadState={loadState} />

        {/* Tier 3 — backends grid */}
        <BackendsGrid
          backends={backends}
          loadState={loadState}
          onPing={onPing}
          pingBusy={pingBusy}
        />

        {/* Tier 5 — knowledge graph (M4) */}
        <KnowledgeGraph graphStats={graphStats} graphFull={graphFull} loadState={loadState} />
      </div>

      <Toast toast={toast} />
      <PluginSlot name="memory:bottom" />
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Tier 1 — Health banner
// ───────────────────────────────────────────────────────────────────────────

interface HealthBannerProps {
  status: MemoryStatusResponse | null;
  loadState: LoadState;
  lastError: string | null;
  lastRefreshedAt: number | null;
}

function HealthBanner({
  status,
  loadState,
  lastError,
  lastRefreshedAt,
}: HealthBannerProps) {
  // `sticky top-0` keeps the banner glued to the top of the scroll container
  // (the main panel scrolls, not the viewport — App.tsx wraps the routes in
  // an overflow-auto pane). z-10 keeps it above the rest of the cards.
  const overall = status?.overall ?? "inactive";

  const refreshedText = useMemo(() => {
    if (!lastRefreshedAt) return "—";
    return formatRelative(new Date(lastRefreshedAt).toISOString());
  }, [lastRefreshedAt]);

  if (loadState === "loading" && !status) {
    return (
      <Card className="sticky top-0 z-10">
        <CardContent className="flex items-center gap-3 py-3">
          <Spinner />
          <span className="text-[0.8rem] text-midforeground/65">
            Loading memory status…
          </span>
        </CardContent>
      </Card>
    );
  }

  if (loadState === "error" && !status) {
    return (
      <Card className="sticky top-0 z-10 border-red-500/40">
        <CardContent className="flex items-center gap-3 py-3">
          <span className={cn("h-2 w-2 rounded-full", statusDot("error"))} />
          <span className="text-[0.8rem]">
            Memory dashboard unavailable
            {lastError ? ` — ${lastError}` : ""}
          </span>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="sticky top-0 z-10">
      <CardContent className="flex flex-wrap items-center gap-4 py-3">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            className={cn(
              "h-2.5 w-2.5 rounded-full",
              statusDot(overall),
            )}
          />
          <Badge tone={statusTone(overall)}>{overall.toUpperCase()}</Badge>
        </div>

        <div className="flex items-center gap-2 text-[0.75rem] tracking-[0.06em]">
          <span className="font-semibold">{status?.provider ?? "—"}</span>
          <span className="text-midforeground/50">·</span>
          <Badge tone={status?.active ? "success" : "outline"}>
            {status?.active ? "active" : "inactive"}
          </Badge>
        </div>

        <div className="ml-auto flex items-center gap-3 text-[0.7rem] text-midforeground/55 normal-case">
          <span>refreshed {refreshedText}</span>
        </div>
      </CardContent>
    </Card>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Tier 2 — Counters strip
// ───────────────────────────────────────────────────────────────────────────

interface CountersStripProps {
  counters: MemoryCountersResponse | null;
  loadState: LoadState;
}

interface CounterTile {
  label: string;
  value: string | number;
  delta?: string | null;
  alert?: boolean;
}

function CountersStrip({ counters, loadState }: CountersStripProps) {
  if (loadState === "loading" && !counters) {
    return (
      <Card>
        <CardContent className="flex items-center gap-2 py-6">
          <Spinner />
          <span className="text-[0.75rem] text-midforeground/65">
            Loading counters…
          </span>
        </CardContent>
      </Card>
    );
  }

  if (!counters) {
    return (
      <Card>
        <CardContent className="py-6">
          <p className="text-[0.75rem] text-midforeground/55 normal-case">
            Counters unavailable — metrics.json has not been written yet.
          </p>
        </CardContent>
      </Card>
    );
  }

  const tiles: CounterTile[] = [
    {
      label: "facts (active)",
      value: counters.facts_active,
      delta: formatDelta(counters.deltas_24h?.facts_active),
    },
    { label: "turns 24h", value: counters.captured_turns_24h },
    { label: "chunks 24h", value: counters.chunks_indexed_24h },
    {
      label: "chunks pending",
      value: counters.chunks_pending,
      alert: counters.chunks_pending > 100,
    },
    {
      label: "qdrant points",
      value: counters.qdrant_points,
      delta: formatDelta(counters.deltas_24h?.qdrant_points),
    },
    {
      label: "last dream",
      value: formatRelative(counters.last_dream_run_at),
      delta:
        counters.last_dream_status === "completed"
          ? "✓ ok"
          : counters.last_dream_status ?? null,
    },
  ];

  return (
    <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
      {tiles.map((tile) => (
        <Card key={tile.label}>
          <CardContent className="flex flex-col gap-1 px-4 py-3">
            <span
              className={cn(
                "font-mondwest text-[1.1rem] leading-none tracking-[0.02em]",
                tile.alert ? "text-amber-300" : undefined,
              )}
            >
              {tile.value}
            </span>
            <span className="text-[0.6rem] tracking-[0.12em] text-midforeground/55 normal-case">
              {tile.label}
            </span>
            {tile.delta ? (
              <span className="text-[0.6rem] tracking-[0.08em] text-midforeground/65 normal-case">
                {tile.delta}
              </span>
            ) : null}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Tier 3 — Backends grid
// ───────────────────────────────────────────────────────────────────────────

interface BackendsGridProps {
  backends: MemoryBackendsResponse | null;
  loadState: LoadState;
  onPing: (name: MemoryBackendName) => Promise<void>;
  pingBusy: MemoryBackendName | null;
}

function BackendsGrid({
  backends,
  loadState,
  onPing,
  pingBusy,
}: BackendsGridProps) {
  return (
    <div className="flex flex-col gap-3">
      <h3 className="font-mondwest text-[0.75rem] tracking-[0.12em] text-midground/85">
        Backends
      </h3>

      {loadState === "loading" && !backends ? (
        <div className="flex items-center gap-2 py-8 text-[0.8rem] text-midforeground/65">
          <Spinner />
          <span>Probing backends…</span>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {BACKEND_ORDER.map(({ key, title, icon: Icon }) => (
            <BackendCard
              key={key}
              name={key}
              title={title}
              icon={Icon}
              info={backends?.[key] ?? null}
              busy={pingBusy === key}
              disabled={pingBusy !== null && pingBusy !== key}
              onPing={() => void onPing(key)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface BackendCardProps {
  name: MemoryBackendName;
  title: string;
  icon: typeof Database;
  info: MemoryBackendInfo | null;
  busy: boolean;
  disabled: boolean;
  onPing: () => void;
}

function BackendCard({
  title,
  icon: Icon,
  info,
  busy,
  disabled,
  onPing,
}: BackendCardProps) {
  const status: MemoryComponentStatus = info?.status ?? "error";
  const lines: Array<[string, string]> = [];

  if (info) {
    if (info.endpoint) lines.push(["endpoint", info.endpoint]);
    if (info.path) lines.push(["path", info.path]);
    if (info.model) lines.push(["model", info.model]);
    if (info.dim !== undefined) lines.push(["dim", String(info.dim)]);
    if (info.journal_mode)
      lines.push(["journal", info.journal_mode]);
    if (info.size_bytes !== undefined)
      lines.push(["size", formatBytes(info.size_bytes)]);
    if (info.collections_count !== undefined)
      lines.push(["collections", String(info.collections_count)]);
    if (info.points_count !== undefined)
      lines.push(["points", String(info.points_count)]);
    if (info.free_gb !== undefined)
      lines.push(["free", `${info.free_gb.toFixed(1)} GB`]);
    if (info.last_latency_ms !== undefined)
      lines.push(["latency", `${info.last_latency_ms} ms`]);
    if (info.last_success_at)
      lines.push(["last ok", formatRelative(info.last_success_at)]);
  }

  return (
    <Card className={cn(busy ? "opacity-70" : undefined)}>
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-2">
        <CardTitle className="flex items-center gap-2 text-[0.8rem]">
          <Icon className="h-3.5 w-3.5 shrink-0" />
          <span>{title}</span>
        </CardTitle>
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            className={cn("h-2 w-2 rounded-full", statusDot(status))}
          />
          <Badge tone={statusTone(status)}>{status}</Badge>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3 px-6 pb-4">
        {info?.message ? (
          <p className="text-[0.7rem] text-amber-300/85 normal-case">
            {info.message}
          </p>
        ) : null}

        {lines.length > 0 ? (
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[0.65rem] tracking-[0.06em] text-midforeground/70 normal-case">
            {lines.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-midforeground/45">{k}</dt>
                <dd className="truncate text-midforeground/85" title={v}>
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="text-[0.65rem] text-midforeground/45 normal-case">
            No backend detail returned.
          </p>
        )}

        <div className="flex">
          <Button
            ghost
            size="sm"
            className="gap-2"
            disabled={disabled || busy}
            onClick={onPing}
          >
            {busy ? <Spinner /> : <RefreshCw className="h-3 w-3" />}
            Ping
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Tier 5 — Knowledge graph (M4)
// ───────────────────────────────────────────────────────────────────────────

type GraphView = "stats" | "graph";

function KnowledgeGraph({
  graphStats,
  graphFull,
  loadState,
}: {
  graphStats: MemoryGraphStatsResponse | null;
  graphFull: MemoryGraphFullResponse | null;
  loadState: LoadState;
}) {
  const [view, setView] = useState<GraphView>("stats");
  const graphRef = useRef<any>(null);
  const isLoading = loadState === "loading" && !graphStats && !graphFull;

  // Center the graph when switching to graph view
  const showGraph = useCallback(() => {
    setView("graph");
    // Give the DOM a tick to render, then center
    setTimeout(() => {
      graphRef.current?.d3Force("charge")?.strength(-120);
      graphRef.current?.zoomToFit(400, 50);
    }, 80);
  }, []);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Network className="h-4 w-4" />
            Knowledge graph
          </CardTitle>
          {/* View toggle */}
          <div className="flex items-center gap-1 rounded-md border border-border/50 p-0.5">
            <button
              onClick={() => setView("stats")}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-[0.7rem] transition-colors ${
                view === "stats"
                  ? "bg-primary text-primary-foreground"
                  : "text-midforeground/55 hover:text-midforeground"
              }`}
            >
              <BarChart3 className="h-3 w-3" />
              Stats
            </button>
            <button
              onClick={showGraph}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-[0.7rem] transition-colors ${
                view === "graph"
                  ? "bg-primary text-primary-foreground"
                  : "text-midforeground/55 hover:text-midforeground"
              }`}
            >
              <GitBranch className="h-3 w-3" />
              Graph
              {graphFull && (
                <span className="text-[0.6rem] opacity-70">
                  ({graphFull.nodes.length})
                </span>
              )}
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {isLoading && (
          <div className="flex h-48 items-center justify-center">
            <Spinner />
          </div>
        )}

        {view === "stats" && !isLoading && graphStats && (
          <>
            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatPill label="Facts" value={graphStats.facts.total} icon={Brain} />
              <StatPill label="Active" value={graphStats.facts.active} icon={Sparkles} />
              <StatPill label="Relations" value={graphStats.entity_relations.total} icon={Network} />
              <StatPill label="Links" value={graphStats.fact_links.total} icon={Database} />
            </div>

            {/* Top entities */}
            {graphStats.top_entities.length > 0 && (
              <div>
                <p className="mb-2 text-[0.7rem] tracking-[0.08em] text-midground/55 uppercase">
                  Top entities
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {graphStats.top_entities.slice(0, 12).map(({ entity, fact_count }) => (
                    <Badge key={entity} tone="outline" className="text-[0.7rem]">
                      {entity} <span className="text-midforeground/45">({fact_count})</span>
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Fact→entity mapping count */}
            <div className="flex items-center justify-between border-t border-border/50 pt-3 text-[0.7rem] text-midforeground/55">
              <span>Entity-fact mappings</span>
              <span>{graphStats.fact_entities.total.toLocaleString()}</span>
            </div>
          </>
        )}

        {view === "graph" && (
          <div className="flex flex-col gap-2">
            {isLoading ? (
              <div className="flex h-64 items-center justify-center">
                <Spinner />
              </div>
            ) : graphFull && graphFull.nodes.length > 0 ? (
              <>
                <div className="flex items-center gap-3 text-[0.65rem] text-midforeground/55">
                  <span>{graphFull.nodes.length} nodes · {graphFull.links.length} edges</span>
                  <span className="text-midforeground/30">·</span>
                  <span>drag to move · scroll to zoom · hover for label</span>
                </div>
                <div
                  className="overflow-hidden rounded-md border border-border/40"
                  style={{ height: 480 }}
                >
                  <ForceGraph2D
                    ref={graphRef}
                    graphData={graphFull}
                    nodeId="id"
                    nodeVal="val"
                    nodeColor={(n: any) => n.color ?? "#64748b"}
                    nodeCanvasObjectMode={() => 'replace'}
                    nodeCanvasObject={(node: any, ctx: any, globalScale: number) => {
                      const r = Math.max(4, Math.sqrt(node.val || 1) * 1.5);
                      const fontSize = Math.max(7, 9 / globalScale);
                      // Draw node circle
                      ctx.beginPath();
                      ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
                      ctx.fillStyle = node.color ?? "#64748b";
                      ctx.fill();
                      ctx.strokeStyle = "rgba(255,255,255,0.15)";
                      ctx.lineWidth = 0.5;
                      ctx.stroke();
                      // Draw label below node
                      const label = String(node.name ?? node.id ?? "");
                      const maxW = Math.max(r * 5, 50);
                      const words = label.split(/\s+/);
                      let line = "", lines: string[] = [];
                      for (const w of words) {
                        const test = line ? `${line} ${w}` : w;
                        if (ctx.measureText(test).width > maxW && line) { lines.push(line); line = w; }
                        else { line = test; }
                      }
                      if (line) lines.push(line);
                      const lh = fontSize * 1.25;
                      const startY = (node.y ?? 0) + r + 2;
                      ctx.font = `500 ${fontSize}px Sans-Serif`;
                      ctx.textAlign = "center";
                      ctx.textBaseline = "top";
                      ctx.fillStyle = "rgba(203,213,225,0.9)";
                      lines.forEach((l, i) => ctx.fillText(l, node.x ?? 0, startY + i * lh));
                    }}
                    linkColor={() => "rgba(148,163,184,0.25)"}
                    linkWidth={(l: any) => Math.max(0.5, (l.weight ?? 0.8) * 2)}
                    linkDirectionalArrowLength={4}
                    linkDirectionalArrowRelPos={0.9}
                    backgroundColor="transparent"
                    warmupTicks={60}
                    cooldownTicks={120}
                    onNodeClick={(node: any) => {
                      // Center on click
                      graphRef.current?.centerAt(node.x, node.y, 500);
                      graphRef.current?.zoom(3, 500);
                    }}
                  />
                </div>
              </>
            ) : (
              <div className="flex h-32 items-center justify-center text-[0.75rem] text-midforeground/55">
                No graph data available — run Dreamer to populate entity relations.
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatPill({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: number;
  icon: typeof Brain;
}) {
  return (
    <div className="flex items-center gap-2 rounded-md border border-border/50 bg-background/50 px-3 py-2">
      <Icon className="h-3.5 w-3.5 text-midforeground/55" />
      <div>
        <p className="text-[0.65rem] text-midforeground/55">{label}</p>
        <p className="font-mono text-sm font-semibold">{value.toLocaleString()}</p>
      </div>
    </div>
  );
}
