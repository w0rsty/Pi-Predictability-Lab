import { useEffect, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";
import type { ThemeMode } from "../../types/lab";
import { buildPlotlyLayout, PLOTLY_CONFIG } from "../../utils/plotlyTheme";

interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  theme: ThemeMode;
  className?: string;
  height?: number;
}

type PlotlyModule = typeof import("plotly.js-cartesian-dist-min");

let plotlyPromise: Promise<PlotlyModule> | null = null;
function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-cartesian-dist-min");
  }
  return plotlyPromise;
}

export function PlotlyChart({
  data,
  layout = {},
  config = {},
  theme,
  className = "",
  height = 360,
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let plotlyRef: PlotlyModule | null = null;
    const node = containerRef.current;
    if (!node) return;

    void loadPlotly().then((mod) => {
      if (cancelled || !containerRef.current) return;
      plotlyRef = mod;
      const Plotly = (mod.default ?? mod) as unknown as typeof import("plotly.js");
      const fullLayout = buildPlotlyLayout(theme, { height, ...layout });
      const fullConfig = { ...PLOTLY_CONFIG, ...config };
      void Plotly.react(containerRef.current, data, fullLayout, fullConfig).then(() => {
        if (!cancelled) setReady(true);
      });
    });

    return () => {
      cancelled = true;
      if (plotlyRef && node) {
        const Plotly = (plotlyRef.default ?? plotlyRef) as unknown as typeof import("plotly.js");
        try {
          Plotly.purge(node);
        } catch {
          // игнорируем при размонтировании
        }
      }
    };
  }, [data, layout, config, theme, height]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (!plotlyPromise) return;
      void plotlyPromise.then((mod) => {
        if (!containerRef.current) return;
        const Plotly = (mod.default ?? mod) as unknown as typeof import("plotly.js");
        try {
          Plotly.Plots.resize(containerRef.current);
        } catch {
          // игнорируем
        }
      });
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  return (
    <div className={`relative w-full ${className}`} style={{ minHeight: height }}>
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center rounded-[4px] border border-[var(--rule)] bg-[var(--bg-alt)]/50 text-xs font-mono text-[var(--ink-muted)]">
          Загрузка графика…
        </div>
      )}
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
