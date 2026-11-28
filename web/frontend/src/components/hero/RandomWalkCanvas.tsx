import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import type { LabData, ThemeMode } from "../../types/lab";
import { MathTex } from "../ui/MathTex";

export type WalkPresetKey =
  | "true_pi"
  | "rational_1_97"
  | "champernowne"
  | "e_cf"
  | "csprng_backdoor";

interface RandomWalkCanvasProps {
  samples?: LabData["samples"];
  customSequence?: {
    label: string;
    digits: number[];
    mask?: number[] | null;
    radix: number;
  };
  theme: ThemeMode;
  height?: number;
  /** Правая встроенная приборная панель (для широкоформатной сцены на Главной). */
  sidePanel?: (activeMeta: {
    preset: WalkPresetKey;
    label: string;
    hint: string;
    presetButtons: ReactNode;
  }) => ReactNode;
}

const PRESETS: Array<{ key: WalkPresetKey; label: string; hint: string }> = [
  { key: "true_pi", label: "π (H₀)", hint: "Броуновское блуждание" },
  { key: "rational_1_97", label: "1/97 (P=96)", hint: "Замкнутый симметричный цикл" },
  { key: "champernowne", label: "Champernowne", hint: "Направленные дуги счётчика" },
  { key: "e_cf", label: "e_cf", hint: "Регулярная лестница [1; 2k, 1]" },
  { key: "csprng_backdoor", label: "Backdoor", hint: "Импульсы триггеров (t-24, t-7)" },
];

export function RandomWalkCanvas({
  samples,
  customSequence,
  theme,
  height = 360,
  sidePanel,
}: RandomWalkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [preset, setPreset] = useState<WalkPresetKey>("true_pi");
  const [speed, setSpeed] = useState<number>(18);
  const [isPlaying, setIsPlaying] = useState<boolean>(() => {
    if (typeof window !== "undefined") {
      return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }
    return true;
  });
  const [cursor, setCursor] = useState<number>(220);

  const activeSeq = useMemo(() => {
    if (customSequence) return customSequence;
    if (!samples) {
      return { label: "π", digits: [1, 4, 1, 5, 9, 2, 6, 5, 3, 5], mask: null, radix: 10 };
    }
    const s = samples[preset];
    return {
      label: PRESETS.find((p) => p.key === preset)?.label ?? preset,
      digits: s.digits,
      mask: "mask" in s ? s.mask : null,
      radix: s.radix,
    };
  }, [customSequence, preset, samples]);

  const points = useMemo(() => {
    const digits = activeSeq.digits;
    const n = Math.min(digits.length, 2200);
    const pts = new Float32Array((n + 1) * 2);
    let x = 0;
    let y = 0;
    pts[0] = 0;
    pts[1] = 0;
    const stepAngle = (2 * Math.PI) / Math.max(activeSeq.radix, 2);
    for (let i = 0; i < n; i++) {
      const ang = digits[i] * stepAngle - Math.PI / 2;
      x += Math.cos(ang);
      y += Math.sin(ang);
      pts[(i + 1) * 2] = x;
      pts[(i + 1) * 2 + 1] = y;
    }
    return { pts, count: n };
  }, [activeSeq]);

  useEffect(() => {
    setCursor(Math.min(180, points.count));
  }, [points.count, preset, customSequence]);

  const cursorRef = useRef<number>(cursor);
  cursorRef.current = cursor;
  const camRef = useRef({ minX: -6, maxX: 6, minY: -6, maxY: 6 });

  useEffect(() => {
    camRef.current = { minX: -5, maxX: 5, minY: -5, maxY: 5 };
  }, [preset, customSequence]);

  useEffect(() => {
    let rafId = 0;
    let lastTime = performance.now();

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      if (isPlaying) {
        const next = cursorRef.current + speed * dt * 3.5;
        cursorRef.current = next >= points.count ? 12 : next;
        setCursor(Math.floor(cursorRef.current));
      }

      const canvas = canvasRef.current;
      const box = containerRef.current;
      if (canvas && box) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = box.clientWidth;
        const h = height;
        if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
          canvas.width = Math.floor(w * dpr);
          canvas.height = Math.floor(h * dpr);
        }
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, w, h);

          const drawN = Math.max(2, Math.min(points.count, Math.floor(cursorRef.current)));
          const pts = points.pts;
          let minX = pts[0];
          let maxX = pts[0];
          let minY = pts[1];
          let maxY = pts[1];
          for (let i = 1; i <= drawN; i++) {
            const px = pts[i * 2];
            const py = pts[i * 2 + 1];
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
          }
          const pad = Math.max(2.5, Math.max(maxX - minX, maxY - minY) * 0.14);
          const cam = camRef.current;
          const lerp = isPlaying ? 0.12 : 1;
          cam.minX += (minX - pad - cam.minX) * lerp;
          cam.maxX += (maxX + pad - cam.maxX) * lerp;
          cam.minY += (minY - pad - cam.minY) * lerp;
          cam.maxY += (maxY + pad - cam.maxY) * lerp;

          const spanX = Math.max(4, cam.maxX - cam.minX);
          const spanY = Math.max(4, cam.maxY - cam.minY);
          const scale = Math.min((w - 32) / spanX, (h - 32) / spanY);
          const cx = w / 2 - ((cam.minX + cam.maxX) / 2) * scale;
          const cy = h / 2 - ((cam.minY + cam.maxY) / 2) * scale;

          // Прибор всегда в тёмном утопленном экране (#121110 / #080707)
          ctx.strokeStyle = "rgba(239, 233, 222, 0.08)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(cx, 0);
          ctx.lineTo(cx, h);
          ctx.moveTo(0, cy);
          ctx.lineTo(w, cy);
          ctx.stroke();

          ctx.strokeStyle = "rgba(239, 233, 222, 0.16)";
          ctx.beginPath();
          for (let tx = 24; tx < w; tx += 24) {
            const tickLen = tx % 96 === 0 ? 5 : 2.5;
            ctx.moveTo(tx, 0);
            ctx.lineTo(tx, tickLen);
            ctx.moveTo(tx, h);
            ctx.lineTo(tx, h - tickLen);
          }
          for (let ty = 24; ty < h; ty += 24) {
            const tickLen = ty % 96 === 0 ? 5 : 2.5;
            ctx.moveTo(0, ty);
            ctx.lineTo(tickLen, ty);
            ctx.moveTo(w, ty);
            ctx.lineTo(w - tickLen, ty);
          }
          ctx.stroke();

          const isH1 =
            !customSequence &&
            (preset === "rational_1_97" || preset === "champernowne" || preset === "e_cf");
          const isCrypto = !customSequence && preset === "csprng_backdoor";
          // Научно-картографическая палитра внутри тёмного экрана плоттера:
          // H0 -> прусский сине-зелёный (94,158,168), H1 -> терракота (223,98,68), Crypto -> охра (212,161,72)
          const strokeRgb = isH1
            ? "223, 98, 68"
            : isCrypto
              ? "212, 161, 72"
              : "126, 178, 187";
          const signalColor = isCrypto ? "#D4A148" : "#DF6244";

          const glowWindow = 40;
          const oldEnd = Math.max(0, drawN - glowWindow);
          ctx.lineCap = "round";
          ctx.lineJoin = "round";

          if (oldEnd > 1) {
            const oldBuckets = 6;
            const step = Math.max(1, Math.ceil(oldEnd / oldBuckets));
            for (let b = 0; b < oldBuckets; b++) {
              const sIdx = b * step;
              const eIdx = Math.min(oldEnd, (b + 1) * step);
              if (sIdx >= eIdx) break;
              const alpha = 0.16 + (b / oldBuckets) * 0.30;
              ctx.strokeStyle = `rgba(${strokeRgb}, ${alpha.toFixed(2)})`;
              ctx.lineWidth = 1.05;
              ctx.beginPath();
              ctx.moveTo(cx + pts[sIdx * 2] * scale, cy + pts[sIdx * 2 + 1] * scale);
              for (let i = sIdx + 1; i <= eIdx; i++) {
                ctx.lineTo(cx + pts[i * 2] * scale, cy + pts[i * 2 + 1] * scale);
              }
              ctx.stroke();
            }
          }

          const freshStart = Math.max(0, drawN - glowWindow);
          for (let i = freshStart; i < drawN; i++) {
            const t = (i - freshStart) / Math.max(1, drawN - freshStart);
            const alpha = 0.48 + t * 0.48;
            ctx.strokeStyle = `rgba(${strokeRgb}, ${alpha.toFixed(2)})`;
            ctx.lineWidth = 1.3 + t * 0.7;
            ctx.beginPath();
            ctx.moveTo(cx + pts[i * 2] * scale, cy + pts[i * 2 + 1] * scale);
            ctx.lineTo(cx + pts[(i + 1) * 2] * scale, cy + pts[(i + 1) * 2 + 1] * scale);
            ctx.stroke();
          }

          if (activeSeq.mask) {
            for (let i = 0; i < drawN; i++) {
              if (activeSeq.mask[i] === 1) {
                const tx = cx + pts[(i + 1) * 2] * scale;
                const ty = cy + pts[(i + 1) * 2 + 1] * scale;
                ctx.fillStyle = "#DF6244";
                ctx.beginPath();
                ctx.arc(tx, ty, 3.1, 0, Math.PI * 2);
                ctx.fill();
              }
            }
          }

          ctx.fillStyle = "rgba(239, 233, 222, 0.45)";
          ctx.beginPath();
          ctx.arc(cx, cy, 2.2, 0, Math.PI * 2);
          ctx.fill();

          const hx = cx + pts[drawN * 2] * scale;
          const hy = cy + pts[drawN * 2 + 1] * scale;
          ctx.strokeStyle = "rgba(223, 98, 68, 0.55)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(hx - 9, hy);
          ctx.lineTo(hx - 3.5, hy);
          ctx.moveTo(hx + 3.5, hy);
          ctx.lineTo(hx + 9, hy);
          ctx.moveTo(hx, hy - 9);
          ctx.lineTo(hx, hy - 3.5);
          ctx.moveTo(hx, hy + 3.5);
          ctx.lineTo(hx, hy + 9);
          ctx.stroke();

          ctx.fillStyle = signalColor;
          ctx.beginPath();
          ctx.arc(hx, hy, 3.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (isPlaying) {
        rafId = requestAnimationFrame(draw);
      }
    };

    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [activeSeq.mask, customSequence, height, isPlaying, points, preset, speed, theme]);

  const activePresetMeta = PRESETS.find((p) => p.key === preset);
  const curX = points.pts[Math.min(cursor, points.count) * 2]?.toFixed(1) ?? "0.0";
  const curY = points.pts[Math.min(cursor, points.count) * 2 + 1]?.toFixed(1) ?? "0.0";

  const presetButtons = !customSequence ? (
    <div className="flex flex-wrap gap-1.5">
      {PRESETS.map((p) => {
        const active = preset === p.key;
        return (
          <button
            key={p.key}
            type="button"
            onClick={() => setPreset(p.key)}
            className={`rounded-[3px] border px-2.5 py-1 font-mono text-[11px] transition-colors ${
              active
                ? "border-[var(--dev-signal)] bg-[var(--device-well)] font-medium text-[var(--device-ink)]"
                : "border-[var(--device-rule)] text-[var(--device-ink-2)] hover:border-[var(--device-rule-strong)] hover:text-[var(--device-ink)]"
            }`}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  ) : null;

  return (
    <div className="hardware-device overflow-hidden">
      <div className={sidePanel ? "grid lg:grid-cols-[1.85fr_1fr]" : "flex flex-col"}>
        {/* Основной блок плоттера */}
        <div
          className={`flex flex-col ${
            sidePanel ? "border-b border-[var(--device-rule)] lg:border-r lg:border-b-0" : ""
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--device-rule)] px-4 py-3">
            <div className="flex items-center gap-2 text-xs">
              <span className="font-sans font-medium text-[var(--device-ink)]">
                Плоттер траектории
              </span>
              <span className="text-[var(--device-ink-muted)]">·</span>
              <MathTex
                tex={
                  activeSeq.radix === 10
                    ? "\\theta_d = 36^\\circ \\cdot d"
                    : `\\theta_d = \\frac{360^\\circ}{${activeSeq.radix}} \\cdot d`
                }
                className="text-xs text-[var(--device-ink-2)]"
              />
            </div>
            <div className="font-mono text-[11px] tabular-nums text-[var(--device-ink-2)]">
              ({curX}, {curY}) · {cursor}/{points.count}
            </div>
          </div>

          {!sidePanel && !customSequence && (
            <div className="border-b border-[var(--device-rule)] bg-[var(--device-elevated)]/50 px-4 py-2">
              {presetButtons}
            </div>
          )}

          <div
            ref={containerRef}
            className="hardware-well relative w-full border-x-0 border-y-0"
            style={{ height }}
          >
            <canvas ref={canvasRef} className="block h-full w-full" />
            <div className="pointer-events-none absolute bottom-3 left-4 font-sans text-xs text-[var(--device-ink-2)]">
              {customSequence ? customSequence.label : activePresetMeta?.hint}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--device-rule)] px-4 py-2.5">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setIsPlaying((v) => !v)}
                className="flex items-center gap-1.5 rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] px-3 py-1 font-sans text-xs text-[var(--device-ink)] transition-colors hover:border-[var(--device-ink)]"
              >
                {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                {isPlaying ? "Пауза" : "Пуск"}
              </button>
              <button
                type="button"
                onClick={() => {
                  cursorRef.current = 12;
                  setCursor(12);
                }}
                aria-label="Сбросить траекторию"
                className="flex items-center gap-1 rounded-[4px] border border-[var(--device-rule)] px-2.5 py-1 font-sans text-xs text-[var(--device-ink-2)] transition-colors hover:text-[var(--device-ink)]"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Сброс
              </button>
            </div>

            <label className="flex items-center gap-2.5 font-sans text-xs text-[var(--device-ink-2)]">
              <span>Скорость</span>
              <input
                type="range"
                min={4}
                max={45}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="instrument-slider w-24"
              />
            </label>
          </div>
        </div>

        {/* Боковая приборная стойка (только в режиме сцены на Главной) */}
        {sidePanel && (
          <div className="flex flex-col justify-between p-5">
            {sidePanel({
              preset,
              label: activePresetMeta?.label ?? activeSeq.label,
              hint: activePresetMeta?.hint ?? "",
              presetButtons,
            })}
          </div>
        )}
      </div>
    </div>
  );
}
