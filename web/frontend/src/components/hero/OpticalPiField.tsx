import { useEffect, useMemo, useRef } from "react";

interface OpticalPiFieldProps {
  digits: number[];
}

/**
 * Тонкий водяной знак цифр π на бумажном поле: под курсором мягко проявляются чернильные знаки и миллиметровка.
 */
export function OpticalPiField({ digits }: OpticalPiFieldProps) {
  const lensRef = useRef<HTMLDivElement | null>(null);
  const pointerPosRef = useRef<{ x: number; y: number; inside: boolean }>({
    x: -999,
    y: -999,
    inside: false,
  });

  const rows = useMemo(() => {
    const out: Array<Array<{ d: number; highlight: boolean }>> = [];
    const cols = 42;
    const totalRows = 7;
    for (let r = 0; r < totalRows; r++) {
      const row: Array<{ d: number; highlight: boolean }> = [];
      for (let c = 0; c < cols; c++) {
        const idx = (r * cols + c) % Math.max(1, digits.length);
        const d = digits[idx] ?? 0;
        row.push({ d, highlight: (r * 7 + c) % 23 === 0 });
      }
      out.push(row);
    }
    return out;
  }, [digits]);

  useEffect(() => {
    const lens = lensRef.current;
    const parent = lens?.parentElement;
    if (!lens || !parent) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
    if (reducedMotion || coarsePointer) return;

    let raf = 0;
    let dirty = false;

    const commit = () => {
      raf = 0;
      if (!dirty || !lensRef.current) return;
      dirty = false;
      const { x, y, inside } = pointerPosRef.current;
      if (inside) {
        lensRef.current.style.setProperty("--mx", `${x.toFixed(1)}px`);
        lensRef.current.style.setProperty("--my", `${y.toFixed(1)}px`);
        lensRef.current.style.opacity = "1";
      } else {
        lensRef.current.style.opacity = "0";
      }
    };

    const schedule = () => {
      dirty = true;
      if (!raf) raf = requestAnimationFrame(commit);
    };

    const onMove = (e: PointerEvent) => {
      const rect = parent.getBoundingClientRect();
      pointerPosRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        inside: true,
      };
      schedule();
    };

    const onLeave = () => {
      pointerPosRef.current.inside = false;
      schedule();
    };

    parent.addEventListener("pointermove", onMove, { passive: true });
    parent.addEventListener("pointerleave", onLeave, { passive: true });
    return () => {
      parent.removeEventListener("pointermove", onMove);
      parent.removeEventListener("pointerleave", onLeave);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const renderMatrix = (isLensLayer: boolean) => (
    <div className="flex flex-col gap-3 px-2 py-2 font-mono text-[11px] leading-none tracking-[0.65em] select-none">
      {rows.map((row, rIdx) => (
        <div key={rIdx} className="whitespace-nowrap">
          {row.map((cell, cIdx) => (
            <span
              key={cIdx}
              className={
                isLensLayer && cell.highlight
                  ? "font-medium text-[var(--sig-h1)]"
                  : "text-[var(--ink)]"
              }
            >
              {cell.d}
            </span>
          ))}
        </div>
      ))}
    </div>
  );

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-0 overflow-hidden select-none"
    >
      <div className="swiss-grid-bg absolute inset-0 opacity-[0.035]">{renderMatrix(false)}</div>

      <div
        ref={lensRef}
        className="absolute inset-0 opacity-0 transition-opacity duration-200"
        style={{
          WebkitMaskImage:
            "radial-gradient(190px circle at var(--mx, -999px) var(--my, -999px), black 0%, transparent 72%)",
          maskImage:
            "radial-gradient(190px circle at var(--mx, -999px) var(--my, -999px), black 0%, transparent 72%)",
        }}
      >
        <div className="swiss-grid-bg absolute inset-0 opacity-[0.42]" />
        <div className="relative opacity-[0.20]">{renderMatrix(true)}</div>
      </div>
    </div>
  );
}
