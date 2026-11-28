import { useEffect, useState } from "react";

interface ScrambleNumberProps {
  value: string;
  durationMs?: number;
  className?: string;
}

const DIGITS = "0123456789";

export function ScrambleNumber({
  value,
  durationMs = 520,
  className = "",
}: ScrambleNumberProps) {
  const [display, setDisplay] = useState<string>(value);

  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      setDisplay(value);
      return;
    }

    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / durationMs);
      const lockedCount = Math.floor(progress * value.length);
      let next = "";
      for (let i = 0; i < value.length; i++) {
        const ch = value[i];
        if (i < lockedCount || !/[0-9]/.test(ch)) {
          next += ch;
        } else {
          next += DIGITS[(Math.floor(now / 45) + i * 3) % 10];
        }
      }
      setDisplay(next);
      if (progress < 1) {
        frame = requestAnimationFrame(tick);
      } else {
        setDisplay(value);
      }
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, durationMs]);

  return <span className={`font-mono tabular-nums ${className}`}>{display}</span>;
}
