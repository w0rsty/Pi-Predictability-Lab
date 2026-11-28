import { formatToken } from "../../utils/arenaStats";

interface ArenaProbSpectrumProps {
  radix: number;
  target: number;
  tfProbs: number[];
  ppmProbs: number[];
}

export function ArenaProbSpectrum({
  radix,
  target,
  tfProbs,
  ppmProbs,
}: ArenaProbSpectrumProps) {
  const tokens = Array.from({ length: radix }, (_, i) => i);
  const chancePct = 100 / Math.max(2, radix);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-sans text-[var(--device-ink-2)]">
          Распределение вероятностей моделей на следующем символе
        </span>
        <div className="flex items-center gap-4 font-sans text-[11px]">
          <span className="flex items-center gap-1.5 text-[var(--device-ink-2)]">
            <span className="inline-block h-2 w-2 rounded-[1px] bg-[var(--dev-signal)]" />
            DigitTransformer
          </span>
          <span className="flex items-center gap-1.5 text-[var(--device-ink-2)]">
            <span className="inline-block h-2 w-2 rounded-[1px] bg-[var(--dev-h0)]" />
            PPM_o8
          </span>
        </div>
      </div>

      <div className="hardware-well relative rounded-[4px] px-3 pt-4 pb-1.5">
        {/* Пунктир случайного уровня 1/K */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute right-3 left-3 border-t border-dashed border-[var(--device-rule-strong)]"
          style={{ bottom: `${22 + (chancePct / 100) * 50}px` }}
          title={`Случайное угадывание ${chancePct.toFixed(1)}%`}
        />

        <div
          className="grid items-end gap-1.5"
          style={{ gridTemplateColumns: `repeat(${radix}, minmax(0, 1fr))` }}
        >
          {tokens.map((tok) => {
            const pTf = (tfProbs[tok] ?? 0) * 100;
            const pPpm = (ppmProbs[tok] ?? 0) * 100;
            const hTf = Math.max(2, Math.round((pTf / 100) * 50));
            const hPpm = Math.max(2, Math.round((pPpm / 100) * 50));
            const isTarget = tok === target;

            return (
              <div key={tok} className="flex flex-col items-center gap-1">
                <div className="flex h-[52px] w-full items-end justify-center gap-0.5">
                  <div
                    className="w-[42%] max-w-[12px] rounded-t-[1px] transition-[height] duration-200"
                    style={{
                      height: `${hTf}px`,
                      backgroundColor: "#DF6244",
                    }}
                    title={`Transformer (${formatToken(tok, radix)}): ${pTf.toFixed(1)}%`}
                  />
                  <div
                    className="w-[42%] max-w-[12px] rounded-t-[1px] transition-[height] duration-200"
                    style={{
                      height: `${hPpm}px`,
                      backgroundColor: "#5E9EA8",
                    }}
                    title={`PPM_o8 (${formatToken(tok, radix)}): ${pPpm.toFixed(1)}%`}
                  />
                </div>
                <span
                  className={`font-mono text-[10px] ${
                    isTarget
                      ? "rounded-[2px] bg-[#2F7A4D] px-1 font-semibold text-white"
                      : "text-[var(--device-ink-muted)]"
                  }`}
                >
                  {formatToken(tok, radix)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
