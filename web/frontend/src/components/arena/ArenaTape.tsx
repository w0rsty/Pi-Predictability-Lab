import { useState } from "react";
import type { ArenaRound } from "../../types/lab";
import { formatToken } from "../../utils/arenaStats";

interface ArenaTapeProps {
  round: ArenaRound;
  radix: number;
  revealed: boolean;
  hideHints: boolean;
  humanChoice: number | null;
}

export function ArenaTape({
  round,
  radix,
  revealed,
  hideHints,
  humanChoice,
}: ArenaTapeProps) {
  const [showAll64, setShowAll64] = useState(false);

  const visibleCount = showAll64 ? round.context.length : Math.min(32, round.context.length);
  const startOffset = round.context.length - visibleCount;
  const visibleTokens = round.context.slice(startOffset);

  const topLags = revealed && !hideHints ? round.transformer.top_lags : [];
  const primaryLag = topLags[0];
  const secondaryLag = topLags[1];
  const showAttentionArcs =
    revealed &&
    !hideHints &&
    (round.is_trigger || (primaryLag !== undefined && primaryLag.weight > 0.25));

  const cellW = 28;
  const totalSlots = visibleTokens.length + 1; // +1 под ячейку [ ? ] / target
  const svgW = totalSlots * cellW;
  const targetSlotX = visibleTokens.length * cellW + cellW / 2;

  const humanRight = humanChoice !== null && humanChoice === round.target;

  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="font-sans text-[var(--device-ink-2)]">
          Окно контекста{" "}
          <span className="font-mono text-[var(--device-ink)]">
            L={visibleTokens.length}
          </span>{" "}
          <span className="text-[var(--device-ink-muted)]">
            {hideHints ? "· источник скрыт" : `· позиция #${round.global_pos.toLocaleString("ru-RU")}`}
          </span>
        </div>

        <button
          type="button"
          onClick={() => setShowAll64((v) => !v)}
          className="rounded-[3px] border border-[var(--device-rule)] bg-[var(--device-well)] px-2.5 py-0.5 font-sans text-[11px] text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
        >
          {showAll64 ? "Свернуть до 32" : "Показать все 64"}
        </button>
      </div>

      <div className="hardware-well overflow-x-auto rounded-[4px] px-3 py-3">
        <svg
          viewBox={`0 0 ${svgW} ${showAttentionArcs ? 116 : 62}`}
          className="mx-auto block w-full"
          style={{
            minWidth: `${Math.max(660, totalSlots * 26)}px`,
            height: showAttentionArcs ? 116 : 62,
          }}
          role="img"
          aria-label="Лента контекста и дуги внимания модели"
        >
          {/* SVG-дуги внимания в фазе revealed */}
          {showAttentionArcs &&
            topLags.slice(0, 2).map((item, idx) => {
              const srcSlot = visibleTokens.length - item.lag;
              if (srcSlot < 0 || srcSlot >= visibleTokens.length) return null;
              const xSrc = srcSlot * cellW + cellW / 2;
              const midX = (xSrc + targetSlotX) / 2;
              const arcPeak = idx === 0 ? 8 : 26;
              const stroke = idx === 0 ? "#DF6244" : "#D4A148";
              const yBase = 62;
              return (
                <g key={`lag-arc-${item.lag}`}>
                  <path
                    d={`M ${xSrc} ${yBase} Q ${midX} ${arcPeak} ${targetSlotX} ${yBase}`}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={idx === 0 ? 2.1 : 1.7}
                  />
                  <text
                    x={midX}
                    y={arcPeak + (idx === 0 ? 12 : 13)}
                    textAnchor="middle"
                    className="font-mono text-[9px]"
                    fill={stroke}
                  >
                    t−{item.lag} ({(item.weight * 100).toFixed(0)}%)
                  </text>
                </g>
              );
            })}

          {/* Ячейки контекста */}
          {visibleTokens.map((tok, idx) => {
            const lag = visibleTokens.length - idx;
            const isPrimary = showAttentionArcs && primaryLag?.lag === lag;
            const isSecondary = showAttentionArcs && secondaryLag?.lag === lag;
            const isCondLag1 =
              revealed && !hideHints && round.is_trigger && round.trigger_info && lag === 1;
            const yTop = showAttentionArcs ? 64 : 10;
            const x = idx * cellW + 2;

            const stroke = isPrimary
              ? "#DF6244"
              : isSecondary
                ? "#D4A148"
                : isCondLag1
                  ? "#EFE9DE"
                  : "rgba(239,233,222,0.14)";
            const fill = isPrimary
              ? "rgba(223,98,68,0.20)"
              : isSecondary
                ? "rgba(212,161,72,0.18)"
                : isCondLag1
                  ? "rgba(239,233,222,0.08)"
                  : "#1B1A18";

            return (
              <g key={`ctx-${idx}`}>
                <rect
                  x={x}
                  y={yTop}
                  width={cellW - 4}
                  height={32}
                  rx={3}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={isPrimary || isSecondary || isCondLag1 ? 1.4 : 1}
                />
                <text
                  x={x + (cellW - 4) / 2}
                  y={yTop + 20}
                  textAnchor="middle"
                  className="font-mono text-xs"
                  fill="#EFE9DE"
                >
                  {formatToken(tok, radix)}
                </text>
                {(isPrimary || isSecondary || isCondLag1) && (
                  <text
                    x={x + (cellW - 4) / 2}
                    y={yTop + 44}
                    textAnchor="middle"
                    className="font-mono text-[9px]"
                    fill={isPrimary ? "#DF6244" : isSecondary ? "#D4A148" : "#B4ABA0"}
                  >
                    t−{lag}
                  </text>
                )}
              </g>
            );
          })}

          {/* Ячейка-мишень справа [ ? ] -> target */}
          {(() => {
            const yTop = showAttentionArcs ? 64 : 10;
            const x = visibleTokens.length * cellW + 2;
            const targetFill = revealed
              ? humanRight
                ? "#2F7A4D"
                : "#DF6244"
              : "rgba(223,98,68,0.12)";
            const targetStroke = revealed ? (humanRight ? "#4D9E6D" : "#DF6244") : "#DF6244";
            return (
              <g>
                <rect
                  x={x}
                  y={yTop}
                  width={cellW - 4}
                  height={32}
                  rx={3}
                  fill={targetFill}
                  stroke={targetStroke}
                  strokeWidth={1.6}
                />
                <text
                  x={x + (cellW - 4) / 2}
                  y={yTop + 20}
                  textAnchor="middle"
                  className="font-mono text-xs font-semibold"
                  fill="#FAF9F5"
                >
                  {revealed ? formatToken(round.target, radix) : "?"}
                </text>
                <text
                  x={x + (cellW - 4) / 2}
                  y={yTop + 44}
                  textAnchor="middle"
                  className="font-mono text-[9px]"
                  fill="#EFE9DE"
                >
                  t
                </text>
              </g>
            );
          })()}
        </svg>
      </div>

      {/* Расшифровка триггера после вскрытия (скрыта в режиме blind_audit до финала) */}
      {revealed && !hideHints && round.trigger_info && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-[4px] border border-[var(--dev-signal)]/45 bg-[#DF6244]/10 px-3 py-1.5 font-mono text-xs text-[var(--device-ink)]">
          <span>
            Позиция #{round.global_pos} · триггер (t−24={round.trigger_info.a}, t−7=
            {round.trigger_info.b}, t−1={round.trigger_info.c_prev} &lt; 4)
          </span>
          <span className="font-medium text-[var(--dev-signal)]">
            {round.trigger_info.formula}
          </span>
        </div>
      )}
    </div>
  );
}
