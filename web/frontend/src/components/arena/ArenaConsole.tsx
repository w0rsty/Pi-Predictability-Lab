import { useEffect } from "react";
import { RotateCcw, Volume2, VolumeX } from "lucide-react";
import type { ArenaModeId, ArenaRound } from "../../types/lab";
import {
  type ConfidenceLevel,
  type PlayedRoundRecord,
  formatToken,
} from "../../utils/arenaStats";
import { ArenaKeypad } from "./ArenaKeypad";
import { ArenaProbSpectrum } from "./ArenaProbSpectrum";
import { ArenaTape } from "./ArenaTape";

export type ArenaPhase = "playing" | "revealed" | "blind_verdict" | "debrief";

interface ArenaConsoleProps {
  mode: ArenaModeId;
  radix: number;
  rounds: ArenaRound[];
  currentIdx: number;
  phase: ArenaPhase;
  records: PlayedRoundRecord[];
  confidence: ConfidenceLevel;
  soundEnabled: boolean;
  inspectIdx: number | null;
  blindVerdict: "H0" | "H1" | null;
  trueDataset: string;
  onSelectToken: (token: number) => void;
  onNextRound: () => void;
  onChangeConfidence: (c: ConfidenceLevel) => void;
  onToggleSound: () => void;
  onNewGame: () => void;
  onSubmitBlindVerdict: (verdict: "H0" | "H1") => void;
}

const CONFIDENCE_OPTIONS: Array<{ val: ConfidenceLevel; label: string; keyHint: string }> = [
  { val: 0.25, label: "25%", keyHint: "Q" },
  { val: 0.6, label: "60%", keyHint: "W" },
  { val: 0.95, label: "95%", keyHint: "E" },
];

export function ArenaConsole({
  mode,
  radix,
  rounds,
  currentIdx,
  phase,
  records,
  confidence,
  soundEnabled,
  inspectIdx,
  blindVerdict,
  trueDataset,
  onSelectToken,
  onNextRound,
  onChangeConfidence,
  onToggleSound,
  onNewGame,
  onSubmitBlindVerdict,
}: ArenaConsoleProps) {
  const activeIdx = inspectIdx !== null ? inspectIdx : currentIdx;
  const activeRound = rounds[activeIdx] ?? rounds[0];
  const activeRecord = records[activeIdx] ?? null;
  const isRevealed = phase === "revealed" || phase === "debrief" || phase === "blind_verdict";
  const hideHints = mode === "blind_audit" && phase !== "debrief";

  // Клавиатурное управление (0..9, ';', Q/W/E, Enter/Space)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const low = e.key.toLowerCase();
      if (low === "q" || low === "й") {
        onChangeConfidence(0.25);
        return;
      }
      if (low === "w" || low === "ц") {
        onChangeConfidence(0.6);
        return;
      }
      if (low === "e" || low === "у") {
        onChangeConfidence(0.95);
        return;
      }

      if (phase === "playing") {
        if (e.key >= "0" && e.key <= "9") {
          const val = Number(e.key);
          if (val < radix) onSelectToken(val);
        } else if ((e.key === ";" || e.key === ":") && radix === 11) {
          onSelectToken(10);
        }
      } else if (phase === "revealed" && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault();
        onNextRound();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onChangeConfidence, onNextRound, onSelectToken, phase, radix]);

  if (!activeRound) return null;

  const playedCount = records.length;
  const humanHits = records.filter((r) => r.humanCorrect).length;
  const tfHits = records.filter((r) => r.round.transformer.pred === r.round.target).length;
  const ppmHits = records.filter((r) => r.round.ppm.pred === r.round.target).length;
  const humanBitsAvg =
    playedCount > 0 ? records.reduce((a, r) => a + r.humanBits, 0) / playedCount : 0;
  const tfBitsAvg =
    playedCount > 0 ? records.reduce((a, r) => a + r.round.transformer.bits, 0) / playedCount : 0;
  const ppmBitsAvg =
    playedCount > 0 ? records.reduce((a, r) => a + r.round.ppm.bits, 0) / playedCount : 0;

  return (
    <section className="hardware-device overflow-hidden">
      {/* Верхняя утопленная полоса телеметрии */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--device-rule)] px-5 py-3.5">
        {/* Прогресс раундов */}
        <div className="space-y-1.5">
          <div className="font-mono text-xs tabular-nums text-[var(--device-ink)]">
            Раунд {String(Math.min(activeIdx + 1, rounds.length)).padStart(2, "0")} /{" "}
            {String(rounds.length).padStart(2, "0")}
          </div>
          <div className="flex flex-wrap gap-1" aria-label="Прогресс раундов">
            {rounds.map((r, i) => {
              const rec = records[i];
              let bg = "rgba(239,233,222,0.12)";
              if (rec) {
                const aiHit = rec.round.transformer.pred === rec.round.target;
                if (rec.humanCorrect) bg = "#2F7A4D";
                else if (aiHit) bg = "#DF6244";
                else bg = "#635D55";
              }
              return (
                <span
                  key={r.id}
                  className={`h-[4px] w-[10px] rounded-[1px] ${
                    i === activeIdx ? "ring-1 ring-[var(--device-ink)]" : ""
                  }`}
                  style={{ backgroundColor: bg }}
                />
              );
            })}
          </div>
        </div>

        {/* Счётчик результатов */}
        <div className="flex flex-wrap items-center gap-5 font-mono text-xs tabular-nums">
          <div>
            <span className="font-sans text-[11px] text-[var(--device-ink-muted)]">Человек: </span>
            <span className="text-[var(--device-ink)]">
              {humanHits}/{playedCount}
            </span>
            {playedCount > 0 && (
              <span className="ml-1 text-[11px] text-[var(--device-ink-2)]">
                ({humanBitsAvg.toFixed(2)} b)
              </span>
            )}
          </div>
          <div>
            <span className="font-sans text-[11px] text-[var(--device-ink-muted)]">
              Transformer:{" "}
            </span>
            <span className="text-[var(--dev-signal)]">
              {tfHits}/{playedCount}
            </span>
            {playedCount > 0 && (
              <span className="ml-1 text-[11px] text-[var(--device-ink-2)]">
                ({tfBitsAvg.toFixed(2)} b)
              </span>
            )}
          </div>
          <div>
            <span className="font-sans text-[11px] text-[var(--device-ink-muted)]">PPM_o8: </span>
            <span className="text-[var(--dev-h0)]">
              {ppmHits}/{playedCount}
            </span>
            {playedCount > 0 && (
              <span className="ml-1 text-[11px] text-[var(--device-ink-2)]">
                ({ppmBitsAvg.toFixed(2)} b)
              </span>
            )}
          </div>
        </div>

        {/* Аппаратные переключатели */}
        <div className="flex flex-wrap items-center gap-2">
          <div
            className="flex items-center rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] p-0.5"
            title="Уверенность ставки (горячие клавиши Q / W / E)"
          >
            {CONFIDENCE_OPTIONS.map((opt) => {
              const active = confidence === opt.val;
              return (
                <button
                  key={opt.val}
                  type="button"
                  onClick={() => onChangeConfidence(opt.val)}
                  className={`rounded-[3px] px-2 py-0.5 font-mono text-[11px] transition-colors ${
                    active
                      ? "bg-[var(--device-elevated)] font-medium text-[var(--device-ink)]"
                      : "text-[var(--device-ink-muted)] hover:text-[var(--device-ink-2)]"
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>

          <button
            type="button"
            onClick={onToggleSound}
            aria-label="Звук клавиш"
            className="rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] p-1.5 text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
          >
            {soundEnabled ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
          </button>

          <button
            type="button"
            onClick={onNewGame}
            data-testid="arena-new-game-btn"
            className="flex items-center gap-1.5 rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] px-2.5 py-1 font-sans text-xs text-[var(--device-ink)] hover:border-[var(--device-ink)]"
          >
            <RotateCcw className="h-3 w-3" />
            Новая партия
          </button>
        </div>
      </div>

      {/* Тело прибора: лента + вскрытие прогнозов + клавиатура */}
      <div className="space-y-3.5 p-4">
        <ArenaTape
          round={activeRound}
          radix={radix}
          revealed={isRevealed && activeRecord !== null}
          hideHints={hideHints}
          humanChoice={activeRecord ? activeRecord.humanChoice : null}
        />

        {/* Панель слепого вердикта аудитора (режим blind_audit после 10 раунда) */}
        {phase === "blind_verdict" && (
          <div className="hardware-well space-y-4 rounded-[4px] border border-[var(--dev-signal)]/50 p-5">
            <div className="space-y-1">
              <div className="font-sans text-sm font-medium text-[var(--device-ink)]">
                Финальный вердикт аудитора
              </div>
              <p className="text-xs leading-relaxed text-[var(--device-ink-2)]">
                Вы прошли {rounds.length} раундов неизвестного потока. Судя по поведению цифр и
                прогнозам моделей, что перед вами?
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                data-testid="blind-verdict-h0"
                onClick={() => onSubmitBlindVerdict("H0")}
                className="flex-1 rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-bg)] px-4 py-2.5 font-sans text-xs font-medium text-[var(--device-ink)] transition-colors hover:border-[var(--dev-h0)]"
              >
                Шум / нормальная константа (H₀)
              </button>
              <button
                type="button"
                data-testid="blind-verdict-h1"
                onClick={() => onSubmitBlindVerdict("H1")}
                className="flex-1 rounded-[4px] bg-[var(--dev-signal)] px-4 py-2.5 font-sans text-xs font-medium text-white transition-opacity hover:opacity-90"
              >
                Алгоритм или скрытая закладка (H₁)
              </button>
            </div>
          </div>
        )}

        {/* Блок одновременного вскрытия прогнозов */}
        {phase === "playing" && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)]/60 px-4 py-3 text-xs text-[var(--device-ink-2)]">
            <span>
              Выберите следующий символ клавишей 0–{radix === 11 ? "9 или ;" : radix - 1} —
              нейросеть и PPM-C уже зафиксировали свой прогноз.
            </span>
            <span className="font-mono text-[11px] text-[var(--device-ink-muted)]">
              Ставка: {Math.round(confidence * 100)}% (Q / W / E)
            </span>
          </div>
        )}

        {(phase === "revealed" || (phase === "debrief" && activeRecord !== null)) && activeRecord && (
          <div className="space-y-4 rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-elevated)]/40 p-4">
            {/* Вердикт раунда в одну строку */}
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-x-5 gap-y-1 font-mono tabular-nums">
                <span className="font-medium text-[var(--device-ink)]">
                  Истина: {formatToken(activeRound.target, radix)}
                </span>
                <span
                  className={
                    activeRecord.humanCorrect
                      ? "text-[var(--dev-led-on)]"
                      : "text-[var(--dev-signal)]"
                  }
                >
                  Ваш выбор: {formatToken(activeRecord.humanChoice, radix)} —{" "}
                  {activeRecord.humanCorrect ? "точно" : "промах"},{" "}
                  {activeRecord.humanBits.toFixed(2)} бит
                </span>
                <span className="text-[var(--device-ink-2)]">
                  Transformer: {formatToken(activeRound.transformer.pred, radix)} (
                  {activeRound.transformer.bits.toFixed(2)} бит)
                </span>
                <span className="text-[var(--device-ink-2)]">
                  PPM_o8: {formatToken(activeRound.ppm.pred, radix)} (
                  {activeRound.ppm.bits.toFixed(2)} бит)
                </span>
              </div>

              {phase === "revealed" && (
                <button
                  type="button"
                  data-testid="arena-next-round-btn"
                  onClick={onNextRound}
                  className="rounded-[4px] bg-[var(--device-ink)] px-3.5 py-1.5 font-sans text-xs font-medium text-[var(--device-bg)] transition-opacity hover:opacity-90"
                >
                  {currentIdx + 1 >= rounds.length
                    ? mode === "blind_audit"
                      ? "К диагнозу →"
                      : "Итоги партии →"
                    : "Следующий раунд →"}
                </button>
              )}
            </div>

            <ArenaProbSpectrum
              radix={radix}
              target={activeRound.target}
              tfProbs={activeRound.transformer.probs}
              ppmProbs={activeRound.ppm.probs}
            />
          </div>
        )}

        {phase === "debrief" && blindVerdict !== null && mode === "blind_audit" && (
          <div className="rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] px-4 py-3 text-xs text-[var(--device-ink-2)]">
            Рассекреченный источник: <strong className="font-mono text-[var(--device-ink)]">{trueDataset}</strong> · Ваш диагноз:{" "}
            <strong className="font-mono text-[var(--device-ink)]">{blindVerdict}</strong>{" "}
            ({(trueDataset === "true_pi" && blindVerdict === "H0") ||
            (trueDataset !== "true_pi" && blindVerdict === "H1")
              ? "диагноз верен"
              : "диагноз расходится с природой потока"})
          </div>
        )}

        {/* Клавиатура слоновой кости */}
        <ArenaKeypad
          radix={radix}
          disabled={phase !== "playing"}
          revealed={isRevealed && activeRecord !== null}
          target={activeRound.target}
          humanChoice={activeRecord ? activeRecord.humanChoice : null}
          aiChoice={activeRound.transformer.pred}
          onSelect={onSelectToken}
        />
      </div>
    </section>
  );
}
