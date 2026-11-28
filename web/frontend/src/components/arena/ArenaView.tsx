import { useCallback, useEffect, useState } from "react";
import { usePiSynth } from "../../hooks/usePiSynth";
import type {
  ArenaDatasetId,
  ArenaModeId,
  ArenaRound,
  ArenaRoundsBank,
  ArenaSessionResponse,
  ThemeMode,
} from "../../types/lab";
import {
  type ConfidenceLevel,
  type PlayedRoundRecord,
  humanBitsForChoice,
} from "../../utils/arenaStats";
import { playToggleClick } from "../../utils/tactileAudio";
import { ArenaConsole, type ArenaPhase } from "./ArenaConsole";
import { ArenaDebrief } from "./ArenaDebrief";

interface ArenaViewProps {
  theme: ThemeMode;
  isGpuConnected: boolean;
  onFetchLiveSession: (payload: {
    dataset: ArenaModeId;
    n_rounds: number;
    seed?: number;
  }) => Promise<ArenaSessionResponse>;
}

const ARENA_MODES: Array<{ id: ArenaModeId; label: string; roundsCount: number }> = [
  { id: "true_pi", label: "Контроль π (H₀)", roundsCount: 15 },
  { id: "csprng_backdoor", label: "Закладка (t-24, t-7)", roundsCount: 15 },
  { id: "champernowne", label: "Чемперноун", roundsCount: 15 },
  { id: "e_cf", label: "Цепная дробь e", roundsCount: 15 },
  { id: "blind_audit", label: "Слепой тест", roundsCount: 10 },
];

const BLIND_POOL: ArenaDatasetId[] = [
  "true_pi",
  "rational_1_97",
  "champernowne",
  "csprng_backdoor",
];

function sampleStaticRounds(
  bank: ArenaRoundsBank,
  mode: ArenaModeId,
  nRounds: number
): { rounds: ArenaRound[]; trueDataset: ArenaDatasetId; radix: number } {
  const trueDataset: ArenaDatasetId =
    mode === "blind_audit"
      ? BLIND_POOL[Math.floor(Math.random() * BLIND_POOL.length)]
      : mode;

  const entry = bank.datasets[trueDataset];
  const pool = [...entry.rounds];

  // Перемешиваем Fisher–Yates и берём сбалансированные раунды для каждой новой партии
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }

  let picked = pool.slice(0, Math.min(nRounds, pool.length));
  if (trueDataset === "csprng_backdoor") {
    const trigs = pool.filter((r) => r.is_trigger);
    const bgs = pool.filter((r) => !r.is_trigger);
    const wantTrigs = Math.min(trigs.length, Math.ceil(nRounds / 2));
    const wantBg = Math.min(bgs.length, nRounds - wantTrigs);
    picked = [...trigs.slice(0, wantTrigs), ...bgs.slice(0, wantBg)];
    for (let i = picked.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [picked[i], picked[j]] = [picked[j], picked[i]];
    }
  }

  return { rounds: picked, trueDataset, radix: entry.radix };
}

export function ArenaView({ theme, isGpuConnected, onFetchLiveSession }: ArenaViewProps) {
  const synth = usePiSynth();
  const [bank, setBank] = useState<ArenaRoundsBank | null>(null);
  const [bankError, setBankError] = useState<string | null>(null);

  const [mode, setMode] = useState<ArenaModeId>("true_pi");
  const [preferLiveGpu, setPreferLiveGpu] = useState<boolean>(true);
  const [sourceLabel, setSourceLabel] = useState<"static" | "live_gpu">("static");
  const [loadingSession, setLoadingSession] = useState<boolean>(false);

  const [rounds, setRounds] = useState<ArenaRound[]>([]);
  const [trueDataset, setTrueDataset] = useState<ArenaDatasetId>("true_pi");
  const [radix, setRadix] = useState<number>(10);
  const [currentIdx, setCurrentIdx] = useState<number>(0);
  const [phase, setPhase] = useState<ArenaPhase>("playing");
  const [records, setRecords] = useState<PlayedRoundRecord[]>([]);
  const [confidence, setConfidence] = useState<ConfidenceLevel>(0.6);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
  const [inspectIdx, setInspectIdx] = useState<number | null>(null);
  const [blindVerdict, setBlindVerdict] = useState<"H0" | "H1" | null>(null);

  // Ленивая загрузка arena_rounds.json только при открытии Арены
  useEffect(() => {
    let cancelled = false;
    const base = import.meta.env.BASE_URL || "/";
    const url = `${base.endsWith("/") ? base : `${base}/`}data/arena_rounds.json`;
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<ArenaRoundsBank>;
      })
      .then((json) => {
        if (!cancelled) setBank(json);
      })
      .catch((err) => {
        if (!cancelled) {
          setBankError(err instanceof Error ? err.message : "Ошибка загрузки arena_rounds.json");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const startSession = useCallback(
    async (targetMode: ArenaModeId, useLive: boolean) => {
      const cfg = ARENA_MODES.find((m) => m.id === targetMode) ?? ARENA_MODES[0];
      setCurrentIdx(0);
      setPhase("playing");
      setRecords([]);
      setInspectIdx(null);
      setBlindVerdict(null);

      if (isGpuConnected && useLive) {
        setLoadingSession(true);
        try {
          const live = await onFetchLiveSession({
            dataset: targetMode,
            n_rounds: cfg.roundsCount,
          });
          setRounds(live.rounds);
          setTrueDataset(live.true_dataset);
          setRadix(live.radix);
          setSourceLabel("live_gpu");
          setLoadingSession(false);
          return;
        } catch {
          // Бесшовный фоллбэк на статический банк реальных чекпойнтов
        } finally {
          setLoadingSession(false);
        }
      }

      if (bank) {
        const sampled = sampleStaticRounds(bank, targetMode, cfg.roundsCount);
        setRounds(sampled.rounds);
        setTrueDataset(sampled.trueDataset);
        setRadix(sampled.radix);
        setSourceLabel("static");
      }
    },
    [bank, isGpuConnected, onFetchLiveSession]
  );

  useEffect(() => {
    if (bank && rounds.length === 0) {
      void startSession(mode, preferLiveGpu);
    }
  }, [bank, mode, preferLiveGpu, rounds.length, startSession]);

  const handleModeChange = (nextMode: ArenaModeId) => {
    setMode(nextMode);
    void startSession(nextMode, preferLiveGpu);
  };

  const handleToggleSource = () => {
    const nextUseLive = !preferLiveGpu;
    setPreferLiveGpu(nextUseLive);
    void startSession(mode, nextUseLive);
  };

  const handleSelectToken = useCallback(
    (token: number) => {
      if (phase !== "playing" || !rounds[currentIdx]) return;
      const r = rounds[currentIdx];
      const bits = humanBitsForChoice(token, r.target, radix, confidence);
      const correct = token === r.target;

      if (soundEnabled) {
        synth.playNote(token, r.is_trigger);
      }

      setRecords((prev) => [
        ...prev,
        {
          round: r,
          humanChoice: token,
          confidence,
          humanBits: bits,
          humanCorrect: correct,
        },
      ]);
      setPhase("revealed");
    },
    [confidence, currentIdx, phase, radix, rounds, soundEnabled, synth]
  );

  const handleNextRound = useCallback(() => {
    if (phase !== "revealed") return;
    if (currentIdx + 1 >= rounds.length) {
      if (mode === "blind_audit") {
        setPhase("blind_verdict");
      } else {
        setPhase("debrief");
        setInspectIdx(0);
      }
    } else {
      setCurrentIdx((i) => i + 1);
      setPhase("playing");
    }
  }, [currentIdx, mode, phase, rounds.length]);

  const activeModeLabel = ARENA_MODES.find((m) => m.id === mode)?.label ?? mode;

  return (
    <div className="space-y-6 pb-20">
      {/* Открытая шапка прямо на тёплой архивной бумаге */}
      <div className="grid gap-3 border-b border-[var(--rule)] pb-4 md:grid-cols-[1fr_auto] md:items-end">
        <div>
          <h1 className="font-serif text-3xl font-medium tracking-tight text-[var(--ink)] sm:text-4xl">
            Арена: человек против нейросети
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--ink-2)]">
            Одновременное предсказание следующего символа против DigitTransformer и PPM-C на
            реальных тестовых окнах.
          </p>
        </div>

        <div className="flex items-center gap-3 font-sans text-xs text-[var(--ink-muted)]">
          {isGpuConnected ? (
            <button
              type="button"
              data-testid="arena-source-toggle"
              onClick={handleToggleSource}
              className="flex items-center gap-2 rounded-[4px] border border-[var(--rule-strong)] px-3 py-1.5 text-[var(--ink)] transition-colors hover:border-[var(--sig-h1)]"
              title="Переключить источник раундов (Static банк / Live GPU случайный срез)"
            >
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor:
                    sourceLabel === "live_gpu" ? "var(--sig-gpu)" : "var(--ink-muted)",
                }}
              />
              {loadingSession
                ? "Генерация на GPU…"
                : sourceLabel === "live_gpu"
                  ? "Live GPU · случайный срез"
                  : "Реальные чекпойнты · Static"}
            </button>
          ) : (
            <span>Реальные чекпойнты · Static</span>
          )}
        </div>
      </div>

      {/* Подчёркнутые вкладки 5 режимов */}
      <div className="flex flex-nowrap gap-5 overflow-x-auto border-b border-[var(--rule)] sm:gap-7">
        {ARENA_MODES.map((m) => {
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              type="button"
              data-testid={`arena-mode-${m.id}`}
              onClick={() => handleModeChange(m.id)}
              className={`-mb-px shrink-0 border-b-2 pb-2.5 font-sans text-sm font-medium transition-colors focus:outline-none focus-visible:text-[var(--sig-h1)] ${
                active
                  ? "border-[var(--sig-h1)] text-[var(--ink)]"
                  : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              {m.label}
            </button>
          );
        })}
      </div>

      {bankError ? (
        <div className="rounded-[4px] border border-[var(--sig-h1)]/50 bg-[var(--bg-alt)] p-5 font-mono text-xs text-[var(--sig-h1)]">
          Не удалось загрузить банк раундов Арены: {bankError}
        </div>
      ) : rounds.length === 0 ? (
        <div className="flex min-h-[240px] items-center justify-center font-mono text-xs text-[var(--ink-muted)]">
          Подготовка тестовых раундов…
        </div>
      ) : (
        <>
          <ArenaConsole
            mode={mode}
            radix={radix}
            rounds={rounds}
            currentIdx={currentIdx}
            phase={phase}
            records={records}
            confidence={confidence}
            soundEnabled={soundEnabled}
            inspectIdx={inspectIdx}
            blindVerdict={blindVerdict}
            trueDataset={trueDataset}
            onSelectToken={handleSelectToken}
            onNextRound={handleNextRound}
            onChangeConfidence={(c) => {
              setConfidence(c);
              if (soundEnabled) playToggleClick(true);
            }}
            onToggleSound={() => setSoundEnabled((s) => !s)}
            onNewGame={() => void startSession(mode, preferLiveGpu)}
            onSubmitBlindVerdict={(v) => {
              setBlindVerdict(v);
              setPhase("debrief");
              setInspectIdx(0);
            }}
          />

          {phase === "debrief" && (
            <ArenaDebrief
              mode={mode}
              modeLabel={activeModeLabel}
              trueDataset={trueDataset}
              radix={radix}
              records={records}
              theme={theme}
              blindVerdict={blindVerdict}
              inspectIdx={inspectIdx}
              onInspectRound={(idx) => setInspectIdx(idx)}
              onPlayAgain={() => void startSession(mode, preferLiveGpu)}
              onSwitchMode={handleModeChange}
            />
          )}
        </>
      )}
    </div>
  );
}
