import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import type {
  DreamCycleInfo,
  DreamDatasetId,
  DreamGpuResponse,
  DreamSeedEntry,
  DreamSeedsBank,
  DreamSelfTrigger,
  ThemeMode,
} from "../../types/lab";
import { applyCustomBackdoorBrowser, generateCryptoRandomDigits } from "../../utils/browserMath";
import {
  createMulberry32,
  detectRepetitionCycle,
  runDreamSingleStep,
  streamDreamGeneration,
  type DreamSpeedMode,
} from "../../utils/dreamEngine";
import { PlotlyChart } from "../ui/PlotlyChart";

interface DreamLabPanelProps {
  theme: ThemeMode;
  isGpuConnected: boolean;
  onSendToPiano: (stream: { label: string; digits: number[]; mask?: number[] | null }) => void;
  onGenerateDreamGpu: (payload: {
    dataset: DreamDatasetId;
    seed_tokens: number[];
    steps: number;
    temperature: number;
    top_k: number;
    random_seed?: number;
  }) => Promise<DreamGpuResponse>;
}

const MODEL_CARDS: Array<{
  id: DreamDatasetId;
  title: string;
  subtitle: string;
  badge: string;
}> = [
  {
    id: "true_pi",
    title: "true_pi (H₀)",
    subtitle: "Коллапс в циклы при низком τ и расхождение с π",
    badge: "Base-10",
  },
  {
    id: "champernowne",
    title: "champernowne (H₁)",
    subtitle: "Автономный подсчёт 6-значных чисел",
    badge: "Base-10",
  },
  {
    id: "e_cf",
    title: "e_cf (H₁)",
    subtitle: "Самостоятельная генерация цепной дроби e",
    badge: "Base-11",
  },
  {
    id: "csprng_backdoor",
    title: "csprng_backdoor",
    subtitle: "Проверка скрытой закладки внутри собственных снов",
    badge: "t−24, t−7",
  },
];

const TAU_NOTCHES = [
  { val: 0.0, label: "0.00 Жадный" },
  { val: 0.3, label: "0.30 Холодный" },
  { val: 1.0, label: "1.00 Норма" },
  { val: 1.4, label: "1.40 Хаос" },
];

function formatTok(tok: number): string {
  return tok === 10 ? ";" : tok.toString(10);
}

function DualDreamWalkCanvas({
  dreamTokens,
  truthTokens,
  radix,
  hasTruth,
}: {
  dreamTokens: number[];
  truthTokens: number[];
  radix: number;
  hasTruth: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = canvas.clientWidth || 620;
    const cssH = 230;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Очистка тёмного экрана
    ctx.fillStyle = "#121110";
    ctx.fillRect(0, 0, cssW, cssH);

    // Тонкая сетка
    ctx.strokeStyle = "rgba(232,226,214,0.06)";
    ctx.lineWidth = 1;
    for (let x = 40; x < cssW; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, cssH);
      ctx.stroke();
    }
    for (let y = 40; y < cssH; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(cssW, y);
      ctx.stroke();
    }

    const buildPts = (seq: number[]) => {
      const pts: Array<[number, number]> = [[0, 0]];
      let x = 0;
      let y = 0;
      const stepAngle = (Math.PI * 2) / Math.max(2, radix);
      for (let i = 0; i < seq.length; i++) {
        const a = seq[i] * stepAngle;
        x += Math.cos(a);
        y += Math.sin(a);
        pts.push([x, y]);
      }
      return pts;
    };

    const dreamPts = buildPts(dreamTokens);
    const compareLen = hasTruth ? Math.min(truthTokens.length, Math.max(dreamTokens.length, 1)) : 0;
    const truthPts = compareLen > 0 ? buildPts(truthTokens.slice(0, compareLen)) : [];

    let minX = -2;
    let maxX = 2;
    let minY = -2;
    let maxY = 2;
    for (const [x, y] of [...dreamPts, ...truthPts]) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const spanX = Math.max(4, maxX - minX);
    const spanY = Math.max(4, maxY - minY);
    const pad = 22;
    const scale = Math.min((cssW - pad * 2) / spanX, (cssH - pad * 2) / spanY);
    const cx = cssW / 2 - ((minX + maxX) / 2) * scale;
    const cy = cssH / 2 - ((minY + maxY) / 2) * scale;

    const toScreen = (x: number, y: number): [number, number] => [cx + x * scale, cy + y * scale];

    // 1. Траектория истинного продолжения
    if (truthPts.length > 1) {
      ctx.strokeStyle = "rgba(94, 158, 168, 0.75)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      const [sx, sy] = toScreen(truthPts[0][0], truthPts[0][1]);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < truthPts.length; i++) {
        const [px, py] = toScreen(truthPts[i][0], truthPts[i][1]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // 2. Сон трансформера
    if (dreamPts.length > 1) {
      ctx.strokeStyle = "#DF6244";
      ctx.lineWidth = 1.85;
      ctx.beginPath();
      const [sx, sy] = toScreen(dreamPts[0][0], dreamPts[0][1]);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < dreamPts.length; i++) {
        const [px, py] = toScreen(dreamPts[i][0], dreamPts[i][1]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();

      const last = dreamPts[dreamPts.length - 1];
      const [lx, ly] = toScreen(last[0], last[1]);
      ctx.fillStyle = "#DF6244";
      ctx.beginPath();
      ctx.arc(lx, ly, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Старт (0,0)
    const [ox, oy] = toScreen(0, 0);
    ctx.fillStyle = "#EAE4DA";
    ctx.beginPath();
    ctx.arc(ox, oy, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }, [dreamTokens, truthTokens, radix, hasTruth]);

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
        <span className="text-[var(--device-ink-2)]">Двойной 2D Random Walk (из одной точки)</span>
        <div className="flex items-center gap-4 text-[10px]">
          {hasTruth && (
            <span className="flex items-center gap-1.5 text-[#5E9EA8]">
              <span className="inline-block h-0.5 w-3 bg-[#5E9EA8]" />
              Реальное продолжение
            </span>
          )}
          <span className="flex items-center gap-1.5 text-[#DF6244]">
            <span className="inline-block h-0.5 w-3 bg-[#DF6244]" />
            Сон трансформера
          </span>
        </div>
      </div>
      <div className="overflow-hidden rounded-[4px] border border-[var(--device-rule)] bg-[#121110]">
        <canvas ref={canvasRef} className="block h-[230px] w-full" />
      </div>
    </div>
  );
}

export function DreamLabPanel({
  theme,
  isGpuConnected,
  onSendToPiano,
  onGenerateDreamGpu,
}: DreamLabPanelProps) {
  const [seedsBank, setSeedsBank] = useState<DreamSeedsBank | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const [dataset, setDataset] = useState<DreamDatasetId>("true_pi");
  const [selectedSeedId, setSelectedSeedId] = useState<string>("true_pi_start");
  const [customSeedText, setCustomSeedText] = useState<string>("");
  const [freshCryptoSeed, setFreshCryptoSeed] = useState<{
    seedTokens: number[];
    groundTruth: number[];
  } | null>(null);

  const [temperature, setTemperature] = useState<number>(0.0);
  const [topK, setTopK] = useState<number>(0);
  const [targetSteps, setTargetSteps] = useState<number>(180);
  const [speed, setSpeed] = useState<DreamSpeedMode>("fast");
  const [useGpuMode, setUseGpuMode] = useState<boolean>(false);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [generated, setGenerated] = useState<number[]>([]);
  const [rawEntropies, setRawEntropies] = useState<number[]>([]);
  const [tempEntropies, setTempEntropies] = useState<number[]>([]);
  const [lastProbs, setLastProbs] = useState<number[]>([]);
  const [selfTriggers, setSelfTriggers] = useState<DreamSelfTrigger[]>([]);
  const [cycleInfo, setCycleInfo] = useState<DreamCycleInfo | null>(null);
  const [selectedTriggerStep, setSelectedTriggerStep] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [engineUsed, setEngineUsed] = useState<"ONNX WASM" | "Local GPU">("ONNX WASM");
  const [statusError, setStatusError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const rngRef = useRef<() => number>(createMulberry32(42));

  // Ленивая загрузка dream_seeds.json только при открытии вкладки «Сны нейросети»
  useEffect(() => {
    let cancelled = false;
    const rawBase = import.meta.env.BASE_URL || "/";
    const base = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
    fetch(`${base}data/dream_seeds.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<DreamSeedsBank>;
      })
      .then((json) => {
        if (!cancelled) setSeedsBank(json);
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadErr(err instanceof Error ? err.message : "Ошибка загрузки dream_seeds.json");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const datasetSeeds = useMemo(
    () => (seedsBank?.seeds ?? []).filter((s) => s.dataset === dataset),
    [seedsBank, dataset]
  );

  const activeSeedEntry: DreamSeedEntry | null = useMemo(() => {
    if (datasetSeeds.length === 0) return null;
    return datasetSeeds.find((s) => s.id === selectedSeedId) ?? datasetSeeds[0];
  }, [datasetSeeds, selectedSeedId]);

  const vocabSize = dataset === "e_cf" ? 11 : 10;

  const { seedTokens, groundTruth, hasGroundTruth } = useMemo(() => {
    if (customSeedText.trim().length > 0) {
      const cleaned = customSeedText
        .replace(/[\s,]+/g, "")
        .split("")
        .map((ch) => (dataset === "e_cf" && ch === ";" ? 10 : Number.parseInt(ch, 10)))
        .filter((d) => Number.isFinite(d) && d >= 0 && d < vocabSize);
      if (cleaned.length >= 64) {
        return {
          seedTokens: cleaned.slice(0, 64),
          groundTruth: [] as number[],
          hasGroundTruth: false,
        };
      }
    }
    if (dataset === "csprng_backdoor" && freshCryptoSeed) {
      return {
        seedTokens: freshCryptoSeed.seedTokens,
        groundTruth: freshCryptoSeed.groundTruth,
        hasGroundTruth: true,
      };
    }
    return {
      seedTokens: activeSeedEntry?.seed_tokens ?? new Array<number>(64).fill(0),
      groundTruth: activeSeedEntry?.ground_truth ?? ([] as number[]),
      hasGroundTruth: (activeSeedEntry?.ground_truth?.length ?? 0) > 0,
    };
  }, [activeSeedEntry, customSeedText, dataset, freshCryptoSeed, vocabSize]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsRunning(false);
  }, []);

  const resetDream = useCallback(() => {
    stopGeneration();
    setGenerated([]);
    setRawEntropies([]);
    setTempEntropies([]);
    setLastProbs([]);
    setSelfTriggers([]);
    setCycleInfo(null);
    setSelectedTriggerStep(null);
    setElapsedMs(0);
    setStatusError(null);
    rngRef.current = createMulberry32(42 + Math.round(temperature * 100));
  }, [stopGeneration, temperature]);

  const handleSelectModel = (nextDs: DreamDatasetId) => {
    resetDream();
    setDataset(nextDs);
    setFreshCryptoSeed(null);
    setCustomSeedText("");
    const first = seedsBank?.seeds.find((s) => s.dataset === nextDs);
    if (first) setSelectedSeedId(first.id);
    if (nextDs !== "csprng_backdoor" && targetSteps > 500 && !useGpuMode) {
      setTargetSteps(300);
    }
  };

  const makeFreshCrypto = () => {
    resetDream();
    const raw = generateCryptoRandomDigits(64 + 500);
    const bd = applyCustomBackdoorBrowser(raw, {
      lagA: 24,
      lagB: 7,
      prevThreshold: 4,
      coeffA: 4,
      coeffB: 1,
      coeffConst: 5,
    });
    setFreshCryptoSeed({
      seedTokens: bd.digits.slice(0, 64),
      groundTruth: bd.digits.slice(64, 564),
    });
  };

  const applyGpuDreamResult = (res: DreamGpuResponse) => {
    setGenerated(res.generated_tokens);
    setRawEntropies(res.step_raw_entropies);
    setTempEntropies(res.step_temp_entropies);
    setLastProbs(res.last_probs);
    setSelfTriggers(res.self_triggers);
    setCycleInfo(res.cycle_info);
    setElapsedMs(Math.round(res.latency_ms));
    setEngineUsed("Local GPU");
    if (res.self_triggers.length > 0) {
      setSelectedTriggerStep(res.self_triggers[res.self_triggers.length - 1].step);
    }
  };

  const handleRunOrPause = async () => {
    if (isRunning) {
      stopGeneration();
      return;
    }
    setStatusError(null);

    if (useGpuMode && isGpuConnected) {
      setIsRunning(true);
      try {
        const res = await onGenerateDreamGpu({
          dataset,
          seed_tokens: seedTokens,
          steps: targetSteps,
          temperature,
          top_k: topK,
          random_seed: 42 + Math.round(temperature * 100),
        });
        applyGpuDreamResult(res);
      } catch (err) {
        setStatusError(err instanceof Error ? err.message : "Ошибка Local GPU генерации");
      } finally {
        setIsRunning(false);
      }
      return;
    }

    const startFrom = generated.length >= targetSteps ? [] : generated;
    if (generated.length >= targetSteps) {
      resetDream();
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setIsRunning(true);
    setEngineUsed("ONNX WASM");
    const tStart = performance.now();

    try {
      await streamDreamGeneration({
        dataset,
        seedTokens,
        initialGenerated: startFrom,
        targetSteps,
        temperature,
        topK,
        speed,
        rng: rngRef.current,
        signal: controller.signal,
        onBatch: (outputs) => {
          setGenerated((prev) => {
            const next = [...prev, ...outputs.map((o) => o.token)];
            const cyc = detectRepetitionCycle(next);
            if (cyc) setCycleInfo((old) => old ?? cyc);
            return next;
          });
          setRawEntropies((prev) => [...prev, ...outputs.map((o) => o.rawEntropyBits)]);
          setTempEntropies((prev) => [...prev, ...outputs.map((o) => o.tempEntropyBits)]);
          const lastOut = outputs[outputs.length - 1];
          setLastProbs(lastOut.effectiveProbs);
          const newTrigs = outputs.map((o) => o.selfTrigger).filter((t): t is DreamSelfTrigger => t !== null);
          if (newTrigs.length > 0) {
            setSelfTriggers((prev) => [...prev, ...newTrigs]);
            setSelectedTriggerStep(newTrigs[newTrigs.length - 1].step);
          }
          setElapsedMs(Math.round(performance.now() - tStart));
        },
      });
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Ошибка ONNX-генерации");
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setIsRunning(false);
    }
  };

  const handleStepOnce = async () => {
    if (isRunning) return;
    setStatusError(null);
    setEngineUsed("ONNX WASM");
    try {
      const history = [...seedTokens, ...generated];
      const out = await runDreamSingleStep(dataset, history, generated, temperature, topK, rngRef.current);
      setGenerated((prev) => {
        const next = [...prev, out.token];
        const cyc = detectRepetitionCycle(next);
        if (cyc) setCycleInfo((old) => old ?? cyc);
        return next;
      });
      setRawEntropies((prev) => [...prev, out.rawEntropyBits]);
      setTempEntropies((prev) => [...prev, out.tempEntropyBits]);
      setLastProbs(out.effectiveProbs);
      if (out.selfTrigger) {
        const trig = out.selfTrigger;
        setSelfTriggers((prev) => [...prev, trig]);
        setSelectedTriggerStep(trig.step);
      }
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Ошибка шага ONNX");
    }
  };

  // Метрики совпадения с ground_truth
  const matchStats = useMemo(() => {
    if (!hasGroundTruth || generated.length === 0) {
      return { compared: 0, matched: 0, acc: null as number | null, exactPrefix: 0 };
    }
    const n = Math.min(generated.length, groundTruth.length);
    let matched = 0;
    let exactPrefix = 0;
    let prefixOpen = true;
    for (let i = 0; i < n; i++) {
      if (generated[i] === groundTruth[i]) {
        matched += 1;
        if (prefixOpen) exactPrefix += 1;
      } else {
        prefixOpen = false;
      }
    }
    return {
      compared: n,
      matched,
      acc: n > 0 ? matched / n : null,
      exactPrefix,
    };
  }, [generated, groundTruth, hasGroundTruth]);

  const selfTrigMap = useMemo(() => {
    const m = new Map<number, DreamSelfTrigger>();
    for (const st of selfTriggers) m.set(st.step, st);
    return m;
  }, [selfTriggers]);

  const activeTriggerObj =
    selectedTriggerStep !== null ? selfTrigMap.get(selectedTriggerStep) ?? null : null;

  const shannonLimit = Math.log2(vocabSize);
  const currentEntropy = rawEntropies.length > 0 ? rawEntropies[rawEntropies.length - 1] : shannonLimit;
  const tokPerSec = elapsedMs > 50 && generated.length > 0 ? (generated.length / (elapsedMs / 1000)).toFixed(1) : "—";

  // Данные для научного графика под прибором (Plotly)
  const chartData = useMemo((): Data[] => {
    if (rawEntropies.length < 10) return [];
    const xs = rawEntropies.map((_, i) => i + 1);
    const traces: Data[] = [
      {
        x: xs,
        y: rawEntropies,
        type: "scatter",
        mode: "lines",
        name: "Энтропия модели H(P)",
        line: { color: "#BF4328", width: 2 },
      },
      {
        x: [1, xs.length],
        y: [shannonLimit, shannonLimit],
        type: "scatter",
        mode: "lines",
        name: `Предел log₂(${vocabSize})`,
        line: { color: "#6E6A63", width: 1.2, dash: "dash" },
      },
    ];

    if (hasGroundTruth) {
      const rollingAcc: Array<number | null> = [];
      const win = 20;
      for (let i = 0; i < generated.length; i++) {
        if (i >= groundTruth.length) {
          rollingAcc.push(null);
          continue;
        }
        const s = Math.max(0, i - win + 1);
        let hits = 0;
        let cnt = 0;
        for (let j = s; j <= i && j < groundTruth.length; j++) {
          cnt += 1;
          if (generated[j] === groundTruth[j]) hits += 1;
        }
        rollingAcc.push(cnt > 0 ? (hits / cnt) * 100 : null);
      }
      traces.push({
        x: xs,
        y: rollingAcc,
        type: "scatter",
        mode: "lines",
        name: "Точность (окно 20)",
        yaxis: "y2",
        line: { color: "#2F7A4D", width: 1.8 },
      });
    }
    return traces;
  }, [rawEntropies, shannonLimit, vocabSize, hasGroundTruth, generated, groundTruth]);

  const chartLayout = useMemo((): Partial<Layout> => {
    return {
      title: { text: "Динамика энтропии шага и совпадения с реальной константой" },
      margin: { l: 52, r: hasGroundTruth ? 54 : 20, t: 46, b: 68 },
      xaxis: { title: { text: "Шаг генерации t" } },
      yaxis: {
        title: { text: "Энтропия H_t (бит)" },
        range: [0, shannonLimit + 0.35],
      },
      ...(hasGroundTruth
        ? {
            yaxis2: {
              title: { text: "Точность (%)" },
              overlaying: "y" as const,
              side: "right" as const,
              range: [0, 105],
              showgrid: false,
            },
          }
        : {}),
      legend: { orientation: "h", y: -0.28, x: 0 },
    };
  }, [hasGroundTruth, shannonLimit]);

  const fullDreamStream = useMemo(() => [...seedTokens, ...generated], [seedTokens, generated]);
  const triggerMaskForPiano = useMemo(() => {
    const mask = new Array<number>(fullDreamStream.length).fill(0);
    for (const tr of selfTriggers) {
      const idx = seedTokens.length + tr.step;
      if (idx < mask.length) mask[idx] = 1;
    }
    return mask;
  }, [fullDreamStream.length, seedTokens.length, selfTriggers]);

  return (
    <div className="lg:col-span-12 space-y-8">
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* ЛЕВАЯ УПРАВЛЯЮЩАЯ КОЛОНКА НА БУМАГЕ */}
        <section className="lg:col-span-5 space-y-5">
          <div>
            <div className="mb-2 font-sans text-xs font-medium text-[var(--ink-2)]">
              1. Выберите обученную модель DigitTransformer
            </div>
            <div className="grid grid-cols-2 gap-2">
              {MODEL_CARDS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  data-testid={`dream-model-${m.id}`}
                  onClick={() => handleSelectModel(m.id)}
                  className={`rounded-[4px] border px-3 py-2 text-left transition-colors ${
                    dataset === m.id
                      ? "border-[var(--sig-h1)] bg-[var(--bg-alt)] text-[var(--ink)]"
                      : "border-[var(--rule)] text-[var(--ink-2)] hover:border-[var(--rule-strong)]"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1 font-mono text-xs font-medium text-[var(--ink)]">
                    <span>{m.title}</span>
                    <span className="text-[10px] text-[var(--ink-muted)]">{m.badge}</span>
                  </div>
                  <div className="mt-0.5 font-sans text-[11px] leading-snug text-[var(--ink-muted)]">
                    {m.subtitle}
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2 border-t border-[var(--rule)] pt-4">
            <div className="flex items-center justify-between gap-2">
              <span className="font-sans text-xs font-medium text-[var(--ink-2)]">
                2. Каноническая затравка (64 токена)
              </span>
              {dataset === "csprng_backdoor" && (
                <button
                  type="button"
                  onClick={makeFreshCrypto}
                  className="rounded-[3px] border border-[var(--rule-strong)] px-2 py-0.5 font-sans text-[11px] text-[var(--ink)] hover:border-[var(--sig-h1)]"
                >
                  Свежий крипто-шум
                </button>
              )}
            </div>
            {loadErr ? (
              <div className="font-mono text-xs text-[var(--sig-h1)]">{loadErr}</div>
            ) : (
              <select
                data-testid="dream-seed-select"
                value={activeSeedEntry?.id ?? ""}
                onChange={(e) => {
                  resetDream();
                  setFreshCryptoSeed(null);
                  setSelectedSeedId(e.target.value);
                }}
                className="desk-select w-full py-1.5 text-xs"
              >
                {datasetSeeds.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            )}
            <details className="rounded-[4px] border border-[var(--rule)] bg-[var(--bg-alt)]/40 px-3 py-1.5 text-xs">
              <summary className="cursor-pointer font-sans text-[11px] text-[var(--ink-2)] hover:text-[var(--ink)]">
                Изменить затравку вручную (64 символа)
              </summary>
              <textarea
                rows={2}
                value={customSeedText}
                onChange={(e) => {
                  resetDream();
                  setCustomSeedText(e.target.value);
                }}
                placeholder={
                  dataset === "e_cf"
                    ? "Введите >= 64 символов из 0..9 и ';'..."
                    : "Введите >= 64 цифр 0..9..."
                }
                className="mt-2 w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] p-2 font-mono text-xs text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
              />
            </details>
          </div>

          {/* Температура tau + параметры декодирования */}
          <div className="space-y-3 border-t border-[var(--rule)] pt-4">
            <div>
              <div className="flex items-center justify-between font-sans text-xs">
                <span className="font-medium text-[var(--ink-2)]">Температура τ</span>
                <span className="font-mono text-xs font-medium text-[var(--ink)]">
                  {temperature.toFixed(2)} {temperature <= 0.02 ? "(Greedy)" : ""}
                </span>
              </div>
              <input
                type="range"
                data-testid="dream-tau-slider"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setTemperature(val);
                  rngRef.current = createMulberry32(42 + Math.round(val * 100));
                }}
                className="mt-1.5 w-full accent-[var(--sig-h1)]"
              />
              <div className="mt-1.5 grid grid-cols-4 gap-1.5">
                {TAU_NOTCHES.map((n) => (
                  <button
                    key={n.val}
                    type="button"
                    data-testid={`dream-tau-${n.val.toFixed(2)}`}
                    onClick={() => {
                      setTemperature(n.val);
                      rngRef.current = createMulberry32(42 + Math.round(n.val * 100));
                    }}
                    className={`rounded-[3px] border px-1.5 py-1 font-mono text-[10px] transition-colors ${
                      Math.abs(temperature - n.val) < 0.03
                        ? "border-[var(--sig-h1)] bg-[var(--bg-alt)] text-[var(--ink)]"
                        : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)]"
                    }`}
                  >
                    {n.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <label className="block font-sans text-[11px] text-[var(--ink-2)]">
                Top-k
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="desk-select mt-1 w-full py-1 text-xs"
                >
                  <option value={0}>Выкл</option>
                  <option value={2}>Top-2</option>
                  <option value={3}>Top-3</option>
                  <option value={5}>Top-5</option>
                </select>
              </label>
              <label className="block font-sans text-[11px] text-[var(--ink-2)]">
                Длина сна
                <select
                  value={targetSteps}
                  onChange={(e) => setTargetSteps(Number(e.target.value))}
                  className="desk-select mt-1 w-full py-1 text-xs"
                >
                  <option value={80}>80 шагов</option>
                  <option value={180}>180 шагов</option>
                  <option value={300}>300 шагов</option>
                  <option value={500}>500 шагов</option>
                  {isGpuConnected && useGpuMode && <option value={1000}>1000 (GPU)</option>}
                </select>
              </label>
              <label className="block font-sans text-[11px] text-[var(--ink-2)]">
                Скорость
                <select
                  value={speed}
                  onChange={(e) => setSpeed(e.target.value as DreamSpeedMode)}
                  className="desk-select mt-1 w-full py-1 text-xs"
                >
                  <option value="step">Пошагово</option>
                  <option value="smooth">Плавно</option>
                  <option value="fast">Быстро</option>
                </select>
              </label>
            </div>

            {isGpuConnected && (
              <label className="flex items-center gap-2 font-sans text-xs text-[var(--ink-2)]">
                <input
                  type="checkbox"
                  checked={useGpuMode}
                  onChange={(e) => {
                    stopGeneration();
                    setUseGpuMode(e.target.checked);
                  }}
                />
                Считать мгновенно на Local GPU (до 1000 шагов)
              </label>
            )}
          </div>

          <div className="flex flex-wrap gap-2 border-t border-[var(--rule)] pt-4">
            <button
              type="button"
              data-testid="dream-run-btn"
              onClick={() => void handleRunOrPause()}
              className="flex-1 rounded-[4px] bg-[var(--ink)] px-4 py-2 font-sans text-xs font-medium text-[var(--bg)] transition-opacity hover:opacity-90"
            >
              {isRunning
                ? "Пауза"
                : generated.length > 0 && generated.length < targetSteps
                  ? "Продолжить сон"
                  : "Запустить сон"}
            </button>
            <button
              type="button"
              data-testid="dream-step-btn"
              disabled={isRunning}
              onClick={() => void handleStepOnce()}
              className="rounded-[4px] border border-[var(--rule-strong)] px-3 py-2 font-mono text-xs text-[var(--ink)] hover:border-[var(--ink)] disabled:opacity-40"
            >
              +1 шаг
            </button>
            <button
              type="button"
              data-testid="dream-reset-btn"
              onClick={resetDream}
              className="rounded-[4px] border border-[var(--rule-strong)] px-3 py-2 font-sans text-xs text-[var(--ink-2)] hover:text-[var(--ink)]"
            >
              Сброс
            </button>
          </div>

          {statusError && (
            <div className="font-mono text-xs text-[var(--sig-h1)]">{statusError}</div>
          )}
        </section>

        {/* ПРАВЫЙ ПРИБОР: КАМЕРА СНОВ */}
        <section className="lg:col-span-7 space-y-6">
          <div className="hardware-device space-y-4 p-5">
            {/* Телеметрия прибора */}
            <div className="grid grid-cols-2 divide-y divide-[var(--device-rule)] rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] sm:grid-cols-4 sm:divide-x sm:divide-y-0">
              <div className="p-3">
                <div className="font-sans text-[10px] text-[var(--device-ink-muted)]">
                  Сгенерировано ({engineUsed})
                </div>
                <div
                  data-testid="dream-progress-counter"
                  className="mt-0.5 font-mono text-sm tabular-nums text-[var(--device-ink)]"
                >
                  {generated.length} / {targetSteps}
                </div>
                <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
                  {tokPerSec !== "—" ? `${tokPerSec} ток/с` : "готово к запуску"}
                </div>
              </div>
              <div className="p-3">
                <div className="font-sans text-[10px] text-[var(--device-ink-muted)]">
                  Энтропия шага H_t
                </div>
                <div className="mt-0.5 font-mono text-sm tabular-nums text-[var(--device-ink)]">
                  {currentEntropy.toFixed(3)} бит
                </div>
                <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
                  предел {shannonLimit.toFixed(2)} b
                </div>
              </div>
              <div className="p-3">
                <div className="font-sans text-[10px] text-[var(--device-ink-muted)]">
                  Совпадение с истиной
                </div>
                <div
                  data-testid="dream-truth-acc"
                  className={`mt-0.5 font-mono text-sm tabular-nums ${
                    matchStats.acc !== null && matchStats.acc > 0.65
                      ? "text-[#4AA874]"
                      : "text-[var(--device-ink)]"
                  }`}
                >
                  {matchStats.acc !== null ? `${(matchStats.acc * 100).toFixed(1)}%` : "—"}
                </div>
                <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
                  {hasGroundTruth ? `точный префикс: ${matchStats.exactPrefix}` : "своя затравка"}
                </div>
              </div>
              <div className="p-3">
                <div className="font-sans text-[10px] text-[var(--device-ink-muted)]">
                  Циклы / Триггеры
                </div>
                <div
                  data-testid="dream-detector-badge"
                  className={`mt-0.5 font-mono text-xs tabular-nums ${
                    cycleInfo || selfTriggers.length > 0
                      ? "text-[var(--dev-signal)]"
                      : "text-[var(--device-ink-2)]"
                  }`}
                >
                  {dataset === "csprng_backdoor"
                    ? selfTriggers.length > 0
                      ? `Само-триг.: ${selfTriggers.filter((s) => s.matched).length}/${selfTriggers.length}`
                      : "Триггеров нет"
                    : cycleInfo
                      ? `Цикл P=${cycleInfo.period} (#${cycleInfo.start_step})`
                      : "Циклов нет"}
                </div>
                <div className="truncate font-mono text-[10px] text-[var(--device-ink-muted)]">
                  {cycleInfo ? `мотив: ${cycleInfo.motif.slice(0, 12)}` : "поиск периода 1..48"}
                </div>
              </div>
            </div>

            {/* Дуги само-триггеров для csprng_backdoor */}
            {dataset === "csprng_backdoor" && activeTriggerObj && (
              <div className="rounded-[4px] border border-[var(--dev-signal)]/40 bg-[var(--device-well)] px-3 py-2 font-mono text-[11px]">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[var(--dev-signal)]">
                    Самоиндуцированный триггер на шаге #{activeTriggerObj.step + 1} (лаги t−24, t−7)
                  </span>
                  <span className="text-[var(--device-ink)]">
                    (4×{activeTriggerObj.a} + {activeTriggerObj.b} + 5) mod 10 ={" "}
                    {activeTriggerObj.expected} · модель выдала {activeTriggerObj.chosen}{" "}
                    {activeTriggerObj.matched ? "✓" : "×"}
                  </span>
                </div>
                <svg viewBox="0 0 360 38" className="mt-1.5 h-[38px] w-full" aria-hidden="true">
                  <path
                    d="M 42 20 Q 180 2 318 20"
                    fill="none"
                    stroke="#DF6244"
                    strokeWidth="1.5"
                  />
                  <path
                    d="M 226 20 Q 272 7 318 20"
                    fill="none"
                    stroke="#D9A04A"
                    strokeWidth="1.4"
                  />
                  <text x="26" y="34" fill="#DF6244" fontSize="9">
                    t−24={activeTriggerObj.a}
                  </text>
                  <text x="210" y="34" fill="#D9A04A" fontSize="9">
                    t−7={activeTriggerObj.b}
                  </text>
                  <text x="322" y="23" fill="#EAE4DA" fontSize="9">
                    → {activeTriggerObj.chosen}
                  </text>
                </svg>
              </div>
            )}

            {/* Лента Затравка -> Сон */}
            <div
              data-testid="dream-token-tape"
              className="hardware-well max-h-[156px] overflow-y-auto rounded-[4px] p-3 font-mono text-xs leading-6"
            >
              <span className="text-[var(--device-ink-muted)]">
                {seedTokens.map(formatTok).join("")}
              </span>
              <span className="mx-1.5 border-l border-[var(--dev-signal)] pl-1.5 text-[10px] text-[var(--dev-signal)]">
                Сон →
              </span>
              {generated.length === 0 ? (
                <span className="text-[var(--device-ink-muted)]">ожидание запуска…</span>
              ) : (
                generated.map((tok, i) => {
                  const isMatch = hasGroundTruth && i < groundTruth.length && tok === groundTruth[i];
                  const inCycle = cycleInfo !== null && i >= cycleInfo.start_step;
                  const trig = selfTrigMap.get(i);
                  return (
                    <span
                      key={i}
                      onClick={() => trig && setSelectedTriggerStep(i)}
                      title={
                        trig
                          ? `Само-триггер шаг #${i + 1}: правило ${trig.expected}, сон ${trig.chosen}`
                          : hasGroundTruth && i < groundTruth.length
                            ? `Шаг ${i + 1}: сон ${formatTok(tok)} / факт ${formatTok(groundTruth[i])}`
                            : `Шаг ${i + 1}`
                      }
                      className={`inline-block px-[0.5px] ${
                        trig
                          ? "cursor-pointer rounded-[2px] bg-[#DF6244]/25 font-semibold text-[#DF6244]"
                          : inCycle
                            ? "bg-[#D97706]/18 text-[#F2D5A7]"
                            : "text-[#EAE4DA]"
                      } ${isMatch ? "border-b-2 border-[#2F7A4D]" : ""}`}
                    >
                      {formatTok(tok)}
                    </span>
                  );
                })
              )}
            </div>

            {/* Мини-индикатор вероятностей последнего шага */}
            <div className="space-y-1">
              <div className="flex items-center justify-between font-mono text-[10px] text-[var(--device-ink-muted)]">
                <span>Мысль модели на текущем шаге P_τ(d)</span>
                <span>
                  {tempEntropies.length > 0
                    ? `H_τ = ${tempEntropies[tempEntropies.length - 1].toFixed(2)} бит`
                    : "нет шагов"}
                </span>
              </div>
              <div
                className="grid gap-1 rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] px-2 pt-2 pb-1"
                style={{ gridTemplateColumns: `repeat(${vocabSize}, minmax(0, 1fr))` }}
              >
                {Array.from({ length: vocabSize }, (_, d) => {
                  const p = lastProbs[d] ?? 1 / vocabSize;
                  const h = Math.max(2, Math.round(p * 28));
                  return (
                    <div key={d} className="flex flex-col items-center gap-0.5">
                      <div className="flex h-[30px] w-full items-end justify-center">
                        <div
                          className="w-full max-w-[14px] rounded-t-[1px]"
                          style={{
                            height: `${h}px`,
                            backgroundColor: p > 0.35 ? "#DF6244" : "rgba(232,226,214,0.45)",
                          }}
                          title={`${formatTok(d)}: ${(p * 100).toFixed(1)}%`}
                        />
                      </div>
                      <span className="font-mono text-[9px] text-[var(--device-ink-muted)]">
                        {formatTok(d)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Двойной 2D Random Walk */}
            <DualDreamWalkCanvas
              dreamTokens={generated}
              truthTokens={groundTruth}
              radix={vocabSize}
              hasTruth={hasGroundTruth}
            />

            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
                {activeSeedEntry ? activeSeedEntry.description : "Пользовательский контекст 64 токена"}
              </div>
              <button
                type="button"
                data-testid="dream-send-piano-btn"
                disabled={generated.length === 0}
                onClick={() =>
                  onSendToPiano({
                    label: `Сон ${dataset} (τ=${temperature.toFixed(2)})`,
                    digits: fullDreamStream,
                    mask: dataset === "csprng_backdoor" ? triggerMaskForPiano : null,
                  })
                }
                className="rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] px-3 py-1.5 font-sans text-xs font-medium text-[var(--device-ink)] hover:border-[var(--dev-signal)] disabled:opacity-40"
              >
                Послушать сон на Пианино π
              </button>
            </div>
          </div>

          {/* Научный график прямо на тёплой бумаге под прибором */}
          {chartData.length > 0 && (
            <div className="pt-2">
              <PlotlyChart data={chartData} layout={chartLayout} theme={theme} height={310} />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
