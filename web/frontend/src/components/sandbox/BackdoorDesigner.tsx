import { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import type { BackdoorTrainResponse } from "../../types/lab";
import {
  analyzeCustomBackdoorBrowser,
  applyCustomBackdoorBrowser,
  generateCryptoRandomDigits,
  type CustomBackdoorParams,
} from "../../utils/browserMath";
import { inferBatchWindows, type OnnxBatchResult } from "../../utils/onnxEngine";

interface BackdoorDesignerProps {
  isGpuConnected: boolean;
  onOpenGpuModal: () => void;
  onSequenceGenerated: (payload: { label: string; digits: number[]; mask: number[] }) => void;
  onSendToPiano: (payload: { label: string; digits: number[]; mask?: number[] | null }) => void;
  onTrainBackdoorGpu: (payload: {
    lag_a: number;
    lag_b: number;
    prev_threshold: number;
    coeff_a: number;
    coeff_b: number;
    coeff_const: number;
    length?: number;
    epochs?: number;
    seed?: number;
  }) => Promise<BackdoorTrainResponse>;
}

interface PresetSpec {
  id: string;
  label: string;
  note: string;
  params: CustomBackdoorParams;
}

const PRESETS: PresetSpec[] = [
  {
    id: "v2",
    label: "Каноническая V2",
    note: "t-24, t-7 · биекция",
    params: { lagA: 24, lagB: 7, prevThreshold: 4, coeffA: 4, coeffB: 1, coeffConst: 5 },
  },
  {
    id: "v1",
    label: "Сломанная V1",
    note: "gcd(2,10)=2 · только нечётные",
    params: { lagA: 24, lagB: 7, prevThreshold: 4, coeffA: 2, coeffB: 0, coeffConst: 5 },
  },
  {
    id: "far",
    label: "Дальние лаги t-31, t-11",
    note: "требует GPU-обучения",
    params: { lagA: 31, lagB: 11, prevThreshold: 4, coeffA: 4, coeffB: 1, coeffConst: 5 },
  },
  {
    id: "near",
    label: "Ближние лаги t-6, t-2",
    note: "короткая память",
    params: { lagA: 6, lagB: 2, prevThreshold: 4, coeffA: 4, coeffB: 1, coeffConst: 5 },
  },
];

function KatexFormula({ tex }: { tex: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.innerHTML = katex.renderToString(tex, {
      displayMode: false,
      throwOnError: false,
    });
  }, [tex]);
  return <div ref={ref} className="overflow-x-auto py-1 text-xs text-[var(--ink)]" />;
}

/** Рисует аппаратный SVG-профиль внимания по лагам t-1..t-64 и выделяет заданные lag_a / lag_b */
function LagAttentionPlot({
  profile,
  lagA,
  lagB,
  title,
}: {
  profile: number[];
  lagA: number;
  lagB: number;
  title: string;
}) {
  const maxVal = Math.max(...profile, 0.05);
  return (
    <div className="space-y-2 rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
        <span className="text-[var(--device-ink-2)]">{title}</span>
        <span className="text-[var(--device-ink-muted)]">
          целевые лаги: <span className="text-[var(--dev-signal)]">t-{lagA}</span>,{" "}
          <span className="text-[var(--dev-signal)]">t-{lagB}</span>
        </span>
      </div>
      <svg viewBox="0 0 640 84" className="h-[84px] w-full" aria-label={title}>
        {profile.map((w, idx) => {
          const lag = 64 - idx;
          const isTarget = lag === lagA || lag === lagB;
          const h = Math.max(2, (w / maxVal) * 58);
          const x = idx * 10 + 1;
          const y = 66 - h;
          return (
            <g key={lag}>
              <rect
                x={x}
                y={y}
                width={7}
                height={h}
                rx={1}
                fill={isTarget ? "var(--dev-signal)" : "rgba(232,226,214,0.34)"}
              />
              {isTarget && (
                <text
                  x={x + 3.5}
                  y={79}
                  textAnchor="middle"
                  fill="var(--dev-signal)"
                  fontSize="9"
                  fontFamily="JetBrains Mono, monospace"
                >
                  -{lag}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function BackdoorDesigner({
  isGpuConnected,
  onOpenGpuModal,
  onSequenceGenerated,
  onSendToPiano,
  onTrainBackdoorGpu,
}: BackdoorDesignerProps) {
  const [activePreset, setActivePreset] = useState<string>("v2");
  const [params, setParams] = useState<CustomBackdoorParams>(PRESETS[0].params);
  const [streamLength, setStreamLength] = useState<number>(3000);
  const [baseNoise, setBaseNoise] = useState<number[]>(() => generateCryptoRandomDigits(3000));

  const [onnxStatus, setOnnxStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [onnxProgress, setOnnxProgress] = useState<{ done: number; total: number } | null>(null);
  const [onnxOut, setOnnxOut] = useState<{
    batch: OnnxBatchResult;
    targetedAcc: number | null;
    targetedCount: number;
    avgTrigConf: number | null;
    avgNoiseConf: number | null;
  } | null>(null);
  const [onnxError, setOnnxError] = useState<string | null>(null);

  const [gpuBusy, setGpuBusy] = useState(false);
  const [gpuResult, setGpuResult] = useState<BackdoorTrainResponse | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);

  const generated = useMemo(() => {
    const res = applyCustomBackdoorBrowser(baseNoise, params);
    const audit = analyzeCustomBackdoorBrowser(res.digits, res.mask, params);
    return { ...res, audit };
  }, [baseNoise, params]);

  const ruleLabel = `Backdoor (t-${params.lagA}, t-${params.lagB})`;

  useEffect(() => {
    onSequenceGenerated({
      label: ruleLabel,
      digits: generated.digits,
      mask: generated.mask,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generated.digits, generated.mask, ruleLabel]);

  const regenerateNoise = (lenVal = streamLength) => {
    setBaseNoise(generateCryptoRandomDigits(lenVal));
    setOnnxOut(null);
    setOnnxStatus("idle");
    setGpuResult(null);
  };

  const applyPreset = (p: PresetSpec) => {
    setActivePreset(p.id);
    setParams(p.params);
    setOnnxOut(null);
    setOnnxStatus("idle");
    setGpuResult(null);
  };

  const updateParam = <K extends keyof CustomBackdoorParams>(key: K, raw: number) => {
    setActivePreset("custom");
    setParams((prev) => {
      const next = { ...prev, [key]: raw };
      if (key === "lagA" && next.lagA === next.lagB) {
        next.lagB = next.lagA === 48 ? 47 : next.lagA + 1;
      } else if (key === "lagB" && next.lagB === next.lagA) {
        next.lagA = next.lagB === 48 ? 47 : next.lagB + 1;
      }
      return next;
    });
    setOnnxOut(null);
    setGpuResult(null);
  };

  const runBrowserOnnxScan = async () => {
    if (generated.digits.length < 70) return;
    setOnnxStatus("running");
    setOnnxError(null);
    try {
      const startPos = 64;
      const maxWin = Math.min(256, generated.digits.length - startPos);
      const windows: number[][] = [];
      const targets: number[] = [];
      const trigFlags: number[] = [];
      for (let i = 0; i < maxWin; i++) {
        const t = startPos + i;
        windows.push(generated.digits.slice(t - 64, t));
        targets.push(generated.digits[t]);
        trigFlags.push(generated.mask[t]);
      }
      const res = await inferBatchWindows("csprng_backdoor", windows, targets, {
        batchSize: 32,
        baseOffset: startPos,
        onProgress: (done, total) => setOnnxProgress({ done, total }),
      });

      let trigHit = 0;
      let trigN = 0;
      let trigConfSum = 0;
      let noiseConfSum = 0;
      let noiseN = 0;
      for (let i = 0; i < maxWin; i++) {
        const hit = res.predictions[i] === targets[i] ? 1 : 0;
        if (trigFlags[i] === 1) {
          trigHit += hit;
          trigConfSum += res.confidences[i];
          trigN += 1;
        } else {
          noiseConfSum += res.confidences[i];
          noiseN += 1;
        }
      }
      setOnnxOut({
        batch: res,
        targetedAcc: trigN > 0 ? trigHit / trigN : null,
        targetedCount: trigN,
        avgTrigConf: trigN > 0 ? trigConfSum / trigN : null,
        avgNoiseConf: noiseN > 0 ? noiseConfSum / noiseN : null,
      });
      setOnnxStatus("done");
    } catch (err) {
      setOnnxError(err instanceof Error ? err.message : "Ошибка ONNX-инференса");
      setOnnxStatus("error");
    } finally {
      setOnnxProgress(null);
    }
  };

  const runGpuTrain = async () => {
    if (!isGpuConnected) {
      onOpenGpuModal();
      return;
    }
    setGpuBusy(true);
    setGpuError(null);
    try {
      const out = await onTrainBackdoorGpu({
        lag_a: params.lagA,
        lag_b: params.lagB,
        prev_threshold: params.prevThreshold,
        coeff_a: params.coeffA,
        coeff_b: params.coeffB,
        coeff_const: params.coeffConst,
        length: 32_000,
        epochs: 18,
        seed: 42,
      });
      setGpuResult(out);
    } catch (err) {
      setGpuError(err instanceof Error ? err.message : "Ошибка GPU-обучения");
    } finally {
      setGpuBusy(false);
    }
  };

  const formulaTex = `((x_{t-${params.lagA}}+x_{t-${params.lagB}})\\bmod 10=0 \\land x_{t-1}<${params.prevThreshold}) \\Rightarrow x_t=(${params.coeffA}x_{t-${params.lagA}}+${params.coeffB}x_{t-${params.lagB}}+${params.coeffConst})\\bmod 10`;
  const { audit } = generated;
  const maxCondCount = Math.max(1, ...audit.conditionalCounts);
  const isCanonicalLags =
    (params.lagA === 24 && params.lagB === 7) || (params.lagA === 7 && params.lagB === 24);

  return (
    <div className="space-y-5 pt-2">
      {/* Пресеты */}
      <div>
        <div className="mb-2 font-sans text-xs font-medium text-[var(--ink-2)]">
          Пресеты закладок
        </div>
        <div className="grid grid-cols-2 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              data-testid={`backdoor-preset-${p.id}`}
              onClick={() => applyPreset(p)}
              className={`rounded-[4px] border px-2.5 py-2 text-left transition-colors ${
                activePreset === p.id
                  ? "border-[var(--sig-h1)] bg-[var(--bg-alt)] text-[var(--ink)]"
                  : "border-[var(--rule)] text-[var(--ink-2)] hover:border-[var(--rule-strong)]"
              }`}
            >
              <div className="font-sans text-xs font-medium">{p.label}</div>
              <div className="mt-0.5 font-mono text-[10px] text-[var(--ink-muted)]">{p.note}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Скомпилированная формула и биективность */}
      <div className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)]/70 p-3 space-y-2">
        <KatexFormula tex={formulaTex} />
        <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
          <span className="text-[var(--ink-2)]">
            gcd(|{params.coeffA}−{params.coeffB}|, 10) = {audit.bijectivity.gcd}
          </span>
          <span
            data-testid="bijectivity-badge"
            className={`rounded-[3px] px-1.5 py-0.5 ${
              audit.bijectivity.isBijective
                ? "bg-[var(--sig-gpu)]/15 text-[var(--sig-gpu)]"
                : "bg-[var(--sig-h1)]/15 text-[var(--sig-h1)]"
            }`}
          >
            {audit.bijectivity.isBijective
              ? "Биекция (10 цифр) — χ² слеп"
              : `Вырождение (${audit.bijectivity.reachableDigits.length} цифр) — χ² ловит!`}
          </span>
        </div>
        <div className="font-mono text-[10px] text-[var(--ink-muted)]">
          Достижимые цифры на триггере: &#123;{audit.bijectivity.reachableDigits.join(", ")}&#125;
        </div>
      </div>

      {/* Регуляторы параметров */}
      <div className="grid grid-cols-2 gap-3">
        <label className="block text-xs text-[var(--ink-2)]">
          <div className="flex justify-between font-mono text-[11px]">
            <span>Лаг A (t−A)</span>
            <span className="text-[var(--ink)]">{params.lagA}</span>
          </div>
          <input
            type="range"
            min={2}
            max={48}
            value={params.lagA}
            onChange={(e) => updateParam("lagA", Number(e.target.value))}
            className="mt-1 w-full accent-[var(--sig-h1)]"
          />
        </label>
        <label className="block text-xs text-[var(--ink-2)]">
          <div className="flex justify-between font-mono text-[11px]">
            <span>Лаг B (t−B)</span>
            <span className="text-[var(--ink)]">{params.lagB}</span>
          </div>
          <input
            type="range"
            min={2}
            max={48}
            value={params.lagB}
            onChange={(e) => updateParam("lagB", Number(e.target.value))}
            className="mt-1 w-full accent-[var(--sig-h1)]"
          />
        </label>
      </div>

      <div className="grid grid-cols-4 gap-2">
        <label className="block text-[11px] text-[var(--ink-2)]">
          x[t−1] &lt; T
          <input
            type="number"
            min={1}
            max={9}
            value={params.prevThreshold}
            onChange={(e) => updateParam("prevThreshold", Math.max(1, Math.min(9, Number(e.target.value) || 4)))}
            className="mt-1 w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-2 py-1 font-mono text-xs text-[var(--ink)]"
          />
        </label>
        <label className="block text-[11px] text-[var(--ink-2)]">
          k (x[t−A])
          <input
            type="number"
            min={0}
            max={9}
            value={params.coeffA}
            onChange={(e) => updateParam("coeffA", Math.max(0, Math.min(9, Number(e.target.value) || 0)))}
            className="mt-1 w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-2 py-1 font-mono text-xs text-[var(--ink)]"
          />
        </label>
        <label className="block text-[11px] text-[var(--ink-2)]">
          m (x[t−B])
          <input
            type="number"
            min={0}
            max={9}
            value={params.coeffB}
            onChange={(e) => updateParam("coeffB", Math.max(0, Math.min(9, Number(e.target.value) || 0)))}
            className="mt-1 w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-2 py-1 font-mono text-xs text-[var(--ink)]"
          />
        </label>
        <label className="block text-[11px] text-[var(--ink-2)]">
          сдвиг s
          <input
            type="number"
            min={0}
            max={9}
            value={params.coeffConst}
            onChange={(e) => updateParam("coeffConst", Math.max(0, Math.min(9, Number(e.target.value) || 0)))}
            className="mt-1 w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-2 py-1 font-mono text-xs text-[var(--ink)]"
          />
        </label>
      </div>

      {/* Длина криптопотока + обновление */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 font-mono text-[11px]">
          {[2000, 3000, 5000, 8000].map((lenVal) => (
            <button
              key={lenVal}
              type="button"
              onClick={() => {
                setStreamLength(lenVal);
                regenerateNoise(lenVal);
              }}
              className={`rounded-[3px] border px-2 py-1 ${
                streamLength === lenVal
                  ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--bg)]"
                  : "border-[var(--rule)] text-[var(--ink-2)] hover:border-[var(--rule-strong)]"
              }`}
            >
              {lenVal}
            </button>
          ))}
        </div>
        <button
          type="button"
          data-testid="backdoor-regen-btn"
          onClick={() => regenerateNoise(streamLength)}
          className="rounded-[4px] border border-[var(--rule-strong)] px-2.5 py-1 font-sans text-xs font-medium text-[var(--ink)] hover:border-[var(--sig-h1)]"
        >
          Новый WebCrypto шум
        </button>
      </div>

      {/* Классический аудит: общий vs условный по триггерам */}
      <div className="hardware-device space-y-3 p-3.5 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
          <span className="text-[var(--device-ink)]">
            Триггеров mask==1: {audit.triggerCount} / {generated.digits.length} (
            {(audit.triggerFraction * 100).toFixed(2)}%)
          </span>
          <span
            className={
              audit.maxAcf > audit.acfBonferroniBound
                ? "text-[var(--dev-signal)]"
                : "text-[var(--device-ink-2)]"
            }
          >
            max|ACF|={audit.maxAcf.toFixed(3)} (t−{audit.maxAcfLag}) vs Бонферрони{" "}
            {audit.acfBonferroniBound.toFixed(3)}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-[3px] border border-[var(--device-rule)] bg-[var(--device-well)] p-2.5">
            <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">Общий χ² (весь поток)</div>
            <div className="mt-0.5 font-mono text-sm text-[var(--device-ink)]">
              p = {audit.overallChi2P.toFixed(4)}
            </div>
            <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
              Markov k=1: {(audit.markovK1OverallAcc * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-[3px] border border-[var(--device-rule)] bg-[var(--device-well)] p-2.5">
            <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">Условный χ² (mask==1)</div>
            <div
              className={`mt-0.5 font-mono text-sm ${
                audit.conditionalChi2P < 0.01 ? "text-[var(--dev-signal)]" : "text-[var(--device-ink)]"
              }`}
            >
              p = {audit.conditionalChi2P < 0.0001 ? audit.conditionalChi2P.toExponential(1) : audit.conditionalChi2P.toFixed(4)}
            </div>
            <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
              Markov k=1 (trig):{" "}
              {audit.markovK1TargetedAcc !== null ? `${(audit.markovK1TargetedAcc * 100).toFixed(1)}%` : "—"}
            </div>
          </div>
        </div>

        {/* Условная гистограмма цифр на mask==1 */}
        <div className="space-y-1">
          <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
            Распределение цифр строго в точках закладки (mask == 1):
          </div>
          <div className="grid grid-cols-10 items-end gap-1 rounded-[3px] border border-[var(--device-rule)] bg-[var(--device-well)] px-2 pt-2 pb-1">
            {audit.conditionalCounts.map((cnt, d) => {
              const h = cnt === 0 ? 2 : Math.max(4, Math.round((cnt / maxCondCount) * 28));
              return (
                <div key={d} className="flex flex-col items-center gap-0.5">
                  <div className="flex h-[30px] items-end">
                    <div
                      className="w-2.5 rounded-t-[1px]"
                      style={{
                        height: `${h}px`,
                        backgroundColor:
                          cnt === 0 ? "rgba(217,119,6,0.9)" : "rgba(232,226,214,0.65)",
                      }}
                      title={`цифра ${d}: ${cnt}`}
                    />
                  </div>
                  <span className="font-mono text-[9px] text-[var(--device-ink-muted)]">{d}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Нейросетевой сканер: Browser ONNX + Local GPU Trainer */}
        <div className="space-y-2.5 border-t border-[var(--device-rule)] pt-3">
          {!isCanonicalLags && (
            <div className="rounded-[3px] border border-[var(--dev-signal)]/40 bg-[var(--device-well)] px-2.5 py-2 font-sans text-[11px] text-[var(--device-ink-2)]">
              Встроенный браузерный ONNX обучен под канонические лаги (t−24, t−7). Для произвольных
              лагов (t−{params.lagA}, t−{params.lagB}) обучите свежий детектор на вашей локальной GPU!
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              data-testid="backdoor-onnx-scan-btn"
              disabled={onnxStatus === "running"}
              onClick={runBrowserOnnxScan}
              className="flex-1 rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] px-3 py-2 font-sans text-xs font-medium text-[var(--device-ink)] hover:border-[var(--dev-signal)] disabled:opacity-50"
            >
              {onnxStatus === "running"
                ? `ONNX-скан ${onnxProgress ? `${onnxProgress.done}/${onnxProgress.total}` : "…"}`
                : "Скан в браузере (ONNX V2)"}
            </button>
            <button
              type="button"
              data-testid="backdoor-gpu-train-btn"
              disabled={gpuBusy}
              onClick={runGpuTrain}
              className="flex-1 rounded-[4px] bg-[var(--dev-signal)] px-3 py-2 font-sans text-xs font-medium text-[#151412] hover:opacity-90 disabled:opacity-50"
            >
              {gpuBusy
                ? "Обучение на GPU…"
                : isGpuConnected
                  ? "Обучить детектор на GPU (~10–15с)"
                  : "Обучить на GPU (подключить мост)"}
            </button>
          </div>

          {onnxError && (
            <div className="font-mono text-[11px] text-[var(--dev-signal)]">{onnxError}</div>
          )}
          {gpuError && (
            <div className="font-mono text-[11px] text-[var(--dev-signal)]">{gpuError}</div>
          )}

          {onnxOut && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
                <span className="text-[var(--device-ink)]">
                  ONNX V2 ({onnxOut.batch.elapsedMs} ms, {onnxOut.batch.windowsScanned} окон):
                </span>
                <span className="text-[var(--dev-signal)]">
                  Targeted Acc:{" "}
                  {onnxOut.targetedAcc !== null
                    ? `${(onnxOut.targetedAcc * 100).toFixed(1)}% (${onnxOut.targetedCount} триг.)`
                    : "нет триггеров в окне"}{" "}
                  · Overall {(onnxOut.batch.overallAcc * 100).toFixed(1)}%
                </span>
              </div>
              <LagAttentionPlot
                profile={onnxOut.batch.peakLagProfile}
                lagA={params.lagA}
                lagB={params.lagB}
                title="Browser ONNX внимание (пик по головам)"
              />
            </div>
          )}

          {gpuResult && (
            <div className="space-y-2" data-testid="backdoor-gpu-result">
              <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
                <span className="text-[var(--device-ink)]">
                  {gpuResult.device} · {gpuResult.elapsed_ms.toFixed(0)} ms ({gpuResult.epochs_ran} эпох)
                </span>
                <span className="text-[var(--dev-signal)]">
                  Targeted Acc:{" "}
                  {gpuResult.targeted_acc !== null
                    ? `${(gpuResult.targeted_acc * 100).toFixed(1)}% (Z=${gpuResult.targeted_z_score?.toFixed(1)}σ)`
                    : "—"}
                </span>
              </div>
              <LagAttentionPlot
                profile={gpuResult.attention_profile}
                lagA={params.lagA}
                lagB={params.lagB}
                title={`Свежий DigitTransformer (${gpuResult.params_count.toLocaleString()} params)`}
              />
            </div>
          )}
        </div>
      </div>

      <button
        type="button"
        onClick={() => onSendToPiano({ label: ruleLabel, digits: generated.digits, mask: generated.mask })}
        className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-transparent px-3 py-1.5 font-sans text-xs font-medium text-[var(--ink)] hover:border-[var(--ink)]"
      >
        Отправить поток в Пианино π (колокольчик на триггерах)
      </button>
    </div>
  );
}
