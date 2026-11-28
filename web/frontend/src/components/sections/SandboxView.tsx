import { useMemo, useState } from "react";
import type {
  BackdoorTrainResponse,
  DreamDatasetId,
  DreamGpuResponse,
  LabData,
  SandboxAnalyzeResponse,
  SandboxGenerateResponse,
  ThemeMode,
} from "../../types/lab";
import {
  analyzeDigitsInBrowser,
  generateChampernowneBrowser,
  generateECfBrowser,
  generateNoisyWithBackdoorBrowser,
  generateRationalBrowser,
  parseDigitString,
} from "../../utils/browserMath";
import { inferBatchWindows, type OnnxBatchResult, type OnnxModelKey } from "../../utils/onnxEngine";
import { RandomWalkCanvas } from "../hero/RandomWalkCanvas";
import { BackdoorDesigner } from "../sandbox/BackdoorDesigner";
import { DreamLabPanel } from "../sandbox/DreamLabPanel";
import { CornerReticles } from "../ui/CornerReticles";

interface SandboxViewProps {
  data: LabData;
  theme: ThemeMode;
  isGpuConnected: boolean;
  onOpenGpuModal: () => void;
  onSendToPiano: (stream: { label: string; digits: number[]; mask?: number[] | null }) => void;
  onGenerateGpu: (payload: {
    mode: "formula" | "rational" | "backdoor";
    expression?: string;
    numerator?: number;
    denominator?: number;
    length: number;
    radix: number;
  }) => Promise<SandboxGenerateResponse>;
  onAnalyzeGpu: (payload: {
    name: string;
    digits: number[];
    mask?: number[] | null;
    radix: number;
    run_ppm?: boolean;
    run_models?: boolean;
  }) => Promise<SandboxAnalyzeResponse>;
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
  onGenerateDreamGpu: (payload: {
    dataset: DreamDatasetId;
    seed_tokens: number[];
    steps: number;
    temperature: number;
    top_k: number;
    random_seed?: number;
  }) => Promise<DreamGpuResponse>;
}

type SandboxTab = "rational" | "formula" | "presets" | "backdoor-designer" | "dream";

const ONNX_CHOICES: Array<{ key: OnnxModelKey; label: string; radix: number }> = [
  { key: "csprng_backdoor", label: "Transformer · Backdoor V2 (base-10)", radix: 10 },
  { key: "true_pi", label: "Transformer · π контроль (base-10)", radix: 10 },
  { key: "champernowne", label: "Transformer · Champernowne (base-10)", radix: 10 },
  { key: "e_cf", label: "Transformer · Цепная дробь e (base-11)", radix: 11 },
];

const QUICK_RATIONALS: Array<{ num: number; den: number; label: string; note: string }> = [
  { num: 1, den: 7, label: "1/7", note: "P = 6" },
  { num: 1, den: 17, label: "1/17", note: "P = 16" },
  { num: 1, den: 97, label: "1/97", note: "P = 96" },
  { num: 355, den: 113, label: "355/113", note: "π ≈" },
  { num: 1, den: 997, label: "1/997", note: "P = 166" },
  { num: 1, den: 999983, label: "1/999983", note: "длинный цикл" },
];

export function SandboxView({
  data,
  theme,
  isGpuConnected,
  onOpenGpuModal,
  onSendToPiano,
  onGenerateGpu,
  onAnalyzeGpu,
  onTrainBackdoorGpu,
  onGenerateDreamGpu,
}: SandboxViewProps) {
  const [tab, setTab] = useState<SandboxTab>("rational");
  const [num, setNum] = useState(1);
  const [den, setDen] = useState(97);
  const [length, setLength] = useState(2000);
  const [radix, setRadix] = useState(10);
  const [formula, setFormula] = useState("sqrt(5)");
  const [customRaw, setCustomRaw] = useState("");
  const [injectBackdoor, setInjectBackdoor] = useState(true);

  const initialRat = useMemo(() => generateRationalBrowser(1, 97, 2000, 10), []);
  const [digits, setDigits] = useState<number[]>(initialRat.digits);
  const [mask, setMask] = useState<number[] | null>(null);
  const [seqRadix, setSeqRadix] = useState<number>(10);
  const [label, setLabel] = useState<string>("1/97");
  const [periodInfo, setPeriodInfo] = useState<{
    period: number | null;
    prePeriod: number;
    minUniqueK: number | null;
  } | null>({
    period: initialRat.period,
    prePeriod: initialRat.prePeriod,
    minUniqueK: initialRat.minUniqueK,
  });

  const [statusMsg, setStatusMsg] = useState<string | null>(
    `Период дроби 1/97: L = ${initialRat.period ?? "—"} (k ≥ ${initialRat.minUniqueK ?? "—"})`
  );
  const [serverResult, setServerResult] = useState<SandboxAnalyzeResponse | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [onnxModelKey, setOnnxModelKey] = useState<OnnxModelKey>("csprng_backdoor");
  const [onnxBusy, setOnnxBusy] = useState(false);
  const [onnxProgress, setOnnxProgress] = useState<{ done: number; total: number } | null>(null);
  const [onnxResult, setOnnxResult] = useState<OnnxBatchResult | null>(null);
  const [onnxError, setOnnxError] = useState<string | null>(null);

  const localStats = useMemo(
    () => analyzeDigitsInBrowser(digits, seqRadix),
    [digits, seqRadix]
  );

  const applyRational = (pVal: number, qVal: number, lenVal: number, radixVal: number) => {
    const safeNum = Math.max(0, Math.floor(pVal) || 1);
    const safeDen = Math.max(2, Math.floor(qVal) || 97);
    const safeLen = Math.max(100, Math.min(20000, Math.floor(lenVal) || 2000));
    const res = generateRationalBrowser(safeNum, safeDen, safeLen, radixVal);
    setNum(safeNum);
    setDen(safeDen);
    setDigits(res.digits);
    setMask(null);
    setSeqRadix(radixVal);
    setLabel(`${safeNum}/${safeDen}`);
    setPeriodInfo({
      period: res.period,
      prePeriod: res.prePeriod,
      minUniqueK: res.minUniqueK,
    });
    setServerResult(null);
    setStatusMsg(
      res.period && res.period > 0
        ? `Период найден: L = ${res.period}, уникальное окно k ≥ ${res.minUniqueK ?? "?"}`
        : res.period === 0
          ? "Конечная дробь (остаток обнулился)"
          : `Период превышает окно анализа (${safeLen} знаков)`
    );
  };

  const runFormula = async () => {
    if (!isGpuConnected) {
      onOpenGpuModal();
      return;
    }
    setIsBusy(true);
    setStatusMsg(null);
    try {
      const safeLen = Math.max(100, Math.min(20000, Math.floor(length) || 2000));
      const res = await onGenerateGpu({
        mode: "formula",
        expression: formula,
        length: safeLen,
        radix,
      });
      setDigits(res.digits);
      setMask(res.mask);
      setSeqRadix(res.radix);
      setLabel(res.label || formula);
      setPeriodInfo(null);
      setServerResult(null);
      setStatusMsg(
        `Сгенерировано ${res.length} цифр (${res.label}) через mpmath за ${res.elapsed_ms.toFixed(1)} ms`
      );
    } catch (err) {
      setStatusMsg(err instanceof Error ? err.message : "Ошибка вычисления формулы");
    } finally {
      setIsBusy(false);
    }
  };

  const runBackdoorSynth = () => {
    const safeLen = Math.max(200, Math.min(10000, Math.floor(length) || 2000));
    const res = generateNoisyWithBackdoorBrowser(safeLen, injectBackdoor, 42);
    const trigCount = res.mask.reduce((a, b) => a + b, 0);
    setDigits(res.digits);
    setMask(res.mask);
    setSeqRadix(10);
    setLabel(injectBackdoor ? "PRNG + Backdoor" : "Mulberry32 шум");
    setPeriodInfo(null);
    setServerResult(null);
    setStatusMsg(
      injectBackdoor
        ? `Синтезирован поток ${safeLen} цифр с ${trigCount} закладками (t-24, t-7)`
        : `Синтезирован чистый шум без закладок (${safeLen} цифр)`
    );
  };

  const runCustomPaste = () => {
    const parsed = parseDigitString(customRaw);
    if (parsed.digits.length < 20) {
      setStatusMsg("Введите не менее 20 цифр для анализа");
      return;
    }
    setDigits(parsed.digits);
    setMask(null);
    setSeqRadix(parsed.radix);
    setLabel(`Пользовательский ввод (base-${parsed.radix})`);
    setPeriodInfo(null);
    setServerResult(null);
    setStatusMsg(`Распознано ${parsed.digits.length} цифр в системе счисления base-${parsed.radix}`);
  };

  const runBrowserOnnxAudit = async () => {
    const selectedChoice = ONNX_CHOICES.find((c) => c.key === onnxModelKey) ?? ONNX_CHOICES[0];
    if (seqRadix > selectedChoice.radix) {
      setOnnxError(
        `Последовательность base-${seqRadix} несовместима со словарём модели (${selectedChoice.radix}). Выберите e_cf для base-11 или переведите ряд в base-10.`
      );
      return;
    }
    if (digits.length < 70) {
      setOnnxError("Для окна контекста 64 нужно минимум 70 цифр последовательности");
      return;
    }
    setOnnxBusy(true);
    setOnnxError(null);
    setOnnxProgress(null);
    try {
      const startPos = 64;
      const totalWin = Math.min(256, digits.length - startPos);
      const windows: number[][] = [];
      const targets: number[] = [];
      for (let i = 0; i < totalWin; i++) {
        const t = startPos + i;
        windows.push(digits.slice(t - 64, t));
        targets.push(digits[t]);
      }
      const res = await inferBatchWindows(onnxModelKey, windows, targets, {
        batchSize: 32,
        baseOffset: startPos,
        onProgress: (done, total) => setOnnxProgress({ done, total }),
      });
      setOnnxResult(res);
      setStatusMsg(
        `Browser ONNX (${onnxModelKey}) просканировал ${res.windowsScanned} окон за ${res.elapsedMs} ms · Acc ${(res.overallAcc * 100).toFixed(2)}%`
      );
    } catch (err) {
      setOnnxError(err instanceof Error ? err.message : "Ошибка браузерного ONNX-инференса");
    } finally {
      setOnnxBusy(false);
      setOnnxProgress(null);
    }
  };

  const runServerAudit = async () => {
    if (!isGpuConnected) {
      onOpenGpuModal();
      return;
    }
    setIsBusy(true);
    try {
      const res = await onAnalyzeGpu({
        name: label,
        digits,
        mask,
        radix: seqRadix,
        run_ppm: true,
        run_models: true,
      });
      setServerResult(res);
      setStatusMsg(`Серверный GPU-аудит завершён за ${res.elapsed_ms.toFixed(1)} ms`);
    } catch (err) {
      setStatusMsg(err instanceof Error ? err.message : "Ошибка серверного аудита");
    } finally {
      setIsBusy(false);
    }
  };

  const expectedFreqPct = 100 / Math.max(2, seqRadix);
  const histMaxPct = Math.max(expectedFreqPct * 1.8, ...localStats.freqs.map((f) => f * 100));

  return (
    <div className="space-y-8 pb-20">
      {/* Единый цвет чернил в Literata (без рыжего курсива) */}
      <div className="grid gap-4 border-b border-[var(--rule)] pb-5 md:grid-cols-[1fr_auto] md:items-end">
        <h1 className="font-serif text-3xl font-medium tracking-tight text-[var(--ink)] sm:text-4xl">
          Генерация дробей, формул mpmath и проверка гипотез
        </h1>
        <div className="font-sans text-xs text-[var(--ink-muted)]">
          Браузерный расчёт · {isGpuConnected ? "Local GPU + PPM-C подключён" : "Автономный режим"}
        </div>
      </div>

      {/* Подчёркнутые вкладки */}
      <div className="flex flex-nowrap gap-5 overflow-x-auto border-b border-[var(--rule)] sm:gap-6">
        {(
          [
            { id: "rational", label: "Дробь p/q (браузер)" },
            { id: "formula", label: "Формула mpmath (GPU)" },
            { id: "presets", label: "Шум / Backdoor / Свой срез" },
            { id: "backdoor-designer", label: "Конструктор закладок" },
            { id: "dream", label: "Сны нейросети" },
          ] as const
        ).map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              data-testid={`sandbox-tab-${t.id}`}
              onClick={() => setTab(t.id)}
              className={`-mb-px shrink-0 border-b-2 pb-3 font-sans text-sm font-medium transition-colors ${
                active
                  ? "border-[var(--sig-h1)] text-[var(--ink)]"
                  : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {tab === "dream" ? (
          <DreamLabPanel
            theme={theme}
            isGpuConnected={isGpuConnected}
            onSendToPiano={onSendToPiano}
            onGenerateDreamGpu={onGenerateDreamGpu}
          />
        ) : (
          <>
            {/* ЛЕВАЯ КОЛОНКА ПРЯМО НА СТОЛЕ (без внешней обводки) */}
            <section className="lg:col-span-5 flex flex-col justify-between space-y-6">
          <div className="space-y-6">
            {tab === "rational" && (
              <div className="space-y-4">
                <div className="font-sans text-sm font-medium text-[var(--ink)]">
                  Рациональная дробь p / q
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="space-y-1">
                    <span className="font-sans text-xs text-[var(--ink-2)]">Числитель (p)</span>
                    <input
                      type="number"
                      data-testid="sandbox-num-input"
                      value={num}
                      onChange={(e) => setNum(Number(e.target.value))}
                      className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-1.5 font-mono text-sm text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="font-sans text-xs text-[var(--ink-2)]">Знаменатель (q)</span>
                    <input
                      type="number"
                      data-testid="sandbox-den-input"
                      value={den}
                      onChange={(e) => setDen(Number(e.target.value))}
                      className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-1.5 font-mono text-sm text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="font-sans text-xs text-[var(--ink-2)]">Длина ряда N</span>
                    <input
                      type="number"
                      value={length}
                      onChange={(e) => setLength(Number(e.target.value))}
                      className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-1.5 font-mono text-sm text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="font-sans text-xs text-[var(--ink-2)]">Система счисления</span>
                    <select
                      value={radix}
                      onChange={(e) => setRadix(Number(e.target.value))}
                      className="desk-select w-full py-1.5"
                    >
                      <option value={2}>2 (двоичная)</option>
                      <option value={10}>10 (десятичная)</option>
                      <option value={16}>16 (шестнадцатеричная)</option>
                    </select>
                  </label>
                </div>
                <button
                  type="button"
                  onClick={() => applyRational(num, den, length, radix)}
                  className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] px-4 py-2 font-sans text-xs font-medium text-[var(--ink)] transition-colors hover:border-[var(--ink)]"
                >
                  Вычислить дробь p/q
                </button>
              </div>
            )}

            {tab === "formula" && (
              <div className="space-y-4">
                <div className="font-sans text-sm font-medium text-[var(--ink)]">
                  Аналитическая формула mpmath (Local GPU)
                </div>
                <label className="block space-y-1">
                  <span className="font-sans text-xs text-[var(--ink-2)]">
                    Выражение AST (pi, e, sqrt(5), (1+sqrt(5))/2, sin(1))
                  </span>
                  <input
                    type="text"
                    data-testid="sandbox-formula-input"
                    value={formula}
                    onChange={(e) => setFormula(e.target.value)}
                    className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-1.5 font-mono text-sm text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
                  />
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <label className="space-y-1">
                    <span className="font-sans text-xs text-[var(--ink-2)]">Длина ряда N</span>
                    <input
                      type="number"
                      value={length}
                      onChange={(e) => setLength(Number(e.target.value))}
                      className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-1.5 font-mono text-sm text-[var(--ink)]"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="font-sans text-xs text-[var(--ink-2)]">Система счисления</span>
                    <select
                      value={radix}
                      onChange={(e) => setRadix(Number(e.target.value))}
                      className="desk-select w-full py-1.5"
                    >
                      <option value={2}>2 (двоичная)</option>
                      <option value={10}>10 (десятичная)</option>
                      <option value={16}>16 (шестнадцатеричная)</option>
                    </select>
                  </label>
                </div>
                <button
                  type="button"
                  data-testid="sandbox-formula-submit"
                  disabled={isBusy}
                  onClick={runFormula}
                  className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] px-4 py-2 font-sans text-xs font-medium text-[var(--ink)] transition-colors hover:border-[var(--ink)] disabled:opacity-50"
                >
                  {isGpuConnected ? "Вычислить через mpmath на сервере" : "Подключить Local GPU"}
                </button>
              </div>
            )}

            {tab === "presets" && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="font-sans text-sm font-medium text-[var(--ink)]">
                    Пресеты и синтез шума с закладкой
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setDigits(data.samples.true_pi.digits);
                        setMask(null);
                        setSeqRadix(10);
                        setLabel("True π");
                        setPeriodInfo(null);
                        setServerResult(null);
                      }}
                      className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-2 text-left font-sans text-xs text-[var(--ink)] hover:border-[var(--sig-h1)]"
                    >
                      True π (H₀)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const champ = generateChampernowneBrowser(2000);
                        setDigits(champ);
                        setMask(null);
                        setSeqRadix(10);
                        setLabel("Champernowne");
                        setPeriodInfo(null);
                        setServerResult(null);
                      }}
                      className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-2 text-left font-sans text-xs text-[var(--ink)] hover:border-[var(--sig-h1)]"
                    >
                      Champernowne
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const ecf = generateECfBrowser(2000);
                        setDigits(ecf);
                        setMask(null);
                        setSeqRadix(11);
                        setLabel("e_cf");
                        setPeriodInfo(null);
                        setServerResult(null);
                      }}
                      className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-2 text-left font-sans text-xs text-[var(--ink)] hover:border-[var(--sig-h1)]"
                    >
                      e_cf (цепная дробь)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setDigits(data.samples.csprng_backdoor.digits);
                        setMask(data.samples.csprng_backdoor.mask);
                        setSeqRadix(10);
                        setLabel("os.urandom + Backdoor");
                        setPeriodInfo(null);
                        setServerResult(null);
                      }}
                      className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-2 text-left font-sans text-xs text-[var(--ink)] hover:border-[var(--sig-h1)]"
                    >
                      CSPRNG + Backdoor
                    </button>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <label className="flex items-center gap-2 font-sans text-xs text-[var(--ink-2)]">
                      <input
                        type="checkbox"
                        checked={injectBackdoor}
                        onChange={(e) => setInjectBackdoor(e.target.checked)}
                      />
                      Вшить закладку (t-24, t-7)
                    </label>
                    <button
                      type="button"
                      onClick={runBackdoorSynth}
                      className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] px-3 py-1 font-sans text-xs font-medium text-[var(--ink)] hover:border-[var(--ink)]"
                    >
                      Синтезировать
                    </button>
                  </div>
                </div>

                <div className="space-y-2 border-t border-[var(--rule)] pt-4">
                  <div className="font-sans text-xs font-medium text-[var(--ink-2)]">
                    Свой срез цифр
                  </div>
                  <textarea
                    rows={3}
                    value={customRaw}
                    onChange={(e) => setCustomRaw(e.target.value)}
                    placeholder="Вставьте строку цифр (base-2, 10 или hex)..."
                    className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] p-2.5 font-mono text-xs text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={runCustomPaste}
                    className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] px-3 py-1.5 font-sans text-xs font-medium text-[var(--ink)] hover:border-[var(--ink)]"
                  >
                    Разобрать и проанализировать
                  </button>
                </div>
              </div>
            )}

            {tab === "backdoor-designer" && (
              <BackdoorDesigner
                isGpuConnected={isGpuConnected}
                onOpenGpuModal={onOpenGpuModal}
                onSequenceGenerated={(seq) => {
                  setDigits(seq.digits);
                  setMask(seq.mask);
                  setSeqRadix(10);
                  setLabel(seq.label);
                  setPeriodInfo(null);
                  setServerResult(null);
                  setOnnxResult(null);
                }}
                onSendToPiano={onSendToPiano}
                onTrainBackdoorGpu={onTrainBackdoorGpu}
              />
            )}

            {tab !== "backdoor-designer" && (
              <>
                {/* Быстрые пресеты дробей прямо на бумаге */}
            <div className="space-y-2.5 border-t border-[var(--rule)] pt-5">
              <div className="font-sans text-xs font-medium text-[var(--ink-2)]">
                Быстрые дроби в один клик
              </div>
              <div className="grid grid-cols-3 gap-2">
                {QUICK_RATIONALS.map((qr) => (
                  <button
                    key={qr.label}
                    type="button"
                    onClick={() => applyRational(qr.num, qr.den, length, radix)}
                    className="group relative rounded-[4px] border border-[var(--rule)] bg-[var(--bg-alt)]/60 px-3 py-2 text-left transition-colors hover:border-[var(--rule-strong)]"
                  >
                    <CornerReticles offset={2} variant="ink" />
                    <div className="font-mono text-xs font-medium text-[var(--ink)]">
                      {qr.label}
                    </div>
                    <div className="font-sans text-[10px] text-[var(--ink-muted)]">{qr.note}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Компактная гистограмма частот цифр прямо на бумаге */}
            <div className="space-y-2 border-t border-[var(--rule)] pt-5">
              <div className="flex items-center justify-between font-sans text-xs text-[var(--ink-2)]">
                <span>Распределение частот цифр (база {seqRadix})</span>
                <span className="font-mono text-[11px] text-[var(--ink-muted)]">
                  ожидание {expectedFreqPct.toFixed(1)}%
                </span>
              </div>
              <div className="relative rounded-[4px] border border-[var(--rule)] bg-[var(--bg-alt)]/50 px-3 pt-4 pb-2">
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute right-3 left-3 border-t border-dashed border-[var(--rule-strong)]"
                  style={{
                    bottom: `${22 + (expectedFreqPct / Math.max(histMaxPct, 1)) * 54}px`,
                  }}
                />
                <div
                  className="grid items-end gap-1.5"
                  style={{
                    gridTemplateColumns: `repeat(${Math.min(seqRadix, 16)}, minmax(0, 1fr))`,
                  }}
                >
                  {localStats.freqs.slice(0, 16).map((f, dIdx) => {
                    const pct = f * 100;
                    const barH = Math.max(3, Math.round((pct / Math.max(histMaxPct, 1)) * 54));
                    const dev = Math.abs(pct - expectedFreqPct);
                    return (
                      <div key={dIdx} className="flex flex-col items-center gap-1">
                        <div className="flex h-[56px] w-full items-end justify-center">
                          <div
                            className="w-full max-w-[14px] rounded-t-[1px] transition-[height] duration-200"
                            style={{
                              height: `${barH}px`,
                              backgroundColor:
                                dev > expectedFreqPct * 0.35 ? "var(--sig-h1)" : "var(--sig-h0)",
                            }}
                            title={`${dIdx.toString(seqRadix)}: ${pct.toFixed(2)}%`}
                          />
                        </div>
                        <span className="font-mono text-[10px] text-[var(--ink-muted)]">
                          {dIdx === 10 && seqRadix === 11 ? ";" : dIdx.toString(seqRadix)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
              </>
            )}
          </div>

          {tab !== "backdoor-designer" && (
            <div className="space-y-3 border-t border-[var(--rule)] pt-4">
              {statusMsg && <div className="font-sans text-xs text-[var(--ink-2)]">{statusMsg}</div>}

              {/* Браузерный ONNX инференс */}
              <div className="space-y-2 rounded-[4px] border border-[var(--rule)] bg-[var(--bg-alt)]/50 p-3">
                <div className="font-sans text-xs font-medium text-[var(--ink)]">
                  Браузерный нейро-инференс (ONNX WASM SIMD)
                </div>
                <div className="flex flex-wrap gap-2">
                  <select
                    value={onnxModelKey}
                    onChange={(e) => setOnnxModelKey(e.target.value as OnnxModelKey)}
                    aria-label="Модель Browser ONNX"
                    className="desk-select flex-1 min-w-[200px] py-1.5 text-xs"
                  >
                    {ONNX_CHOICES.map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    data-testid="sandbox-onnx-audit-btn"
                    disabled={onnxBusy}
                    onClick={runBrowserOnnxAudit}
                    className="rounded-[4px] border border-[var(--sig-h1)] bg-[var(--bg)] px-3.5 py-1.5 font-sans text-xs font-medium text-[var(--ink)] transition-colors hover:bg-[var(--sig-h1)] hover:text-[#151412] disabled:opacity-50"
                  >
                    {onnxBusy
                      ? `ONNX ${onnxProgress ? `${onnxProgress.done}/${onnxProgress.total}` : "…"}`
                      : "Проверить нейросетью в браузере (ONNX)"}
                  </button>
                </div>
                {onnxError && <div className="font-mono text-[11px] text-[var(--sig-h1)]">{onnxError}</div>}
              </div>

              <div className="flex flex-wrap gap-2.5">
                <button
                  type="button"
                  onClick={() => onSendToPiano({ label, digits, mask })}
                  className="rounded-[4px] border border-[var(--rule-strong)] bg-transparent px-3.5 py-2 font-sans text-xs font-medium text-[var(--ink)] transition-colors hover:border-[var(--ink)]"
                >
                  Отправить в Пианино π
                </button>
                <button
                  type="button"
                  data-testid="sandbox-server-audit-btn"
                  disabled={isBusy}
                  onClick={runServerAudit}
                  className="flex-1 rounded-[4px] bg-[var(--ink)] px-4 py-2 font-sans text-xs font-medium text-[var(--bg)] transition-opacity hover:opacity-90 disabled:opacity-50"
                >
                  {isGpuConnected ? "Запустить GPU + PPM-C аудит" : "GPU-аудит (нужен мост)"}
                </button>
              </div>
            </div>
          )}
        </section>

        {/* ПРАВАЯ КОЛОНКА: ТЁМНЫЙ ПРИБОР-ВЫВОД + ПЛОТТЕР */}
        <section className="lg:col-span-7 space-y-6">
          <div className="hardware-device p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-sans text-xs font-medium text-[var(--device-ink-2)]">
                Последовательность:{" "}
                <span className="font-mono text-[var(--device-ink)]">{label}</span> (
                {digits.length} симв.)
              </div>
              {periodInfo?.period !== null && periodInfo?.period !== undefined && (
                <div className="font-mono text-xs text-[var(--dev-signal)]">
                  Период L = {periodInfo.period}
                </div>
              )}
            </div>

            <div className="hardware-well max-h-[112px] overflow-y-auto break-all rounded-[4px] p-3 font-mono text-xs leading-relaxed tracking-[0.06em] text-[var(--device-ink)]">
              {digits
                .slice(0, 280)
                .map((d) => (d === 10 ? ";" : d.toString(seqRadix)))
                .join("")}
              {digits.length > 280 ? "…" : ""}
            </div>

            <div className="grid grid-cols-2 divide-y divide-[var(--device-rule)] rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] sm:grid-cols-4 sm:divide-x sm:divide-y-0">
              <div className="p-3">
                <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                  Энтропия Шеннона
                </div>
                <div className="mt-0.5 font-mono text-sm tabular-nums text-[var(--device-ink)]">
                  {localStats.entropyBits.toFixed(3)} / {localStats.shannonLimitBits.toFixed(2)} b
                </div>
              </div>
              <div className="p-3">
                <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                  χ² p-value
                </div>
                <div className="mt-0.5 font-mono text-sm tabular-nums text-[var(--device-ink)]">
                  {localStats.chi2PValue.toFixed(4)}
                </div>
              </div>
              <div className="p-3">
                <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                  Markov k=1 Acc
                </div>
                <div
                  className={`mt-0.5 font-mono text-sm tabular-nums ${
                    localStats.markovK1Acc > 1 / Math.max(2, seqRadix) + 0.08
                      ? "text-[var(--dev-signal)]"
                      : "text-[var(--device-ink)]"
                  }`}
                >
                  {(localStats.markovK1Acc * 100).toFixed(2)}%
                </div>
              </div>
              <div className="p-3">
                <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                  {serverResult?.ppm ? "PPM-C (GPU)" : "Уник. 2-грамм"}
                </div>
                <div className="mt-0.5 font-mono text-sm tabular-nums text-[var(--device-ink)]">
                  {serverResult?.ppm
                    ? `${(serverResult.ppm.overall_acc * 100).toFixed(2)}%`
                    : `${(localStats.uniqueK2Ratio * 100).toFixed(1)}%`}
                </div>
              </div>
            </div>

            {onnxResult && (
              <div
                data-testid="sandbox-onnx-result"
                className="space-y-2 border-t border-[var(--device-rule)] pt-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
                  <span className="text-[var(--device-ink)]">
                    Browser ONNX ({onnxResult.modelKey}) · {onnxResult.windowsScanned} окон за{" "}
                    {onnxResult.elapsedMs} ms
                  </span>
                  <span className="text-[var(--dev-signal)]">
                    Acc {(onnxResult.overallAcc * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="font-mono text-[11px] text-[var(--device-ink-2)]">
                  Топ-лаги внимания:{" "}
                  {onnxResult.topLags
                    .slice(0, 4)
                    .map((l) => `t−${l.lag} (${(l.weight * 100).toFixed(1)}%)`)
                    .join(" · ")}
                </div>
                {onnxResult.anomalies.length > 0 && (
                  <div className="font-mono text-[10px] text-[var(--device-ink-muted)]">
                    Пики уверенности (p ≥ 35%):{" "}
                    {onnxResult.anomalies
                      .slice(0, 4)
                      .map((a) => `@${a.targetPos}→${a.pred} (${(a.confidence * 100).toFixed(0)}%)`)
                      .join(", ")}
                  </div>
                )}
              </div>
            )}

            {serverResult && (
              <div className="space-y-3 border-t border-[var(--device-rule)] pt-4">
                <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
                  <span
                    className={`rounded-[3px] border px-2 py-0.5 ${
                      serverResult.verdict.hypothesis === "H1"
                        ? "border-[var(--dev-signal)]/70 text-[var(--dev-signal)]"
                        : "border-[var(--device-rule-strong)] text-[var(--device-ink-2)]"
                    }`}
                  >
                    Вердикт: {serverResult.verdict.hypothesis}
                  </span>
                  <span className="text-[var(--device-ink-muted)]">
                    LZMA: {serverResult.stats.lzma_bits_per_digit.toFixed(3)} бит/знак
                  </span>
                </div>
                <p className="font-sans text-xs leading-relaxed text-[var(--device-ink-2)]">
                  {serverResult.verdict.summary_ru}
                </p>

                {serverResult.neural_models.length > 0 && (
                  <div className="grid grid-cols-1 gap-px border border-[var(--device-rule)] bg-[var(--device-rule)] sm:grid-cols-2">
                    {serverResult.neural_models.map((nm) => {
                      const isAnomaly = Math.abs(nm.z_score) >= 3.0;
                      return (
                        <div
                          key={nm.model_key}
                          className="bg-[var(--device-well)] p-3 font-mono text-xs"
                        >
                          <div className="flex justify-between">
                            <span className="text-[var(--device-ink)]">{nm.arch}</span>
                            <span
                              className={
                                isAnomaly
                                  ? "text-[var(--dev-signal)]"
                                  : "text-[var(--device-ink-2)]"
                              }
                            >
                              {(nm.overall_acc * 100).toFixed(2)}% (Z={nm.z_score.toFixed(1)}σ)
                            </span>
                          </div>
                          <div className="mt-1 text-[10px] text-[var(--device-ink-muted)]">
                            ckpt: {nm.dataset_trained} · {nm.bits_per_symbol.toFixed(3)} b/sym
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          <RandomWalkCanvas
            theme={theme}
            height={280}
            customSequence={{ label, digits, mask, radix: seqRadix }}
          />
        </section>
          </>
        )}
      </div>
    </div>
  );
}
