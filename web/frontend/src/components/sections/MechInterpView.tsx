import { useMemo, useState } from "react";
import { Volume2, VolumeX } from "lucide-react";
import type {
  AblationResponse,
  HeadAblationRow,
  LabData,
  ThemeMode,
} from "../../types/lab";
import { getPlotlyColors } from "../../utils/plotlyTheme";
import { playToggleClick } from "../../utils/tactileAudio";
import { CornerReticles } from "../ui/CornerReticles";
import { PlotlyChart } from "../ui/PlotlyChart";

interface MechInterpViewProps {
  data: LabData;
  theme: ThemeMode;
  isGpuConnected: boolean;
  onOpenGpuModal: () => void;
  onAblateGpu: (payload: {
    dataset: string;
    ablate_heads: [number, number][];
    max_windows?: number;
  }) => Promise<AblationResponse>;
}

const ABLATION_DATASETS: Array<{
  id: string;
  label: string;
  subsets: Array<{ id: string; label: string }>;
  defaultHead: { layer: number; head: number };
}> = [
  {
    id: "e_cf",
    label: "e_cf",
    subsets: [
      { id: "all", label: "all" },
      { id: "even_2k", label: "even_2k" },
      { id: "const_1_sep", label: "const_1_sep" },
    ],
    defaultHead: { layer: 0, head: 2 },
  },
  {
    id: "csprng_backdoor",
    label: "Backdoor",
    subsets: [
      { id: "targeted", label: "targeted (mask=1)" },
      { id: "all", label: "all" },
    ],
    defaultHead: { layer: 0, head: 1 },
  },
  {
    id: "champernowne",
    label: "Champernowne",
    subsets: [
      { id: "all", label: "all" },
      { id: "carry_over", label: "carry_over" },
      { id: "non_carry", label: "non_carry" },
      { id: "carry_digit", label: "carry_digit" },
      { id: "inc_digit", label: "inc_digit" },
    ],
    defaultHead: { layer: 0, head: 0 },
  },
  {
    id: "rational_1_97",
    label: "1/97",
    subsets: [{ id: "all", label: "all" }],
    defaultHead: { layer: 0, head: 0 },
  },
  {
    id: "true_pi",
    label: "true_pi (H₀)",
    subsets: [{ id: "all", label: "all" }],
    defaultHead: { layer: 0, head: 2 },
  },
];

const EMB_OPTIONS: Array<{ key: string; label: string }> = [
  { key: "e_cf", label: "e_cf (структура)" },
  { key: "champernowne", label: "Champernowne" },
  { key: "rational_1_97", label: "1/97" },
  { key: "true_pi", label: "true_pi (контроль)" },
  { key: "memorization_10k_heavy", label: "10K Mem" },
];

function formatPVal(raw: number | string): string {
  if (typeof raw === "string") return raw;
  if (raw < 1e-4) return raw.toExponential(2);
  return raw.toFixed(4);
}

export function MechInterpView({
  data,
  theme,
  isGpuConnected,
  onOpenGpuModal,
  onAblateGpu,
}: MechInterpViewProps) {
  const c = getPlotlyColors(theme);

  // 1. Состояние пульта абляции 3×4
  const [ablationDataset, setAblationDataset] = useState<string>("e_cf");
  const [ablationSubset, setAblationSubset] = useState<string>("all");
  const [disabledHeads, setDisabledHeads] = useState<Set<string>>(new Set());
  const [focusedHead, setFocusedHead] = useState<{ layer: number; head: number }>({
    layer: 0,
    head: 2,
  });
  const [toggleSound, setToggleSound] = useState<boolean>(true);
  const [gpuResult, setGpuResult] = useState<AblationResponse | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [isRunningGpu, setIsRunningGpu] = useState(false);

  const activeDsConfig =
    ABLATION_DATASETS.find((d) => d.id === ablationDataset) ?? ABLATION_DATASETS[0];

  const ablationRows = useMemo<HeadAblationRow[]>(() => {
    const src =
      ablationDataset === "csprng_backdoor"
        ? data.exp5_mech_interp.head_ablation_backdoor
        : data.exp5_mech_interp.head_ablation.filter((r) => r.dataset === ablationDataset);

    const bySubset = src.filter((r) => r.subset === ablationSubset);
    if (bySubset.length > 0) return bySubset;
    return src.filter((r) => !r.subset || r.subset === "all");
  }, [ablationDataset, ablationSubset, data.exp5_mech_interp]);

  const headMap = useMemo(() => {
    const map = new Map<string, HeadAblationRow>();
    for (const row of ablationRows) {
      map.set(`${row.layer}-${row.head}`, row);
    }
    return map;
  }, [ablationRows]);

  const runLiveAblation = async (nextDisabled: Set<string>, dataset: string) => {
    if (!isGpuConnected) return;
    setIsRunningGpu(true);
    setGpuError(null);
    try {
      const pairs: [number, number][] = Array.from(nextDisabled).map((k) => {
        const [l, h] = k.split("-").map(Number);
        return [l, h];
      });
      const res = await onAblateGpu({
        dataset,
        ablate_heads: pairs,
        max_windows: 2048,
      });
      setGpuResult(res);
    } catch (err) {
      setGpuError(err instanceof Error ? err.message : "Ошибка GPU-абляции");
    } finally {
      setIsRunningGpu(false);
    }
  };

  const toggleHead = (layer: number, head: number) => {
    const key = `${layer}-${head}`;
    setFocusedHead({ layer, head });
    const next = new Set(disabledHeads);
    const willTurnOff = !next.has(key);
    if (willTurnOff) next.add(key);
    else next.delete(key);

    if (toggleSound) {
      playToggleClick(!willTurnOff);
    }

    setDisabledHeads(next);
    if (isGpuConnected) {
      void runLiveAblation(next, ablationDataset);
    } else {
      setGpuResult(null);
    }
  };

  const handleDatasetChange = (dsId: string) => {
    const cfg = ABLATION_DATASETS.find((d) => d.id === dsId) ?? ABLATION_DATASETS[0];
    setAblationDataset(dsId);
    setAblationSubset(cfg.subsets[0].id);
    setDisabledHeads(new Set());
    setGpuResult(null);
    setGpuError(null);
    setFocusedHead(cfg.defaultHead);
  };

  const resetHeads = () => {
    if (toggleSound && disabledHeads.size > 0) {
      playToggleClick(true);
    }
    setDisabledHeads(new Set());
    setGpuResult(null);
    setGpuError(null);
    if (isGpuConnected) {
      void runLiveAblation(new Set(), ablationDataset);
    }
  };

  const baselineAcc = ablationRows[0]?.base_acc ?? 0;
  const focusedRow = headMap.get(`${focusedHead.layer}-${focusedHead.head}`) ?? ablationRows[0];

  const estimatedConfigDeltaPp = useMemo(() => {
    if (disabledHeads.size === 0) return 0;
    if (disabledHeads.size === 1) {
      const [onlyKey] = Array.from(disabledHeads);
      return headMap.get(onlyKey)?.delta_acc_pp ?? 0;
    }
    let sum = 0;
    for (const k of disabledHeads) {
      sum += headMap.get(k)?.delta_acc_pp ?? 0;
    }
    return sum;
  }, [disabledHeads, headMap]);

  const useTargetedGpu =
    gpuResult &&
    ablationDataset === "csprng_backdoor" &&
    ablationSubset === "targeted" &&
    gpuResult.targeted_baseline_acc !== null &&
    gpuResult.targeted_ablated_acc !== null;

  const displayBaseAccPct = useTargetedGpu
    ? gpuResult.targeted_baseline_acc! * 100
    : gpuResult
      ? gpuResult.baseline_acc * 100
      : baselineAcc * 100;

  const displayAblatedAccPct = useTargetedGpu
    ? gpuResult.targeted_ablated_acc! * 100
    : gpuResult
      ? gpuResult.ablated_acc * 100
      : Math.max(0, Math.min(100, displayBaseAccPct + estimatedConfigDeltaPp));

  const displayConfigDeltaPp = useTargetedGpu
    ? gpuResult.targeted_delta_acc_pp ?? 0
    : gpuResult
      ? gpuResult.delta_acc_pp
      : estimatedConfigDeltaPp;

  // 2. Крипто-осциллограф (CSPRNG Backdoor)
  const backdoor = data.samples.csprng_backdoor;
  const triggers = backdoor.triggers;
  const [triggerCursor, setTriggerCursor] = useState(0);
  const currentTrig = triggers[triggerCursor % Math.max(1, triggers.length)];
  const targetLocalIdx = currentTrig ? currentTrig.t : 35;
  const winStart = Math.max(0, targetLocalIdx - 26);
  const winEnd = Math.min(backdoor.digits.length, targetLocalIdx + 3);
  const winDigits = backdoor.digits.slice(winStart, winEnd);
  const winMask = backdoor.mask.slice(winStart, winEnd);

  const cryptoAttn = data.exp5_mech_interp.crypto_attn_profile;
  const cryptoDetectors = useMemo(
    () => data.exp5_mech_interp.metrics.filter((m) => m.module === "Module2_CryptoBackdoor"),
    [data.exp5_mech_interp.metrics]
  );

  // 3. Геометрия эмбеддингов (PCA) + Linear Probing
  const [embKey, setEmbKey] = useState<string>("e_cf");
  const embEntry =
    data.exp5_mech_interp.embedding_geometry[embKey] ??
    data.exp5_mech_interp.embedding_geometry.e_cf;
  const embPi = data.exp5_mech_interp.embedding_geometry.true_pi;
  const probingRows = data.exp5_mech_interp.linear_probing;

  return (
    <div className="space-y-14 pb-20">
      {/* Открытая шапка на бумаге */}
      <div className="grid gap-4 border-b border-[var(--rule)] pb-5 md:grid-cols-[1fr_auto] md:items-end">
        <div>
          <h1 className="font-serif text-3xl font-medium tracking-tight text-[var(--ink)] sm:text-4xl">
            Абляция схем внимания, крипто-осциллограф и геометрия весов
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--ink-2)]">
            Каузальное выключение 12 голов Mini-Transformer, локализация двухлаговой закладки в{" "}
            <code className="font-mono text-[var(--ink)]">os.urandom</code> и геометрия эмбеддингов
            цифр 0..9.
          </p>
        </div>
        <div className="font-sans text-xs text-[var(--ink-muted)] md:text-right">
          <div>3 слоя × 4 головы · поправка Холма–Бонферрони</div>
          <div className="font-mono text-[11px]">120 tests · 33 significant · 0/12 on π</div>
        </div>
      </div>

      {/* ПРИБОР 1: ТЁМНЫЙ ПУЛЬТ АБЛЯЦИИ (3×4 МАТРИЦА + ПРАВАЯ СТОЙКА) */}
      <section className="hardware-device overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--device-rule)] px-5 py-4">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="font-serif text-xl font-medium text-[var(--device-ink)]">
                Пульт отключения голов внимания
              </h2>
              <button
                type="button"
                onClick={() => setToggleSound((s) => !s)}
                aria-label="Переключить звук тумблеров"
                className="inline-flex items-center gap-1.5 rounded-[3px] border border-[var(--device-rule)] bg-[var(--device-well)] px-2.5 py-0.5 font-sans text-[11px] text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
              >
                {toggleSound ? (
                  <Volume2 className="h-3 w-3 text-[var(--dev-signal)]" />
                ) : (
                  <VolumeX className="h-3 w-3 text-[var(--device-ink-muted)]" />
                )}
                Звук тумблеров: {toggleSound ? "Вкл" : "Выкл"}
              </button>
            </div>
            <p className="font-sans text-xs text-[var(--device-ink-muted)]">
              Нажмите на тумблер для обнуления вклада головы в остаточный поток · срез:{" "}
              <span className="font-mono text-[var(--device-ink)]">{ablationSubset}</span>
            </p>
          </div>

          <div className="flex flex-col items-start gap-2 sm:items-end">
            <div className="flex flex-wrap rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] p-0.5">
              {ABLATION_DATASETS.map((ds) => {
                const active = ablationDataset === ds.id;
                return (
                  <button
                    key={ds.id}
                    type="button"
                    onClick={() => handleDatasetChange(ds.id)}
                    className={`rounded-[3px] border-b-[1.5px] px-2.5 py-1 font-mono text-[11px] transition-colors ${
                      active
                        ? "border-[var(--dev-signal)] bg-[var(--device-bg)] font-medium text-[var(--device-ink)]"
                        : "border-transparent text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
                    }`}
                  >
                    {ds.label}
                  </button>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center gap-1.5 font-sans text-xs">
              <span className="text-[var(--device-ink-muted)]">Срез:</span>
              {activeDsConfig.subsets.map((sub) => {
                const active = ablationSubset === sub.id;
                return (
                  <button
                    key={sub.id}
                    type="button"
                    onClick={() => setAblationSubset(sub.id)}
                    className={`rounded-[3px] border px-2 py-0.5 font-mono text-[10px] transition-colors ${
                      active
                        ? "border-[var(--dev-signal)]/75 bg-[#DF6244]/15 text-[var(--device-ink)]"
                        : "border-[var(--device-rule)] text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
                    }`}
                  >
                    {sub.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-[1.34fr_0.66fr]">
          {/* Монолитная матрица 3×4 */}
          <div className="hardware-well grid grid-cols-2 gap-px border-0 border-b border-[var(--device-rule)] bg-[var(--device-rule)] sm:grid-cols-4 lg:border-r lg:border-b-0">
            {[0, 1, 2].map((layer) =>
              [0, 1, 2, 3].map((head) => {
                const key = `${layer}-${head}`;
                const row = headMap.get(key);
                const isDisabled = disabledHeads.has(key);
                const isSig = row?.mcnemar_sig_holm ?? false;
                const isFocused = focusedHead.layer === layer && focusedHead.head === head;

                return (
                  <button
                    key={key}
                    type="button"
                    data-testid={`ablation-switch-L${layer}-H${head}`}
                    onMouseEnter={() => setFocusedHead({ layer, head })}
                    onClick={() => toggleHead(layer, head)}
                    className={`group relative flex flex-col justify-between p-4 text-left transition-colors ${
                      isDisabled
                        ? "bg-[var(--device-well)]"
                        : isFocused
                          ? "bg-[var(--device-elevated)]"
                          : "bg-[var(--device-bg)] hover:bg-[var(--device-elevated)]"
                    }`}
                  >
                    <CornerReticles
                      active={isDisabled || isFocused}
                      variant={isDisabled ? "signal" : "device"}
                      offset={4}
                    />

                    <div className="flex w-full items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-[5px] w-[5px] rounded-full"
                          style={{
                            backgroundColor: isDisabled
                              ? "var(--dev-led-off)"
                              : "var(--dev-led-on)",
                          }}
                        />
                        <span className="font-mono text-xs font-medium tracking-tight text-[var(--device-ink)]">
                          L{layer}·H{head}
                        </span>
                      </div>

                      <span
                        className="relative inline-flex h-[18px] w-[34px] items-center rounded-[2px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] p-[2px]"
                        style={{ boxShadow: "inset 0 1px 3px rgba(0,0,0,0.75)" }}
                        aria-label={isDisabled ? "Отключена" : "Активна"}
                      >
                        <span
                          className={`h-[12px] w-[14px] rounded-[1px] border-t border-white/25 transition-transform duration-150 ease-[cubic-bezier(0.22,1,0.36,1)] ${
                            isDisabled ? "translate-x-0" : "translate-x-[14px]"
                          }`}
                          style={{
                            backgroundColor: isDisabled
                              ? "var(--dev-led-off)"
                              : "var(--device-ink)",
                          }}
                        />
                      </span>
                    </div>

                    <div className="mt-5 flex items-end justify-between gap-2">
                      <div>
                        <div
                          className={`font-mono text-xs tabular-nums ${
                            isSig
                              ? "font-medium text-[var(--dev-signal)]"
                              : "text-[var(--device-ink-2)]"
                          }`}
                        >
                          {row
                            ? `Δ ${row.delta_acc_pp >= 0 ? "+" : ""}${row.delta_acc_pp.toFixed(2)} п.п.`
                            : "—"}
                        </div>
                        <div className="mt-0.5 font-mono text-[10px] tabular-nums text-[var(--device-ink-muted)]">
                          {row
                            ? `${row.delta_bits >= 0 ? "+" : ""}${row.delta_bits.toFixed(4)} b`
                            : ""}
                        </div>
                      </div>

                      {isSig && (
                        <span className="rounded-[2px] border border-[var(--dev-signal)]/65 px-1 py-0.5 font-mono text-[9px] font-medium text-[var(--dev-signal)]">
                          HOLM*
                        </span>
                      )}
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Правая панель телеметрии */}
          <div className="flex flex-col justify-between bg-[var(--device-bg)] p-5">
            <div className="space-y-5">
              <div className="flex items-center justify-between border-b border-[var(--device-rule)] pb-3 font-sans text-xs">
                <span className="text-[var(--device-ink-2)]">
                  {isRunningGpu
                    ? "Вычисление на GPU..."
                    : gpuResult
                      ? `Live GPU (${gpuResult.elapsed_ms.toFixed(0)} ms)`
                      : "Предпросчёт (Exp 5)"}
                </span>
                <span className="font-mono text-[11px] tabular-nums text-[var(--device-ink-2)]">
                  Откл.: {disabledHeads.size}/12
                </span>
              </div>

              <div>
                <div className="mb-2 font-sans text-xs font-medium text-[var(--device-ink-2)]">
                  {disabledHeads.size === 0
                    ? "Состояние пульта · Все 12/12 голов активны"
                    : `Конфигурация пульта · Откл.: ${Array.from(disabledHeads)
                        .map((k) => `L${k.replace("-", "·H")}`)
                        .join(", ")}`}
                </div>
                <div className="grid grid-cols-2 gap-4 border-b border-[var(--device-rule)] pb-4">
                  <div>
                    <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                      Базовая точность
                    </div>
                    <div className="mt-1 font-mono text-[24px] leading-none font-medium tabular-nums text-[var(--device-ink)]">
                      {displayBaseAccPct.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                      {disabledHeads.size === 0 ? "Текущая Δ пульта" : "После абляции"}
                    </div>
                    <div
                      className={`mt-1 font-mono text-[24px] leading-none font-medium tabular-nums ${
                        disabledHeads.size > 0 && Math.abs(displayConfigDeltaPp) >= 0.1
                          ? "text-[var(--dev-signal)]"
                          : "text-[var(--device-ink)]"
                      }`}
                    >
                      {disabledHeads.size === 0
                        ? "0.00 п.п."
                        : `${displayAblatedAccPct.toFixed(2)}%`}
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <div className="mb-1.5 font-sans text-xs font-medium text-[var(--device-ink-2)]">
                  Инспектор головы L{focusedHead.layer}·H{focusedHead.head} (срез:{" "}
                  <span className="font-mono">{ablationSubset}</span>)
                </div>
                <dl className="divide-y divide-[var(--device-rule)] text-xs">
                  <div className="flex justify-between py-1.5">
                    <dt className="text-[var(--device-ink-muted)]">Эффект отключения</dt>
                    <dd
                      className={`font-mono tabular-nums ${
                        focusedRow?.mcnemar_sig_holm
                          ? "font-medium text-[var(--dev-signal)]"
                          : "text-[var(--device-ink)]"
                      }`}
                    >
                      {focusedRow
                        ? `${focusedRow.delta_acc_pp >= 0 ? "+" : ""}${focusedRow.delta_acc_pp.toFixed(2)} п.п.`
                        : "—"}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <dt className="text-[var(--device-ink-muted)]">Рост битрейта Δ</dt>
                    <dd className="font-mono tabular-nums text-[var(--device-ink-2)]">
                      {focusedRow
                        ? `${focusedRow.delta_bits >= 0 ? "+" : ""}${focusedRow.delta_bits.toFixed(4)} бит`
                        : "—"}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <dt className="text-[var(--device-ink-muted)]">p_holm (McNemar)</dt>
                    <dd className="font-mono tabular-nums text-[var(--device-ink)]">
                      {focusedRow ? formatPVal(focusedRow.mcnemar_p_holm) : "—"}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <dt className="text-[var(--device-ink-muted)]">Роль схемы</dt>
                    <dd className="text-right font-mono text-[var(--device-ink-2)]">
                      {focusedRow?.head_role ??
                        (focusedRow?.mcnemar_sig_holm ? "significant_circuit" : "redundant")}
                    </dd>
                  </div>
                </dl>
              </div>

              {gpuError && (
                <div className="rounded-[3px] border border-[var(--dev-led-off)]/50 px-3 py-2 font-sans text-xs text-[var(--dev-led-off)]">
                  {gpuError}
                </div>
              )}
            </div>

            <div className="mt-5 flex flex-wrap gap-2 border-t border-[var(--device-rule)] pt-4">
              <button
                type="button"
                disabled={isRunningGpu}
                onClick={() => {
                  if (!isGpuConnected) {
                    onOpenGpuModal();
                  } else {
                    void runLiveAblation(disabledHeads, ablationDataset);
                  }
                }}
                className="flex-1 rounded-[4px] bg-[var(--device-ink)] px-3.5 py-2 font-sans text-xs font-medium text-[var(--device-bg)] transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {isGpuConnected ? "Выполнить абляцию на GPU" : "Подключить GPU"}
              </button>
              <button
                type="button"
                onClick={resetHeads}
                className="rounded-[4px] border border-[var(--device-rule-strong)] px-3 py-2 font-sans text-xs text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
              >
                Сбросить
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ПРИБОР 2: ТЁМНЫЙ КРИПТО-ОСЦИЛЛОГРАФ */}
      <section className="hardware-device p-5 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--device-rule)] pb-4">
          <div>
            <h2 className="font-serif text-xl font-medium text-[var(--device-ink)]">
              Локализация двухлаговой закладки и сравнение детекторов
            </h2>
            <p className="mt-1 text-xs text-[var(--device-ink-2)]">
              Условие:{" "}
              <code className="font-mono text-[var(--device-ink)]">
                (d[t-24] + d[t-7]) mod 10 == 0 и d[t-1] &lt; 4
              </code>{" "}
              →{" "}
              <code className="font-mono text-[var(--dev-signal)]">
                d[t] = (4·a + b + 5) mod 10
              </code>
              {currentTrig && (
                <span className="ml-2 font-mono text-[11px] text-[var(--device-ink-muted)]">
                  (t={currentTrig.t}: a={currentTrig.a}, b={currentTrig.b}, c={currentTrig.c} →{" "}
                  {currentTrig.target})
                </span>
              )}
            </p>
          </div>

          <button
            type="button"
            data-testid="next-trigger-btn"
            onClick={() => setTriggerCursor((n) => (n + 1) % Math.max(1, triggers.length))}
            className="rounded-[4px] border border-[var(--device-rule-strong)] bg-[var(--device-well)] px-3.5 py-2 font-sans text-xs font-medium text-[var(--device-ink)] transition-colors hover:border-[var(--dev-signal)]"
          >
            Следующий триггер ({(triggerCursor % Math.max(1, triggers.length)) + 1}/
            {triggers.length}) →
          </button>
        </div>

        {/* Утопленный экран дуг внимания */}
        <div className="hardware-well overflow-x-auto rounded-[4px] p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-3 text-xs">
            <span className="font-sans text-[var(--device-ink-2)]">
              Фокус внимания головы{" "}
              <strong className="font-mono text-[var(--device-ink)]">
                {cryptoAttn.focus_head_label}
              </strong>
              :
            </span>
            <div className="flex flex-wrap items-center gap-4 font-mono text-[11px]">
              <span className="flex items-center gap-1.5 text-[var(--dev-signal)]">
                <span className="h-0.5 w-3 bg-[var(--dev-signal)]" />
                lag t−24 ({(cryptoAttn.trigger_lag24_attn * 100).toFixed(2)}%)
              </span>
              <span className="flex items-center gap-1.5 text-[var(--dev-ochre)]">
                <span className="h-0.5 w-3 bg-[var(--dev-ochre)]" />
                lag t−7 ({(cryptoAttn.trigger_lag7_attn * 100).toFixed(2)}%)
              </span>
              <span className="flex items-center gap-1.5 text-[var(--device-ink-2)]">
                <span className="h-2 w-2 border border-[var(--device-rule-strong)]" />
                условие t−1 (&lt;4)
              </span>
            </div>
          </div>

          <svg
            viewBox={`0 0 ${winDigits.length * 34} 134`}
            className="h-[134px] w-full min-w-[740px]"
            role="img"
            aria-label="Крипто-осциллограф внимания по лагам t-24, t-7 и условию t-1"
          >
            {currentTrig &&
              winDigits.map((_, idx) => {
                const absIdx = winStart + idx;
                const targetVisIdx = currentTrig.t - winStart;
                const xTarget = targetVisIdx * 34 + 17;
                const xSrc = idx * 34 + 17;
                if (absIdx === currentTrig.lag24_idx) {
                  const mid = (xSrc + xTarget) / 2;
                  return (
                    <path
                      key="arc-24"
                      d={`M ${xSrc} 78 Q ${mid} 6 ${xTarget} 78`}
                      fill="none"
                      stroke="#DF6244"
                      strokeWidth={2.2}
                    />
                  );
                }
                if (absIdx === currentTrig.lag7_idx) {
                  const mid = (xSrc + xTarget) / 2;
                  return (
                    <path
                      key="arc-7"
                      d={`M ${xSrc} 78 Q ${mid} 34 ${xTarget} 78`}
                      fill="none"
                      stroke="#D4A148"
                      strokeWidth={1.8}
                    />
                  );
                }
                return null;
              })}

            {winDigits.map((d, idx) => {
              const absIdx = winStart + idx;
              const isTarget = currentTrig ? absIdx === currentTrig.t : false;
              const isLag24 = currentTrig ? absIdx === currentTrig.lag24_idx : false;
              const isLag7 = currentTrig ? absIdx === currentTrig.lag7_idx : false;
              const isLag1 = currentTrig ? absIdx === currentTrig.lag1_idx : false;
              const isOtherTrig = !isTarget && winMask[idx] === 1;
              const x = idx * 34 + 2;

              const strokeColor =
                isTarget || isLag24
                  ? "#DF6244"
                  : isLag7
                    ? "#D4A148"
                    : isLag1
                      ? "#EFE9DE"
                      : isOtherTrig
                        ? "#DF6244"
                        : "rgba(239,233,222,0.14)";
              const fillColor = isTarget
                ? "#DF6244"
                : isLag24
                  ? "rgba(223,98,68,0.18)"
                  : isLag7
                    ? "rgba(212,161,72,0.18)"
                    : isLag1
                      ? "rgba(239,233,222,0.10)"
                      : "#1B1A18";

              return (
                <g key={absIdx}>
                  <rect
                    x={x}
                    y={80}
                    width={30}
                    height={34}
                    rx={3}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={isTarget || isLag24 || isLag7 || isLag1 ? 1.4 : 1}
                  />
                  <text
                    x={x + 15}
                    y={101}
                    textAnchor="middle"
                    className="font-mono text-xs"
                    fill={isTarget ? "#FAF9F5" : "#EFE9DE"}
                  >
                    {d}
                  </text>
                  {(isTarget || isLag24 || isLag7 || isLag1) && (
                    <text
                      x={x + 15}
                      y={128}
                      textAnchor="middle"
                      className="font-mono text-[9px]"
                      fill={isLag7 ? "#D4A148" : isLag1 ? "#B4ABA0" : "#DF6244"}
                    >
                      {isTarget ? "t" : isLag24 ? "t-24" : isLag7 ? "t-7" : "t-1"}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* 4 детектора с достаточным воздухом */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {cryptoDetectors.map((det) => {
            const isWinner = det.model === "DigitTransformer";
            const label = det.model === "Uniform" ? "Классика (χ² / Uniform)" : det.model;
            return (
              <div
                key={det.model}
                className={`hardware-well rounded-[4px] p-4 ${
                  isWinner ? "border-t-2 border-t-[var(--dev-signal)]" : ""
                }`}
              >
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-sans font-medium text-[var(--device-ink)]">{label}</span>
                  <span className="font-mono text-[11px] text-[var(--device-ink-muted)]">
                    {det.targeted_z_score !== null
                      ? `Z=${det.targeted_z_score >= 0 ? "+" : ""}${det.targeted_z_score.toFixed(1)}σ`
                      : `Z=${det.z_score.toFixed(1)}σ`}
                  </span>
                </div>
                <div className="mt-4 flex items-baseline justify-between gap-3 font-mono tabular-nums">
                  <div>
                    <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                      Targeted Acc
                    </div>
                    <div
                      className={`mt-0.5 text-xl font-medium ${
                        isWinner ? "text-[var(--dev-signal)]" : "text-[var(--device-ink)]"
                      }`}
                    >
                      {((det.targeted_acc ?? det.overall_acc) * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-sans text-[11px] text-[var(--device-ink-muted)]">
                      Overall
                    </div>
                    <div className="mt-0.5 text-sm text-[var(--device-ink-2)]">
                      {(det.overall_acc * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Профиль внимания внутри прибора */}
        <div>
          <div className="mb-2 font-sans text-xs text-[var(--device-ink-2)]">
            Профиль внимания головы {cryptoAttn.focus_head_label} по лагам (n=
            {cryptoAttn.trigger_samples_averaged} окон)
          </div>
          <PlotlyChart
            theme="dark"
            height={250}
            data={[
              {
                x: cryptoAttn.lags,
                y: cryptoAttn.trigger_mean_attn.map((v) => v * 100),
                type: "bar",
                name: "При триггере (mask=1)",
                marker: { color: "#DF6244" },
              },
              {
                x: cryptoAttn.lags,
                y: cryptoAttn.nontrigger_mean_attn.map((v) => v * 100),
                type: "scatter",
                mode: "lines",
                name: "Фоновый шум (mask=0)",
                line: { color: "#5E9EA8", width: 1.5 },
              },
            ]}
            layout={{
              margin: { l: 48, r: 16, t: 16, b: 44 },
              xaxis: { title: { text: "Лаг относительно целевой позиции t" } },
              yaxis: { title: { text: "Вес внимания (%)" } },
              legend: { orientation: "h", y: 1.15 },
            }}
          />
        </div>
      </section>

      {/* НАУЧНЫЕ ГРАФИКИ НАПРЯМУЮ НА БУМАГЕ (БЕЗ КОРОБОК!) */}
      <div className="grid grid-cols-1 gap-10 border-t border-[var(--rule)] pt-10 lg:grid-cols-2">
        <section className="space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--rule)] pb-3">
            <div>
              <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
                Геометрия эмбеддингов цифр 0..9 (PCA 2D)
              </h2>
              <div className="mt-1 font-sans text-xs text-[var(--ink-2)]">
                Перестановочный тест (10,000 перестановок) · d_model = {embEntry.d_model}
              </div>
            </div>

            <select
              aria-label="Выбор модели для PCA"
              value={embKey}
              onChange={(e) => setEmbKey(e.target.value)}
              className="desk-select"
            >
              {EMB_OPTIONS.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          <PlotlyChart
            theme={theme}
            height={300}
            data={[
              {
                x: embEntry.coords.slice(0, 10).map((p) => p[0]),
                y: embEntry.coords.slice(0, 10).map((p) => p[1]),
                text: Array.from({ length: 10 }, (_, i) => String(i)),
                mode: "text+lines+markers",
                type: "scatter",
                name: embKey,
                textposition: "top center",
                line: { color: c.h1, width: 1.2, dash: "dot" },
                marker: { color: c.h1, size: 8 },
                textfont: { color: c.text, size: 11, family: "JetBrains Mono" },
              },
              ...(embKey !== "true_pi"
                ? [
                    {
                      x: embPi.coords.slice(0, 10).map((p) => p[0]),
                      y: embPi.coords.slice(0, 10).map((p) => p[1]),
                      text: Array.from({ length: 10 }, (_, i) => String(i)),
                      mode: "markers" as const,
                      type: "scatter" as const,
                      name: "true_pi (H₀)",
                      marker: { color: c.h0, size: 6, symbol: "diamond" },
                    },
                  ]
                : []),
            ]}
            layout={{
              margin: { l: 42, r: 16, t: 16, b: 42 },
              xaxis: { title: { text: "PC1" } },
              yaxis: { title: { text: "PC2" } },
              legend: { orientation: "h", y: 1.14 },
            }}
          />

          <div className="grid grid-cols-3 divide-x divide-[var(--rule)] border-t border-[var(--rule)] pt-3 text-xs">
            <div>
              <div className="font-sans text-[11px] text-[var(--ink-muted)]">Дисперсия PC1–PC2</div>
              <div className="font-mono tabular-nums text-[var(--ink)]">
                {(embEntry.explained_variance_pc1_pc2 * 100).toFixed(1)}%
              </div>
            </div>
            <div className="pl-3">
              <div className="font-sans text-[11px] text-[var(--ink-muted)]">Соседний cos 0..8</div>
              <div className="font-mono tabular-nums text-[var(--ink)]">
                {embEntry.adjacent_cosine_0_to_8.toFixed(3)}
              </div>
            </div>
            <div className="pl-3">
              <div className="font-sans text-[11px] text-[var(--ink-muted)]">Перест. p-value</div>
              <div className="font-mono tabular-nums text-[var(--sig-h1)]">
                {embEntry.adj_0_to_8_perm_p.toFixed(4)}
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="border-b border-[var(--rule)] pb-3">
            <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
              Линейное зондирование скрытых состояний (GRU против Transformer)
            </h2>
            <p className="mt-1 text-xs leading-relaxed text-[var(--ink-2)]">
              Обученные модели против случайной инициализации (для несбалансированного класса{" "}
              <code className="font-mono text-[var(--ink)]">champ_carry</code> используется Balanced
              Accuracy). Фазы <code className="font-mono text-[var(--ink)]">e_cf_phase</code> и{" "}
              <code className="font-mono text-[var(--ink)]">rat97_phase</code> разделимы уже до
              обучения.
            </p>
          </div>

          <PlotlyChart
            theme={theme}
            height={335}
            data={[
              {
                x: probingRows
                  .filter((r) => r.state === "Trained")
                  .map((r) => `${r.task} (${r.model.replace("Digit", "")})`),
                y: probingRows
                  .filter((r) => r.state === "Trained")
                  .map((r) => r.balanced_acc * 100),
                type: "bar",
                name: "Обученная модель",
                marker: { color: c.h1 },
              },
              {
                x: probingRows
                  .filter((r) => r.state === "Untrained")
                  .map((r) => `${r.task} (${r.model.replace("Digit", "")})`),
                y: probingRows
                  .filter((r) => r.state === "Untrained")
                  .map((r) => r.balanced_acc * 100),
                type: "bar",
                name: "До обучения (контроль)",
                marker: { color: c.h0 },
              },
            ]}
            layout={{
              barmode: "group",
              margin: { l: 48, r: 16, t: 16, b: 120 },
              xaxis: { automargin: true, tickangle: -28 },
              yaxis: { title: { text: "Balanced Accuracy (%)" }, range: [0, 108] },
              legend: { orientation: "h", y: 1.12 },
            }}
          />
        </section>
      </div>
    </div>
  );
}
