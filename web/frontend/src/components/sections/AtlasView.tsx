import { useMemo, useState } from "react";
import type { LabData, ThemeMode } from "../../types/lab";
import { getPlotlyColors } from "../../utils/plotlyTheme";
import { CornerReticles } from "../ui/CornerReticles";
import { PlotlyChart } from "../ui/PlotlyChart";

interface AtlasViewProps {
  data: LabData;
  theme: ThemeMode;
}

type AtlasSubTab = "exp1" | "exp2" | "exp3" | "exp4" | "leakage";

const SHORT_DS: Record<string, string> = {
  true_pi: "π",
  hacked_pi_tier1: "T1",
  hacked_pi_tier2: "T2",
  hacked_pi_tier3: "T3",
};

const SHORT_MODEL: Record<string, string> = {
  PPM_o8: "PPM",
  TRANSFORMER: "TR",
  MLP: "MLP",
  CNN: "CNN",
  GRU: "GRU",
};

export function AtlasView({ data, theme }: AtlasViewProps) {
  const [subTab, setSubTab] = useState<AtlasSubTab>("exp1");
  const [leakageMode, setLeakageMode] = useState<"shuffle" | "chronological">("shuffle");
  const c = getPlotlyColors(theme);

  const tabs: Array<{ id: AtlasSubTab; label: string }> = [
    { id: "exp1", label: "Sanity Check и PPM-C" },
    { id: "exp2", label: "Девять последовательностей" },
    { id: "exp3", label: "Базы 2 / 10 / 16" },
    { id: "exp4", label: "Скейлинг и запоминание" },
    { id: "leakage", label: "Демо утечки данных" },
  ];

  const exp1Labels = useMemo(
    () =>
      data.exp1_sanity.ppm_vs_neural.map(
        (r) => `${SHORT_MODEL[r.model] ?? r.model} · ${SHORT_DS[r.dataset] ?? r.dataset}`
      ),
    [data.exp1_sanity.ppm_vs_neural]
  );

  // Exp 2: матрица Overall Acc (%)
  const exp2Rows = data.exp2_constants.metrics;
  const exp2Datasets = useMemo(
    () => Array.from(new Set(exp2Rows.map((r) => r.dataset))),
    [exp2Rows]
  );
  const exp2Models = useMemo(
    () => Array.from(new Set(exp2Rows.map((r) => r.model))),
    [exp2Rows]
  );
  const exp2Z = useMemo(
    () =>
      exp2Datasets.map((ds) =>
        exp2Models.map((m) => {
          const found = exp2Rows.find((r) => r.dataset === ds && r.model === m);
          return found ? Number((found.test_acc * 100).toFixed(3)) : null;
        })
      ),
    [exp2Datasets, exp2Models, exp2Rows]
  );

  // Картографическая цветовая шкала
  const heatmapColorscale: Array<[number, string]> =
    theme === "dark"
      ? [
          [0.0, "#1C1A17"],
          [0.25, "#5E9EA8"],
          [0.55, "#D4A148"],
          [0.82, "#DF6244"],
          [1.0, "#EAE4DA"],
        ]
      : [
          [0.0, "#EBE5D9"],
          [0.25, "#235A63"],
          [0.55, "#B8862D"],
          [0.82, "#BF4328"],
          [1.0, "#181614"],
        ];

  const blockARows = useMemo(
    () => data.exp4_scaling.metrics.filter((r) => r.block === "Block_A_Window_H0"),
    [data.exp4_scaling.metrics]
  );
  const blockCRows = useMemo(
    () => data.exp4_scaling.metrics.filter((r) => r.block === "Block_C_Capacity_Scaling"),
    [data.exp4_scaling.metrics]
  );
  const memCurve = data.exp4_scaling.memorization_curve;

  return (
    <div className="space-y-8 pb-20">
      <div className="grid gap-3 border-b border-[var(--rule)] pb-5 md:grid-cols-[1fr_auto] md:items-end">
        <h1 className="font-serif text-3xl font-medium tracking-tight text-[var(--ink)] sm:text-4xl">
          Атлас экспериментов: от базового контроля к скейлингу и утечкам
        </h1>
        <div className="font-sans text-xs text-[var(--ink-muted)]">
          Эксперименты 1–4 · Хронологический срез 80/10/10
        </div>
      </div>

      {/* Подчёркнутые вкладки */}
      <div className="flex flex-wrap gap-6 border-b border-[var(--rule)]">
        {tabs.map((t) => {
          const active = subTab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              data-testid={`atlas-subtab-${t.id}`}
              onClick={() => setSubTab(t.id)}
              className={`-mb-px border-b-2 pb-3 font-sans text-sm font-medium transition-colors ${
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

      {subTab === "exp1" && (
        <section className="space-y-8">
          <div className="max-w-3xl">
            <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
              Контроль пайплайна: истинное π против трёх уровней синтетических правил
            </h2>
            <p className="mt-1.5 text-sm text-[var(--ink-2)]">
              Раздельная оценка общей точности (Overall Acc) по всему ряду и целевой точности
              (Targeted Acc) строго на позициях модифицированной маски.
            </p>
          </div>

          <PlotlyChart
            theme={theme}
            height={370}
            data={[
              {
                x: exp1Labels,
                y: data.exp1_sanity.ppm_vs_neural.map((r) => r.overall_acc * 100),
                type: "bar",
                name: "Overall Acc (%)",
                marker: { color: c.h0 },
              },
              {
                x: exp1Labels,
                y: data.exp1_sanity.ppm_vs_neural.map(
                  (r) => (r.targeted_acc ?? r.overall_acc) * 100
                ),
                type: "bar",
                name: "Targeted Mask Acc (%)",
                marker: { color: c.h1 },
              },
            ]}
            layout={{
              barmode: "group",
              margin: { l: 54, r: 28, t: 28, b: 92 },
              xaxis: { automargin: true, tickangle: -32 },
              yaxis: { title: { text: "Точность (%)" }, range: [0, 108] },
              legend: { orientation: "h", y: 1.12 },
            }}
          />

          <div className="lab-table-wrapper">
            <table className="booktabs-table min-w-[640px] text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="font-sans text-xs font-medium text-[var(--ink-2)]">
                  <th className="py-2.5 pr-4">Датасет</th>
                  <th className="py-2.5 pr-4">Модель</th>
                  <th className="py-2.5 pr-4">Overall</th>
                  <th className="py-2.5 pr-4">Targeted</th>
                  <th className="py-2.5 pr-4">Бит/симв</th>
                  <th className="py-2.5">Энтропия</th>
                </tr>
              </thead>
              <tbody>
                {data.exp1_sanity.ppm_vs_neural.slice(0, 12).map((row, i) => (
                  <tr key={i} className="hover:bg-[var(--bg-hover)]">
                    <td className="py-2.5 pr-4 text-[var(--ink)]">{row.dataset}</td>
                    <td className="py-2.5 pr-4 text-[var(--ink-2)]">{row.model}</td>
                    <td className="py-2.5 pr-4 text-[var(--ink)]">
                      {(row.overall_acc * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pr-4 font-medium text-[var(--sig-h1)]">
                      {row.targeted_acc !== null ? `${(row.targeted_acc * 100).toFixed(2)}%` : "—"}
                    </td>
                    <td className="py-2.5 pr-4 text-[var(--ink-2)]">
                      {row.bits_per_symbol.toFixed(4)}
                    </td>
                    <td className="py-2.5 text-[var(--ink-muted)]">
                      {row.pred_entropy.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {subTab === "exp2" && (
        <section className="space-y-6">
          <div className="max-w-3xl">
            <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
              Тепловая карта точности по константам, ГПСЧ и цепным дробям
            </h2>
            <p className="mt-1.5 text-sm text-[var(--ink-2)]">
              Девять последовательностей (окно L=64, бюджет ~250K параметров): от случайного шума до
              100%-го освоения цепной дроби e и последовательности Чемперноуна.
            </p>
          </div>

          <PlotlyChart
            theme={theme}
            height={440}
            data={[
              {
                z: exp2Z,
                x: exp2Models,
                y: exp2Datasets,
                type: "heatmap",
                colorscale: heatmapColorscale,
                hoverongaps: false,
              },
            ]}
            layout={{
              margin: { l: 155, r: 28, t: 16, b: 72 },
              xaxis: { automargin: true },
              yaxis: { automargin: true },
            }}
          />
        </section>
      )}

      {subTab === "exp3" && (
        <section className="space-y-8">
          <div className="max-w-3xl">
            <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
              Смена системы счисления: проверка в направлении формулы Бэйли–Боруэйна–Плаффа
            </h2>
            <p className="mt-1.5 text-sm text-[var(--ink-2)]">
              Битрейт предсказания Base-2, Base-10 и Base-16 в сравнении с теоретическим пределом
              Шеннона log₂(K).
            </p>
          </div>

          <PlotlyChart
            theme={theme}
            height={360}
            data={[
              {
                x: data.exp3_bases.metrics.map((r) => `${r.base_label} · ${r.model}`),
                y: data.exp3_bases.metrics.map((r) => r.bits_per_symbol),
                type: "bar",
                name: "Факт бит/символ",
                marker: { color: c.h1 },
              },
              {
                x: data.exp3_bases.metrics.map((r) => `${r.base_label} · ${r.model}`),
                y: data.exp3_bases.metrics.map((r) => r.shannon_limit_bits),
                type: "scatter",
                mode: "markers",
                name: "Предел Шеннона log₂(K)",
                marker: { color: c.h0, size: 9, symbol: "line-ew-open" },
              },
            ]}
            layout={{
              margin: { l: 54, r: 24, t: 24, b: 108 },
              xaxis: { automargin: true, tickangle: -28 },
              yaxis: { title: { text: "Бит на символ" } },
              legend: { orientation: "h", y: 1.12 },
            }}
          />

          <div className="lab-table-wrapper">
            <table className="booktabs-table min-w-[600px] text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="font-sans text-xs font-medium text-[var(--ink-2)]">
                  <th className="py-2.5 pr-4">Модель</th>
                  <th className="py-2.5 pr-4">Base-2 (бит Acc)</th>
                  <th className="py-2.5 pr-4">Base-2 экв. 4-бит</th>
                  <th className="py-2.5 pr-4">Base-16 (ниббл Acc)</th>
                  <th className="py-2.5">Δ Ниббл (бит)</th>
                </tr>
              </thead>
              <tbody>
                {data.exp3_bases.block_4bit_vs_nibble.map((row) => (
                  <tr key={row.model} className="hover:bg-[var(--bg-hover)]">
                    <td className="py-2.5 pr-4 text-[var(--ink)]">{row.model}</td>
                    <td className="py-2.5 pr-4 text-[var(--ink-2)]">
                      {row.base2_bit_acc_pct.toFixed(3)}%
                    </td>
                    <td className="py-2.5 pr-4 text-[var(--ink-2)]">
                      {row.base2_equiv_4bit_acc_pct.toFixed(3)}%
                    </td>
                    <td className="py-2.5 pr-4 text-[var(--ink)]">
                      {row.base16_nibble_acc_pct.toFixed(3)}%
                    </td>
                    <td className="py-2.5 text-[var(--sig-h1)]">
                      {row.delta_nibble_bits_base16_minus_base2 >= 0 ? "+" : ""}
                      {row.delta_nibble_bits_base16_minus_base2.toFixed(5)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {subTab === "exp4" && (
        <div className="space-y-12">
          <section className="space-y-4">
            <div className="max-w-3xl">
              <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
                Кривая запоминания: 1.54M Transformer зазубривает 10K цифр π без обобщения
              </h2>
              <p className="mt-1.5 text-sm text-[var(--ink-2)]">
                За 200 эпох без EarlyStopping точность на обучении достигает 99.27%, но тестовая
                точность не уходит от 10.0%, а тестовая перекрёстная энтропия растёт до 27.4 бит.
              </p>
            </div>

            <PlotlyChart
              theme={theme}
              height={350}
              data={[
                {
                  x: memCurve.map((p) => p.epoch),
                  y: memCurve.map((p) => p.train_acc * 100),
                  type: "scatter",
                  mode: "lines",
                  name: "Train Acc (%)",
                  line: { color: c.h1, width: 2.2 },
                },
                {
                  x: memCurve.map((p) => p.epoch),
                  y: memCurve.map((p) => p.test_acc * 100),
                  type: "scatter",
                  mode: "lines",
                  name: "Test Acc (%)",
                  line: { color: c.h0, width: 2, dash: "dot" },
                },
              ]}
              layout={{
                margin: { l: 54, r: 24, t: 24, b: 52 },
                xaxis: { title: { text: "Эпоха" } },
                yaxis: { title: { text: "Точность (%)" }, range: [0, 105] },
                legend: { orientation: "h", y: 1.12 },
              }}
            />
          </section>

          <section className="space-y-4 border-t border-[var(--rule)] pt-8">
            <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
              Масштабирование окна L и бюджета параметров на истинном π
            </h2>
            <PlotlyChart
              theme={theme}
              height={310}
              data={[
                {
                  x: blockARows.map((r) => `L=${r.window_size} · ${r.model.replace("Digit", "")}`),
                  y: blockARows.map((r) => r.test_acc * 100),
                  type: "bar",
                  name: "Срез по длине окна",
                  marker: { color: c.h0 },
                },
                {
                  x: blockCRows.map((r) => `${r.size_tier} · ${r.model.replace("Digit", "")}`),
                  y: blockCRows.map((r) => r.test_acc * 100),
                  type: "bar",
                  name: "Срез по числу весов",
                  marker: { color: c.h1 },
                },
              ]}
              layout={{
                margin: { l: 54, r: 24, t: 24, b: 96 },
                xaxis: { automargin: true, tickangle: -28 },
                yaxis: { title: { text: "Test Acc (%)" }, range: [8.5, 11.5] },
                legend: { orientation: "h", y: 1.12 },
              }}
            />
          </section>
        </div>
      )}

      {subTab === "leakage" && (
        <section className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--rule)] pb-4">
            <h2 className="font-serif text-xl font-medium text-[var(--ink)]">
              Почему случайное перемешивание окон (shuffle=True) создаёт иллюзию «взлома π»
            </h2>

            <div className="inline-flex rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] p-0.5">
              <button
                type="button"
                data-testid="leakage-shuffle-btn"
                onClick={() => setLeakageMode("shuffle")}
                className={`rounded-[3px] px-3 py-1.5 font-sans text-xs font-medium transition-colors ${
                  leakageMode === "shuffle"
                    ? "bg-[var(--sig-h1)] text-white"
                    : "text-[var(--ink-2)] hover:text-[var(--ink)]"
                }`}
              >
                С утечкой (shuffle=True)
              </button>
              <button
                type="button"
                data-testid="leakage-chrono-btn"
                onClick={() => setLeakageMode("chronological")}
                className={`rounded-[3px] px-3 py-1.5 font-sans text-xs font-medium transition-colors ${
                  leakageMode === "chronological"
                    ? "bg-[var(--ink)] text-[var(--bg)]"
                    : "text-[var(--ink-2)] hover:text-[var(--ink)]"
                }`}
              >
                Хронологический 80/10/10
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 divide-y divide-[var(--rule)] border-y border-[var(--rule-strong)] sm:grid-cols-3 sm:divide-x sm:divide-y-0">
            <div className="group relative p-5">
              <CornerReticles offset={4} variant="ink" />
              <div className="font-sans text-xs text-[var(--ink-muted)]">
                Кажущаяся / реальная точность
              </div>
              <div
                className={`mt-1.5 font-mono text-2xl font-medium tabular-nums ${
                  leakageMode === "shuffle" ? "text-[var(--sig-h1)]" : "text-[var(--sig-h0)]"
                }`}
              >
                {leakageMode === "shuffle" ? "~96.40%" : "9.99%"}
              </div>
            </div>
            <div className="group relative p-5">
              <CornerReticles offset={4} variant="ink" />
              <div className="font-sans text-xs text-[var(--ink-muted)]">
                Перекрытие окон Train и Test
              </div>
              <div className="mt-1.5 font-mono text-2xl font-medium tabular-nums text-[var(--ink)]">
                {leakageMode === "shuffle" ? "63 из 64 знаков" : "0.0%"}
              </div>
            </div>
            <div className="group relative p-5">
              <CornerReticles offset={4} variant="ink" />
              <div className="font-sans text-xs text-[var(--ink-muted)]">Вердикт аудита</div>
              <div className="mt-1.5 text-sm leading-relaxed text-[var(--ink-2)]">
                {leakageMode === "shuffle"
                  ? "Артефакт утечки: соседнее окно со сдвигом на 1 шаг уже было в Train."
                  : "Чистый срез 0–80% / 80–90% / 90–100%: истинная непредсказуемость π."}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
