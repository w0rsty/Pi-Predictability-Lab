import { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";
import type { ArenaModeId, ThemeMode } from "../../types/lab";
import {
  type ParticipantSummary,
  type PlayedRoundRecord,
  buildDebriefConclusion,
  buildMarkdownProtocol,
  exactMcNemar,
  summarizeHuman,
  summarizeModel,
} from "../../utils/arenaStats";
import { getPlotlyColors } from "../../utils/plotlyTheme";
import { PlotlyChart } from "../ui/PlotlyChart";

interface ArenaDebriefProps {
  mode: ArenaModeId;
  modeLabel: string;
  trueDataset: string;
  radix: number;
  records: PlayedRoundRecord[];
  theme: ThemeMode;
  blindVerdict: "H0" | "H1" | null;
  inspectIdx: number | null;
  onInspectRound: (idx: number) => void;
  onPlayAgain: () => void;
  onSwitchMode: (mode: ArenaModeId) => void;
}

export function ArenaDebrief({
  mode,
  modeLabel,
  trueDataset,
  radix,
  records,
  theme,
  blindVerdict,
  inspectIdx,
  onInspectRound,
  onPlayAgain,
  onSwitchMode,
}: ArenaDebriefProps) {
  const c = getPlotlyColors(theme);
  const [chartMetric, setChartMetric] = useState<"hits" | "bits">("hits");
  const [copied, setCopied] = useState(false);

  const summary = useMemo(() => {
    const human = summarizeHuman(records, radix);
    const tf = summarizeModel("DigitTransformer", records, radix, (r) => r.transformer)!;
    const gru = summarizeModel("DigitGRU", records, radix, (r) => r.gru);
    const ppm = summarizeModel("PPM_o8", records, radix, (r) => r.ppm)!;

    const n = records.length;
    const p0 = 1 / Math.max(2, radix);
    const uniformRow: ParticipantSummary = {
      name: `Uniform (1/${radix})`,
      hits: Math.round(n * p0 * 10) / 10,
      n,
      acc: p0,
      ciLow: p0,
      ciHigh: p0,
      avgBits: Math.log2(radix),
      zScore: 0,
      pValue: 1,
    };

    const rows: ParticipantSummary[] = [human, tf, ...(gru ? [gru] : []), ppm, uniformRow];
    const mcnemar = exactMcNemar(
      records.map((r) => r.humanCorrect),
      records.map((r) => r.round.transformer.pred === r.round.target)
    );
    const conclusion = buildDebriefConclusion(trueDataset, human, tf, ppm, radix);
    return { rows, mcnemar, conclusion, human, tf, ppm };
  }, [radix, records, trueDataset]);

  // Накопленные ряды для Plotly
  const cumulative = useMemo(() => {
    const x: number[] = [];
    const hHits: number[] = [];
    const tfHits: number[] = [];
    const ppmHits: number[] = [];
    const hBits: number[] = [];
    const tfBits: number[] = [];
    const ppmBits: number[] = [];

    let hh = 0;
    let th = 0;
    let ph = 0;
    let hb = 0;
    let tb = 0;
    let pb = 0;
    records.forEach((rec, i) => {
      const k = i + 1;
      if (rec.humanCorrect) hh += 1;
      if (rec.round.transformer.pred === rec.round.target) th += 1;
      if (rec.round.ppm.pred === rec.round.target) ph += 1;
      hb += rec.humanBits;
      tb += rec.round.transformer.bits;
      pb += rec.round.ppm.bits;
      x.push(k);
      hHits.push(hh);
      tfHits.push(th);
      ppmHits.push(ph);
      hBits.push(Number((hb / k).toFixed(3)));
      tfBits.push(Number((tb / k).toFixed(3)));
      ppmBits.push(Number((pb / k).toFixed(3)));
    });
    return { x, hHits, tfHits, ppmHits, hBits, tfBits, ppmBits };
  }, [records]);

  const shannonLimit = Math.log2(radix);

  const handleCopyProtocol = async () => {
    const md = buildMarkdownProtocol(
      modeLabel,
      trueDataset,
      radix,
      summary.rows,
      summary.mcnemar
    );
    try {
      await navigator.clipboard.writeText(md);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // игнорируем
    }
  };

  const blindRight =
    blindVerdict !== null &&
    ((trueDataset === "true_pi" && blindVerdict === "H0") ||
      (trueDataset !== "true_pi" && blindVerdict === "H1"));

  return (
    <section data-testid="arena-debrief" className="space-y-10 border-t border-[var(--rule)] pt-10">
      {/* Вывод по результатам партии прямо на бумаге */}
      <div className="space-y-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--ink-muted)]">
          <span>Протокол дуэли · источник: {trueDataset}</span>
          {blindVerdict !== null && mode === "blind_audit" && (
            <span className="font-mono text-[var(--ink)]">
              Диагноз {blindVerdict}: {blindRight ? "подтверждён" : "ошибка классификации"}
            </span>
          )}
        </div>
        <h2 className="font-serif text-2xl font-medium tracking-tight text-[var(--ink)]">
          Статистический разбор партии
        </h2>
        <p className="max-w-4xl text-sm leading-relaxed text-[var(--ink-2)]">
          {summary.conclusion}
        </p>
      </div>

      {/* Строгая booktabs-таблица */}
      <div className="space-y-2">
        <div className="lab-table-wrapper">
          <table className="booktabs-table min-w-[660px] text-left font-mono text-xs tabular-nums">
            <thead>
              <tr className="font-sans text-xs font-medium text-[var(--ink-2)]">
                <th className="py-2.5 pr-4">Участник</th>
                <th className="py-2.5 pr-4">Попадания</th>
                <th className="py-2.5 pr-4">Точность (95% Уилсон)</th>
                <th className="py-2.5 pr-4">Битрейт (бит/симв)</th>
                <th className="py-2.5 pr-4">Z-score</th>
                <th className="py-2.5">Биномиальный p</th>
              </tr>
            </thead>
            <tbody>
              {summary.rows.map((row) => (
                <tr key={row.name} className="hover:bg-[var(--bg-hover)]">
                  <td className="py-2.5 pr-4 font-sans font-medium text-[var(--ink)]">{row.name}</td>
                  <td className="py-2.5 pr-4 text-[var(--ink)]">
                    {row.hits} / {row.n}
                  </td>
                  <td className="py-2.5 pr-4 text-[var(--ink)]">
                    {(row.acc * 100).toFixed(1)}%{" "}
                    <span className="text-[11px] text-[var(--ink-muted)]">
                      [{(row.ciLow * 100).toFixed(1)}%, {(row.ciHigh * 100).toFixed(1)}%]
                    </span>
                  </td>
                  <td
                    className={`py-2.5 pr-4 ${
                      row.avgBits < shannonLimit * 0.85
                        ? "font-medium text-[var(--sig-h1)]"
                        : "text-[var(--ink-2)]"
                    }`}
                  >
                    {row.avgBits.toFixed(3)}
                  </td>
                  <td className="py-2.5 pr-4 text-[var(--ink-2)]">
                    {row.zScore >= 0 ? "+" : ""}
                    {row.zScore.toFixed(2)}σ
                  </td>
                  <td className="py-2.5 text-[var(--ink-muted)]">
                    {row.pValue < 1e-4 ? row.pValue.toExponential(2) : row.pValue.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="font-mono text-[11px] text-[var(--ink-muted)]">
          Парный тест Макнемара (Человек vs DigitTransformer): b={summary.mcnemar.b}, c=
          {summary.mcnemar.c}, p=
          {summary.mcnemar.pValue < 1e-4
            ? summary.mcnemar.pValue.toExponential(2)
            : summary.mcnemar.pValue.toFixed(4)}
        </div>
      </div>

      {/* График Plotly прямо на бумаге */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="font-serif text-lg font-medium text-[var(--ink)]">
            Динамика по ходу партии
          </h3>
          <div className="flex items-center gap-4 border-b border-[var(--rule)] font-sans text-xs font-medium">
            <button
              type="button"
              onClick={() => setChartMetric("hits")}
              className={`-mb-px border-b-2 pb-1 transition-colors ${
                chartMetric === "hits"
                  ? "border-[var(--sig-h1)] text-[var(--ink)]"
                  : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              Накопленные попадания
            </button>
            <button
              type="button"
              onClick={() => setChartMetric("bits")}
              className={`-mb-px border-b-2 pb-1 transition-colors ${
                chartMetric === "bits"
                  ? "border-[var(--sig-h1)] text-[var(--ink)]"
                  : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              Накопленный битрейт (бит/симв)
            </button>
          </div>
        </div>

        <PlotlyChart
          theme={theme}
          height={290}
          data={[
            {
              x: cumulative.x,
              y: chartMetric === "hits" ? cumulative.hHits : cumulative.hBits,
              type: "scatter",
              mode: "lines+markers",
              name: "Человек",
              line: { color: c.ink, width: 2 },
              marker: { size: 5 },
            },
            {
              x: cumulative.x,
              y: chartMetric === "hits" ? cumulative.tfHits : cumulative.tfBits,
              type: "scatter",
              mode: "lines+markers",
              name: "DigitTransformer",
              line: { color: c.h1, width: 2.2 },
              marker: { size: 5 },
            },
            {
              x: cumulative.x,
              y: chartMetric === "hits" ? cumulative.ppmHits : cumulative.ppmBits,
              type: "scatter",
              mode: "lines+markers",
              name: "PPM_o8",
              line: { color: c.h0, width: 1.8 },
              marker: { size: 4 },
            },
            ...(chartMetric === "bits"
              ? [
                  {
                    x: cumulative.x,
                    y: cumulative.x.map(() => shannonLimit),
                    type: "scatter" as const,
                    mode: "lines" as const,
                    name: `Предел log₂(${radix}) = ${shannonLimit.toFixed(2)}`,
                    line: { color: c.muted, width: 1.2, dash: "dot" as const },
                  },
                ]
              : []),
          ]}
          layout={{
            margin: { l: 50, r: 20, t: 18, b: 44 },
            xaxis: { title: { text: "Раунд" }, dtick: 1 },
            yaxis: {
              title: {
                text: chartMetric === "hits" ? "Верных предсказаний" : "Бит на символ",
              },
            },
            legend: { orientation: "h", y: 1.15 },
          }}
        />
      </div>

      {/* Интерактивная лента разбора раундов */}
      <div className="space-y-2.5 border-t border-[var(--rule)] pt-5">
        <div className="font-sans text-xs text-[var(--ink-2)]">
          Нажмите на номер раунда, чтобы открыть его дуги внимания и спектр вероятностей на консоли сверху:
        </div>
        <div className="flex flex-wrap gap-1.5">
          {records.map((rec, i) => {
            const active = inspectIdx === i;
            const aiHit = rec.round.transformer.pred === rec.round.target;
            return (
              <button
                key={rec.round.id}
                type="button"
                onClick={() => onInspectRound(i)}
                className={`rounded-[3px] border px-2.5 py-1 font-mono text-xs transition-colors ${
                  active
                    ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--bg)]"
                    : rec.humanCorrect
                      ? "border-[var(--sig-gpu)]/60 text-[var(--sig-gpu)] hover:bg-[var(--bg-hover)]"
                      : aiHit
                        ? "border-[var(--sig-h1)]/50 text-[var(--sig-h1)] hover:bg-[var(--bg-hover)]"
                        : "border-[var(--rule)] text-[var(--ink-muted)] hover:text-[var(--ink)]"
                }`}
              >
                #{i + 1}
              </button>
            );
          })}
        </div>
      </div>

      {/* Действия после матча */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onPlayAgain}
          className="rounded-[4px] bg-[var(--ink)] px-4 py-2 font-sans text-xs font-medium text-[var(--bg)] transition-opacity hover:opacity-90"
        >
          Сыграть ещё партию
        </button>

        <button
          type="button"
          onClick={() =>
            onSwitchMode(mode === "csprng_backdoor" ? "blind_audit" : "csprng_backdoor")
          }
          className="rounded-[4px] border border-[var(--rule-strong)] px-4 py-2 font-sans text-xs font-medium text-[var(--ink)] hover:border-[var(--ink)]"
        >
          {mode === "csprng_backdoor" ? "Перейти к слепому тесту →" : "Перейти к крипто-закладке →"}
        </button>

        <button
          type="button"
          data-testid="arena-copy-protocol-btn"
          onClick={handleCopyProtocol}
          className="inline-flex items-center gap-1.5 rounded-[4px] border border-[var(--rule-strong)] px-3.5 py-2 font-sans text-xs font-medium text-[var(--ink-2)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Скопировано" : "Скопировать протокол дуэли"}
        </button>
      </div>
    </section>
  );
}
