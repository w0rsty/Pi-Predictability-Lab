import { useMemo, useState } from "react";
import type { LabData, TabId, ThemeMode } from "../../types/lab";
import { OpticalPiField } from "../hero/OpticalPiField";
import { PiPianoSynth, type PianoStreamKey } from "../hero/PiPianoSynth";
import { RandomWalkCanvas } from "../hero/RandomWalkCanvas";
import { CornerReticles } from "../ui/CornerReticles";
import { MathTex } from "../ui/MathTex";

interface OverviewViewProps {
  data: LabData;
  theme: ThemeMode;
  onNavigate: (tab: TabId, subHint?: string) => void;
  externalPianoStream: { label: string; digits: number[]; mask?: number[] | null } | null;
  onClearExternalPianoStream?: () => void;
  pianoOverride?: PianoStreamKey;
}

interface ClaimRow {
  group: "H0" | "H1";
  claim: string;
  proof: string;
  limit: string;
  tex?: string;
}

const CLAIMS: ClaimRow[] = [
  {
    group: "H0",
    claim: "Константы π, e, √2 и ГПСЧ (MT19937, CSPRNG) неразличимы моделями от равномерного кода",
    proof: "9.74–10.16% в Base-10 (3.3219–3.3248 бит при пределе 3.3219), McNemar p ∈ [0.17, 0.97] против Markov_k3",
    limit: "При N=99,936 окон MDE = 0.27 п.п. (power 80%, seed=42); корреляции ниже 0.27 п.п. не детектируются",
    tex: "\\log_2 10 \\approx 3.3219",
  },
  {
    group: "H0",
    claim: "Base-16 (мотивация BBP) и 20-битный координатный оракул n → dₙ не дают авторегрессионного выигрыша",
    proof: "Base-16: 6.22–6.32% (4.0000–4.0019 бит при пределе 4.0); n → dₙ: 6.20–6.26% (Base-16), 9.85–10.00% (Base-10), p ≥ 0.205",
    limit: "Знаки вычислены через mpmath, а не формулу BBP по модулю; в оракуле нет синтетического контроля вроде n mod 7",
    tex: "n \\to d_n",
  },
  {
    group: "H1",
    claim: "На алгоритмических рядах (e_cf, champernowne) нейросети радикально обходят суффиксную память PPM_o8",
    proof: "e_cf: DigitTransformer 99.996% (0.00024 бит) vs PPM_o8 57.71%; champernowne: DigitMLP 99.998% vs PPM_o8 19.99%",
    limit: "После позиции 488,889 в champernowne идут только 6-значные числа 100000..185184 — сети выучили шаг 6",
    tex: "e = [2; 1, 2, 1, 1, 4, 1, \\dots]",
  },
  {
    group: "H0",
    claim: "Масштабирование окна (L=32..128) и параметров (50K..1.5M) не помогает на π; снятие регуляризации ведёт к зубрёжке",
    proof: "true_pi: 9.73–10.14%, |Z| ≤ 2.84σ; на 10K знаков за 200 эпох 1.54M модель даёт 99.03% Train (99.27% пик) и 27.402 бит Test",
    limit: "В блоке окна L={32,64,128} на H₀ стоял patience=1; 1/97 решается уже при L=32 из-за уникальности 2-грамм (10² > 97)",
    tex: "10^2 > 97",
  },
  {
    group: "H1",
    claim: "DigitTransformer находит двухлаговую крипто-закладку в os.urandom, невидимую для χ², KS, ACF и PPM_o8",
    proof: "Overall 13.83% (Z=+40.36σ), Targeted на mask==1 100.00% (4,119/4,119, Z=+192.54σ); χ² p=0.957, PPM_o8 9.93%",
    limit: "Плотность триггеров 4.066%; не проверялись BigCrush/PractRand, MLP, CNN и перебор всех пар лагов",
    tex: "(4a + b + 5) \\bmod 10",
  },
  {
    group: "H1",
    claim: "Абляция голов внимания и линейное зондирование выделяют функциональные схемы без ложных срабатываний на π",
    proof: "Поправка Холма (120 тестов): 33/120 значимы, на true_pi 0/12; L0-H2 на e_cf -1.93 п.п. (2k), L0-H1 на backdoor -13.11 п.п.",
    limit: "e_cf_phase и rat97_phase линейно разделимы и без обучения (100%); скрытые размерности GRU (160) и Transformer (96) разные",
    tex: "p_{\\text{holm}} < 0.05",
  },
];

const STAGES = [
  {
    code: "01",
    title: "Проверка пайплайна: π против 3-Tier Hacked π",
    metric: "100.0% Targeted",
    summary: "Четыре архитектуры (~250K параметров): случайный шум на π и детекция синтетических правил Tier 1–3.",
    target: "atlas" as TabId,
  },
  {
    code: "02",
    title: "Сравнение 9 последовательностей (L = 64)",
    metric: "99.998% vs 9.99%",
    summary: "Иррациональные константы и ГПСЧ у порога Шеннона; e_cf и Champernowne обходят суффиксную модель PPM_o8.",
    target: "atlas" as TabId,
  },
  {
    code: "03",
    title: "Системы счисления Base-2 / Base-10 / Base-16",
    metric: "4.000–4.002 бит",
    summary: "Сверка 250K нибблов Base-16 с 1M бит Base-2 (мотивация BBP): локальный авторегрессионный выигрыш отсутствует.",
    target: "atlas" as TabId,
  },
  {
    code: "04",
    title: "Масштабирование окна и тест запоминания 10K",
    metric: "27.402 бит Test",
    summary: "Рост окна (32..128) и ёмкости (50K..1.54M) не обобщает π; отключение регуляризации ведёт к зубрёжке 99.27% Train.",
    target: "atlas" as TabId,
  },
  {
    code: "05",
    title: "Мехинтерп, геометрия эмбеддингов и криптоаудит",
    metric: "Z = +192.5σ",
    summary: "Абляция 12 голов по Холму (0/12 ложных на π), кольца PCA и локализация двухлаговой закладки (t−24, t−7).",
    target: "mech-interp" as TabId,
  },
];

export function OverviewView({
  data,
  theme,
  onNavigate,
  externalPianoStream,
  onClearExternalPianoStream,
  pianoOverride,
}: OverviewViewProps) {
  const [claimFilter, setClaimFilter] = useState<"ALL" | "H0" | "H1">("ALL");

  const filteredClaims = useMemo(
    () => (claimFilter === "ALL" ? CLAIMS : CLAIMS.filter((c) => c.group === claimFilter)),
    [claimFilter]
  );

  return (
    <div className="space-y-16 pb-20">
      {/* ОТКРЫТАЯ РЕДАКЦИОННАЯ ШАПКА НА АРХИВНОЙ БУМАГЕ (единый цвет H1 без рыжего курсива) */}
      <section className="relative pt-2 pb-4">
        <OpticalPiField digits={data.samples.true_pi.digits} />

        <div className="relative z-10 grid gap-8 lg:grid-cols-[1.35fr_0.65fr] lg:items-end">
          <h1 className="font-serif text-[38px] leading-[1.08] font-medium tracking-[-0.02em] text-[var(--ink)] sm:text-[50px] lg:text-[56px]">
            Пределы предсказуемости цифр числа π и нейросетевой криптоаудит
          </h1>

          <div className="space-y-3 border-t border-[var(--rule)] pt-4 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6">
            <p className="text-sm leading-relaxed text-[var(--ink-2)]">
              Эмпирическая проверка гипотезы нормальности{" "}
              <span className="whitespace-nowrap">(<MathTex tex="H_0" />)</span> на{" "}
              <MathTex tex="\pi, e, \sqrt{2}" /> и ГПСЧ против алгоритмических рядов{" "}
              <span className="whitespace-nowrap">(<MathTex tex="H_1" />)</span> на четырёх
              архитектурах, <code className="font-mono text-[var(--ink)]">PPM-C</code> и каузальной
              абляции.
            </p>
            <div className="font-mono text-[11px] text-[var(--ink-muted)]">
              10⁶ цифр · окно L = 64 · 4 архитектуры · 26 автотестов
            </div>
          </div>
        </div>
      </section>

      {/* ШИРОКОФОРМАТНАЯ СЦЕНА: ТЁМНЫЙ ПЛОТТЕР RANDOM WALK НА ТЁПЛОМ СТОЛЕ */}
      <RandomWalkCanvas
        samples={data.samples}
        theme={theme}
        height={380}
        sidePanel={({ label, hint, presetButtons }) => (
          <>
            <div className="space-y-5">
              <div>
                <div className="font-sans text-xs font-medium text-[var(--device-ink-2)]">
                  Выбор последовательности
                </div>
                <div className="mt-2">{presetButtons}</div>
                <p className="mt-2 font-sans text-xs text-[var(--device-ink-muted)]">
                  Активно: <span className="text-[var(--device-ink)]">{label}</span> — {hint}
                </p>
              </div>

              {/* 4 ориентира эксперимента прямо внутри приборной стойки */}
              <div className="grid grid-cols-2 gap-x-4 gap-y-3 border-y border-[var(--device-rule)] py-4">
                <div>
                  <div className="font-sans text-[11px] text-[var(--device-ink-2)]">
                    Предел log₂ 10
                  </div>
                  <div className="mt-0.5 font-mono text-base font-medium tabular-nums text-[var(--dev-h0)]">
                    3.3219 бит
                  </div>
                </div>
                <div>
                  <div className="font-sans text-[11px] text-[var(--device-ink-2)]">
                    Обобщение H₁
                  </div>
                  <div className="mt-0.5 font-mono text-base font-medium tabular-nums text-[var(--device-ink)]">
                    99.998%
                  </div>
                </div>
                <div>
                  <div className="font-sans text-[11px] text-[var(--device-ink-2)]">
                    Таргет закладки
                  </div>
                  <div className="mt-0.5 font-mono text-base font-medium tabular-nums text-[var(--dev-signal)]">
                    100.00%
                  </div>
                </div>
                <div>
                  <div className="font-sans text-[11px] text-[var(--device-ink-2)]">
                    Ложных голов на π
                  </div>
                  <div className="mt-0.5 font-mono text-base font-medium tabular-nums text-[var(--device-ink)]">
                    0 / 12
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap gap-2.5">
              <button
                type="button"
                onClick={() => onNavigate("mech-interp")}
                className="flex-1 rounded-[4px] bg-[var(--device-ink)] px-3.5 py-2 font-sans text-xs font-medium text-[var(--device-bg)] transition-opacity hover:opacity-90"
              >
                Открыть Мехинтерп →
              </button>
              <button
                type="button"
                onClick={() => onNavigate("sandbox")}
                className="rounded-[4px] border border-[var(--device-rule-strong)] px-3.5 py-2 font-sans text-xs font-medium text-[var(--device-ink)] transition-colors hover:border-[var(--device-ink)]"
              >
                Песочница
              </button>
            </div>
          </>
        )}
      />

      {/* ПИАНИНО ПИ (НАСТОЛЬНЫЙ СИНТЕЗАТОР С КЛАВИШАМИ СЛОНОВОЙ КОСТИ) */}
      <PiPianoSynth
        samples={data.samples}
        theme={theme}
        externalStream={externalPianoStream}
        onClearExternalStream={onClearExternalPianoStream}
        activeStreamOverride={pianoOverride}
      />

      {/* ПЯТЬ ЭТАПОВ ИССЛЕДОВАНИЯ ПРЯМО НА БУМАГЕ (БЕЗ ВНЕШНИХ КОРОБОК) */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="font-serif text-2xl font-medium tracking-tight text-[var(--ink)]">
            Структура экспериментального стенда
          </h2>
          <span className="font-sans text-xs text-[var(--ink-muted)]">
            Нажмите на этап, чтобы открыть графики и таблицы
          </span>
        </div>

        <div className="divide-y divide-[var(--rule)] border-y border-[var(--rule-strong)]">
          {STAGES.map((stage) => (
            <button
              key={stage.code}
              type="button"
              onClick={() => onNavigate(stage.target)}
              className="group relative flex w-full flex-col gap-3 px-2 py-5 text-left transition-colors hover:bg-[var(--bg-hover)] sm:flex-row sm:items-center sm:gap-6"
            >
              <CornerReticles offset={4} variant="ink" />

              <span className="w-[48px] shrink-0 font-mono text-base font-medium tabular-nums text-[var(--ink-muted)] transition-colors group-hover:text-[var(--sig-h1)]">
                {stage.code}
              </span>

              <div className="min-w-0 flex-1">
                <div className="font-sans text-[15px] font-medium text-[var(--ink)]">
                  {stage.title}
                </div>
                <div className="mt-0.5 text-xs leading-relaxed text-[var(--ink-2)]">
                  {stage.summary}
                </div>
              </div>

              <div className="flex shrink-0 items-center justify-between gap-4 sm:w-[220px] sm:justify-end">
                <span className="font-mono text-xs font-medium tabular-nums text-[var(--sig-h1)]">
                  {stage.metric}
                </span>
                <span className="font-mono text-sm text-[var(--ink-muted)] transition-transform duration-150 group-hover:translate-x-1 group-hover:text-[var(--ink)]">
                  →
                </span>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* ВТОРИЧНАЯ ПОЛОСА БЕЗ РАМОК: BOOKTABS-ТАБЛИЦА ВЕРИФИКАЦИИ */}
      <section className="-mx-5 bg-[var(--bg-alt)] px-5 py-10 sm:-mx-6 sm:px-6 lg:-mx-12 lg:px-12">
        <div className="space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="font-serif text-2xl font-medium tracking-tight text-[var(--ink)]">
              Заявление → Доказательство → Ограничение
            </h2>

            <div className="flex items-center gap-5 border-b border-[var(--rule)] font-sans text-xs font-medium">
              {(["ALL", "H0", "H1"] as const).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setClaimFilter(f)}
                  className={`-mb-px border-b-2 pb-1.5 transition-colors ${
                    claimFilter === f
                      ? "border-[var(--sig-h1)] text-[var(--ink)]"
                      : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink-2)]"
                  }`}
                >
                  {f === "ALL" ? "Все (6)" : f}
                </button>
              ))}
            </div>
          </div>

          <div
            tabIndex={0}
            role="region"
            aria-label="Таблица заявлений, доказательств и ограничений"
            className="lab-table-wrapper"
          >
            <table className="booktabs-table min-w-[680px] text-left text-xs">
              <thead>
                <tr className="font-sans text-xs font-medium text-[var(--ink-2)]">
                  <th className="py-3 pr-4">Гипотеза</th>
                  <th className="py-3 pr-5">Заявление</th>
                  <th className="py-3 pr-5">Доказательство (results/)</th>
                  <th className="py-3">Ограничение</th>
                </tr>
              </thead>
              <tbody>
                {filteredClaims.map((row, idx) => (
                  <tr key={idx} className="align-top transition-colors hover:bg-[var(--bg-hover)]">
                    <td className="py-4 pr-4">
                      <span
                        className={`inline-block rounded-[3px] border px-1.5 py-0.5 font-mono text-[10px] font-medium ${
                          row.group === "H1"
                            ? "border-[var(--sig-h1)]/50 text-[var(--sig-h1)]"
                            : "border-[var(--sig-h0)]/45 text-[var(--sig-h0)]"
                        }`}
                      >
                        {row.group}
                      </span>
                    </td>
                    <td className="py-4 pr-5 font-medium text-[var(--ink)]">
                      <div>{row.claim}</div>
                      {row.tex && (
                        <div className="mt-1 text-[11px] text-[var(--ink-muted)]">
                          <MathTex tex={row.tex} />
                        </div>
                      )}
                    </td>
                    <td className="py-4 pr-5 font-mono text-[11px] leading-relaxed text-[var(--ink-2)]">
                      {row.proof}
                    </td>
                    <td className="py-4 text-xs leading-relaxed text-[var(--ink-muted)]">
                      {row.limit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
