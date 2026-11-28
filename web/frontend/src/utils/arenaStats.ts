import type { ArenaModelPrediction, ArenaRound } from "../types/lab";

export type ConfidenceLevel = 0.25 | 0.6 | 0.95;

export interface PlayedRoundRecord {
  round: ArenaRound;
  humanChoice: number;
  confidence: ConfidenceLevel;
  humanBits: number;
  humanCorrect: boolean;
}

export interface ParticipantSummary {
  name: string;
  hits: number;
  n: number;
  acc: number;
  ciLow: number;
  ciHigh: number;
  avgBits: number;
  zScore: number;
  pValue: number;
}

export function formatToken(token: number, radix: number): string {
  if (radix === 11 && token === 10) return ";";
  return token.toString(Math.min(16, Math.max(2, radix))).toUpperCase();
}

export function humanBitsForChoice(
  chosen: number,
  target: number,
  radix: number,
  confidence: ConfidenceLevel
): number {
  const k = Math.max(2, radix);
  const p = chosen === target ? confidence : (1 - confidence) / (k - 1);
  return -Math.log2(Math.max(p, 1e-12));
}

export function wilsonInterval(hits: number, n: number, z = 1.95996398454): [number, number] {
  if (n <= 0) return [0, 0];
  const p = hits / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denom;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

function logFactorial(n: number): number {
  let acc = 0;
  for (let i = 2; i <= n; i++) acc += Math.log(i);
  return acc;
}

function logChoose(n: number, k: number): number {
  if (k < 0 || k > n) return -Infinity;
  return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}

/** Односторонний биномиальный p-value P(X >= hits) при X ~ Bin(n, p0) и Z-score. */
export function binomialStats(
  hits: number,
  n: number,
  p0: number
): { zScore: number; pValue: number } {
  if (n <= 0) return { zScore: 0, pValue: 1 };
  const mean = n * p0;
  const sd = Math.sqrt(Math.max(1e-12, n * p0 * (1 - p0)));
  const zScore = (hits - mean) / sd;

  let pSum = 0;
  for (let k = hits; k <= n; k++) {
    const lp =
      logChoose(n, k) + k * Math.log(Math.max(p0, 1e-12)) + (n - k) * Math.log(Math.max(1 - p0, 1e-12));
    pSum += Math.exp(lp);
  }
  return { zScore, pValue: Math.min(1, Math.max(0, pSum)) };
}

/** Точный двусторонний тест Макнемара по несовпадающим парам b (A=1,B=0) и c (A=0,B=1). */
export function exactMcNemar(
  aHits: boolean[],
  bHits: boolean[]
): { b: number; c: number; pValue: number } {
  let b = 0;
  let c = 0;
  const n = Math.min(aHits.length, bHits.length);
  for (let i = 0; i < n; i++) {
    if (aHits[i] && !bHits[i]) b += 1;
    else if (!aHits[i] && bHits[i]) c += 1;
  }
  const m = b + c;
  if (m === 0) return { b: 0, c: 0, pValue: 1 };
  const kMax = Math.min(b, c);
  let tail = 0;
  for (let k = 0; k <= kMax; k++) {
    tail += Math.exp(logChoose(m, k) - m * Math.LN2);
  }
  return { b, c, pValue: Math.min(1, 2 * tail) };
}

export function summarizeModel(
  name: string,
  records: PlayedRoundRecord[],
  radix: number,
  extract: (r: ArenaRound) => ArenaModelPrediction | null
): ParticipantSummary | null {
  const valid = records.filter((rec) => extract(rec.round) !== null);
  if (valid.length === 0) return null;
  const n = valid.length;
  const hits = valid.filter((rec) => extract(rec.round)!.pred === rec.round.target).length;
  const avgBits =
    valid.reduce((acc, rec) => acc + extract(rec.round)!.bits, 0) / Math.max(1, n);
  const [ciLow, ciHigh] = wilsonInterval(hits, n);
  const { zScore, pValue } = binomialStats(hits, n, 1 / Math.max(2, radix));
  return {
    name,
    hits,
    n,
    acc: hits / n,
    ciLow,
    ciHigh,
    avgBits,
    zScore,
    pValue,
  };
}

export function summarizeHuman(
  records: PlayedRoundRecord[],
  radix: number
): ParticipantSummary {
  const n = records.length;
  const hits = records.filter((r) => r.humanCorrect).length;
  const avgBits = n > 0 ? records.reduce((a, r) => a + r.humanBits, 0) / n : Math.log2(radix);
  const [ciLow, ciHigh] = wilsonInterval(hits, n);
  const { zScore, pValue } = binomialStats(hits, n, 1 / Math.max(2, radix));
  return {
    name: "Человек",
    hits,
    n,
    acc: n > 0 ? hits / n : 0,
    ciLow,
    ciHigh,
    avgBits,
    zScore,
    pValue,
  };
}

export function buildDebriefConclusion(
  trueDataset: string,
  human: ParticipantSummary,
  tf: ParticipantSummary,
  ppm: ParticipantSummary,
  radix: number
): string {
  const shannon = Math.log2(radix);
  if (trueDataset === "true_pi") {
    return `На контроле true_pi (H₀) ни человек (${(human.acc * 100).toFixed(1)}%, Z=${human.zScore.toFixed(2)}σ), ни DigitTransformer (${(tf.acc * 100).toFixed(1)}%, ${tf.avgBits.toFixed(2)} бит/симв) не преодолели барьер случайности ${shannon.toFixed(2)} бит.`;
  }
  if (trueDataset === "csprng_backdoor") {
    return `В потоке os.urandom с двухлаговой закладкой человек (${(human.acc * 100).toFixed(1)}%) и PPM_o8 (${(ppm.acc * 100).toFixed(1)}%) слепы к нелинейной зависимости t−24, t−7, тогда как DigitTransformer достиг ${(tf.acc * 100).toFixed(1)}% (${tf.avgBits.toFixed(2)} бит/симв, Z=${tf.zScore.toFixed(2)}σ).`;
  }
  if (trueDataset === "champernowne") {
    return `На счётчике Чемперноуна человек (${(human.acc * 100).toFixed(1)}%) и DigitTransformer (${(tf.acc * 100).toFixed(1)}%, ${tf.avgBits.toFixed(2)} бит) уверенно опережают суффиксную таблицу PPM_o8 (${(ppm.acc * 100).toFixed(1)}%).`;
  }
  if (trueDataset === "e_cf") {
    return `На цепной дроби e человек (${(human.acc * 100).toFixed(1)}%) считывает ритм [1; 2k; 1], а DigitTransformer показывает ${(tf.acc * 100).toFixed(1)}% (${tf.avgBits.toFixed(2)} бит/симв при пределе ${shannon.toFixed(2)} бит).`;
  }
  return `На периодической дроби 1/97 обе модели и внимательный наблюдатель фиксируют детерминированный цикл (DigitTransformer: ${(tf.acc * 100).toFixed(1)}%, PPM_o8: ${(ppm.acc * 100).toFixed(1)}%).`;
}

export function buildMarkdownProtocol(
  modeLabel: string,
  trueDataset: string,
  radix: number,
  rows: ParticipantSummary[],
  mcnemar: { b: number; c: number; pValue: number }
): string {
  const lines = [
    `### Протокол дуэли Арены: ${modeLabel} (источник: ${trueDataset}, base-${radix})`,
    "",
    "| Участник | Попадания | Точность (95% Уилсон) | Битрейт (бит/симв) | Z-score | p-value |",
    "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ...rows.map(
      (r) =>
        `| ${r.name} | ${r.hits}/${r.n} | ${(r.acc * 100).toFixed(1)}% [${(r.ciLow * 100).toFixed(1)}%, ${(r.ciHigh * 100).toFixed(1)}%] | ${r.avgBits.toFixed(3)} | ${r.zScore >= 0 ? "+" : ""}${r.zScore.toFixed(2)}σ | ${r.pValue < 1e-4 ? r.pValue.toExponential(2) : r.pValue.toFixed(4)} |`
    ),
    "",
    `Парный тест Макнемара (Человек vs DigitTransformer): b=${mcnemar.b}, c=${mcnemar.c}, p=${mcnemar.pValue < 1e-4 ? mcnemar.pValue.toExponential(2) : mcnemar.pValue.toFixed(4)}.`,
  ];
  return lines.join("\n");
}
