import type { DreamCycleInfo, DreamDatasetId, DreamSelfTrigger } from "../types/lab";
import { inferNextProbsOnly } from "./onnxEngine";

export type DreamSpeedMode = "step" | "smooth" | "fast";

export interface SampleTokenResult {
  token: number;
  chosenProb: number;
  rawEntropyBits: number;
  tempEntropyBits: number;
  effectiveProbs: number[];
}

export interface DreamStepOutput {
  step: number;
  token: number;
  chosenProb: number;
  rawEntropyBits: number;
  tempEntropyBits: number;
  effectiveProbs: number[];
  selfTrigger: DreamSelfTrigger | null;
  cycleInfo: DreamCycleInfo | null;
}

export function createMulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shannonEntropyBits(probs: number[]): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 1e-12) {
      h -= p * Math.log2(p);
    }
  }
  return h;
}

/**
 * Температурное масштабирование вероятностей ONNX (восстановление логитов ln(max(p, 1e-12)) / τ + top-k).
 */
export function sampleNextToken(
  probs: number[],
  temperature: number,
  topK: number,
  rng: () => number = Math.random
): SampleTokenResult {
  const n = probs.length;
  const rawEntropyBits = shannonEntropyBits(probs);

  if (temperature <= 0.02) {
    let bestIdx = 0;
    for (let i = 1; i < n; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    return {
      token: bestIdx,
      chosenProb: probs[bestIdx] ?? 0,
      rawEntropyBits,
      tempEntropyBits: 0,
      effectiveProbs: probs.slice(),
    };
  }

  const logits = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    logits[i] = Math.log(Math.max(probs[i], 1e-12)) / temperature;
  }

  let allowed: Set<number> | null = null;
  if (topK > 0 && topK < n) {
    const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => logits[b] - logits[a]);
    allowed = new Set(indices.slice(0, topK));
  }

  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!allowed || allowed.has(i)) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
  }

  const effectiveProbs = new Array<number>(n).fill(0);
  let sumExp = 0;
  for (let i = 0; i < n; i++) {
    if (!allowed || allowed.has(i)) {
      const e = Math.exp(logits[i] - maxLogit);
      effectiveProbs[i] = e;
      sumExp += e;
    }
  }
  const norm = sumExp > 0 ? sumExp : 1;
  for (let i = 0; i < n; i++) {
    effectiveProbs[i] /= norm;
  }

  const tempEntropyBits = shannonEntropyBits(effectiveProbs);
  const r = rng();
  let acc = 0;
  let chosen = n - 1;
  for (let i = 0; i < n; i++) {
    acc += effectiveProbs[i];
    if (r <= acc) {
      chosen = i;
      break;
    }
  }

  return {
    token: chosen,
    chosenProb: effectiveProbs[chosen] ?? 0,
    rawEntropyBits,
    tempEntropyBits,
    effectiveProbs,
  };
}

/**
 * Ищет минимальный устойчивый период P в [1..48], повторившийся >= 3 циклов и >= 18 символов.
 */
export function detectRepetitionCycle(generated: number[]): DreamCycleInfo | null {
  const n = generated.length;
  if (n < 18) return null;
  const maxP = Math.min(48, Math.floor(n / 3));
  for (let p = 1; p <= maxP; p++) {
    const repeats = Math.max(3, Math.ceil(18 / p));
    const span = p * repeats;
    if (n < span) continue;
    const start = n - span;
    const patStart = n - p;
    let ok = true;
    for (let i = 0; i < span; i++) {
      if (generated[start + i] !== generated[patStart + (i % p)]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      const motif = generated
        .slice(patStart)
        .map((d) => (d === 10 ? ";" : d.toString(10)))
        .join("");
      return {
        period: p,
        start_step: start,
        motif,
      };
    }
  }
  return null;
}

export async function runDreamSingleStep(
  dataset: DreamDatasetId,
  history: number[],
  generatedSoFar: number[],
  temperature: number,
  topK: number,
  rng: () => number
): Promise<DreamStepOutput> {
  const ctx = history.slice(history.length - 64);
  const probs = await inferNextProbsOnly(dataset, ctx);
  const sampled = sampleNextToken(probs, temperature, topK, rng);
  const stepIdx = generatedSoFar.length;

  let selfTrigger: DreamSelfTrigger | null = null;
  if (dataset === "csprng_backdoor" && history.length >= 24) {
    const a = history[history.length - 24];
    const b = history[history.length - 7];
    const c = history[history.length - 1];
    if ((a + b) % 10 === 0 && c < 4) {
      const expected = (4 * a + b + 5) % 10;
      selfTrigger = {
        step: stepIdx,
        expected,
        chosen: sampled.token,
        matched: sampled.token === expected,
        a,
        b,
        c,
      };
    }
  }

  const nextGen = [...generatedSoFar, sampled.token];
  const cycleInfo = detectRepetitionCycle(nextGen);

  return {
    step: stepIdx,
    token: sampled.token,
    chosenProb: sampled.chosenProb,
    rawEntropyBits: sampled.rawEntropyBits,
    tempEntropyBits: sampled.tempEntropyBits,
    effectiveProbs: sampled.effectiveProbs,
    selfTrigger,
    cycleInfo,
  };
}

export async function streamDreamGeneration(params: {
  dataset: DreamDatasetId;
  seedTokens: number[];
  initialGenerated: number[];
  targetSteps: number;
  temperature: number;
  topK: number;
  speed: DreamSpeedMode;
  rng: () => number;
  signal: AbortSignal;
  onBatch: (outputs: DreamStepOutput[]) => void;
}): Promise<void> {
  const history = [...params.seedTokens, ...params.initialGenerated];
  const generated = [...params.initialGenerated];
  const chunkSize = params.speed === "fast" ? 6 : 1;
  const delayMs = params.speed === "step" ? 95 : params.speed === "smooth" ? 38 : 0;

  while (generated.length < params.targetSteps) {
    if (params.signal.aborted) break;
    const batch: DreamStepOutput[] = [];
    for (let i = 0; i < chunkSize && generated.length < params.targetSteps; i++) {
      if (params.signal.aborted) break;
      const out = await runDreamSingleStep(
        params.dataset,
        history,
        generated,
        params.temperature,
        params.topK,
        params.rng
      );
      history.push(out.token);
      generated.push(out.token);
      batch.push(out);
    }
    if (batch.length > 0 && !params.signal.aborted) {
      params.onBatch(batch);
    }
    if (generated.length < params.targetSteps && !params.signal.aborted) {
      if (delayMs > 0) {
        await new Promise((r) => window.setTimeout(r, delayMs));
      } else {
        await new Promise<void>((r) => window.requestAnimationFrame(() => r()));
      }
    }
  }
}
