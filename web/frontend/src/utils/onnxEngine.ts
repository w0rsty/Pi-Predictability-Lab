import * as ort from "onnxruntime-web/wasm";

export type OnnxModelKey = "true_pi" | "csprng_backdoor" | "champernowne" | "e_cf";

export interface OnnxLagWeight {
  lag: number;
  weight: number;
}

export interface OnnxSinglePrediction {
  probs: number[];
  pred: number;
  confidence: number;
  attnLast: number[][][]; // [n_layers][n_heads][64]
  lagProfile: number[]; // длина 64, индекс i -> лаг (64 - i)
  topLags: OnnxLagWeight[];
}

export interface OnnxBatchResult {
  modelKey: OnnxModelKey;
  elapsedMs: number;
  windowsScanned: number;
  predictions: number[];
  confidences: number[];
  /** Среднее распределение внимания по лагам среди наиболее уверенных окон */
  peakLagProfile: number[];
  topLags: OnnxLagWeight[];
  anomalies: Array<{
    windowIdx: number;
    targetPos: number;
    pred: number;
    actual: number;
    confidence: number;
    topLags: OnnxLagWeight[];
  }>;
  overallAcc: number;
}

const MODEL_FILES: Record<OnnxModelKey, string> = {
  true_pi: "transformer_true_pi.onnx",
  csprng_backdoor: "transformer_csprng_backdoor.onnx",
  champernowne: "transformer_champernowne.onnx",
  e_cf: "transformer_e_cf.onnx",
};

let envInitialized = false;
const sessionCache = new Map<OnnxModelKey, Promise<ort.InferenceSession>>();
const loadedKeys = new Set<OnnxModelKey>();
const listeners = new Set<() => void>();

function ensureOrtEnv(): string {
  const rawBase = import.meta.env.BASE_URL || "/";
  const base = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
  if (!envInitialized) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = `${base}ort/`;
    envInitialized = true;
  }
  return base;
}

function notifyListeners(): void {
  listeners.forEach((fn) => fn());
}

export function subscribeOnnxState(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getLoadedOnnxKeys(): OnnxModelKey[] {
  return Array.from(loadedKeys);
}

export async function loadOnnxModel(key: OnnxModelKey): Promise<ort.InferenceSession> {
  const existing = sessionCache.get(key);
  if (existing) return existing;

  const base = ensureOrtEnv();
  const modelUrl = `${base}models/${MODEL_FILES[key]}`;
  const promise = (async () => {
    const res = await fetch(modelUrl);
    if (!res.ok) {
      throw new Error(`Не удалось загрузить ONNX-модель ${key} (HTTP ${res.status})`);
    }
    const buffer = await res.arrayBuffer();
    const session = await ort.InferenceSession.create(new Uint8Array(buffer), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    loadedKeys.add(key);
    notifyListeners();
    return session;
  })().catch((err) => {
    sessionCache.delete(key);
    throw err;
  });

  sessionCache.set(key, promise);
  return promise;
}

function buildInputTensor(windows: number[][]): ort.Tensor {
  const batch = windows.length;
  const seqLen = 64;
  const arr = new BigInt64Array(batch * seqLen);
  for (let b = 0; b < batch; b++) {
    const win = windows[b];
    const offset = b * seqLen;
    for (let i = 0; i < seqLen; i++) {
      arr[offset + i] = BigInt(win[i] ?? 0);
    }
  }
  return new ort.Tensor("int64", arr, [batch, seqLen]);
}

function extractTopLags(profile: number[], topK = 6): OnnxLagWeight[] {
  const items = profile.map((w, idx) => ({ lag: 64 - idx, weight: w }));
  items.sort((a, b) => b.weight - a.weight);
  return items.slice(0, topK);
}

export async function inferSingleWindow(
  key: OnnxModelKey,
  context64: number[]
): Promise<OnnxSinglePrediction> {
  if (context64.length !== 64) {
    throw new Error(`Ожидалось окно длиной 64, получено ${context64.length}`);
  }
  const session = await loadOnnxModel(key);
  const inputTensor = buildInputTensor([context64]);
  const outputs = await session.run({ tokens: inputTensor });
  try {
    const probsData = outputs.probs.data as Float32Array;
    const attnData = outputs.attn_last.data as Float32Array;
    const attnDims = outputs.attn_last.dims; // [1, n_layers, n_heads, 64]

    const vocab = outputs.probs.dims[1] ?? probsData.length;
    const probs = Array.from(probsData.slice(0, vocab));
    let pred = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[pred]) pred = i;
    }

    const nLayers = Number(attnDims[1] ?? 3);
    const nHeads = Number(attnDims[2] ?? 4);
    const seqLen = Number(attnDims[3] ?? 64);

    const attnLast: number[][][] = [];
    const lagProfile = new Array<number>(seqLen).fill(0);

    // Максимум по головам даёт контрастную локализацию (t-24, t-7 и т.д.)
    for (let l = 0; l < nLayers; l++) {
      const headsArr: number[][] = [];
      for (let h = 0; h < nHeads; h++) {
        const start = (l * nHeads + h) * seqLen;
        const row = Array.from(attnData.slice(start, start + seqLen));
        headsArr.push(row);
        for (let i = 0; i < seqLen; i++) {
          if (row[i] > lagProfile[i]) lagProfile[i] = row[i];
        }
      }
      attnLast.push(headsArr);
    }

    const sumProf = lagProfile.reduce((a, b) => a + b, 0) || 1;
    const normProfile = lagProfile.map((v) => v / sumProf);

    return {
      probs,
      pred,
      confidence: probs[pred] ?? 0,
      attnLast,
      lagProfile: normProfile,
      topLags: extractTopLags(normProfile, 6),
    };
  } finally {
    inputTensor.dispose?.();
    outputs.probs?.dispose?.();
    outputs.attn_last?.dispose?.();
  }
}

/** Быстрый авторегрессионный шаг для Снов нейросети (возвращает только probs и очищает WASM-тензоры). */
export async function inferNextProbsOnly(
  key: OnnxModelKey,
  context64: number[]
): Promise<number[]> {
  if (context64.length !== 64) {
    throw new Error(`Ожидалось окно длиной 64, получено ${context64.length}`);
  }
  const session = await loadOnnxModel(key);
  const inputTensor = buildInputTensor([context64]);
  const outputs = await session.run({ tokens: inputTensor });
  try {
    const probsData = outputs.probs.data as Float32Array;
    const vocab = Number(outputs.probs.dims[1] ?? probsData.length);
    return Array.from(probsData.slice(0, vocab));
  } finally {
    inputTensor.dispose?.();
    outputs.probs?.dispose?.();
    outputs.attn_last?.dispose?.();
  }
}

export async function inferBatchWindows(
  key: OnnxModelKey,
  windows: number[][],
  targets: number[],
  options?: {
    batchSize?: number;
    baseOffset?: number;
    onProgress?: (done: number, total: number) => void;
  }
): Promise<OnnxBatchResult> {
  const t0 = performance.now();
  const batchSize = options?.batchSize ?? 32;
  const baseOffset = options?.baseOffset ?? 64;
  const session = await loadOnnxModel(key);

  const total = windows.length;
  const predictions: number[] = new Array(total);
  const confidences: number[] = new Array(total);
  const windowProfiles: number[][] = new Array(total);
  let correctCount = 0;

  for (let start = 0; start < total; start += batchSize) {
    const end = Math.min(total, start + batchSize);
    const slice = windows.slice(start, end);
    const inputTensor = buildInputTensor(slice);
    const outputs = await session.run({ tokens: inputTensor });
    const probsData = outputs.probs.data as Float32Array;
    const attnData = outputs.attn_last.data as Float32Array;

    const vocab = Number(outputs.probs.dims[1] ?? 10);
    const nLayers = Number(outputs.attn_last.dims[1] ?? 3);
    const nHeads = Number(outputs.attn_last.dims[2] ?? 4);
    const seqLen = Number(outputs.attn_last.dims[3] ?? 64);

    for (let b = 0; b < slice.length; b++) {
      const idx = start + b;
      const pOffset = b * vocab;
      let pred = 0;
      for (let v = 1; v < vocab; v++) {
        if (probsData[pOffset + v] > probsData[pOffset + pred]) pred = v;
      }
      predictions[idx] = pred;
      confidences[idx] = probsData[pOffset + pred];
      if (pred === targets[idx]) correctCount += 1;

      const prof = new Array<number>(seqLen).fill(0);
      const bAttnOffset = b * nLayers * nHeads * seqLen;
      for (let l = 0; l < nLayers; l++) {
        for (let h = 0; h < nHeads; h++) {
          const rowStart = bAttnOffset + (l * nHeads + h) * seqLen;
          for (let i = 0; i < seqLen; i++) {
            const w = attnData[rowStart + i];
            if (w > prof[i]) prof[i] = w;
          }
        }
      }
      const pSum = prof.reduce((a, x) => a + x, 0) || 1;
      windowProfiles[idx] = prof.map((x) => x / pSum);
    }

    options?.onProgress?.(end, total);
    if (end < total) {
      await new Promise((r) => window.setTimeout(r, 0));
    }
  }

  // Агрегируем профили внимания по аномальным (высокоуверенным) окнам, если они есть, иначе по всем
  const highIndices: number[] = [];
  for (let i = 0; i < total; i++) {
    if (confidences[i] >= 0.45) highIndices.push(i);
  }
  const sourceIdx = highIndices.length > 0 ? highIndices : Array.from({ length: total }, (_, i) => i);
  const peakLagProfile = new Array<number>(64).fill(0);
  for (const i of sourceIdx) {
    const prof = windowProfiles[i];
    for (let pos = 0; pos < 64; pos++) peakLagProfile[pos] += prof[pos];
  }
  const aggSum = peakLagProfile.reduce((a, b) => a + b, 0) || 1;
  for (let pos = 0; pos < 64; pos++) peakLagProfile[pos] /= aggSum;

  const anomalies: OnnxBatchResult["anomalies"] = [];
  for (let i = 0; i < total; i++) {
    if (confidences[i] >= 0.35) {
      anomalies.push({
        windowIdx: i,
        targetPos: baseOffset + i,
        pred: predictions[i],
        actual: targets[i],
        confidence: confidences[i],
        topLags: extractTopLags(windowProfiles[i], 3),
      });
    }
  }
  anomalies.sort((a, b) => b.confidence - a.confidence);

  return {
    modelKey: key,
    elapsedMs: Math.round(performance.now() - t0),
    windowsScanned: total,
    predictions,
    confidences,
    peakLagProfile,
    topLags: extractTopLags(peakLagProfile, 6),
    anomalies: anomalies.slice(0, 12),
    overallAcc: total > 0 ? correctCount / total : 0,
  };
}
