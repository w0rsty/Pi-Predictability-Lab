import type {
  ArenaDatasetId,
  ArenaModeId,
  ArenaRound,
  ArenaRoundsBank,
} from "../types/lab";
import {
  applyCustomBackdoorBrowser,
  generateChampernowneBrowser,
  generateCryptoRandomDigits,
  generateECfBrowser,
  generateRationalBrowser,
} from "./browserMath";
import { inferSingleWindow, type OnnxModelKey } from "./onnxEngine";

const BLIND_POOL: ArenaDatasetId[] = [
  "true_pi",
  "rational_1_97",
  "champernowne",
  "csprng_backdoor",
];

function computeBrowserSuffixPpmProbs(
  history: number[],
  radix: number,
  maxOrder = 6
): number[] {
  const n = history.length;
  for (let order = Math.min(maxOrder, n); order >= 1; order--) {
    const suffix = history.slice(n - order);
    const counts = new Array<number>(radix).fill(0);
    let total = 0;
    for (let i = 0; i + order < n; i++) {
      let match = true;
      for (let j = 0; j < order; j++) {
        if (history[i + j] !== suffix[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        const nextTok = history[i + order];
        if (nextTok >= 0 && nextTok < radix) {
          counts[nextTok] += 1;
          total += 1;
        }
      }
    }
    if (total > 0) {
      // Простое аддитивное сглаживание (Jeffrey/Laplace fallback) по найденным суффиксам
      const denom = total + 0.5 * radix;
      return counts.map((c) => (c + 0.5) / denom);
    }
  }
  return new Array<number>(radix).fill(1 / radix);
}

function argmax(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[best]) best = i;
  }
  return best;
}

export async function generateBrowserOnnxArenaSession(
  mode: ArenaModeId,
  nRounds: number,
  bank: ArenaRoundsBank | null
): Promise<{ rounds: ArenaRound[]; trueDataset: ArenaDatasetId; radix: number }> {
  const trueDataset: ArenaDatasetId =
    mode === "blind_audit"
      ? BLIND_POOL[Math.floor(Math.random() * BLIND_POOL.length)]
      : mode;

  let radix = 10;
  let seq: number[] = [];
  let mask: number[] = [];
  let onnxKey: OnnxModelKey = "true_pi";

  if (trueDataset === "csprng_backdoor") {
    const base = generateCryptoRandomDigits(2000);
    const bd = applyCustomBackdoorBrowser(base, {
      lagA: 24,
      lagB: 7,
      prevThreshold: 4,
      coeffA: 4,
      coeffB: 1,
      coeffConst: 5,
    });
    seq = bd.digits;
    mask = bd.mask;
    onnxKey = "csprng_backdoor";
  } else if (trueDataset === "champernowne") {
    seq = generateChampernowneBrowser(3600);
    mask = new Array(seq.length).fill(0);
    onnxKey = "champernowne";
  } else if (trueDataset === "e_cf") {
    radix = 11;
    seq = generateECfBrowser(2600);
    mask = new Array(seq.length).fill(0);
    onnxKey = "e_cf";
  } else if (trueDataset === "rational_1_97") {
    seq = generateRationalBrowser(1, 97, 1600, 10).digits;
    mask = new Array(seq.length).fill(0);
    onnxKey = "true_pi";
  } else {
    onnxKey = "true_pi";
  }

  const rounds: ArenaRound[] = [];

  if (trueDataset === "true_pi" && bank) {
    const pool = [...bank.datasets.true_pi.rounds];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const picked = pool.slice(0, Math.min(nRounds, pool.length));
    for (let i = 0; i < picked.length; i++) {
      const r = picked[i];
      const onnxRes = await inferSingleWindow("true_pi", r.context);
      const tBits = -Math.log2(Math.max(onnxRes.probs[r.target] ?? 1e-12, 1e-12));
      rounds.push({
        ...r,
        id: `onnx-pi-${i}-${r.global_pos}`,
        transformer: {
          probs: onnxRes.probs,
          pred: onnxRes.pred,
          bits: Number(tBits.toFixed(4)),
          top_lags: onnxRes.topLags.map((l) => ({ lag: l.lag, weight: Number(l.weight.toFixed(4)) })),
        },
        gru: null,
      });
    }
    return { rounds, trueDataset, radix: 10 };
  }

  // Выбор позиций из свежего потока
  const minPos = 64;
  const maxPos = seq.length - 1;
  const candidates: number[] = [];
  if (trueDataset === "csprng_backdoor") {
    const trigPos: number[] = [];
    const bgPos: number[] = [];
    for (let t = minPos; t < maxPos; t++) {
      if (mask[t] === 1) trigPos.push(t);
      else bgPos.push(t);
    }
    const wantTrig = Math.min(trigPos.length, Math.ceil(nRounds / 2));
    const wantBg = nRounds - wantTrig;
    for (let i = 0; i < wantTrig; i++) {
      const pick = trigPos.splice(Math.floor(Math.random() * trigPos.length), 1)[0];
      candidates.push(pick);
    }
    for (let i = 0; i < wantBg && bgPos.length > 0; i++) {
      const pick = bgPos.splice(Math.floor(Math.random() * bgPos.length), 1)[0];
      candidates.push(pick);
    }
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
  } else {
    const used = new Set<number>();
    while (candidates.length < nRounds && used.size < maxPos - minPos) {
      const pos = minPos + Math.floor(Math.random() * (maxPos - minPos));
      if (!used.has(pos)) {
        used.add(pos);
        candidates.push(pos);
      }
    }
  }

  for (let i = 0; i < candidates.length; i++) {
    const pos = candidates[i];
    const context = seq.slice(pos - 64, pos);
    const target = seq[pos];
    const isTrig = mask[pos] === 1;
    const onnxRes = await inferSingleWindow(onnxKey, context);
    const tBits = -Math.log2(Math.max(onnxRes.probs[target] ?? 1e-12, 1e-12));

    const ppmProbs = computeBrowserSuffixPpmProbs(seq.slice(0, pos), radix, 6);
    const ppmPred = argmax(ppmProbs);
    const ppmBits = -Math.log2(Math.max(ppmProbs[target] ?? 1e-12, 1e-12));

    rounds.push({
      id: `onnx-${trueDataset}-${i}-${pos}`,
      global_pos: pos,
      context,
      target,
      is_trigger: isTrig,
      trigger_info: isTrig
        ? {
            lag_a: 24,
            lag_b: 7,
            a: seq[pos - 24],
            b: seq[pos - 7],
            c_prev: seq[pos - 1],
            formula: `(4×${seq[pos - 24]} + ${seq[pos - 7]} + 5) mod 10 = ${target}`,
          }
        : null,
      transformer: {
        probs: onnxRes.probs,
        pred: onnxRes.pred,
        bits: Number(tBits.toFixed(4)),
        top_lags: onnxRes.topLags.map((l) => ({ lag: l.lag, weight: Number(l.weight.toFixed(4)) })),
      },
      gru: null,
      ppm: {
        probs: ppmProbs,
        pred: ppmPred,
        bits: Number(ppmBits.toFixed(4)),
      },
    });
  }

  return { rounds, trueDataset, radix };
}
