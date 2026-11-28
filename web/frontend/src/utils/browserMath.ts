export interface RationalResult {
  digits: number[];
  period: number | null;
  prePeriod: number;
  minUniqueK: number | null;
}

export interface BrowserStatsResult {
  length: number;
  radix: number;
  counts: number[];
  freqs: number[];
  chi2Stat: number;
  chi2PValue: number;
  chi2Passed: boolean;
  entropyBits: number;
  shannonLimitBits: number;
  markovK1Acc: number;
  markovK1Bits: number;
  uniqueK2Ratio: number;
  uniqueK3Ratio: number;
}

// Гамма-функция и регуляризованная верхняя неполная гамма Q(a, x) для точного p-value хи-квадрат
function logGamma(z: number): number {
  const p = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  let x = 0.99999999999980993;
  const zm1 = z - 1;
  for (let i = 0; i < p.length; i++) {
    x += p[i] / (zm1 + i + 1);
  }
  const t = zm1 + p.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (zm1 + 0.5) * Math.log(t) - t + Math.log(x);
}

export function chiSquareSurvival(chi2: number, df: number): number {
  if (chi2 <= 0) return 1.0;
  if (df <= 0) return 0.0;
  const a = df / 2;
  const x = chi2 / 2;
  if (x < a + 1) {
    let ap = a;
    let sum = 1 / a;
    let del = sum;
    for (let n = 1; n <= 200; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-12) break;
    }
    const pLower = sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    return Math.max(0, Math.min(1, 1 - pLower));
  }
  let b = x + 1 - a;
  let c = 1 / 1e-30;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= 200; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = b + an / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  const q = Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  return Math.max(0, Math.min(1, q));
}

export function generateRationalBrowser(
  numerator: number,
  denominator: number,
  length: number,
  radix = 10
): RationalResult {
  const num = Math.max(0, Math.floor(numerator));
  const den = Math.max(1, Math.min(200_000, Math.floor(denominator)));
  const n = Math.max(50, Math.min(30_000, Math.floor(length)));
  const r = Math.max(2, Math.min(16, Math.floor(radix)));

  const digits: number[] = new Array(n);
  const seenRemainders = new Map<number, number>();
  let rem = num % den;
  let period: number | null = null;
  let prePeriod = 0;

  for (let i = 0; i < n; i++) {
    if (period === null && rem !== 0) {
      const prevIdx = seenRemainders.get(rem);
      if (prevIdx !== undefined) {
        prePeriod = prevIdx;
        period = i - prevIdx;
      } else if (seenRemainders.size < 200_005) {
        seenRemainders.set(rem, i);
      }
    } else if (period === null && rem === 0) {
      period = 0;
      prePeriod = i;
    }
    rem *= r;
    const q = Math.floor(rem / den);
    rem = rem % den;
    digits[i] = q;
  }

  let minUniqueK: number | null = null;
  if (period && period > 0) {
    for (let k = 1; k <= 8; k++) {
      if (Math.pow(r, k) > period) {
        const windowSet = new Set<string>();
        const cycleSlice = digits.slice(prePeriod, prePeriod + period + k);
        let ok = cycleSlice.length >= period + k - 1;
        if (ok) {
          for (let j = 0; j < period; j++) {
            const key = cycleSlice.slice(j, j + k).join(",");
            if (windowSet.has(key)) {
              ok = false;
              break;
            }
            windowSet.add(key);
          }
        }
        if (ok) {
          minUniqueK = k;
          break;
        }
      }
    }
  }

  return { digits, period, prePeriod, minUniqueK };
}

export function generateChampernowneBrowser(length: number): number[] {
  const n = Math.max(50, Math.min(30_000, Math.floor(length)));
  const out: number[] = [];
  let k = 1;
  while (out.length < n) {
    const s = String(k);
    for (let i = 0; i < s.length && out.length < n; i++) {
      out.push(s.charCodeAt(i) - 48);
    }
    k += 1;
  }
  return out;
}

export function generateECfBrowser(length: number): number[] {
  const n = Math.max(50, Math.min(30_000, Math.floor(length)));
  const out: number[] = [2, 10];
  let k = 1;
  while (out.length < n) {
    out.push(1, 10);
    const even = String(2 * k);
    for (let i = 0; i < even.length; i++) {
      out.push(even.charCodeAt(i) - 48);
    }
    out.push(10, 1, 10);
    k += 1;
  }
  return out.slice(0, n);
}

export function generateNoisyWithBackdoorBrowser(
  length: number,
  injectBackdoor: boolean,
  seed = 42
): { digits: number[]; mask: number[] } {
  const n = Math.max(100, Math.min(30_000, Math.floor(length)));
  // Простой Mulberry32 PRNG для воспроизводимости шума в браузере
  let s = seed >>> 0;
  const nextU32 = () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return (t ^ (t >>> 14)) >>> 0;
  };
  const digits = new Array<number>(n);
  const mask = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    digits[i] = nextU32() % 10;
  }
  if (injectBackdoor) {
    for (let t = 24; t < n; t++) {
      const a = digits[t - 24];
      const b = digits[t - 7];
      const c = digits[t - 1];
      if ((a + b) % 10 === 0 && c < 4) {
        digits[t] = (4 * a + b + 5) % 10;
        mask[t] = 1;
      }
    }
  }
  return { digits, mask };
}

export function analyzeDigitsInBrowser(digits: number[], radix: number): BrowserStatsResult {
  const n = digits.length;
  const counts = new Array<number>(radix).fill(0);
  for (let i = 0; i < n; i++) {
    const d = digits[i];
    if (d >= 0 && d < radix) counts[d] += 1;
  }
  const freqs = counts.map((c) => c / Math.max(n, 1));
  const expected = n / radix;
  let chi2Stat = 0;
  for (let d = 0; d < radix; d++) {
    const diff = counts[d] - expected;
    chi2Stat += (diff * diff) / Math.max(expected, 1e-9);
  }
  const chi2PValue = chiSquareSurvival(chi2Stat, Math.max(radix - 1, 1));

  let entropyNats = 0;
  for (let d = 0; d < radix; d++) {
    if (freqs[d] > 0) entropyNats -= freqs[d] * Math.log(freqs[d]);
  }
  const entropyBits = entropyNats / Math.LN2;
  const shannonLimitBits = Math.log2(radix);

  // Быстрый хронологический Markov_k1 (80% train -> 20% test)
  const splitIdx = Math.max(20, Math.floor(n * 0.8));
  const trans = Array.from({ length: radix }, () => new Array<number>(radix).fill(1e-3));
  for (let i = 0; i < splitIdx - 1; i++) {
    const u = digits[i];
    const v = digits[i + 1];
    if (u >= 0 && u < radix && v >= 0 && v < radix) trans[u][v] += 1;
  }
  const probs = trans.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map((v) => v / sum);
  });
  let correct = 0;
  let ceSum = 0;
  let testN = 0;
  for (let i = splitIdx; i < n - 1; i++) {
    const u = digits[i];
    const target = digits[i + 1];
    if (u >= 0 && u < radix && target >= 0 && target < radix) {
      const row = probs[u];
      let bestD = 0;
      for (let d = 1; d < radix; d++) {
        if (row[d] > row[bestD]) bestD = d;
      }
      if (bestD === target) correct += 1;
      ceSum -= Math.log(Math.max(row[target], 1e-12));
      testN += 1;
    }
  }

  const k2 = new Set<number>();
  const k3 = new Set<number>();
  for (let i = 0; i + 2 < n; i++) {
    k2.add(digits[i] * radix + digits[i + 1]);
    k3.add((digits[i] * radix + digits[i + 1]) * radix + digits[i + 2]);
  }

  return {
    length: n,
    radix,
    counts,
    freqs,
    chi2Stat,
    chi2PValue,
    chi2Passed: chi2PValue > 0.05,
    entropyBits,
    shannonLimitBits,
    markovK1Acc: testN > 0 ? correct / testN : 0,
    markovK1Bits: testN > 0 ? ceSum / testN / Math.LN2 : shannonLimitBits,
    uniqueK2Ratio: k2.size / Math.pow(radix, 2),
    uniqueK3Ratio: k3.size / Math.min(Math.pow(radix, 3), Math.max(n - 2, 1)),
  };
}

export function parseDigitString(raw: string): { digits: number[]; radix: number } {
  const cleaned = raw.replace(/[\s,._;]+/g, "").toLowerCase();
  if (!cleaned) return { digits: [], radix: 10 };
  const hasHex = /[a-f]/.test(cleaned);
  const radix = hasHex ? 16 : /^[01]+$/.test(cleaned) ? 2 : 10;
  const digits: number[] = [];
  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    const code = ch.charCodeAt(0);
    if (code >= 48 && code <= 57) {
      const val = code - 48;
      if (val < radix) digits.push(val);
    } else if (hasHex && code >= 97 && code <= 102) {
      digits.push(code - 97 + 10);
    }
  }
  return { digits, radix };
}
