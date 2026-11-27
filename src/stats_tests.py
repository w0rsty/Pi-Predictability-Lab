# Проверка равномерности (chi2, блочный KS, ACF с поправкой Бонферрони) и оценка сжимаемости (gzip/lzma)
import gzip
import lzma
import math
from typing import Any

import numpy as np
from scipy import stats


def chi_square_uniformity(digits: np.ndarray, radix: int, alpha: float = 0.05) -> dict[str, Any]:
    """Проверяет равномерность частот цифр тестом Хи-квадрат Пирсона."""
    counts = np.bincount(digits, minlength=radix)
    n = int(len(digits))
    expected = np.full(radix, n / radix, dtype=np.float64)
    res = stats.chisquare(f_obs=counts, f_exp=expected)

    probs = counts / n
    # Эмпирическая энтропия Шеннона в битах и натах
    nonzero = probs[probs > 0]
    entropy_nats = float(-np.sum(nonzero * np.log(nonzero)))
    entropy_bits = float(entropy_nats / math.log(2.0))

    return {
        "chi2_stat": float(res.statistic),
        "chi2_p_value": float(res.pvalue),
        "chi2_passed": bool(res.pvalue > alpha),
        "counts": counts.tolist(),
        "freqs": probs.tolist(),
        "min_freq": float(probs.min()),
        "max_freq": float(probs.max()),
        "empirical_entropy_nats": entropy_nats,
        "empirical_entropy_bits": entropy_bits,
        "theoretical_max_bits": float(math.log2(radix)),
    }


def ks_uniformity_blocks(digits: np.ndarray, radix: int, alpha: float = 0.05) -> dict[str, Any]:
    """Проверяет равномерность через тест Колмогорова-Смирнова на непересекающихся блоках цифр.

    Одиночные дискретные цифры (0..9) сравнивать с непрерывным uniform(0, 1) нельзя —
    ступеньки по 0.1 всегда дают D >= 0.05 и p = 0 даже на идеальном генераторе.
    Поэтому собираем блоки по m цифр в числа из [0, 1).
    """
    if radix == 2:
        block_size = 24
    elif radix == 10:
        block_size = 6
    elif radix == 16:
        block_size = 6
    else:
        block_size = max(1, int(math.ceil(20 / math.log2(radix))))

    n_blocks = len(digits) // block_size
    if n_blocks < 10:
        raise ValueError("Слишком короткая последовательность для блочного KS-теста")

    trimmed = digits[: n_blocks * block_size].reshape(n_blocks, block_size).astype(np.float64)
    # Веса d_j * radix^(-j-1) переводят m-значный блок в непрерывное число из [0, 1) с дискретностью <= 10^-6
    powers = float(radix) ** (-np.arange(1, block_size + 1, dtype=np.float64))
    u_samples = trimmed @ powers

    res = stats.kstest(u_samples, "uniform")
    return {
        "ks_block_size": block_size,
        "ks_num_blocks": n_blocks,
        "ks_stat": float(res.statistic),
        "ks_p_value": float(res.pvalue),
        "ks_passed": bool(res.pvalue > alpha),
    }


def autocorrelation_function(digits: np.ndarray, max_lag: int = 64) -> dict[str, Any]:
    """Вычисляет автокорреляцию Пирсона (ACF) для лагов 1..max_lag."""
    if len(digits) < 2:
        raise ValueError(f"Для расчёта ACF требуется >= 2 элементов, получено {len(digits)}")
    if max_lag < 1:
        raise ValueError(f"max_lag должен быть >= 1, получено {max_lag}")

    eff_lag = min(int(max_lag), len(digits) - 1)
    x = digits.astype(np.float64)
    x_centered = x - np.mean(x)
    var = float(np.dot(x_centered, x_centered))

    n = len(x)
    lags = list(range(1, eff_lag + 1))
    if var == 0.0:
        # Вырожденная константа (дисперсия = 0): автокорреляция не определена, отдаём нули
        acf_values = [0.0] * len(lags)
    else:
        acf_values = [
            float(np.dot(x_centered[:-lag], x_centered[lag:]) / var)
            for lag in lags
        ]

    # Доверительный интервал 95%: сырой 1.96/sqrt(N) и с поправкой Бонферрони на eff_lag тестов
    conf_bound_raw = 1.96 / math.sqrt(n)
    conf_bound_bonferroni = float(stats.norm.ppf(1.0 - 0.05 / (2.0 * eff_lag)) / math.sqrt(n))
    abs_vals = [abs(v) for v in acf_values]
    max_idx = int(np.argmax(abs_vals))
    significant_lags_raw = [lag for lag, val in zip(lags, acf_values) if abs(val) > conf_bound_raw]
    significant_lags_bonferroni = [lag for lag, val in zip(lags, acf_values) if abs(val) > conf_bound_bonferroni]

    return {
        "acf_lags": lags,
        "acf_values": acf_values,
        "acf_conf_bound_95": conf_bound_raw,
        "acf_conf_bound_raw": conf_bound_raw,
        "acf_conf_bound_bonferroni": conf_bound_bonferroni,
        "acf_max_abs": float(abs_vals[max_idx]),
        "acf_max_lag": lags[max_idx],
        "acf_significant_count": len(significant_lags_bonferroni),
        "acf_significant_count_raw": len(significant_lags_raw),
        "acf_significant_count_bonferroni": len(significant_lags_bonferroni),
        "acf_significant_lags_sample": significant_lags_bonferroni[:10],
        "acf_significant_lags_raw_sample": significant_lags_raw[:10],
    }


def compression_benchmark(digits: np.ndarray, radix: int) -> dict[str, Any]:
    """Оценивает сжимаемость массива uint8 (1 цифра = 1 байт) алгоритмами gzip и lzma."""
    raw_bytes = digits.astype(np.uint8, copy=False).tobytes()
    raw_len = len(raw_bytes)

    gz_bytes = gzip.compress(raw_bytes, compresslevel=6)
    xz_bytes = lzma.compress(raw_bytes, preset=6)

    gz_len = len(gz_bytes)
    xz_len = len(xz_bytes)

    return {
        "raw_bytes": raw_len,
        "gzip_bytes": gz_len,
        "gzip_ratio": float(gz_len / raw_len),
        "gzip_bits_per_digit": float((gz_len * 8.0) / raw_len),
        "lzma_bytes": xz_len,
        "lzma_ratio": float(xz_len / raw_len),
        "lzma_bits_per_digit": float((xz_len * 8.0) / raw_len),
        "shannon_limit_bits_per_digit": float(math.log2(radix)),
    }


def run_full_statistical_suite(
    name: str,
    digits: np.ndarray,
    radix: int,
    max_lag: int = 64,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Запускает полный набор классических тестов для одной последовательности."""
    chi2_res = chi_square_uniformity(digits, radix=radix, alpha=alpha)
    ks_res = ks_uniformity_blocks(digits, radix=radix, alpha=alpha)
    acf_res = autocorrelation_function(digits, max_lag=max_lag)
    comp_res = compression_benchmark(digits, radix=radix)

    return {
        "name": name,
        "radix": radix,
        "length": int(len(digits)),
        **chi2_res,
        **ks_res,
        **acf_res,
        **comp_res,
    }


run_classical_tests = run_full_statistical_suite
