# Тесты классических бейзлайнов (PPM-C), крипто-закладки и статистического аудита
import math

import numpy as np
import pytest

from src.baselines import PPMCompressor, compute_prediction_metrics
from src.data_generator import _legacy_biased_backdoor_v1, apply_csprng_backdoor, generate_rational_digits
from src.evaluator import compute_bitrate_metrics, compute_mcnemar_exact, wilson_ci
from src.stats_tests import chi_square_uniformity


def test_ppmc_normalization_and_periodic_compression():
    digits = generate_rational_digits(1, 97, length=3000, radix=10)
    train_digits = digits[:2400]
    test_digits = digits[2400:]

    ppm = PPMCompressor(radix=10, max_order=8)
    ppm.fit(train_digits)
    probs = ppm.predict_proba(test_digits, window_size=32)

    assert probs.shape == (len(test_digits) - 32, 10)
    assert np.isfinite(probs).all()
    assert (probs > 0.0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    targets = test_digits[32:]
    metrics = compute_prediction_metrics(probs, targets, mask=None)
    assert metrics["overall_acc"] > 0.99
    assert (metrics["cross_entropy"] / math.log(2.0)) < 0.10


def test_csprng_backdoor_uniformity_vs_legacy_biased_v1():
    rng = np.random.default_rng(123)
    noise = rng.integers(0, 10, size=60_000, dtype=np.uint8)

    legacy_digits, legacy_mask = _legacy_biased_backdoor_v1(noise, lag1=24, lag2=7)
    legacy_triggered = legacy_digits[legacy_mask == 1]
    assert len(legacy_triggered) > 100
    assert set(np.unique(legacy_triggered).tolist()).issubset({1, 3, 5, 7, 9})

    clean_digits, backdoor_mask = apply_csprng_backdoor(noise, lag1=24, lag2=7)
    trig_vals = clean_digits[backdoor_mask == 1]
    assert 0.025 < backdoor_mask.mean() < 0.055
    assert set(np.unique(trig_vals).tolist()) == set(range(10))

    chi_res = chi_square_uniformity(clean_digits, radix=10)
    assert chi_res["chi2_p_value"] > 0.01


def test_statistical_metrics_wilson_mcnemar_bitrate():
    lo, hi = wilson_ci(100, 1000)
    assert 0.08 < lo < 0.10 < hi < 0.125

    res_tied = compute_mcnemar_exact(
        np.array([True, False, True, False]),
        np.array([True, False, True, False]),
    )
    assert res_tied["p_value"] == 1.0
    assert res_tied["significant_0_05"] is False

    corr_a = np.array([True] * 40 + [False] * 10)
    corr_b = np.array([True] * 10 + [False] * 40)
    res_diff = compute_mcnemar_exact(corr_a, corr_b)
    assert res_diff["significant_0_05"] is True
    assert res_diff["p_value"] < 1e-4

    bm = compute_bitrate_metrics(math.log(10), radix=10, correct=100, total=1000)
    assert math.isclose(bm["shannon_limit_bits"], math.log2(10), rel_tol=1e-6)
    assert math.isclose(bm["normalized_bits"], 1.0, rel_tol=1e-6)
