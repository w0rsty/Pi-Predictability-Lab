# Тесты статистики, поправки Холма — Бонферрони, бейзлайнов (Markov_k0, PPM-C) и графиков
import math

import numpy as np
import plotly.graph_objects as go

from src.baselines import MarkovK0Baseline, PPMCompressor, fit_and_evaluate_ppm
from src.data_generator import generate_rational_digits
from src.dataset import split_sequence_chronologically
from src.evaluator import (
    block_bootstrap_accuracy_ci,
    compute_binomial_z_score,
    compute_mcnemar_exact,
    compute_mde,
    holm_bonferroni_correction,
)
from src.stats_tests import autocorrelation_function
from src.viz.classical_and_exp1 import build_exp1_test_xray
from src.viz.theme import save_plotly_figure


def test_holm_bonferroni_and_mde_and_bootstrap() -> None:
    p_adj_1, sig_1 = holm_bonferroni_correction([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert np.allclose(p_adj_1, [0.03, 0.06, 0.06, 0.02])
    assert sig_1 == [True, False, False, True]

    p_adj_2, sig_2 = holm_bonferroni_correction([0.004, 0.02, 0.03, 0.4], alpha=0.05)
    assert np.allclose(p_adj_2, [0.016, 0.06, 0.06, 0.4])
    assert sig_2 == [True, False, False, False]

    mde = compute_mde(n=99_936, p0=0.1, alpha=0.05, power=0.80)
    assert abs(mde["se"] - 0.000949) < 1e-6
    assert abs(mde["mde_two_sided"] - 0.002659) < 1e-5

    periodic_corr = np.tile(np.array([1, 1, 0, 1, 0], dtype=np.float64), 400)
    lo, hi = block_bootstrap_accuracy_ci(periodic_corr, block_size=25, n_resamples=300, seed=42)
    assert 0.0 <= lo <= float(periodic_corr.mean()) <= hi <= 1.0


def test_binomial_z_mcnemar_and_acf_bonferroni() -> None:
    extreme = compute_binomial_z_score(acc=1.0, radix=10, n_samples=4_119)
    assert extreme["z_score"] > 190.0
    assert extreme["z_p_value"] == 0.0
    assert float(extreme["z_log10_p"]) < -300.0

    corr_a = np.ones(1_500, dtype=bool)
    corr_b = np.zeros(1_500, dtype=bool)
    mc_ext = compute_mcnemar_exact(corr_a, corr_b)
    assert mc_ext["p_value_str"].startswith("< 1e-300")
    assert mc_ext["significant_1e_100"] is True

    mc_zero = compute_mcnemar_exact(corr_a, corr_a)
    assert mc_zero["discordant_pairs"] == 0 and mc_zero["p_value"] == 1.0

    rng = np.random.default_rng(42)
    noise = rng.integers(0, 10, size=20_000, dtype=np.uint8)
    acf = autocorrelation_function(noise, max_lag=64)
    assert acf["acf_conf_bound_bonferroni"] > acf["acf_conf_bound_raw"]
    assert acf["acf_significant_count"] == acf["acf_significant_count_bonferroni"]


def test_markov_k0_and_ppm_compressor() -> None:
    biased = np.array(([0] * 80 + [1] * 20) * 10, dtype=np.uint8)
    k0 = MarkovK0Baseline(radix=10, smoothing=0.0).fit(biased)
    assert k0.probs_ is not None
    assert abs(float(k0.probs_[0]) - 0.8) < 1e-9
    assert abs(float(k0.probs_[1]) - 0.2) < 1e-9
    assert np.all(k0.predict(10) == 0)

    split_b = split_sequence_chronologically(biased, None, 0.8, 0.1, 0.1)
    m_k0 = k0.evaluate(split_b, window_size=8)
    assert m_k0["model"] == "Markov_k0" and abs(m_k0["overall_acc"] - 0.8) < 0.05

    # На равномерном i.i.d. шуме PPM-C при order=2 держится около log2(10) ~ 3.322 бит
    noise = np.random.default_rng(7).integers(0, 10, size=40_000, dtype=np.uint8)
    split_n = split_sequence_chronologically(noise, None, 0.8, 0.1, 0.1)
    m_o2, _ = fit_and_evaluate_ppm(split_n, radix=10, window_size=32, max_order=2)
    bits_o2 = m_o2["cross_entropy"] / math.log(2.0)
    assert 3.20 < bits_o2 < 3.45 and m_o2["overall_acc"] < 0.13

    # Radix=16, order=8 должен корректно пройти sparse int64 ветку без ошибок
    hex_seq = np.random.default_rng(9).integers(0, 16, size=3_000, dtype=np.uint8)
    ppm_hex = PPMCompressor(radix=16, max_order=8).fit(hex_seq[:2_000])
    assert 8 in ppm_hex.sparse_tables
    probs_hex = ppm_hex.predict_proba(hex_seq[2_000:], window_size=16)
    assert probs_hex.shape == (1_000 - 16, 16)
    assert np.allclose(probs_hex.sum(axis=1), 1.0, atol=1e-5)

    # На 1/97 (P=96) PPM-C даёт ~100% точность и < 0.1 бита на символ
    rat97 = generate_rational_digits(1, 97, length=5_000, radix=10)
    split_r = split_sequence_chronologically(rat97, None, 0.8, 0.1, 0.1)
    m_r, _ = fit_and_evaluate_ppm(split_r, radix=10, window_size=32, max_order=8)
    assert m_r["overall_acc"] > 0.99 and (m_r["cross_entropy"] / math.log(2.0)) < 0.1


def test_viz_plotly_figure_smoke(tmp_path) -> None:
    runs_by_ds = {
        "true_pi": {
            "test_metrics": {
                "cross_entropy": 2.3025,
                "targeted_ce": None,
                "targeted_confidence": None,
                "unmasked_confidence": 0.100,
            }
        },
        "hacked_pi_tier1": {
            "test_metrics": {
                "cross_entropy": 2.150,
                "targeted_ce": 0.002,
                "targeted_confidence": 0.998,
                "unmasked_confidence": 0.100,
            }
        },
    }
    fig = build_exp1_test_xray("transformer", runs_by_ds)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4
    assert list(fig.data[0].x) == ["true_pi", "hacked_pi_tier1"]
    assert np.allclose(fig.data[0].y, [2.3025, 2.150])
    assert np.allclose(fig.data[2].y, [0.0, 99.8])

    html_p, png_p = save_plotly_figure(fig, tmp_path, "smoke_xray", save_png=False)
    assert png_p is None
    assert html_p.exists() and html_p.stat().st_size > 0
    html_text = html_p.read_text(encoding="utf-8")
    assert "true_pi" in html_text and "hacked_pi_tier1" in html_text
