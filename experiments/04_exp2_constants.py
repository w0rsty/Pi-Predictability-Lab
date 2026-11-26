# Эксперимент 2: битва 9 последовательностей (константы, ГПСЧ, 1/999983, Чемперноун, цепные дроби e и pi).
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baselines import (
    PPMCompressor,
    _encode_markov_states,
    compute_prediction_metrics,
    evaluate_uniform_baseline,
    fit_and_evaluate_lightgbm,
    fit_and_evaluate_markov,
    fit_and_evaluate_ppm,
)
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import ChronologicalSplit, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.models import build_model, count_parameters
from src.stats_tests import run_full_statistical_suite
from src.trainer import create_run_dir, resolve_device, train_model
from src.visualizer import (
    plot_exp2_cf_and_champernowne_contrast,
    plot_exp2_constants_comparison,
    plot_exp2_embedding_pca_preview,
    plot_exp2_heatmap_bits,
    plot_exp2_radar_pyecharts,
)

EXP2_DATASETS = [
    "true_pi",
    "e",
    # На pi_cf точность ~69.3% — не взлом числа pi, а статистика разделителей и распределение Гаусса-Кузьмина (уже в Markov_k1 = 1.5978 бит)
    # Отдельный флаг --arch нужен, чтобы прогонять серии по одной архитектуре и не ронять весь запуск по VRAM
    "sqrt2",
    "mt19937",
    "csprng",
    "pi_cf",
    "rational_1_999983",
    "champernowne",
    "e_cf",
]
NEW_DATASETS = ["champernowne", "e_cf", "pi_cf"]
H0_GROUP = {"true_pi", "e", "sqrt2", "mt19937", "csprng", "pi_cf"}
H0_DECIMAL_DATASETS = ["true_pi", "e", "sqrt2", "mt19937", "csprng"]

CANONICAL_MODEL_MAP = {
    "MLP": "DigitMLP",
    "CNN": "DigitCNN",
    "GRU": "DigitGRU",
    "TRANSFORMER": "DigitTransformer",
}
CANONICAL_TO_ARCH = {
    "DigitMLP": "mlp",
    "DigitCNN": "cnn",
    "DigitGRU": "gru",
    "DigitTransformer": "transformer",
}
MODEL_ORDER = [
    "Uniform",
    "Markov_k1",
    "Markov_k2",
    "Markov_k3",
    "LightGBM",
    "PPM_o8",
    "DigitMLP",
    "DigitCNN",
    "DigitGRU",
    "DigitTransformer",
]

# Целевые чекпойнты для Эксперимента 5 (механистическая интерпретируемость)
EXP5_CHECKPOINT_TARGETS: dict[str, list[str]] = {
    "DigitTransformer": ["e_cf", "champernowne", "rational_1_999983", "pi_cf", "true_pi"],
    "DigitGRU": ["rational_1_999983", "champernowne", "e_cf", "true_pi"],
}


def canonical_model_name(name: str) -> str:
    """Приводит имена нейросетей (MLP/CNN/GRU/TRANSFORMER) к каноническим Digit*."""
    return CANONICAL_MODEL_MAP.get(name, name)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Точный 95% доверительный интервал Уилсона для доли k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)


def compute_mcnemar_exact(corr_a: np.ndarray, corr_b: np.ndarray) -> dict[str, Any]:
    """Считает точный парный тест Макнемара (биномиальный + Эдвардс chi2) между двумя моделями."""
    if len(corr_a) != len(corr_b):
        raise ValueError(f"Длины массивов не совпадают: {len(corr_a)} vs {len(corr_b)}")
    n = int(len(corr_a))
    both_correct = int(np.sum(corr_a & corr_b))
    b_wins = int(np.sum(corr_a & ~corr_b))   # Модель A угадала, модель B ошиблась
    c_losses = int(np.sum(~corr_a & corr_b)) # Модель A ошиблась, модель B угадала
    both_wrong = int(np.sum(~corr_a & ~corr_b))
    n_disc = b_wins + c_losses

    if n_disc == 0:
        return {
            "n_samples": n,
            "acc_a": round(float(np.mean(corr_a)), 6),
            "acc_b": round(float(np.mean(corr_b)), 6),
            "both_correct": both_correct,
            "b_a_wins_only": b_wins,
            "c_b_wins_only": c_losses,
            "both_wrong": both_wrong,
            "discordant_pairs": 0,
            "chi2_edwards": 0.0,
            "p_value": 1.0,
            "log10_p_value": 0.0,
            "p_value_str": "1.0000",
            "significant_0_05": False,
            "significant_1e_100": False,
        }

    chi2_cc = ((abs(b_wins - c_losses) - 1.0) ** 2) / n_disc
    k_min = min(b_wins, c_losses)
    # Используем logcdf, а при -inf на сверхмалых хвостах — точную нормальную лог-асимптотику
    log_one_side = float(stats.binom.logcdf(k_min, n_disc, 0.5))
    if not math.isfinite(log_one_side):
        log_one_side = float(stats.norm.logsf(math.sqrt(max(chi2_cc, 1e-12))))
    log10_p = min(0.0, (log_one_side + math.log(2.0)) / math.log(10.0))
    p_val = float(stats.binomtest(k_min, n=n_disc, p=0.5, alternative="two-sided").pvalue)

    if log10_p < -300.0:
        p_str = f"< 1e-300 (log10_p ≈ {log10_p:.1f})"
    else:
        p_str = f"{p_val:.4e}" if p_val < 1e-4 else f"{p_val:.6f}"

    return {
        "n_samples": n,
        "acc_a": round(float(np.mean(corr_a)), 6),
        "acc_b": round(float(np.mean(corr_b)), 6),
        "both_correct": both_correct,
        "b_a_wins_only": b_wins,
        "c_b_wins_only": c_losses,
        "both_wrong": both_wrong,
        "discordant_pairs": int(n_disc),
        "chi2_edwards": round(float(chi2_cc), 4),
        "p_value": p_val,
        "log10_p_value": round(float(log10_p), 2),
        "p_value_str": p_str,
        "significant_0_05": bool(log10_p < math.log10(0.05)),
        "significant_1e_100": bool(log10_p < -100.0),
    }


def _find_latest_csv(runs_root: Path, prefix: str, filename: str = "metrics.csv") -> Path | None:
    cands = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in cands:
        p = d / filename
        if p.exists():
            return p
    return None


def _resolve_run_dir(source_str: str | None) -> Path | None:
    if not source_str:
        return None
    p = Path(source_str)
    if p.is_absolute() and p.exists():
        return p
    rel = ROOT_DIR / source_str
    if rel.exists():
        return rel
    in_runs = ROOT_DIR / "runs" / source_str
    if in_runs.exists():
        return in_runs
    return None


def load_cached_results(runs_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Подтягивает уже посчитанные пары (dataset, model) из Фаз 2, 3, 3.5 и прошлых Exp 2."""
    cached: dict[tuple[str, str], dict[str, Any]] = {}

    # 1. Бейзлайны из Фазы 2
    p2_csv = _find_latest_csv(runs_root, "phase2_baselines_")
    if p2_csv is not None:
        df2 = pd.read_csv(p2_csv)
        for _, row in df2.iterrows():
            ds = str(row["dataset"])
            if ds not in EXP2_DATASETS:
                continue
            model = canonical_model_name(str(row["model"]))
            cached[(ds, model)] = {
                "dataset": ds,
                "radix": int(row["radix"]),
                "model": model,
                "test_acc": float(row["overall_acc"]),
                "test_ce": float(row["cross_entropy"]),
                "test_ppl": float(row["perplexity"]),
                "pred_entropy": float(row["pred_entropy"]),
                "num_eval_samples": int(row["num_eval_samples"]),
                "train_time_sec": float(row["train_time_sec"]),
                "source": p2_csv.parent.name,
            }

    # 2. PPM-C из Фазы 3.5 (true_pi)
    p35_csv = _find_latest_csv(runs_root, "phase3_5_ppm_summary_")
    if p35_csv is not None:
        df35 = pd.read_csv(p35_csv)
        for _, row in df35.iterrows():
            ds = str(row["dataset"])
            if ds not in EXP2_DATASETS:
                continue
            model = canonical_model_name(str(row["model"]))
            cached[(ds, model)] = {
                "dataset": ds,
                "radix": 10,
                "model": model,
                "test_acc": float(row["overall_acc"]),
                "test_ce": float(row["cross_entropy"]),
                "test_ppl": float(row["perplexity"]),
                "pred_entropy": float(row["pred_entropy"]),
                "num_eval_samples": 99936,
                "train_time_sec": float(row["train_time_sec"]),
                "source": p35_csv.parent.name,
            }

    # 3. Нейросети из Фазы 3 (true_pi)
    for arch in ("mlp", "cnn", "gru", "transformer"):
        p3_csv = _find_latest_csv(runs_root, f"exp1_{arch}_summary_")
        if p3_csv is None:
            continue
        df3 = pd.read_csv(p3_csv)
        for _, row in df3.iterrows():
            ds = str(row["dataset"])
            if ds not in EXP2_DATASETS:
                continue
            m_name = canonical_model_name(arch.upper())
            cached[(ds, m_name)] = {
                "dataset": ds,
                "radix": 10,
                "model": m_name,
                "test_acc": float(row["test_overall_acc"]),
                "test_ce": float(row["test_cross_entropy"]),
                "test_ppl": float(row["test_perplexity"]),
                "pred_entropy": float(row["test_pred_entropy"]),
                "num_eval_samples": 99936,
                "train_time_sec": float(row["train_time_sec"]),
                "param_count": int(row["param_count"]),
                "epochs_ran": int(row["epochs_ran"]),
                "best_epoch": int(row["best_epoch"]),
                "source": str(row["run_dir"]),
            }

    # 4. Готовые результаты предыдущих запусков Exp 2 (сортировка от ранних к свежим)
    exp2_dirs = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("exp2_constants_")],
        key=lambda p: p.name,
    )
    for d in exp2_dirs:
        csv_p = d / "metrics.csv"
        if not csv_p.exists():
            continue
        df_e2 = pd.read_csv(csv_p)
        for _, row in df_e2.iterrows():
            ds = str(row["dataset"])
            m_name = canonical_model_name(str(row["model"]))
            ce_val = row["test_ce_nats"] if "test_ce_nats" in df_e2.columns else row["test_ce"]
            p_cnt = row.get("params_count", row.get("param_count"))
            cached[(ds, m_name)] = {
                "dataset": ds,
                "radix": int(row["radix"]),
                "model": m_name,
                "test_acc": float(row["test_acc"]),
                "test_ce": float(ce_val),
                "test_ppl": float(row["test_ppl"]),
                "pred_entropy": float(row["pred_entropy"]) if pd.notna(row.get("pred_entropy")) else None,
                "num_eval_samples": int(row["num_eval_samples"]),
                "train_time_sec": float(row["train_time_sec"]) if pd.notna(row.get("train_time_sec")) else None,
                "param_count": int(p_cnt) if pd.notna(p_cnt) else None,
                "epochs_ran": int(row["epochs_ran"]) if pd.notna(row.get("epochs_ran")) else None,
                "best_epoch": int(row["best_epoch"]) if pd.notna(row.get("best_epoch")) else None,
                "sec_per_epoch": float(row["sec_per_epoch"]) if pd.notna(row.get("sec_per_epoch")) else None,
                "samples_per_sec": float(row["samples_per_sec"]) if pd.notna(row.get("samples_per_sec")) else None,
                "source": str(row.get("source", d.name)),
            }

    # 5. Индивидуальные прогоны exp2_<arch>_<dataset>_* (подхват, если сводка ещё не записалась)
    for arch in ("mlp", "cnn", "gru", "transformer"):
        m_name = canonical_model_name(arch.upper())
        for ds in EXP2_DATASETS:
            if (ds, m_name) in cached:
                continue
            pat = re.compile(rf"^exp2_{re.escape(arch)}_{re.escape(ds)}_\d{{4}}-\d{{2}}-\d{{2}}_")
            ind_dirs = sorted(
                [p for p in runs_root.iterdir() if p.is_dir() and pat.match(p.name)],
                key=lambda p: p.name,
            )
            for d in reversed(ind_dirs):
                pd_path = d / "plot_data.json"
                ck_path = d / "best_model.pt"
                if not pd_path.exists() or not ck_path.exists():
                    continue
                data = json.loads(pd_path.read_text(encoding="utf-8"))
                s = data.get("summary", {})
                tm = s.get("test_metrics", {})
                hist = data.get("history", [])
                ep_times = [float(h["epoch_time_sec"]) for h in hist if h.get("epoch_time_sec") is not None]
                sec_ep = round(float(np.mean(ep_times)), 3) if ep_times else None
                cached[(ds, m_name)] = {
                    "dataset": ds,
                    "radix": 11 if ds in ("e_cf", "pi_cf") else 10,
                    "model": m_name,
                    "test_acc": float(tm["overall_acc"]),
                    "test_ce": float(tm["cross_entropy"]),
                    "test_ppl": float(tm["perplexity"]),
                    "pred_entropy": float(tm["pred_entropy"]) if tm.get("pred_entropy") is not None else None,
                    "num_eval_samples": int(tm["num_eval_samples"]),
                    "train_time_sec": float(s["total_train_time_sec"]),
                    "param_count": int(s["param_count"]),
                    "epochs_ran": int(s["epochs_ran"]),
                    "best_epoch": int(s["best_epoch"]),
                    "sec_per_epoch": sec_ep,
                    "source": str(s.get("run_dir", d)),
                }
                break

    return cached


def _read_sec_per_epoch(run_dir: Path | None, fallback_total: float | None, epochs_ran: int | None) -> float | None:
    if run_dir is not None:
        pd_path = run_dir / "plot_data.json"
        if pd_path.exists():
            data = json.loads(pd_path.read_text(encoding="utf-8"))
            hist = data.get("history", [])
            ep_times = [float(h["epoch_time_sec"]) for h in hist if h.get("epoch_time_sec") is not None]
            if ep_times:
                return round(float(np.mean(ep_times)), 3)
    if fallback_total is not None and epochs_ran and epochs_ran > 0:
        return round(float(fallback_total) / float(epochs_ran), 3)
    return None


def _eval_markov_preds_and_speed(split: ChronologicalSplit, radix: int, order: int, window_size: int) -> tuple[np.ndarray, float]:
    num_states = radix**order
    train_states = _encode_markov_states(split.train_digits, order=order, radix=radix)
    train_targets = split.train_digits[order:].astype(np.int64)
    flat_idx = train_states * radix + train_targets
    counts = np.bincount(flat_idx, minlength=num_states * radix).reshape(num_states, radix).astype(np.float64)
    probs_table = (counts + 1e-3) / (counts.sum(axis=1, keepdims=True) + 1e-3 * radix)

    test_slice = split.test_digits[window_size - order :]
    test_targets = split.test_digits[window_size:]
    t0 = time.perf_counter()
    test_states = _encode_markov_states(test_slice, order=order, radix=radix)
    pred_probs = probs_table[test_states]
    preds = np.argmax(pred_probs, axis=1)
    elapsed = max(time.perf_counter() - t0, 1e-5)
    return (preds == test_targets), round(len(test_targets) / elapsed, 1)


def _eval_ppm_preds_and_speed(split: ChronologicalSplit, radix: int, window_size: int, max_order: int = 8) -> tuple[np.ndarray, float]:
    ppm = PPMCompressor(radix=radix, max_order=max_order).fit(split.train_digits)
    test_targets = split.test_digits[window_size:]
    t0 = time.perf_counter()
    probs = ppm.predict_proba(split.test_digits, window_size=window_size)
    preds = np.argmax(probs, axis=1)
    elapsed = max(time.perf_counter() - t0, 1e-5)
    return (preds == test_targets), round(len(test_targets) / elapsed, 1)


def _remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    """Конвертирует ключи nn.MultiheadAttention (Фаза 3) в ключи нативного SDPA (Фаза 4)."""
    remapped: dict[str, torch.Tensor] = {}
    changed = False
    for k, v in state_dict.items():
        nk = (
            k.replace(".attn.in_proj_weight", ".qkv_proj.weight")
            .replace(".attn.in_proj_bias", ".qkv_proj.bias")
            .replace(".attn.out_proj.", ".out_proj.")
        )
        if nk != k:
            changed = True
        remapped[nk] = v
    return remapped, changed


@torch.no_grad()
def _benchmark_and_predict_neural(
    arch: str,
    run_dir: Path,
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
    size_tier: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Загружает best_model.pt, замеряет GPU throughput на Test и возвращает bool-вектор верных ответов + метаданные."""
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Не найден чекпойнт {ckpt_path}")

    loaders = create_dataloaders(
        split=split,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict, was_remapped = _remap_legacy_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    amp_enabled = device.type == "cuda"
    test_loader = loaders["test"]

    # Короткий прогрев GPU перед точным замером скорости инференса
    for contexts, _, _ in test_loader:
        contexts = contexts.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _ = model(contexts)
        break

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    correct_chunks: list[np.ndarray] = []
    for contexts, targets, _ in test_loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(contexts)
        preds = torch.argmax(logits.float(), dim=-1)
        correct_chunks.append((preds == targets).cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-5)

    correct_bool = np.concatenate(correct_chunks, axis=0)
    samples_per_sec = round(float(len(correct_bool) / elapsed), 1)

    meta: dict[str, Any] = {
        "arch": arch,
        "radix": int(radix),
        "window_size": int(window_size),
        "size_tier": size_tier,
        "param_count": int(count_parameters(model)),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
        "ckpt_path": ckpt_path,
        "was_remapped_legacy": was_remapped,
    }
    if arch == "transformer":
        meta.update(
            {
                "d_model": int(model.d_model),
                "n_heads": int(model.n_heads),
                "n_layers": int(model.n_layers),
                "token_emb": model.token_emb.weight.detach().cpu().numpy().copy(),
            }
        )
    elif arch == "gru":
        meta.update(
            {
                "hidden_size": int(model.hidden_size),
                "num_layers": int(model.num_layers),
                "emb_dim": int(model.emb_dim),
            }
        )
    elif arch == "mlp":
        meta.update({"emb_dim": int(model.emb_dim)})
    elif arch == "cnn":
        meta.update({"channels": int(model.channels), "head_hidden": int(model.head_hidden)})

    return correct_bool, samples_per_sec, meta


def _compute_radar_scores(df_final: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Считает 5 осей радара в единой шкале [0, 100] для PPM_o8 и 4 нейроархитектур."""
    families = ["PPM_o8", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    shannon_10 = math.log2(10.0)
    piv_bits = df_final.pivot(index="dataset", columns="model", values="bits_per_symbol")
    piv_speed = df_final.pivot(index="dataset", columns="model", values="samples_per_sec")

    median_speeds: dict[str, float] = {}
    for m_name in families:
        if m_name in piv_speed.columns:
            vals = piv_speed[m_name].dropna().astype(float)
            median_speeds[m_name] = float(vals.median()) if not vals.empty else 0.0
        else:
            median_speeds[m_name] = 0.0
    max_speed = max(max(median_speeds.values()), 1.0)

    scores: dict[str, dict[str, float]] = {}
    for m_name in families:
        if m_name not in piv_bits.columns:
            continue
        col = piv_bits[m_name]
        mean_h0 = float(col.reindex(H0_DECIMAL_DATASETS).mean())
        bits_rat = float(col["rational_1_999983"])
        bits_champ = float(col["champernowne"])
        bits_ecf = float(col["e_cf"])
        bits_picf = float(col["pi_cf"])

        noise_cal = 100.0 * math.exp(-10.0 * abs(mean_h0 - shannon_10))
        periodic_mem = max(0.0, min(100.0, 100.0 * (1.0 - bits_rat / shannon_10)))
        counting = max(0.0, min(100.0, 100.0 * (1.0 - bits_champ / shannon_10)))
        cf_arith = max(0.0, min(100.0, 100.0 * (1.0 - bits_ecf / max(bits_picf, 1e-9))))
        # Квадратный корень отношения пропускной способности честно разделяет модели без сплющивания хвоста
        throughput = 100.0 * math.sqrt(median_speeds[m_name] / max_speed)

        scores[m_name] = {
            "noise_calibration": round(noise_cal, 2),
            "periodic_memory": round(periodic_mem, 2),
            "multi_digit_counting": round(counting, 2),
            "cf_arithmetic": round(cf_arith, 2),
            "throughput": round(throughput, 2),
            "median_samples_per_sec": round(median_speeds[m_name], 1),
        }
    return scores


def run_experiment_2(
    archs: list[str],
    size_tier: str = "medium",
    max_epochs: int = 10,
    patience: int = 3,
    reuse_completed: bool = True,
    stage_tag: str = "final",
    finalize: bool = True,
) -> Path:
    """Выполняет генерацию, бейзлайны, обучение нейросетей, статистический аудит и финальные графики."""
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed = int(config["seed"])
    seed_everything(seed)

    window_size = int(config["data"]["default_window_size"])
    split_cfg = config["data"]["split"]
    train_cfg = config["training"]
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 4))

    cached = load_cached_results(runs_root) if reuse_completed else {}
    records: dict[tuple[str, str], dict[str, Any]] = dict(cached)

    # 1. Загрузка/генерация 9 датасетов и статистика новых констант
    artifacts = {}
    splits: dict[str, ChronologicalSplit] = {}
    new_stats_rows: list[dict[str, Any]] = []
    new_stats_full: list[dict[str, Any]] = []

    print("=== Шаг 1: Загрузка/генерация 9 датасетов и проверка leakage-free разбиения ===")
    for ds_name in EXP2_DATASETS:
        t0 = time.perf_counter()
        art = get_or_create_dataset(name=ds_name, config=config)
        gen_sec = time.perf_counter() - t0
        if ds_name in NEW_DATASETS:
            status_lbl = "из кэша" if art.from_cache else "сгенерирован с нуля"
            print(f"[{ds_name}] {status_lbl}: {art.length:,} элементов (radix={art.radix}) за {gen_sec:.3f} с")

        artifacts[ds_name] = art
        split = split_sequence_chronologically(
            digits=art.digits,
            mask=art.mask,
            train_ratio=float(split_cfg["train"]),
            val_ratio=float(split_cfg["val"]),
            test_ratio=float(split_cfg["test"]),
        )
        verify_zero_leakage(split, window_size=window_size)
        splits[ds_name] = split

        if ds_name in NEW_DATASETS:
            st = run_full_statistical_suite(ds_name, art.digits, radix=art.radix, max_lag=64)
            new_stats_full.append(st)
            new_stats_rows.append(
                {
                    "name": st["name"],
                    "radix": st["radix"],
                    "length": st["length"],
                    "chi2_stat": round(st["chi2_stat"], 4),
                    "chi2_p_value": st["chi2_p_value"],
                    "chi2_passed": st["chi2_passed"],
                    "ks_stat": round(st["ks_stat"], 6),
                    "ks_p_value": st["ks_p_value"],
                    "ks_passed": st["ks_passed"],
                    "acf_max_abs": round(st["acf_max_abs"], 6),
                    "acf_max_lag": st["acf_max_lag"],
                    "acf_significant_count": st["acf_significant_count"],
                    "gzip_ratio": round(st["gzip_ratio"], 5),
                    "lzma_ratio": round(st["lzma_ratio"], 5),
                    "lzma_bits_per_digit": round(st["lzma_bits_per_digit"], 4),
                    "shannon_limit_bits": round(st["shannon_limit_bits_per_digit"], 4),
                    "empirical_entropy_nats": round(st["empirical_entropy_nats"], 6),
                }
            )

    # 2. Недостающие бейзлайны
    print("\n=== Шаг 2: Проверка/досчёт классических бейзлайнов и PPM-C ===")
    for ds_name in EXP2_DATASETS:
        art = artifacts[ds_name]
        split = splits[ds_name]
        n_eval = len(split.test_digits) - window_size

        if (ds_name, "Uniform") not in records:
            u = evaluate_uniform_baseline(split, radix=art.radix, window_size=window_size)
            records[(ds_name, "Uniform")] = {
                "dataset": ds_name,
                "radix": art.radix,
                "model": "Uniform",
                "test_acc": u["overall_acc"],
                "test_ce": u["cross_entropy"],
                "test_ppl": u["perplexity"],
                "pred_entropy": u["pred_entropy"],
                "num_eval_samples": n_eval,
                "train_time_sec": 0.0,
                "samples_per_sec": round(n_eval / max(float(u["eval_time_sec"]), 1e-4), 1),
                "source": "exp2_fresh",
            }

        for k in (1, 2, 3):
            m_key = f"Markov_k{k}"
            if (ds_name, m_key) not in records:
                mk, _ = fit_and_evaluate_markov(split, radix=art.radix, order=k, window_size=window_size)
                records[(ds_name, m_key)] = {
                    "dataset": ds_name,
                    "radix": art.radix,
                    "model": m_key,
                    "test_acc": mk["overall_acc"],
                    "test_ce": mk["cross_entropy"],
                    "test_ppl": mk["perplexity"],
                    "pred_entropy": mk["pred_entropy"],
                    "num_eval_samples": n_eval,
                    "train_time_sec": mk["train_time_sec"],
                    "samples_per_sec": round(n_eval / max(float(mk["eval_time_sec"]), 1e-4), 1),
                    "source": "exp2_fresh",
                }

        if (ds_name, "LightGBM") not in records:
            lgb_res = fit_and_evaluate_lightgbm(
                split,
                radix=art.radix,
                window_size=window_size,
                seed=seed,
                max_train_samples=200_000,
                n_estimators=150,
            )
            records[(ds_name, "LightGBM")] = {
                "dataset": ds_name,
                "radix": art.radix,
                "model": "LightGBM",
                "test_acc": lgb_res["overall_acc"],
                "test_ce": lgb_res["cross_entropy"],
                "test_ppl": lgb_res["perplexity"],
                "pred_entropy": lgb_res["pred_entropy"],
                "num_eval_samples": n_eval,
                "train_time_sec": lgb_res["train_time_sec"],
                "samples_per_sec": round(n_eval / max(float(lgb_res["eval_time_sec"]), 1e-4), 1),
                "source": "exp2_fresh",
            }

        if (ds_name, "PPM_o8") not in records:
            ppm_res, _ = fit_and_evaluate_ppm(split, radix=art.radix, window_size=window_size, max_order=8)
            records[(ds_name, "PPM_o8")] = {
                "dataset": ds_name,
                "radix": art.radix,
                "model": "PPM_o8",
                "test_acc": ppm_res["overall_acc"],
                "test_ce": ppm_res["cross_entropy"],
                "test_ppl": ppm_res["perplexity"],
                "pred_entropy": ppm_res["pred_entropy"],
                "num_eval_samples": n_eval,
                "train_time_sec": ppm_res["train_time_sec"],
                "samples_per_sec": round(n_eval / max(float(ppm_res["eval_time_sec"]), 1e-4), 1),
                "source": "exp2_fresh",
            }

    # 3. Обучение запрошенных нейросетей (Шаг 4B-2: DigitTransformer на 8 датасетах)
    for arch in archs:
        m_name = canonical_model_name(arch.upper())
        for ds_name in EXP2_DATASETS:
            if (ds_name, m_name) in records:
                print(f"[{m_name}] Датасет {ds_name}: используем кэш ({records[(ds_name, m_name)]['source']})")
                continue

            seed_everything(seed)
            art = artifacts[ds_name]
            split = splits[ds_name]
            loaders = create_dataloaders(
                split=split,
                window_size=window_size,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            model = build_model(arch=arch, radix=art.radix, window_size=window_size, size_tier=size_tier)
            print(f"\n=== [Exp 2 | {m_name} | {size_tier}] Датасет: {ds_name} (radix={art.radix}) | Параметров: {count_parameters(model):,} ===")
            res = train_model(
                model=model,
                loaders=loaders,
                config=config,
                exp_name=f"exp2_{arch}_{ds_name}",
                radix=art.radix,
                epochs=max_epochs,
                early_stopping_patience=patience,
                enable_early_stopping=True,
            )
            s = res["summary"]
            tm = res["test_metrics"]
            hist = res["history"]
            sec_ep = round(float(np.mean([h["epoch_time_sec"] for h in hist])), 3) if hist else None

            records[(ds_name, m_name)] = {
                "dataset": ds_name,
                "radix": art.radix,
                "model": m_name,
                "test_acc": float(tm["overall_acc"]),
                "test_ce": float(tm["cross_entropy"]),
                "test_ppl": float(tm["perplexity"]),
                "pred_entropy": float(tm["pred_entropy"]),
                "num_eval_samples": int(tm["num_eval_samples"]),
                "train_time_sec": float(s["total_train_time_sec"]),
                "param_count": int(s["param_count"]),
                "epochs_ran": int(s["epochs_ran"]),
                "best_epoch": int(s["best_epoch"]),
                "sec_per_epoch": sec_ep,
                "source": s["run_dir"],
            }

    summary_dir = create_run_dir(runs_root, f"exp2_constants_{stage_tag}")
    (summary_dir / "config_snapshot.yaml").write_text(
        (ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # 4. Аудит производительности на RTX 3060 Ti, по-примерные предсказания для McNemar и сбор весов для Exp 5
    preds_cache: dict[tuple[str, str], np.ndarray] = {}
    neural_meta: dict[tuple[str, str], dict[str, Any]] = {}
    pca_data: dict[str, dict[str, Any]] = {}

    if finalize:
        print("\n=== Шаг 4: Бенчмарк throughput на RTX 3060 Ti, McNemar-аудит и сбор чекпойнтов для Exp 5 ===")
        for ds_name in EXP2_DATASETS:
            art = artifacts[ds_name]
            split = splits[ds_name]
            n_eval = len(split.test_digits) - window_size

            # Оцениваем по-примерно Markov_k1..k3 и PPM_o8 (мгновенно) + скорости
            for k in (1, 2, 3):
                mk_corr, mk_sps = _eval_markov_preds_and_speed(split, radix=art.radix, order=k, window_size=window_size)
                if k == 3:
                    preds_cache[(ds_name, "Markov_k3")] = mk_corr
                if (ds_name, f"Markov_k{k}") in records:
                    records[(ds_name, f"Markov_k{k}")]["samples_per_sec"] = mk_sps

            ppm_corr, ppm_sps = _eval_ppm_preds_and_speed(split, radix=art.radix, window_size=window_size, max_order=8)
            preds_cache[(ds_name, "PPM_o8")] = ppm_corr
            if (ds_name, "PPM_o8") in records:
                records[(ds_name, "PPM_o8")]["samples_per_sec"] = ppm_sps

            if (ds_name, "Uniform") in records and not records[(ds_name, "Uniform")].get("samples_per_sec"):
                records[(ds_name, "Uniform")]["samples_per_sec"] = round(n_eval / 0.001, 1)

            # Чекпойнты нейросетей на RTX 3060 Ti
            for m_name, arch in CANONICAL_TO_ARCH.items():
                rec = records.get((ds_name, m_name))
                if rec is None:
                    continue
                run_dir = _resolve_run_dir(rec.get("source"))
                if run_dir is None:
                    continue
                rec["sec_per_epoch"] = _read_sec_per_epoch(
                    run_dir=run_dir,
                    fallback_total=rec.get("train_time_sec"),
                    epochs_ran=rec.get("epochs_ran"),
                )
                corr, sps, meta = _benchmark_and_predict_neural(
                    arch=arch,
                    run_dir=run_dir,
                    split=split,
                    radix=art.radix,
                    window_size=window_size,
                    size_tier=size_tier,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                )
                preds_cache[(ds_name, m_name)] = corr
                rec["samples_per_sec"] = sps
                meta["source_run_dir"] = str(run_dir)
                neural_meta[(ds_name, m_name)] = meta

                # Предварительный 2D PCA эмбеддингов DigitTransformer для ключевых 3 датасетов
                if m_name == "DigitTransformer" and ds_name in ("true_pi", "e_cf", "champernowne"):
                    emb_w = meta.pop("token_emb")
                    pca = PCA(n_components=2, random_state=seed)
                    coords = pca.fit_transform(emb_w)
                    pca_data[ds_name] = {
                        "radix": int(art.radix),
                        "coords": np.round(coords, 6).tolist(),
                        "explained_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_],
                        "explained_variance_sum": round(float(np.sum(pca.explained_variance_ratio_)), 6),
                    }
                else:
                    meta.pop("token_emb", None)

        # Копируем целевые чекпойнты для Эксперимента 5
        ckpt_dir = summary_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_manifest: list[dict[str, Any]] = []
        for m_name, ds_list in EXP5_CHECKPOINT_TARGETS.items():
            arch = CANONICAL_TO_ARCH[m_name]
            for ds_name in ds_list:
                meta = neural_meta.get((ds_name, m_name))
                if meta is None:
                    continue
                src_ckpt: Path = meta["ckpt_path"]
                dst_name = f"{arch}_{ds_name}.pt"
                dst_ckpt = ckpt_dir / dst_name
                if meta.get("was_remapped_legacy"):
                    raw_ck = torch.load(src_ckpt, map_location="cpu", weights_only=True)
                    remapped_sd, _ = _remap_legacy_state_dict(raw_ck["model_state_dict"])
                    raw_ck["model_state_dict"] = remapped_sd
                    torch.save(raw_ck, dst_ckpt)
                else:
                    shutil.copy2(src_ckpt, dst_ckpt)
                entry = {k: v for k, v in meta.items() if k != "ckpt_path"}
                entry["model"] = m_name
                entry["dataset"] = ds_name
                entry["checkpoint_file"] = dst_name
                ckpt_manifest.append(entry)

        (ckpt_dir / "checkpoints_metadata.json").write_text(
            json.dumps(ckpt_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Сохранено чекпойнтов для Эксперимента 5: {len(ckpt_manifest)} -> {ckpt_dir}")

    # 5. Расчёт итоговых метрик, Wilson 95% CI и gain_over_markov_bits
    best_markov_bits_by_ds: dict[str, float] = {}
    for ds_name in EXP2_DATASETS:
        mk_bits = [
            records[(ds_name, f"Markov_k{k}")]["test_ce"] / math.log(2.0)
            for k in (1, 2, 3)
            if (ds_name, f"Markov_k{k}") in records
        ]
        best_markov_bits_by_ds[ds_name] = min(mk_bits) if mk_bits else math.log2(artifacts[ds_name].radix)

    final_rows: list[dict[str, Any]] = []
    for ds_name in EXP2_DATASETS:
        art = artifacts[ds_name]
        shannon_bits = math.log2(art.radix)
        best_mk_bits = best_markov_bits_by_ds[ds_name]
        hyp_group = "H0" if ds_name in H0_GROUP else "H1"

        for m_name in MODEL_ORDER:
            if (ds_name, m_name) not in records:
                continue
            r = dict(records[(ds_name, m_name)])
            n_eval = int(r["num_eval_samples"])
            k_corr = int(round(r["test_acc"] * n_eval))
            ci_lo, ci_hi = wilson_ci(k_corr, n_eval)
            bits = r["test_ce"] / math.log(2.0)

            final_rows.append(
                {
                    "dataset": ds_name,
                    "radix": art.radix,
                    "hypothesis_group": hyp_group,
                    "model": m_name,
                    "params_count": r.get("param_count"),
                    "test_acc": round(r["test_acc"], 6),
                    "wilson_ci_low": ci_lo,
                    "wilson_ci_high": ci_hi,
                    "test_ce_nats": round(r["test_ce"], 6),
                    "test_ppl": round(r["test_ppl"], 4),
                    "bits_per_symbol": round(bits, 6),
                    "shannon_limit_bits": round(shannon_bits, 6),
                    "gain_over_markov_bits": round(best_mk_bits - bits, 6),
                    "sec_per_epoch": r.get("sec_per_epoch"),
                    "samples_per_sec": r.get("samples_per_sec"),
                    "pred_entropy": round(r["pred_entropy"], 6) if r.get("pred_entropy") is not None else None,
                    "num_eval_samples": n_eval,
                    "epochs_ran": r.get("epochs_ran"),
                    "best_epoch": r.get("best_epoch"),
                    "train_time_sec": r.get("train_time_sec"),
                    "length": art.length,
                    "source": r.get("source"),
                }
            )

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(summary_dir / "metrics.csv", index=False)
    if new_stats_rows:
        pd.DataFrame(new_stats_rows).to_csv(summary_dir / "classical_stats_new_constants.csv", index=False)

    # 6. Парный тест Макнемара и запись stat_audit.json
    stat_audit: dict[str, Any] = {}
    if finalize:
        for ds_name in EXP2_DATASETS:
            hyp_group = "H0" if ds_name in H0_GROUP else "H1"
            neural_rows = [
                row for row in final_rows if row["dataset"] == ds_name and row["model"] in CANONICAL_TO_ARCH
            ]
            best_neural = min(neural_rows, key=lambda r: (r["bits_per_symbol"], -r["test_acc"]))
            best_n_name = best_neural["model"]

            ds_audit: dict[str, Any] = {
                "hypothesis_group": hyp_group,
                "best_neural_model": best_n_name,
                "best_neural_bits": best_neural["bits_per_symbol"],
                "best_neural_acc": best_neural["test_acc"],
                "comparisons": {},
            }
            pairs = [
                ("DigitTransformer_vs_Markov_k3", "DigitTransformer", "Markov_k3"),
                ("DigitTransformer_vs_PPM_o8", "DigitTransformer", "PPM_o8"),
                ("BestNeural_vs_Markov_k3", best_n_name, "Markov_k3"),
                ("BestNeural_vs_PPM_o8", best_n_name, "PPM_o8"),
            ]
            for label, ma, mb in pairs:
                if (ds_name, ma) in preds_cache and (ds_name, mb) in preds_cache:
                    res_mc = compute_mcnemar_exact(preds_cache[(ds_name, ma)], preds_cache[(ds_name, mb)])
                    res_mc["model_a"] = ma
                    res_mc["model_b"] = mb
                    ds_audit["comparisons"][label] = res_mc
            stat_audit[ds_name] = ds_audit

        (summary_dir / "stat_audit.json").write_text(
            json.dumps(stat_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 7. Финальные графики (4 графика .html + .png) + plot_data.json
    radar_scores = _compute_radar_scores(df_final) if finalize else {}
    with (summary_dir / "plot_data.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": final_rows,
                "classical_stats_new_constants": new_stats_full,
                "radar_scores": radar_scores,
                "radar_formulas": {
                    "noise_calibration": "100 * exp(-10 * |mean(bits_H0_decimal) - log2(10)|)",
                    "periodic_memory": "100 * (1 - bits_1_999983 / log2(10))",
                    "multi_digit_counting": "100 * (1 - bits_champernowne / log2(10))",
                    "cf_arithmetic": "100 * (1 - bits_e_cf / bits_pi_cf)",
                    "throughput": "100 * sqrt(median_samples_per_sec / max_median_samples_per_sec)",
                },
                "embedding_pca_preview": pca_data,
                "datasets": EXP2_DATASETS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    saved_plots: list[Path] = []
    if finalize:
        heatmap_models = ["PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
        h1, p1 = plot_exp2_heatmap_bits(df_final, summary_dir, datasets=EXP2_DATASETS, models=heatmap_models)
        h2, p2 = plot_exp2_cf_and_champernowne_contrast(df_final, summary_dir)
        h3, p3 = plot_exp2_radar_pyecharts(radar_scores, summary_dir)
        h4, p4 = plot_exp2_embedding_pca_preview(pca_data, summary_dir)
        saved_plots.extend([h1, p1, h2, p2, h3, p3, h4, p4])
    else:
        models_for_plot = [
            m for m in ("PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer")
            if m in df_final["model"].unique()
        ]
        h_c, p_c = plot_exp2_constants_comparison(
            df=df_final,
            summary_dir=summary_dir,
            datasets=EXP2_DATASETS,
            models_to_show=models_for_plot,
            title_suffix=stage_tag.upper(),
        )
        saved_plots.extend([h_c, p_c])

    print(f"\nСводка Эксперимента 2 ({stage_tag}) сохранена в: {summary_dir}")
    piv_bits = df_final.pivot(index="dataset", columns="model", values="bits_per_symbol").reindex(EXP2_DATASETS)
    show_cols = [c for c in ("Markov_k1", "Markov_k3", "PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer") if c in piv_bits.columns]
    print("\nСводная таблица (bits_per_symbol = CE / ln 2):")
    print(piv_bits[show_cols].round(4).to_string())

    piv_acc = df_final.pivot(index="dataset", columns="model", values="test_acc").reindex(EXP2_DATASETS)
    print("\nСводная таблица (Test Accuracy, %):")
    print((piv_acc[show_cols] * 100.0).round(2).to_string())

    if radar_scores:
        print("\nБаллы 5-осевого радара (0–100):")
        print(pd.DataFrame(radar_scores).T.to_string())

    print("\nГрафики:")
    for sp in saved_plots:
        print(f"- {sp}")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 2: Битва констант, ГПСЧ и цепных дробей")
    parser.add_argument(
        "--arch",
        type=str,
        default="transformer",
        help="Архитектура: mlp, cnn, gru, transformer, all, или baselines_only",
    )
    parser.add_argument("--size", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--stage-tag", type=str, default="final")
    parser.add_argument("--no-finalize", action="store_true", help="Не запускать McNemar-аудит и финальные 4 графика")
    args = parser.parse_args()

    if args.arch == "all":
        archs = ["mlp", "cnn", "gru", "transformer"]
    elif args.arch in ("baselines_only", "none"):
        archs = []
    else:
        archs = [a.strip().lower() for a in args.arch.split(",") if a.strip()]

    run_experiment_2(
        archs=archs,
        size_tier=args.size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        reuse_completed=args.reuse_completed,
        stage_tag=args.stage_tag,
        finalize=not args.no_finalize,
    )


if __name__ == "__main__":
    main()
