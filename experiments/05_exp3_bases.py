"""Фаза 4 — Эксперимент 3: Влияние системы счисления (Base-2, Base-10, Base-16) и проверка гипотезы BBP."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baselines import (
    PPMCompressor,
    _encode_markov_states,
    evaluate_uniform_baseline,
    fit_and_evaluate_lightgbm,
    fit_and_evaluate_markov,
    fit_and_evaluate_ppm,
)
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import ChronologicalSplit, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import compute_binomial_z_score, compute_mcnemar_exact, wilson_ci
from src.models import build_model, count_parameters
from src.stats_tests import run_full_statistical_suite
from src.trainer import create_run_dir, resolve_device, train_model
from src.visualizer import plot_exp3_bases_normalized_bits, plot_exp3_bases_zscore_ci

EXP3_BASES = [
    ("pi_base2", "Base-2", 2),
    ("true_pi", "Base-10", 10),
    ("pi_base16", "Base-16", 16),
]

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


def _canonical_model(name: str) -> str:
    return CANONICAL_MODEL_MAP.get(name, name)


def _find_latest_exp2_final(runs_root: Path) -> Path | None:
    cands = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("exp2_constants_final_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in cands:
        if (d / "metrics.csv").exists():
            return d / "metrics.csv"
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


def _read_sec_per_epoch(run_dir: Path | None, total_time: float | None, epochs_ran: int | None) -> float | None:
    if run_dir is not None and (run_dir / "plot_data.json").exists():
        data = json.loads((run_dir / "plot_data.json").read_text(encoding="utf-8"))
        ep_times = [float(h["epoch_time_sec"]) for h in data.get("history", []) if h.get("epoch_time_sec") is not None]
        if ep_times:
            return round(float(np.mean(ep_times)), 3)
    if total_time is not None and epochs_ran and epochs_ran > 0:
        return round(float(total_time) / float(epochs_ran), 3)
    return None


def _remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = (
            k.replace(".attn.in_proj_weight", ".qkv_proj.weight")
            .replace(".attn.in_proj_bias", ".qkv_proj.bias")
            .replace(".attn.out_proj.", ".out_proj.")
        )
        out[nk] = v
    return out


def _eval_markov_preds(split: ChronologicalSplit, radix: int, order: int, window_size: int) -> tuple[np.ndarray, float]:
    num_states = radix**order
    tr_s = _encode_markov_states(split.train_digits, order=order, radix=radix)
    tr_y = split.train_digits[order:].astype(np.int64)
    counts = np.bincount(tr_s * radix + tr_y, minlength=num_states * radix).reshape(num_states, radix).astype(np.float64)
    probs = (counts + 1e-3) / (counts.sum(axis=1, keepdims=True) + 1e-3 * radix)

    te_slice = split.test_digits[window_size - order :]
    te_y = split.test_digits[window_size:]
    t0 = time.perf_counter()
    te_s = _encode_markov_states(te_slice, order=order, radix=radix)
    preds = np.argmax(probs[te_s], axis=1)
    elapsed = max(time.perf_counter() - t0, 1e-5)
    return (preds == te_y), round(len(te_y) / elapsed, 1)


def _eval_ppm_preds(split: ChronologicalSplit, radix: int, window_size: int, max_order: int = 8) -> tuple[np.ndarray, float]:
    ppm = PPMCompressor(radix=radix, max_order=max_order).fit(split.train_digits)
    te_y = split.test_digits[window_size:]
    t0 = time.perf_counter()
    preds = np.argmax(ppm.predict_proba(split.test_digits, window_size=window_size), axis=1)
    elapsed = max(time.perf_counter() - t0, 1e-5)
    return (preds == te_y), round(len(te_y) / elapsed, 1)


@torch.no_grad()
def _eval_neural_preds_and_speed(
    arch: str,
    run_dir: Path,
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
    size_tier: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    ckpt_path = run_dir / "best_model.pt"
    loaders = create_dataloaders(
        split=split,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(_remap_legacy_state_dict(ckpt["model_state_dict"]))
    model = model.to(device).eval()

    amp_enabled = device.type == "cuda"
    test_loader = loaders["test"]
    for contexts, _, _ in test_loader:
        contexts = contexts.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _ = model(contexts)
        break

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for contexts, targets, _ in test_loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(contexts)
        preds = torch.argmax(logits.float(), dim=-1)
        chunks.append((preds == targets).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-5)
    corr = np.concatenate(chunks, axis=0)
    return corr, round(float(len(corr) / elapsed), 1)


def _verify_bit_nibble_equivalence(bits_arr: np.ndarray, hex_arr: np.ndarray) -> dict[str, Any]:
    """Проверяет математическое тождество между первыми N/4 нибблами pi_base16 и битами pi_base2."""
    n_nibbles = min(len(hex_arr), len(bits_arr) // 4)
    grouped = bits_arr[: n_nibbles * 4].reshape(n_nibbles, 4).astype(np.uint8)
    reconstructed = grouped[:, 0] * 8 + grouped[:, 1] * 4 + grouped[:, 2] * 2 + grouped[:, 3]
    mismatch = int(np.sum(reconstructed != hex_arr[:n_nibbles]))
    return {
        "checked_nibbles": int(n_nibbles),
        "checked_bits": int(n_nibbles * 4),
        "exact_match": bool(mismatch == 0),
        "mismatches": mismatch,
    }


def run_experiment_3(
    archs: list[str],
    size_tier: str = "medium",
    max_epochs: int = 10,
    patience: int = 3,
    reuse_base10: bool = True,
) -> Path:
    """Проводит эксперимент сравнения систем счисления Base-2, Base-10 и Base-16 (BBP)."""
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

    # 1. Загрузка 3 датасетов и классическая статистика
    artifacts = {}
    splits: dict[str, ChronologicalSplit] = {}
    stats_rows: list[dict[str, Any]] = []
    stats_full: list[dict[str, Any]] = []

    print("=== Шаг 1: Загрузка кэша Pi (Base-2, Base-10, Base-16), проверка эквивалентности и статистика ===")
    for ds_name, base_label, expected_radix in EXP3_BASES:
        art = get_or_create_dataset(name=ds_name, config=config)
        assert art.radix == expected_radix
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

        st = run_full_statistical_suite(ds_name, art.digits, radix=art.radix, max_lag=64)
        stats_full.append(st)
        stats_rows.append(
            {
                "dataset": ds_name,
                "base_label": base_label,
                "radix": art.radix,
                "length": art.length,
                "chi2_stat": round(st["chi2_stat"], 4),
                "chi2_p_value": round(st["chi2_p_value"], 6),
                "chi2_passed": st["chi2_passed"],
                "ks_p_value": round(st["ks_p_value"], 6),
                "ks_passed": st["ks_passed"],
                "acf_max_abs": round(st["acf_max_abs"], 6),
                "acf_max_lag": st["acf_max_lag"],
                "acf_significant_count": st["acf_significant_count"],
                "lzma_bits_per_symbol": round(st["lzma_bits_per_digit"], 4),
                "shannon_limit_bits": round(st["shannon_limit_bits_per_digit"], 4),
                "lzma_normalized": round(st["lzma_bits_per_digit"] / st["shannon_limit_bits_per_digit"], 4),
            }
        )

    bit_nibble_check = _verify_bit_nibble_equivalence(artifacts["pi_base2"].digits, artifacts["pi_base16"].digits)
    print("\nКлассическая статистика по трём основаниям:")
    print(pd.DataFrame(stats_rows).to_string(index=False))
    print(
        f"\nПроверка битовой эквивалентности Base-2 <-> Base-16: "
        f"{bit_nibble_check['checked_nibbles']:,} нибблов ({bit_nibble_check['checked_bits']:,} бит), "
        f"несовпадений = {bit_nibble_check['mismatches']}"
    )

    # 2. Подхват готовых метрик Base-10 (true_pi) из Эксперимента 2
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if reuse_base10:
        exp2_csv = _find_latest_exp2_final(runs_root)
        if exp2_csv is not None:
            df_e2 = pd.read_csv(exp2_csv)
            df_pi10 = df_e2[df_e2["dataset"] == "true_pi"]
            for _, row in df_pi10.iterrows():
                m_name = _canonical_model(str(row["model"]))
                records[("true_pi", m_name)] = {
                    "dataset": "true_pi",
                    "radix": 10,
                    "model": m_name,
                    "test_acc": float(row["test_acc"]),
                    "test_ce": float(row["test_ce_nats"]),
                    "test_ppl": float(row["test_ppl"]),
                    "pred_entropy": float(row["pred_entropy"]) if pd.notna(row.get("pred_entropy")) else None,
                    "num_eval_samples": int(row["num_eval_samples"]),
                    "param_count": int(row["params_count"]) if pd.notna(row.get("params_count")) else None,
                    "epochs_ran": int(row["epochs_ran"]) if pd.notna(row.get("epochs_ran")) else None,
                    "best_epoch": int(row["best_epoch"]) if pd.notna(row.get("best_epoch")) else None,
                    "sec_per_epoch": float(row["sec_per_epoch"]) if pd.notna(row.get("sec_per_epoch")) else None,
                    "samples_per_sec": float(row["samples_per_sec"]) if pd.notna(row.get("samples_per_sec")) else None,
                    "train_time_sec": float(row["train_time_sec"]) if pd.notna(row.get("train_time_sec")) else None,
                    "source": str(row.get("source", exp2_csv.parent.name)),
                }
            print(f"\nПодтянуты готовые результаты Base-10 (true_pi) из {exp2_csv.parent.name}: {len(df_pi10)} моделей")

    # 3. Расчёт классических бейзлайнов + PPM_o8 и обучение 4 нейросетей для Base-2 и Base-16
    print("\n=== Шаг 2: Бейзлайны, PPM-C (order=8) и обучение нейросетей на Base-2 и Base-16 ===")
    for ds_name, base_label, radix in EXP3_BASES:
        split = splits[ds_name]
        n_eval = len(split.test_digits) - window_size

        if (ds_name, "Uniform") not in records:
            u = evaluate_uniform_baseline(split, radix=radix, window_size=window_size)
            records[(ds_name, "Uniform")] = {
                "dataset": ds_name,
                "radix": radix,
                "model": "Uniform",
                "test_acc": u["overall_acc"],
                "test_ce": u["cross_entropy"],
                "test_ppl": u["perplexity"],
                "pred_entropy": u["pred_entropy"],
                "num_eval_samples": n_eval,
                "train_time_sec": 0.0,
                "samples_per_sec": round(n_eval / max(float(u["eval_time_sec"]), 1e-4), 1),
                "source": "exp3_fresh",
            }

        for k in (1, 2, 3):
            m_key = f"Markov_k{k}"
            if (ds_name, m_key) not in records:
                mk, _ = fit_and_evaluate_markov(split, radix=radix, order=k, window_size=window_size)
                records[(ds_name, m_key)] = {
                    "dataset": ds_name,
                    "radix": radix,
                    "model": m_key,
                    "test_acc": mk["overall_acc"],
                    "test_ce": mk["cross_entropy"],
                    "test_ppl": mk["perplexity"],
                    "pred_entropy": mk["pred_entropy"],
                    "num_eval_samples": n_eval,
                    "train_time_sec": mk["train_time_sec"],
                    "samples_per_sec": round(n_eval / max(float(mk["eval_time_sec"]), 1e-4), 1),
                    "source": "exp3_fresh",
                }

        if (ds_name, "LightGBM") not in records:
            lgb_res = fit_and_evaluate_lightgbm(
                split,
                radix=radix,
                window_size=window_size,
                seed=seed,
                max_train_samples=200_000,
                n_estimators=150,
            )
            records[(ds_name, "LightGBM")] = {
                "dataset": ds_name,
                "radix": radix,
                "model": "LightGBM",
                "test_acc": lgb_res["overall_acc"],
                "test_ce": lgb_res["cross_entropy"],
                "test_ppl": lgb_res["perplexity"],
                "pred_entropy": lgb_res["pred_entropy"],
                "num_eval_samples": n_eval,
                "train_time_sec": lgb_res["train_time_sec"],
                "samples_per_sec": round(n_eval / max(float(lgb_res["eval_time_sec"]), 1e-4), 1),
                "source": "exp3_fresh",
            }

        if (ds_name, "PPM_o8") not in records:
            ppm_res, _ = fit_and_evaluate_ppm(split, radix=radix, window_size=window_size, max_order=8)
            records[(ds_name, "PPM_o8")] = {
                "dataset": ds_name,
                "radix": radix,
                "model": "PPM_o8",
                "test_acc": ppm_res["overall_acc"],
                "test_ce": ppm_res["cross_entropy"],
                "test_ppl": ppm_res["perplexity"],
                "pred_entropy": ppm_res["pred_entropy"],
                "num_eval_samples": n_eval,
                "train_time_sec": ppm_res["train_time_sec"],
                "samples_per_sec": round(n_eval / max(float(ppm_res["eval_time_sec"]), 1e-4), 1),
                "source": "exp3_fresh",
            }

        for arch in archs:
            m_name = _canonical_model(arch.upper())
            if (ds_name, m_name) in records:
                print(f"[{base_label} | {m_name}] Используем готовый результат из Exp 2 ({records[(ds_name, m_name)]['source']})")
                continue

            seed_everything(seed)
            loaders = create_dataloaders(
                split=split,
                window_size=window_size,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
            model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
            p_cnt = count_parameters(model)
            print(f"\n=== [Exp 3 | {base_label} (K={radix}) | {m_name} | Параметров: {p_cnt:,}] ===")
            res = train_model(
                model=model,
                loaders=loaders,
                config=config,
                exp_name=f"exp3_{arch}_{ds_name}",
                radix=radix,
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
                "radix": radix,
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

    # 4. По-примерные предсказания (для McNemar) и замер GPU throughput для всех 3 баз
    print("\n=== Шаг 3: Статистический аудит (McNemar + Z-score + 4-bit block comparison) ===")
    preds_cache: dict[tuple[str, str], np.ndarray] = {}
    for ds_name, _, radix in EXP3_BASES:
        split = splits[ds_name]
        te_y = split.test_digits[window_size:]
        # Uniform предсказывает класс 0 всегда
        preds_cache[(ds_name, "Uniform")] = (np.zeros_like(te_y) == te_y)

        for k in (1, 3):
            mk_corr, mk_sps = _eval_markov_preds(split, radix=radix, order=k, window_size=window_size)
            preds_cache[(ds_name, f"Markov_k{k}")] = mk_corr
            if (ds_name, f"Markov_k{k}") in records:
                records[(ds_name, f"Markov_k{k}")]["samples_per_sec"] = mk_sps

        ppm_corr, ppm_sps = _eval_ppm_preds(split, radix=radix, window_size=window_size, max_order=8)
        preds_cache[(ds_name, "PPM_o8")] = ppm_corr
        if (ds_name, "PPM_o8") in records:
            records[(ds_name, "PPM_o8")]["samples_per_sec"] = ppm_sps

        for m_name, arch in CANONICAL_TO_ARCH.items():
            rec = records[(ds_name, m_name)]
            run_dir = _resolve_run_dir(rec.get("source"))
            if run_dir is None:
                continue
            if rec.get("sec_per_epoch") is None:
                rec["sec_per_epoch"] = _read_sec_per_epoch(run_dir, rec.get("train_time_sec"), rec.get("epochs_ran"))
            corr, sps = _eval_neural_preds_and_speed(
                arch=arch,
                run_dir=run_dir,
                split=split,
                radix=radix,
                window_size=window_size,
                size_tier=size_tier,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
            preds_cache[(ds_name, m_name)] = corr
            rec["samples_per_sec"] = sps

    # 5. Сборка итоговых метрик
    best_markov_bits: dict[str, float] = {}
    for ds_name, _, radix in EXP3_BASES:
        mk_vals = [records[(ds_name, f"Markov_k{k}")]["test_ce"] / math.log(2.0) for k in (1, 2, 3)]
        best_markov_bits[ds_name] = min(mk_vals)

    rows: list[dict[str, Any]] = []
    for ds_name, base_label, radix in EXP3_BASES:
        shannon_bits = math.log2(radix)
        win_info_bits = round(window_size * shannon_bits, 2)
        best_mk = best_markov_bits[ds_name]

        for m_name in MODEL_ORDER:
            r = records[(ds_name, m_name)]
            n_eval = int(r["num_eval_samples"])
            k_corr = int(round(r["test_acc"] * n_eval))
            ci_lo, ci_hi = wilson_ci(k_corr, n_eval)
            z_info = compute_binomial_z_score(r["test_acc"], radix=radix, n_samples=n_eval, ci_low=ci_lo, ci_high=ci_hi)
            bits = r["test_ce"] / math.log(2.0)
            norm_bits = bits / shannon_bits

            rows.append(
                {
                    "dataset": ds_name,
                    "base_label": base_label,
                    "radix": radix,
                    "window_size": window_size,
                    "window_info_bits": win_info_bits,
                    "model": m_name,
                    "params_count": r.get("param_count"),
                    "test_acc": round(r["test_acc"], 6),
                    "random_chance": z_info["random_chance"],
                    "wilson_ci_low": ci_lo,
                    "wilson_ci_high": ci_hi,
                    "z_score": z_info["z_score"],
                    "z_ci_low": z_info["z_ci_low"],
                    "z_ci_high": z_info["z_ci_high"],
                    "z_p_value": z_info["z_p_value"],
                    "test_ce_nats": round(r["test_ce"], 6),
                    "test_ppl": round(r["test_ppl"], 4),
                    "bits_per_symbol": round(bits, 6),
                    "shannon_limit_bits": round(shannon_bits, 6),
                    "normalized_bits": round(norm_bits, 6),
                    "excess_bits_per_bit": round(norm_bits - 1.0, 6),
                    "equiv_4bit_nibble_bits": round(norm_bits * 4.0, 6),
                    "gain_over_markov_bits": round(best_mk - bits, 6),
                    "sec_per_epoch": r.get("sec_per_epoch"),
                    "samples_per_sec": r.get("samples_per_sec"),
                    "pred_entropy": round(r["pred_entropy"], 6) if r.get("pred_entropy") is not None else None,
                    "num_eval_samples": n_eval,
                    "epochs_ran": r.get("epochs_ran"),
                    "best_epoch": r.get("best_epoch"),
                    "train_time_sec": r.get("train_time_sec"),
                    "source": r.get("source"),
                }
            )

    # 6. Блочное сравнение 4-битных блоков (Base-2 L=64 bit vs Base-16 BBP L=64 nibbles = 256 bit)
    block_rows: list[dict[str, Any]] = []
    df_all = pd.DataFrame(rows)
    for m_name in ("PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"):
        r2 = df_all[(df_all["base_label"] == "Base-2") & (df_all["model"] == m_name)].iloc[0]
        r16 = df_all[(df_all["base_label"] == "Base-16") & (df_all["model"] == m_name)].iloc[0]
        joint_4bit_acc = (float(r2["test_acc"]) ** 4) * 100.0
        block_rows.append(
            {
                "model": m_name,
                "base2_bit_acc_pct": round(float(r2["test_acc"]) * 100.0, 3),
                "base2_equiv_4bit_acc_pct": round(joint_4bit_acc, 3),
                "base16_nibble_acc_pct": round(float(r16["test_acc"]) * 100.0, 3),
                "base2_4bit_entropy_bits": round(float(r2["equiv_4bit_nibble_bits"]), 5),
                "base16_nibble_entropy_bits": round(float(r16["bits_per_symbol"]), 5),
                "delta_nibble_bits_base16_minus_base2": round(
                    float(r16["bits_per_symbol"]) - float(r2["equiv_4bit_nibble_bits"]), 5
                ),
            }
        )

    # 7. Тест Макнемара и Z-тест для stat_audit.json
    stat_audit: dict[str, Any] = {"bit_nibble_equivalence": bit_nibble_check, "bases": {}}
    for ds_name, base_label, radix in EXP3_BASES:
        sub_rows = [r for r in rows if r["dataset"] == ds_name]
        neural_rows = [r for r in sub_rows if r["model"] in CANONICAL_TO_ARCH]
        best_neural = min(neural_rows, key=lambda r: (r["normalized_bits"], -r["test_acc"]))

        comparisons: dict[str, Any] = {}
        for m_name in ("DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer", "PPM_o8"):
            if (ds_name, m_name) in preds_cache and (ds_name, "Uniform") in preds_cache:
                mc_u = compute_mcnemar_exact(preds_cache[(ds_name, m_name)], preds_cache[(ds_name, "Uniform")])
                mc_u["model_a"] = m_name
                mc_u["model_b"] = "Uniform"
                comparisons[f"{m_name}_vs_Uniform"] = mc_u
            if (ds_name, m_name) in preds_cache and (ds_name, "Markov_k1") in preds_cache:
                mc_m = compute_mcnemar_exact(preds_cache[(ds_name, m_name)], preds_cache[(ds_name, "Markov_k1")])
                mc_m["model_a"] = m_name
                mc_m["model_b"] = "Markov_k1"
                comparisons[f"{m_name}_vs_Markov_k1"] = mc_m

        z_audit = {
            r["model"]: {
                "test_acc": r["test_acc"],
                "wilson_ci": [r["wilson_ci_low"], r["wilson_ci_high"]],
                "z_score": r["z_score"],
                "z_p_value": r["z_p_value"],
                "normalized_bits": r["normalized_bits"],
            }
            for r in sub_rows
            if r["model"] != "Uniform"
        }
        stat_audit["bases"][base_label] = {
            "dataset": ds_name,
            "radix": radix,
            "best_neural_by_entropy": best_neural["model"],
            "best_neural_normalized_bits": best_neural["normalized_bits"],
            "z_tests": z_audit,
            "mcnemar_comparisons": comparisons,
        }

    # 8. Сохранение в runs/exp3_bases_<timestamp>/
    summary_dir = create_run_dir(runs_root, "exp3_bases")
    (summary_dir / "config_snapshot.yaml").write_text(
        (ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    df_all.to_csv(summary_dir / "metrics.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(summary_dir / "classical_stats_bases.csv", index=False)
    pd.DataFrame(block_rows).to_csv(summary_dir / "block_4bit_vs_nibble.csv", index=False)
    (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (summary_dir / "plot_data.json").write_text(
        json.dumps(
            {
                "metrics": rows,
                "classical_stats": stats_full,
                "block_4bit_vs_nibble": block_rows,
                "bit_nibble_equivalence": bit_nibble_check,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    h1, p1 = plot_exp3_bases_normalized_bits(df_all, summary_dir)
    h2, p2 = plot_exp3_bases_zscore_ci(df_all, summary_dir)

    print(f"\nЭксперимент 3 завершён! Артефакты сохранены в: {summary_dir}")
    print("\nСводка нормированного битрейта (normalized_bits = bits_per_symbol / log2(K), идеал шума = 1.0000):")
    piv_norm = df_all.pivot(index="model", columns="base_label", values="normalized_bits").reindex(MODEL_ORDER)[
        ["Base-2", "Base-10", "Base-16"]
    ]
    print(piv_norm.round(5).to_string())

    print("\nСводка Z-score (σ отклонения от 1/K):")
    piv_z = df_all.pivot(index="model", columns="base_label", values="z_score").reindex(MODEL_ORDER)[
        ["Base-2", "Base-10", "Base-16"]
    ]
    print(piv_z.round(3).to_string())

    print("\nБлочное сопоставление (Base-2 4 бита vs Base-16 ниббл):")
    print(pd.DataFrame(block_rows).to_string(index=False))

    print(f"\nГрафики:\n- {h1}\n- {p1}\n- {h2}\n- {p2}")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 3: Системы счисления Base-2, Base-10, Base-16 (BBP)")
    parser.add_argument(
        "--arch",
        type=str,
        default="all",
        help="Архитектуры: mlp, cnn, gru, transformer или all",
    )
    parser.add_argument("--size", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-reuse-base10", action="store_true")
    args = parser.parse_args()

    archs = ["mlp", "cnn", "gru", "transformer"] if args.arch == "all" else [a.strip().lower() for a in args.arch.split(",") if a.strip()]
    run_experiment_3(
        archs=archs,
        size_tier=args.size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        reuse_base10=not args.no_reuse_base10,
    )


if __name__ == "__main__":
    main()
