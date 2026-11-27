# Сравнение систем счисления Base-2, Base-10 и Base-16 на знаках pi (BBP-мотивация)
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.baselines import evaluate_uniform_baseline, fit_and_evaluate_lightgbm, fit_and_evaluate_markov, fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import ChronologicalSplit, SequenceWindowDataset, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import compute_binomial_z_score, compute_mcnemar_exact, wilson_ci
from src.exp_utils import (
    CANONICAL_TO_ARCH,
    MODEL_ORDER,
    baseline_record,
    canonical_model_name,
    eval_markov_preds_and_speed,
    eval_ppm_preds_and_speed,
    find_latest_csv,
    predict_and_benchmark,
    read_sec_per_epoch,
    resolve_run_dir,
)
from src.models import build_model
from src.stats_tests import run_full_statistical_suite
from src.trainer import create_run_dir, resolve_device, train_model
from src.viz import plot_exp3_bases_normalized_bits, plot_exp3_bases_zscore_ci

ROOT_DIR = Path(__file__).resolve().parents[1]
EXP3_BASES = [("pi_base2", "Base-2", 2), ("true_pi", "Base-10", 10), ("pi_base16", "Base-16", 16)]


def _verify_bit_nibble_equivalence(bits_arr: np.ndarray, nibbles_arr: np.ndarray) -> dict[str, Any]:
    n_nibbles = min(len(bits_arr) // 4, len(nibbles_arr))
    b4 = bits_arr[: n_nibbles * 4].reshape(n_nibbles, 4).astype(np.int64)
    packed = b4[:, 0] * 8 + b4[:, 1] * 4 + b4[:, 2] * 2 + b4[:, 3]
    mismatches = int(np.sum(packed != nibbles_arr[:n_nibbles].astype(np.int64)))
    return {"checked_nibbles": int(n_nibbles), "checked_bits": int(n_nibbles * 4), "mismatches": mismatches, "exact_match": mismatches == 0}


def _load_and_audit_bases(config: dict[str, Any], window_size: int) -> tuple[dict, dict, list[dict], list[dict], dict[str, Any]]:
    split_cfg = config["data"]["split"]
    artifacts, splits, stats_rows, stats_full = {}, {}, [], []
    for ds_name, base_label, expected_radix in EXP3_BASES:
        art = get_or_create_dataset(name=ds_name, config=config)
        assert art.radix == expected_radix
        artifacts[ds_name] = art
        split = split_sequence_chronologically(
            digits=art.digits, mask=art.mask,
            train_ratio=float(split_cfg["train"]), val_ratio=float(split_cfg["val"]), test_ratio=float(split_cfg["test"]),
        )
        verify_zero_leakage(
            split,
            window_size=window_size,
            train_ds=SequenceWindowDataset(split.train_digits, window_size, split.train_mask, split.train_range[0]),
            val_ds=SequenceWindowDataset(split.val_digits, window_size, split.val_mask, split.val_range[0]),
            test_ds=SequenceWindowDataset(split.test_digits, window_size, split.test_mask, split.test_range[0]),
        )
        splits[ds_name] = split
        st = run_full_statistical_suite(ds_name, art.digits, radix=art.radix, max_lag=64)
        stats_full.append(st)
        stats_rows.append({
            "dataset": ds_name, "base_label": base_label, "radix": art.radix, "length": art.length,
            "chi2_stat": round(st["chi2_stat"], 4), "chi2_p_value": round(st["chi2_p_value"], 6), "chi2_passed": st["chi2_passed"],
            "ks_p_value": round(st["ks_p_value"], 6), "ks_passed": st["ks_passed"], "acf_max_abs": round(st["acf_max_abs"], 6),
            "acf_max_lag": st["acf_max_lag"], "acf_significant_count": st["acf_significant_count"],
            "lzma_bits_per_symbol": round(st["lzma_bits_per_digit"], 4), "shannon_limit_bits": round(st["shannon_limit_bits_per_digit"], 4),
            "lzma_normalized": round(st["lzma_bits_per_digit"] / st["shannon_limit_bits_per_digit"], 4),
        })
    eq_check = _verify_bit_nibble_equivalence(artifacts["pi_base2"].digits, artifacts["pi_base16"].digits)
    return artifacts, splits, stats_rows, stats_full, eq_check


def _seed_base10_cache(runs_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    exp2_csv = find_latest_csv(runs_root, "exp2_constants_final_") or (ROOT_DIR / "results" / "04_exp2_constants" / "metrics.csv")
    if not exp2_csv.exists():
        return records
    df_pi10 = pd.read_csv(exp2_csv)
    for _, row in df_pi10[df_pi10["dataset"] == "true_pi"].iterrows():
        m_name = canonical_model_name(str(row["model"]))
        records[("true_pi", m_name)] = {
            "dataset": "true_pi", "radix": 10, "model": m_name, "test_acc": float(row["test_acc"]),
            "test_ce": float(row["test_ce_nats"]), "test_ppl": float(row["test_ppl"]),
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
    return records


def _train_missing_bases(
    archs: list[str], records: dict[tuple[str, str], dict[str, Any]], splits: dict[str, ChronologicalSplit],
    config: dict[str, Any], size_tier: str, max_epochs: int, patience: int,
) -> None:
    seed, window_size = int(config["seed"]), int(config["data"]["default_window_size"])
    batch_size, num_workers = int(config["training"]["batch_size"]), int(config["training"].get("num_workers", 4))
    for ds_name, base_label, radix in EXP3_BASES:
        split = splits[ds_name]
        n_eval = len(split.test_digits) - window_size
        if (ds_name, "Uniform") not in records:
            records[(ds_name, "Uniform")] = baseline_record(ds_name, radix, "Uniform", evaluate_uniform_baseline(split, radix, window_size), n_eval, "exp3_fresh", 0.0)
        for k in (1, 2, 3):
            if (ds_name, f"Markov_k{k}") not in records:
                mk, _ = fit_and_evaluate_markov(split, radix=radix, order=k, window_size=window_size)
                records[(ds_name, f"Markov_k{k}")] = baseline_record(ds_name, radix, f"Markov_k{k}", mk, n_eval, "exp3_fresh")
        if (ds_name, "LightGBM") not in records:
            lgb_res = fit_and_evaluate_lightgbm(split, radix=radix, window_size=window_size, seed=seed, max_train_samples=200_000, n_estimators=150)
            records[(ds_name, "LightGBM")] = baseline_record(ds_name, radix, "LightGBM", lgb_res, n_eval, "exp3_fresh")
        if (ds_name, "PPM_o8") not in records:
            ppm_res, _ = fit_and_evaluate_ppm(split, radix=radix, window_size=window_size, max_order=8)
            records[(ds_name, "PPM_o8")] = baseline_record(ds_name, radix, "PPM_o8", ppm_res, n_eval, "exp3_fresh")
        for arch in archs:
            m_name = canonical_model_name(arch.upper())
            if (ds_name, m_name) in records:
                continue
            seed_everything(seed)
            loaders = create_dataloaders(split=split, window_size=window_size, batch_size=batch_size, num_workers=num_workers, pin_memory=True, seed=seed)
            model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
            res = train_model(model, loaders, config, f"exp3_{arch}_{ds_name}", radix, epochs=max_epochs, early_stopping_patience=patience, enable_early_stopping=True)
            s, tm, hist = res["summary"], res["test_metrics"], res["history"]
            records[(ds_name, m_name)] = {
                "dataset": ds_name, "radix": radix, "model": m_name, "test_acc": float(tm["overall_acc"]),
                "test_ce": float(tm["cross_entropy"]), "test_ppl": float(tm["perplexity"]), "pred_entropy": float(tm["pred_entropy"]),
                "num_eval_samples": int(tm["num_eval_samples"]), "train_time_sec": float(s["total_train_time_sec"]),
                "param_count": int(s["param_count"]), "epochs_ran": int(s["epochs_ran"]), "best_epoch": int(s["best_epoch"]),
                "sec_per_epoch": round(float(np.mean([h["epoch_time_sec"] for h in hist])), 3) if hist else None, "source": s["run_dir"],
            }


def _collect_preds_and_build_rows(
    records: dict[tuple[str, str], dict[str, Any]], splits: dict[str, ChronologicalSplit], config: dict[str, Any], size_tier: str, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    window_size = int(config["data"]["default_window_size"])
    batch_size, num_workers = int(config["training"]["batch_size"]), int(config["training"].get("num_workers", 4))
    preds_cache: dict[tuple[str, str], np.ndarray] = {}
    seed = int(config["seed"])
    for ds_name, _, radix in EXP3_BASES:
        split = splits[ds_name]
        te_y = split.test_digits[window_size:]
        preds_cache[(ds_name, "Uniform")] = np.zeros_like(te_y) == te_y
        for k in (1, 3):
            mk_c, mk_s = eval_markov_preds_and_speed(split, radix, k, window_size)
            preds_cache[(ds_name, f"Markov_k{k}")] = mk_c
            records[(ds_name, f"Markov_k{k}")]["samples_per_sec"] = mk_s
        ppm_c, ppm_s = eval_ppm_preds_and_speed(split, radix, window_size, max_order=8)
        preds_cache[(ds_name, "PPM_o8")] = ppm_c
        records[(ds_name, "PPM_o8")]["samples_per_sec"] = ppm_s
        for m_name, arch in CANONICAL_TO_ARCH.items():
            rec = records[(ds_name, m_name)]
            run_dir = resolve_run_dir(rec.get("source"), ROOT_DIR)
            if run_dir is None:
                continue
            if rec.get("sec_per_epoch") is None:
                rec["sec_per_epoch"] = read_sec_per_epoch(run_dir, rec.get("train_time_sec"), rec.get("epochs_ran"))
            corr, sps, _ = predict_and_benchmark(arch, run_dir, split, radix, window_size, size_tier, batch_size, num_workers, device, seed=seed)
            preds_cache[(ds_name, m_name)] = corr
            rec["samples_per_sec"] = sps

    rows: list[dict[str, Any]] = []
    for ds_name, base_label, radix in EXP3_BASES:
        shannon = math.log2(radix)
        best_mk = min(records[(ds_name, f"Markov_k{k}")]["test_ce"] / math.log(2.0) for k in (1, 2, 3))
        for m_name in MODEL_ORDER:
            r = records[(ds_name, m_name)]
            n_eval = int(r["num_eval_samples"])
            ci_lo, ci_hi = wilson_ci(int(round(r["test_acc"] * n_eval)), n_eval)
            z_info = compute_binomial_z_score(r["test_acc"], radix=radix, n_samples=n_eval, ci_low=ci_lo, ci_high=ci_hi)
            bits = r["test_ce"] / math.log(2.0)
            norm = bits / shannon
            rows.append({
                "dataset": ds_name, "base_label": base_label, "radix": radix, "window_size": window_size,
                "window_info_bits": round(window_size * shannon, 2), "model": m_name, "params_count": r.get("param_count"),
                "test_acc": round(r["test_acc"], 6), "random_chance": z_info["random_chance"], "wilson_ci_low": ci_lo,
                "wilson_ci_high": ci_hi, "z_score": z_info["z_score"], "z_ci_low": z_info["z_ci_low"], "z_ci_high": z_info["z_ci_high"],
                "z_p_value": z_info["z_p_value"], "test_ce_nats": round(r["test_ce"], 6), "test_ppl": round(r["test_ppl"], 4),
                "bits_per_symbol": round(bits, 6), "shannon_limit_bits": round(shannon, 6), "normalized_bits": round(norm, 6),
                "excess_bits_per_bit": round(norm - 1.0, 6), "equiv_4bit_nibble_bits": round(norm * 4.0, 6),
                "gain_over_markov_bits": round(best_mk - bits, 6), "sec_per_epoch": r.get("sec_per_epoch"),
                "samples_per_sec": r.get("samples_per_sec"), "pred_entropy": round(r["pred_entropy"], 6) if r.get("pred_entropy") is not None else None,
                "num_eval_samples": n_eval, "epochs_ran": r.get("epochs_ran"), "best_epoch": r.get("best_epoch"),
                "train_time_sec": r.get("train_time_sec"), "source": r.get("source"),
            })
    return rows, preds_cache


def _build_block_and_mcnemar_audit(df_all: pd.DataFrame, rows: list[dict[str, Any]], preds_cache: dict[tuple[str, str], np.ndarray], bit_nibble_check: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block_rows = []
    for m_name in ("PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"):
        r2 = df_all[(df_all["base_label"] == "Base-2") & (df_all["model"] == m_name)].iloc[0]
        r16 = df_all[(df_all["base_label"] == "Base-16") & (df_all["model"] == m_name)].iloc[0]
        block_rows.append({
            "model": m_name, "base2_bit_acc_pct": round(float(r2["test_acc"]) * 100.0, 3),
            "base2_equiv_4bit_acc_pct": round((float(r2["test_acc"]) ** 4) * 100.0, 3),
            "base16_nibble_acc_pct": round(float(r16["test_acc"]) * 100.0, 3),
            "base2_4bit_entropy_bits": round(float(r2["equiv_4bit_nibble_bits"]), 5),
            "base16_nibble_entropy_bits": round(float(r16["bits_per_symbol"]), 5),
            "delta_nibble_bits_base16_minus_base2": round(float(r16["bits_per_symbol"]) - float(r2["equiv_4bit_nibble_bits"]), 5),
        })

    stat_audit: dict[str, Any] = {"bit_nibble_equivalence": bit_nibble_check, "bases": {}}
    for ds_name, base_label, radix in EXP3_BASES:
        sub_rows = [r for r in rows if r["dataset"] == ds_name]
        best_n = min([r for r in sub_rows if r["model"] in CANONICAL_TO_ARCH], key=lambda r: (r["normalized_bits"], -r["test_acc"]))
        comps = {}
        for m_name in ("DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer", "PPM_o8"):
            for ref in ("Uniform", "Markov_k1"):
                if (ds_name, m_name) in preds_cache and (ds_name, ref) in preds_cache:
                    comps[f"{m_name}_vs_{ref}"] = compute_mcnemar_exact(preds_cache[(ds_name, m_name)], preds_cache[(ds_name, ref)]) | {"model_a": m_name, "model_b": ref}
        z_audit = {r["model"]: {"test_acc": r["test_acc"], "wilson_ci": [r["wilson_ci_low"], r["wilson_ci_high"]], "z_score": r["z_score"], "z_p_value": r["z_p_value"], "normalized_bits": r["normalized_bits"]} for r in sub_rows if r["model"] != "Uniform"}
        stat_audit["bases"][base_label] = {"dataset": ds_name, "radix": radix, "best_neural_by_entropy": best_n["model"], "best_neural_normalized_bits": best_n["normalized_bits"], "z_tests": z_audit, "mcnemar_comparisons": comps}
    return block_rows, stat_audit


def run_experiment_3(archs: list[str], size_tier: str = "medium", max_epochs: int = 10, patience: int = 3, reuse_base10: bool = True) -> Path:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config["seed"]))
    window_size = int(config["data"]["default_window_size"])
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]
    device = resolve_device(str(config["training"].get("device", "cuda")))

    _, splits, stats_rows, stats_full, bit_nibble_check = _load_and_audit_bases(config, window_size)
    records = _seed_base10_cache(runs_root) if reuse_base10 else {}
    _train_missing_bases(archs, records, splits, config, size_tier, max_epochs, patience)
    rows, preds_cache = _collect_preds_and_build_rows(records, splits, config, size_tier, device)
    df_all = pd.DataFrame(rows)
    block_rows, stat_audit = _build_block_and_mcnemar_audit(df_all, rows, preds_cache, bit_nibble_check)

    summary_dir = create_run_dir(runs_root, "exp3_bases")
    (summary_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    df_all.to_csv(summary_dir / "metrics.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(summary_dir / "classical_stats_bases.csv", index=False)
    pd.DataFrame(block_rows).to_csv(summary_dir / "block_4bit_vs_nibble.csv", index=False)
    (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (summary_dir / "plot_data.json").write_text(json.dumps({"metrics": rows, "classical_stats": stats_full, "block_4bit_vs_nibble": block_rows, "bit_nibble_equivalence": bit_nibble_check}, ensure_ascii=False, indent=2), encoding="utf-8")

    h1, p1 = plot_exp3_bases_normalized_bits(df_all, summary_dir)
    h2, p2 = plot_exp3_bases_zscore_ci(df_all, summary_dir)
    print(f"\nЭксперимент 3 завершён! Артефакты: {summary_dir}\n- {p1}\n- {p2}")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 3: Системы счисления Base-2, Base-10, Base-16 (BBP)")
    parser.add_argument("--arch", type=str, default="all", help="Архитектуры: mlp, cnn, gru, transformer или all")
    parser.add_argument("--size", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-reuse-base10", action="store_true")
    args = parser.parse_args()

    archs = ["mlp", "cnn", "gru", "transformer"] if args.arch == "all" else [a.strip().lower() for a in args.arch.split(",") if a.strip()]
    run_experiment_3(archs=archs, size_tier=args.size, max_epochs=args.max_epochs, patience=args.patience, reuse_base10=not args.no_reuse_base10)


if __name__ == "__main__":
    main()
