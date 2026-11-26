# Эксперимент 2: сравнение 9 последовательностей (константы, ГПСЧ, 1/999983, Чемперноун, цепные дроби e и pi)
import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.baselines import evaluate_uniform_baseline, fit_and_evaluate_lightgbm, fit_and_evaluate_markov, fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import ChronologicalSplit, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import compute_mcnemar_exact, wilson_ci
from src.exp_utils import (
    CANONICAL_TO_ARCH,
    MODEL_ORDER,
    baseline_record,
    canonical_model_name,
    eval_markov_preds_and_speed,
    eval_ppm_preds_and_speed,
    find_latest_csv,
    read_sec_per_epoch,
    resolve_run_dir,
)
from src.models import build_model, count_parameters
from src.stats_tests import run_full_statistical_suite
from src.trainer import create_run_dir, remap_legacy_state_dict, resolve_device, train_model
from src.visualizer import (
    plot_exp2_cf_and_champernowne_contrast,
    plot_exp2_constants_comparison,
    plot_exp2_embedding_pca_preview,
    plot_exp2_heatmap_bits,
    plot_exp2_radar_pyecharts,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
EXP2_DATASETS = ["true_pi", "e", "sqrt2", "mt19937", "csprng", "pi_cf", "rational_1_999983", "champernowne", "e_cf"]
NEW_DATASETS = ["champernowne", "e_cf", "pi_cf"]
H0_GROUP = {"true_pi", "e", "sqrt2", "mt19937", "csprng", "pi_cf"}
H0_DECIMAL_DATASETS = ["true_pi", "e", "sqrt2", "mt19937", "csprng"]
EXP5_CHECKPOINT_TARGETS = {
    "DigitTransformer": ["e_cf", "champernowne", "rational_1_999983", "pi_cf", "true_pi"],
    "DigitGRU": ["rational_1_999983", "champernowne", "e_cf", "true_pi"],
}


def _load_baselines_and_exp1_cache(runs_root: Path, cached: dict[tuple[str, str], dict[str, Any]]) -> None:
    b_csv = find_latest_csv(runs_root, "baselines_") or (ROOT_DIR / "results" / "01_baselines" / "metrics.csv")
    if b_csv.exists():
        for _, row in pd.read_csv(b_csv).iterrows():
            ds = str(row["dataset"])
            if ds in EXP2_DATASETS:
                model = canonical_model_name(str(row["model"]))
                cached[(ds, model)] = {
                    "dataset": ds, "radix": int(row["radix"]), "model": model,
                    "test_acc": float(row["overall_acc"]), "test_ce": float(row["cross_entropy"]),
                    "test_ppl": float(row["perplexity"]), "pred_entropy": float(row["pred_entropy"]),
                    "num_eval_samples": int(row["num_eval_samples"]), "train_time_sec": float(row["train_time_sec"]),
                    "source": b_csv.parent.name,
                }
    ppm_csv = find_latest_csv(runs_root, "ppm_summary_") or (ROOT_DIR / "results" / "02_exp1_sanity" / "ppm_metrics.csv")
    if ppm_csv.exists():
        for _, row in pd.read_csv(ppm_csv).iterrows():
            ds = str(row["dataset"])
            if ds in EXP2_DATASETS:
                model = canonical_model_name(str(row["model"]))
                cached[(ds, model)] = {
                    "dataset": ds, "radix": 10, "model": model, "test_acc": float(row["overall_acc"]),
                    "test_ce": float(row["cross_entropy"]), "test_ppl": float(row["perplexity"]),
                    "pred_entropy": float(row["pred_entropy"]), "num_eval_samples": 99936,
                    "train_time_sec": float(row["train_time_sec"]), "source": ppm_csv.parent.name,
                }
    for arch in ("mlp", "cnn", "gru", "transformer"):
        csv3 = find_latest_csv(runs_root, f"exp1_{arch}_summary_")
        if csv3 is None:
            continue
        for _, row in pd.read_csv(csv3).iterrows():
            ds = str(row["dataset"])
            if ds in EXP2_DATASETS:
                m_name = canonical_model_name(arch.upper())
                cached[(ds, m_name)] = {
                    "dataset": ds, "radix": 10, "model": m_name, "test_acc": float(row["test_overall_acc"]),
                    "test_ce": float(row["test_cross_entropy"]), "test_ppl": float(row["test_perplexity"]),
                    "pred_entropy": float(row["test_pred_entropy"]), "num_eval_samples": 99936,
                    "train_time_sec": float(row["train_time_sec"]), "param_count": int(row["param_count"]),
                    "epochs_ran": int(row["epochs_ran"]), "best_epoch": int(row["best_epoch"]), "source": str(row["run_dir"]),
                }


def load_cached_results(runs_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    _load_baselines_and_exp1_cache(runs_root, cached)
    for d in sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("exp2_constants_")], key=lambda p: p.name):
        if not (d / "metrics.csv").exists():
            continue
        df_e2 = pd.read_csv(d / "metrics.csv")
        for _, row in df_e2.iterrows():
            ds, m_name = str(row["dataset"]), canonical_model_name(str(row["model"]))
            ce_val = row["test_ce_nats"] if "test_ce_nats" in df_e2.columns else row["test_ce"]
            p_cnt = row.get("params_count", row.get("param_count"))
            cached[(ds, m_name)] = {
                "dataset": ds, "radix": int(row["radix"]), "model": m_name, "test_acc": float(row["test_acc"]),
                "test_ce": float(ce_val), "test_ppl": float(row["test_ppl"]),
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
    for arch in ("mlp", "cnn", "gru", "transformer"):
        m_name = canonical_model_name(arch.upper())
        for ds in EXP2_DATASETS:
            if (ds, m_name) in cached:
                continue
            pat = re.compile(rf"^exp2_{re.escape(arch)}_{re.escape(ds)}_\d{{4}}-\d{{2}}-\d{{2}}_")
            for d in sorted([p for p in runs_root.iterdir() if p.is_dir() and pat.match(p.name)], key=lambda p: p.name, reverse=True):
                if (d / "plot_data.json").exists() and (d / "best_model.pt").exists():
                    data = json.loads((d / "plot_data.json").read_text(encoding="utf-8"))
                    s, tm = data["summary"], data["summary"]["test_metrics"]
                    ep_t = [float(h["epoch_time_sec"]) for h in data.get("history", []) if h.get("epoch_time_sec") is not None]
                    cached[(ds, m_name)] = {
                        "dataset": ds, "radix": 11 if ds in ("e_cf", "pi_cf") else 10, "model": m_name,
                        "test_acc": float(tm["overall_acc"]), "test_ce": float(tm["cross_entropy"]),
                        "test_ppl": float(tm["perplexity"]), "pred_entropy": float(tm["pred_entropy"]) if tm.get("pred_entropy") is not None else None,
                        "num_eval_samples": int(tm["num_eval_samples"]), "train_time_sec": float(s["total_train_time_sec"]),
                        "param_count": int(s["param_count"]), "epochs_ran": int(s["epochs_ran"]),
                        "best_epoch": int(s["best_epoch"]), "sec_per_epoch": round(float(np.mean(ep_t)), 3) if ep_t else None,
                        "source": str(s.get("run_dir", d)),
                    }
                    break
    return cached


@torch.no_grad()
def _benchmark_and_predict_neural(
    arch: str, run_dir: Path, split: ChronologicalSplit, radix: int, window_size: int, size_tier: str, batch_size: int, num_workers: int, device: torch.device,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    ckpt_path = run_dir / "best_model.pt"
    loaders = create_dataloaders(split=split, window_size=window_size, batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd, remapped = remap_legacy_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(sd)
    model = model.to(device).eval()

    amp_enabled = device.type == "cuda"
    test_loader = loaders["test"]
    for ctx, _, _ in test_loader:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _ = model(ctx.to(device, non_blocking=True))
        break
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0, chunks = time.perf_counter(), []
    for ctx, tgt, _ in test_loader:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(ctx.to(device, non_blocking=True))
        chunks.append((torch.argmax(logits.float(), dim=-1) == tgt.to(device, non_blocking=True)).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    correct_bool = np.concatenate(chunks, axis=0)
    sps = round(float(len(correct_bool) / max(time.perf_counter() - t0, 1e-5)), 1)

    meta: dict[str, Any] = {
        "arch": arch, "radix": int(radix), "window_size": int(window_size), "size_tier": size_tier,
        "param_count": int(count_parameters(model)), "best_epoch": int(ckpt.get("epoch", 0)),
        "val_loss": float(ckpt.get("val_loss", float("nan"))), "ckpt_path": ckpt_path, "was_remapped_legacy": remapped,
    }
    if arch == "transformer":
        meta.update({"d_model": int(model.d_model), "n_heads": int(model.n_heads), "n_layers": int(model.n_layers), "token_emb": model.token_emb.weight.detach().cpu().numpy().copy()})
    elif arch == "gru":
        meta.update({"hidden_size": int(model.hidden_size), "num_layers": int(model.num_layers), "emb_dim": int(model.emb_dim)})
    elif arch == "mlp":
        meta.update({"emb_dim": int(model.emb_dim)})
    elif arch == "cnn":
        meta.update({"channels": int(model.channels), "head_hidden": int(model.head_hidden)})
    return correct_bool, sps, meta


def _compute_radar_scores(df_final: pd.DataFrame) -> dict[str, dict[str, float]]:
    families = ["PPM_o8", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    shannon_10 = math.log2(10.0)
    piv_bits = df_final.pivot(index="dataset", columns="model", values="bits_per_symbol")
    piv_speed = df_final.pivot(index="dataset", columns="model", values="samples_per_sec")
    med_speeds = {m: float(piv_speed[m].dropna().astype(float).median()) if m in piv_speed.columns and not piv_speed[m].dropna().empty else 0.0 for m in families}
    max_speed = max(max(med_speeds.values()), 1.0)

    scores: dict[str, dict[str, float]] = {}
    for m in families:
        if m not in piv_bits.columns:
            continue
        col = piv_bits[m]
        mean_h0 = float(col.reindex(H0_DECIMAL_DATASETS).mean())
        scores[m] = {
            "noise_calibration": round(100.0 * math.exp(-10.0 * abs(mean_h0 - shannon_10)), 2),
            "periodic_memory": round(max(0.0, min(100.0, 100.0 * (1.0 - float(col["rational_1_999983"]) / shannon_10))), 2),
            "multi_digit_counting": round(max(0.0, min(100.0, 100.0 * (1.0 - float(col["champernowne"]) / shannon_10))), 2),
            "cf_arithmetic": round(max(0.0, min(100.0, 100.0 * (1.0 - float(col["e_cf"]) / max(float(col["pi_cf"]), 1e-9)))), 2),
            "throughput": round(100.0 * math.sqrt(med_speeds[m] / max_speed), 2),
            "median_samples_per_sec": round(med_speeds[m], 1),
        }
    return scores


def _prepare_exp2_datasets(config: dict[str, Any], window_size: int) -> tuple[dict, dict, list[dict], list[dict]]:
    split_cfg = config["data"]["split"]
    artifacts, splits, stats_rows, stats_full = {}, {}, [], []
    for ds_name in EXP2_DATASETS:
        art = get_or_create_dataset(name=ds_name, config=config)
        artifacts[ds_name] = art
        split = split_sequence_chronologically(
            digits=art.digits, mask=art.mask,
            train_ratio=float(split_cfg["train"]), val_ratio=float(split_cfg["val"]), test_ratio=float(split_cfg["test"]),
        )
        verify_zero_leakage(split, window_size=window_size)
        splits[ds_name] = split
        if ds_name in NEW_DATASETS:
            st = run_full_statistical_suite(ds_name, art.digits, radix=art.radix, max_lag=64)
            stats_full.append(st)
            stats_rows.append({
                "name": st["name"], "radix": st["radix"], "length": st["length"], "chi2_stat": round(st["chi2_stat"], 4),
                "chi2_p_value": st["chi2_p_value"], "chi2_passed": st["chi2_passed"], "ks_stat": round(st["ks_stat"], 6),
                "ks_p_value": st["ks_p_value"], "ks_passed": st["ks_passed"], "acf_max_abs": round(st["acf_max_abs"], 6),
                "acf_max_lag": st["acf_max_lag"], "acf_significant_count": st["acf_significant_count"],
                "gzip_ratio": round(st["gzip_ratio"], 5), "lzma_ratio": round(st["lzma_ratio"], 5),
                "lzma_bits_per_digit": round(st["lzma_bits_per_digit"], 4), "shannon_limit_bits": round(st["shannon_limit_bits_per_digit"], 4),
                "empirical_entropy_nats": round(st["empirical_entropy_nats"], 6),
            })
    return artifacts, splits, stats_rows, stats_full


def _ensure_exp2_baselines_and_train(
    archs: list[str], records: dict[tuple[str, str], dict[str, Any]], artifacts: dict, splits: dict,
    config: dict[str, Any], size_tier: str, max_epochs: int, patience: int,
) -> None:
    seed, window_size = int(config["seed"]), int(config["data"]["default_window_size"])
    batch_size, num_workers = int(config["training"]["batch_size"]), int(config["training"].get("num_workers", 4))
    for ds_name in EXP2_DATASETS:
        art, split = artifacts[ds_name], splits[ds_name]
        n_eval = len(split.test_digits) - window_size
        if (ds_name, "Uniform") not in records:
            records[(ds_name, "Uniform")] = baseline_record(ds_name, art.radix, "Uniform", evaluate_uniform_baseline(split, art.radix, window_size), n_eval, "exp2_fresh", 0.0)
        for k in (1, 2, 3):
            if (ds_name, f"Markov_k{k}") not in records:
                mk, _ = fit_and_evaluate_markov(split, radix=art.radix, order=k, window_size=window_size)
                records[(ds_name, f"Markov_k{k}")] = baseline_record(ds_name, art.radix, f"Markov_k{k}", mk, n_eval, "exp2_fresh")
        if (ds_name, "LightGBM") not in records:
            lgb_res = fit_and_evaluate_lightgbm(split, radix=art.radix, window_size=window_size, seed=seed, max_train_samples=200_000, n_estimators=150)
            records[(ds_name, "LightGBM")] = baseline_record(ds_name, art.radix, "LightGBM", lgb_res, n_eval, "exp2_fresh")
        if (ds_name, "PPM_o8") not in records:
            ppm_res, _ = fit_and_evaluate_ppm(split, radix=art.radix, window_size=window_size, max_order=8)
            records[(ds_name, "PPM_o8")] = baseline_record(ds_name, art.radix, "PPM_o8", ppm_res, n_eval, "exp2_fresh")

    for arch in archs:
        m_name = canonical_model_name(arch.upper())
        for ds_name in EXP2_DATASETS:
            if (ds_name, m_name) in records:
                continue
            seed_everything(seed)
            art, split = artifacts[ds_name], splits[ds_name]
            loaders = create_dataloaders(split=split, window_size=window_size, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
            model = build_model(arch=arch, radix=art.radix, window_size=window_size, size_tier=size_tier)
            res = train_model(model, loaders, config, f"exp2_{arch}_{ds_name}", art.radix, epochs=max_epochs, early_stopping_patience=patience, enable_early_stopping=True)
            s, tm, hist = res["summary"], res["test_metrics"], res["history"]
            records[(ds_name, m_name)] = {
                "dataset": ds_name, "radix": art.radix, "model": m_name, "test_acc": float(tm["overall_acc"]),
                "test_ce": float(tm["cross_entropy"]), "test_ppl": float(tm["perplexity"]), "pred_entropy": float(tm["pred_entropy"]),
                "num_eval_samples": int(tm["num_eval_samples"]), "train_time_sec": float(s["total_train_time_sec"]),
                "param_count": int(s["param_count"]), "epochs_ran": int(s["epochs_ran"]), "best_epoch": int(s["best_epoch"]),
                "sec_per_epoch": round(float(np.mean([h["epoch_time_sec"] for h in hist])), 3) if hist else None, "source": s["run_dir"],
            }


def _run_exp2_finalize_audit(
    summary_dir: Path, records: dict[tuple[str, str], dict[str, Any]], artifacts: dict, splits: dict,
    config: dict[str, Any], size_tier: str, device: torch.device,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, dict[str, Any]]]:
    seed, window_size = int(config["seed"]), int(config["data"]["default_window_size"])
    batch_size, num_workers = int(config["training"]["batch_size"]), int(config["training"].get("num_workers", 4))
    preds_cache, neural_meta, pca_data = {}, {}, {}

    for ds_name in EXP2_DATASETS:
        art, split = artifacts[ds_name], splits[ds_name]
        for k in (1, 2, 3):
            mk_corr, mk_sps = eval_markov_preds_and_speed(split, art.radix, k, window_size)
            if k == 3:
                preds_cache[(ds_name, "Markov_k3")] = mk_corr
            if (ds_name, f"Markov_k{k}") in records:
                records[(ds_name, f"Markov_k{k}")]["samples_per_sec"] = mk_sps
        ppm_corr, ppm_sps = eval_ppm_preds_and_speed(split, art.radix, window_size, max_order=8)
        preds_cache[(ds_name, "PPM_o8")] = ppm_corr
        if (ds_name, "PPM_o8") in records:
            records[(ds_name, "PPM_o8")]["samples_per_sec"] = ppm_sps
        if (ds_name, "Uniform") in records and not records[(ds_name, "Uniform")].get("samples_per_sec"):
            records[(ds_name, "Uniform")]["samples_per_sec"] = round((len(split.test_digits) - window_size) / 0.001, 1)

        for m_name, arch in CANONICAL_TO_ARCH.items():
            rec = records.get((ds_name, m_name))
            run_dir = resolve_run_dir(rec.get("source"), ROOT_DIR) if rec else None
            if rec is None or run_dir is None:
                continue
            rec["sec_per_epoch"] = read_sec_per_epoch(run_dir, rec.get("train_time_sec"), rec.get("epochs_ran"))
            corr, sps, meta = _benchmark_and_predict_neural(arch, run_dir, split, art.radix, window_size, size_tier, batch_size, num_workers, device)
            preds_cache[(ds_name, m_name)], rec["samples_per_sec"] = corr, sps
            meta["source_run_dir"] = str(run_dir)
            neural_meta[(ds_name, m_name)] = meta
            if m_name == "DigitTransformer" and ds_name in ("true_pi", "e_cf", "champernowne"):
                pca = PCA(n_components=2, random_state=seed)
                coords = pca.fit_transform(meta.pop("token_emb"))
                pca_data[ds_name] = {
                    "radix": int(art.radix), "coords": np.round(coords, 6).tolist(),
                    "explained_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_],
                    "explained_variance_sum": round(float(np.sum(pca.explained_variance_ratio_)), 6),
                }
            else:
                meta.pop("token_emb", None)

    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for m_name, ds_list in EXP5_CHECKPOINT_TARGETS.items():
        arch = CANONICAL_TO_ARCH[m_name]
        for ds_name in ds_list:
            if (meta := neural_meta.get((ds_name, m_name))) is None:
                continue
            dst = ckpt_dir / f"{arch}_{ds_name}.pt"
            if meta.get("was_remapped_legacy"):
                raw_ck = torch.load(meta["ckpt_path"], map_location="cpu", weights_only=True)
                raw_ck["model_state_dict"], _ = remap_legacy_state_dict(raw_ck["model_state_dict"])
                torch.save(raw_ck, dst)
            else:
                shutil.copy2(meta["ckpt_path"], dst)
            manifest.append({k: v for k, v in meta.items() if k != "ckpt_path"} | {"model": m_name, "dataset": ds_name, "checkpoint_file": dst.name})
    (ckpt_dir / "checkpoints_metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return preds_cache, pca_data


def _build_exp2_final_rows(records: dict[tuple[str, str], dict[str, Any]], artifacts: dict) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    for ds_name in EXP2_DATASETS:
        art = artifacts[ds_name]
        shannon = math.log2(art.radix)
        mk_bits = [records[(ds_name, f"Markov_k{k}")]["test_ce"] / math.log(2.0) for k in (1, 2, 3) if (ds_name, f"Markov_k{k}") in records]
        best_mk = min(mk_bits) if mk_bits else shannon
        for m_name in MODEL_ORDER:
            if (ds_name, m_name) not in records:
                continue
            r = records[(ds_name, m_name)]
            n_eval = int(r["num_eval_samples"])
            ci_lo, ci_hi = wilson_ci(int(round(r["test_acc"] * n_eval)), n_eval)
            bits = r["test_ce"] / math.log(2.0)
            final_rows.append({
                "dataset": ds_name, "radix": art.radix, "hypothesis_group": "H0" if ds_name in H0_GROUP else "H1",
                "model": m_name, "params_count": r.get("param_count"), "test_acc": round(r["test_acc"], 6),
                "wilson_ci_low": ci_lo, "wilson_ci_high": ci_hi, "test_ce_nats": round(r["test_ce"], 6),
                "test_ppl": round(r["test_ppl"], 4), "bits_per_symbol": round(bits, 6), "shannon_limit_bits": round(shannon, 6),
                "gain_over_markov_bits": round(best_mk - bits, 6), "sec_per_epoch": r.get("sec_per_epoch"),
                "samples_per_sec": r.get("samples_per_sec"), "pred_entropy": round(r["pred_entropy"], 6) if r.get("pred_entropy") is not None else None,
                "num_eval_samples": n_eval, "epochs_ran": r.get("epochs_ran"), "best_epoch": r.get("best_epoch"),
                "train_time_sec": r.get("train_time_sec"), "length": art.length, "source": r.get("source"),
            })
    return final_rows


def _save_mcnemar_and_plots(
    summary_dir: Path, df_final: pd.DataFrame, final_rows: list[dict], preds_cache: dict[tuple[str, str], np.ndarray],
    new_stats_rows: list[dict], new_stats_full: list[dict], pca_data: dict[str, dict], finalize: bool, stage_tag: str,
) -> None:
    if finalize:
        stat_audit: dict[str, Any] = {}
        for ds_name in EXP2_DATASETS:
            best_n = min([r for r in final_rows if r["dataset"] == ds_name and r["model"] in CANONICAL_TO_ARCH], key=lambda r: (r["bits_per_symbol"], -r["test_acc"]))
            comps = {}
            for label, ma, mb in (("DigitTransformer_vs_Markov_k3", "DigitTransformer", "Markov_k3"), ("DigitTransformer_vs_PPM_o8", "DigitTransformer", "PPM_o8"), ("BestNeural_vs_Markov_k3", best_n["model"], "Markov_k3"), ("BestNeural_vs_PPM_o8", best_n["model"], "PPM_o8")):
                if (ds_name, ma) in preds_cache and (ds_name, mb) in preds_cache:
                    comps[label] = compute_mcnemar_exact(preds_cache[(ds_name, ma)], preds_cache[(ds_name, mb)]) | {"model_a": ma, "model_b": mb}
            stat_audit[ds_name] = {"hypothesis_group": "H0" if ds_name in H0_GROUP else "H1", "best_neural_model": best_n["model"], "best_neural_bits": best_n["bits_per_symbol"], "best_neural_acc": best_n["test_acc"], "comparisons": comps}
        (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    radar_scores = _compute_radar_scores(df_final) if finalize else {}
    (summary_dir / "plot_data.json").write_text(
        json.dumps({"metrics": final_rows, "classical_stats_new_constants": new_stats_full, "radar_scores": radar_scores, "embedding_pca_preview": pca_data, "datasets": EXP2_DATASETS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if finalize:
        plot_exp2_heatmap_bits(df_final, summary_dir, datasets=EXP2_DATASETS, models=["PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"])
        plot_exp2_cf_and_champernowne_contrast(df_final, summary_dir)
        plot_exp2_radar_pyecharts(radar_scores, summary_dir)
        plot_exp2_embedding_pca_preview(pca_data, summary_dir)
    else:
        models_for_plot = [m for m in ("PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer") if m in df_final["model"].unique()]
        plot_exp2_constants_comparison(df_final, summary_dir, EXP2_DATASETS, models_for_plot, title_suffix=stage_tag.upper())


def run_experiment_2(archs: list[str], size_tier: str = "medium", max_epochs: int = 10, patience: int = 3, reuse_completed: bool = True, stage_tag: str = "final", finalize: bool = True) -> Path:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config["seed"]))
    window_size = int(config["data"]["default_window_size"])
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]
    device = resolve_device(str(config["training"].get("device", "cuda")))

    records: dict[tuple[str, str], dict[str, Any]] = dict(load_cached_results(runs_root) if reuse_completed else {})
    artifacts, splits, new_stats_rows, new_stats_full = _prepare_exp2_datasets(config, window_size)
    _ensure_exp2_baselines_and_train(archs, records, artifacts, splits, config, size_tier, max_epochs, patience)

    summary_dir = create_run_dir(runs_root, f"exp2_constants_{stage_tag}")
    (summary_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    preds_cache, pca_data = _run_exp2_finalize_audit(summary_dir, records, artifacts, splits, config, size_tier, device) if finalize else ({}, {})

    final_rows = _build_exp2_final_rows(records, artifacts)
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(summary_dir / "metrics.csv", index=False)
    if new_stats_rows:
        pd.DataFrame(new_stats_rows).to_csv(summary_dir / "classical_stats_new_constants.csv", index=False)
    _save_mcnemar_and_plots(summary_dir, df_final, final_rows, preds_cache, new_stats_rows, new_stats_full, pca_data, finalize, stage_tag)

    print(f"\nСводка Эксперимента 2 ({stage_tag}) сохранена в: {summary_dir}")
    piv_bits = df_final.pivot(index="dataset", columns="model", values="bits_per_symbol").reindex(EXP2_DATASETS)
    show_cols = [c for c in ("Markov_k1", "Markov_k3", "PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer") if c in piv_bits.columns]
    print(piv_bits[show_cols].round(4).to_string())
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 2: Битва констант, ГПСЧ и цепных дробей")
    parser.add_argument("--arch", type=str, default="transformer", help="Архитектура: mlp, cnn, gru, transformer, all, или baselines_only")
    parser.add_argument("--size", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--stage-tag", type=str, default="final")
    parser.add_argument("--no-finalize", action="store_true", help="Не запускать McNemar-аудит и финальные 4 графика")
    args = parser.parse_args()

    archs = ["mlp", "cnn", "gru", "transformer"] if args.arch == "all" else ([] if args.arch in ("baselines_only", "none") else [a.strip().lower() for a in args.arch.split(",") if a.strip()])
    run_experiment_2(archs=archs, size_tier=args.size, max_epochs=args.max_epochs, patience=args.patience, reuse_completed=args.reuse_completed, stage_tag=args.stage_tag, finalize=not args.no_finalize)


if __name__ == "__main__":
    main()
