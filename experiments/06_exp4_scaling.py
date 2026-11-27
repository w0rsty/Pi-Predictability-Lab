# Масштабирование окна L (32/64/128), бюджета параметров (50K/250K/1.5M) и тест на переобучение
import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.baselines import PPMCompressor, fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import ChronologicalSplit, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import compute_binomial_z_score, compute_mcnemar_exact, evaluate_model, wilson_ci
from src.exp_utils import CANONICAL_TO_ARCH, canonical_model_name, find_latest_csv, free_gpu, predict_and_benchmark, resolve_run_dir
from src.models import build_model, count_parameters
from src.trainer import create_run_dir, resolve_device, train_model
from src.viz import (
    plot_exp4_capacity_scaling,
    plot_exp4_compute_frontier,
    plot_exp4_memorization_scissors,
    plot_exp4_window_phase_transition,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _pick_batch_size(size_tier: str, window_size: int, arch: str = "") -> int:
    # На heavy 1.54M трансформере при bs=1024 упирался в 8 ГБ VRAM, поэтому ставим 256
    if window_size >= 128:
        return 512 if size_tier != "heavy" else 256
    if size_tier == "heavy":
        return 256 if arch == "transformer" else 512
    return 1024 if size_tier == "medium" else 2048


def _build_metric_row(
    block: str, dataset: str, hypothesis_group: str, radix: int, window_size: int, size_tier: str, model_name: str,
    params_count: int | None, test_acc: float, test_ce: float, test_ppl: float, pred_entropy: float | None, num_eval_samples: int,
    sec_per_epoch: float | None, samples_per_sec: float | None, epochs_ran: int | None, best_epoch: int | None, train_time_sec: float | None, source: str,
) -> dict[str, Any]:
    ci_lo, ci_hi = wilson_ci(int(round(test_acc * num_eval_samples)), num_eval_samples)
    z_info = compute_binomial_z_score(test_acc, radix=radix, n_samples=num_eval_samples, ci_low=ci_lo, ci_high=ci_hi)
    shannon = math.log2(radix)
    bits = test_ce / math.log(2.0)
    return {
        "block": block, "dataset": dataset, "hypothesis_group": hypothesis_group, "radix": radix,
        "window_size": window_size, "size_tier": size_tier, "model": model_name, "params_count": params_count,
        "test_acc": round(test_acc, 6), "wilson_ci_low": ci_lo, "wilson_ci_high": ci_hi,
        "z_score": z_info["z_score"], "z_p_value": z_info["z_p_value"], "test_ce_nats": round(test_ce, 6),
        "test_ppl": round(test_ppl, 4), "bits_per_symbol": round(bits, 6), "shannon_limit_bits": round(shannon, 6),
        "normalized_bits": round(bits / shannon, 6), "sec_per_epoch": sec_per_epoch, "samples_per_sec": samples_per_sec,
        "pred_entropy": round(pred_entropy, 6) if pred_entropy is not None else None, "num_eval_samples": num_eval_samples,
        "epochs_ran": epochs_ran, "best_epoch": best_epoch, "train_time_sec": train_time_sec, "source": source,
    }


def _save_partial(rows_map: dict, summary_dir: Path) -> None:
    pd.DataFrame(list(rows_map.values())).to_csv(summary_dir / "partial_metrics.csv", index=False)


def _load_existing_partial(summary_dir: Path) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    part_p = summary_dir / "partial_metrics.csv"
    if not part_p.exists():
        return {}
    rows_map = {}
    for _, row in pd.read_csv(part_p).iterrows():
        key = (str(row["block"]), str(row["dataset"]), int(row["window_size"]), str(row["size_tier"]), str(row["model"]))
        rows_map[key] = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    return rows_map


def _run_or_reuse_neural(
    block: str, ds_name: str, hyp_group: str, arch: str, size_tier: str, window_size: int, split: ChronologicalSplit,
    radix: int, config: dict[str, Any], exp2_cache: dict[tuple[str, str], dict[str, Any]], rows_map: dict,
    summary_dir: Path, device: torch.device, seed: int, max_epochs: int, patience: int, scheduler_type: str | None,
) -> dict[str, Any]:
    m_name = canonical_model_name(arch.upper())
    key = (block, ds_name, int(window_size), size_tier, m_name)
    if key in rows_map:
        return rows_map[key]
    if window_size == 64 and size_tier == "medium" and (ds_name, m_name) in exp2_cache:
        c = exp2_cache[(ds_name, m_name)]
        row = _build_metric_row(
            block, ds_name, hyp_group, radix, 64, "medium", m_name,
            int(c["params_count"]) if c.get("params_count") is not None else None,
            float(c["test_acc"]), float(c["test_ce_nats"]), float(c["test_ppl"]),
            float(c["pred_entropy"]) if c.get("pred_entropy") is not None else None, int(c["num_eval_samples"]),
            float(c["sec_per_epoch"]) if c.get("sec_per_epoch") is not None else None,
            float(c["samples_per_sec"]) if c.get("samples_per_sec") is not None else None,
            int(c["epochs_ran"]) if c.get("epochs_ran") is not None else None,
            int(c["best_epoch"]) if c.get("best_epoch") is not None else None,
            float(c["train_time_sec"]) if c.get("train_time_sec") is not None else None, str(c["source"]),
        )
        rows_map[key] = row
        _save_partial(rows_map, summary_dir)
        return row

    seed_everything(seed)
    bs = _pick_batch_size(size_tier, window_size, arch=arch)
    num_workers = int(config["training"].get("num_workers", 4))
    loaders = create_dataloaders(split=split, window_size=window_size, batch_size=bs, num_workers=num_workers, pin_memory=True, seed=seed)
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    res = train_model(
        model, loaders, config, f"exp4_{block.lower()}_{arch}_{ds_name}_L{window_size}_{size_tier}",
        radix, epochs=max_epochs, learning_rate=float(config["training"]["learning_rate"]), weight_decay=1e-4,
        scheduler_type=scheduler_type, amp_dtype="bfloat16", early_stopping_patience=patience, enable_early_stopping=True,
    )
    del model, loaders
    free_gpu()

    s, tm, hist = res["summary"], res["test_metrics"], res["history"]
    _, sps, _ = predict_and_benchmark(arch, Path(s["run_dir"]), split, radix, window_size, size_tier, bs, num_workers, device, amp_dtype="bfloat16", seed=seed)
    sec_ep = round(float(np.mean([h["epoch_time_sec"] for h in hist])), 3) if hist else None
    row = _build_metric_row(
        block, ds_name, hyp_group, radix, window_size, size_tier, m_name, int(s["param_count"]),
        float(tm["overall_acc"]), float(tm["cross_entropy"]), float(tm["perplexity"]), float(tm["pred_entropy"]),
        int(tm["num_eval_samples"]), sec_ep, sps, int(s["epochs_ran"]), int(s["best_epoch"]), float(s["total_train_time_sec"]), s["run_dir"],
    )
    rows_map[key] = row
    _save_partial(rows_map, summary_dir)
    return row


def _train_mem_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, use_amp: bool) -> tuple[float, float]:
    model.train()
    tr_loss_sum, tr_corr, tr_n = 0.0, 0, 0
    for contexts, targets, _ in loader:
        contexts, targets = contexts.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(contexts)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            tr_n += targets.size(0)
            tr_loss_sum += float(loss.item()) * targets.size(0)
            tr_corr += int((torch.argmax(logits.float(), dim=-1) == targets).sum().item())
    return tr_loss_sum / max(tr_n, 1), tr_corr / max(tr_n, 1)


def run_block_d_memorization(pi_digits: np.ndarray, summary_dir: Path, config: dict[str, Any], seed: int = 42, epochs: int = 200, batch_size: int = 128) -> pd.DataFrame:
    # Блок D: dropout=0, без EarlyStopping — переобучение heavy 1.54M на 10K знаках pi (Train vs Test)
    curve_path = summary_dir / "memorization_curve.csv"
    if curve_path.exists():
        existing = pd.read_csv(curve_path)
        if len(existing) >= epochs:
            return existing

    seed_everything(seed)
    device = resolve_device(str(config["training"].get("device", "cuda")))
    split = split_sequence_chronologically(pi_digits[:10_000], mask=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    verify_zero_leakage(split, window_size=64)
    loaders = create_dataloaders(split=split, window_size=64, batch_size=batch_size, num_workers=2, pin_memory=True, seed=seed)
    model = build_model("transformer", radix=10, window_size=64, size_tier="heavy", dropout=0.0).to(device)
    p_cnt = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    use_amp = device.type == "cuda"
    ln2 = math.log(2.0)

    history, t_start = [], time.perf_counter()
    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        tr_ce, tr_acc = _train_mem_epoch(model, loaders["train"], optimizer, device, use_amp)
        val_m = evaluate_model(model, loaders["val"], device=device, radix=10, use_amp=use_amp, amp_dtype="bfloat16", collect_transitions=False)
        test_m = evaluate_model(model, loaders["test"], device=device, radix=10, use_amp=use_amp, amp_dtype="bfloat16", collect_transitions=False)
        history.append({
            "epoch": ep, "train_loss": round(tr_ce, 6), "train_acc": round(tr_acc, 6), "train_bits": round(tr_ce / ln2, 6),
            "val_loss": round(float(val_m["cross_entropy"]), 6), "val_acc": round(float(val_m["overall_acc"]), 6), "val_bits": round(float(val_m["cross_entropy"]) / ln2, 6),
            "test_loss": round(float(test_m["cross_entropy"]), 6), "test_acc": round(float(test_m["overall_acc"]), 6), "test_bits": round(float(test_m["cross_entropy"]) / ln2, 6),
            "epoch_time_sec": round(time.perf_counter() - ep_t0, 3),
        })
        if ep in (1, 10, 50, 100, 150, epochs) or ep % 40 == 0:
            last = history[-1]
            print(f"  Эпоха {ep:3d}/{epochs} ({time.perf_counter()-t_start:.1f}с) | Train: {last['train_acc']*100:6.2f}% | Test: {last['test_acc']*100:5.2f}%")

    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epochs, "model_state_dict": model.state_dict(), "param_count": p_cnt, "train_acc": history[-1]["train_acc"], "test_acc": history[-1]["test_acc"]}, ckpt_dir / "transformer_memorization_10k_heavy.pt")
    del model, loaders, optimizer
    free_gpu()
    mem_df = pd.DataFrame(history)
    mem_df.to_csv(curve_path, index=False)
    return mem_df


def _run_block_b(rat97_split: ChronologicalSplit, config: dict, exp2_cache: dict, rows_map: dict, summary_dir: Path, device: torch.device, seed: int) -> None:
    # На 1/97 (P=96) уже все 96/96 двузначных подстрок уникальны (10^2=100 > 97), поэтому L=32 и PPM сразу берут 100.0%
    for L in (64, 128):
        ppm_key = ("Block_B_PhaseTransition_H1", "rational_1_97", L, "tabular", "PPM_o8")
        if ppm_key not in rows_map:
            verify_zero_leakage(rat97_split, window_size=L)
            ppm_res, _ = fit_and_evaluate_ppm(rat97_split, radix=10, window_size=L, max_order=8)
            sps = round(ppm_res["num_eval_samples"] / max(float(ppm_res["eval_time_sec"]), 1e-4), 1)
            rows_map[ppm_key] = _build_metric_row("Block_B_PhaseTransition_H1", "rational_1_97", "H1", 10, L, "tabular", "PPM_o8", None, ppm_res["overall_acc"], ppm_res["cross_entropy"], ppm_res["perplexity"], ppm_res["pred_entropy"], ppm_res["num_eval_samples"], None, sps, None, None, ppm_res["train_time_sec"], "exp4_ppm_o8")
            _save_partial(rows_map, summary_dir)
    for L in (32, 64, 128):
        verify_zero_leakage(rat97_split, window_size=L)
        for arch in ("mlp", "gru", "transformer"):
            _run_or_reuse_neural("Block_B_PhaseTransition_H1", "rational_1_97", "H1", arch, "medium", L, rat97_split, 10, config, exp2_cache, rows_map, summary_dir, device, seed, 12, 3, "cosine")


def _finalize_exp4(summary_dir: Path, rows_map: dict, rat97_split: ChronologicalSplit, config: dict, device: torch.device) -> list[Path]:
    df_metrics = pd.DataFrame(list(rows_map.values()))
    df_metrics.to_csv(summary_dir / "metrics.csv", index=False)

    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rat_key = ("Block_B_PhaseTransition_H1", "rational_1_97", 128, "medium", "DigitTransformer")
    ckpt_meta = []
    if rat_key in rows_map:
        src_dir = resolve_run_dir(rows_map[rat_key]["source"], ROOT_DIR)
        if src_dir is not None and (src_dir / "best_model.pt").exists():
            dst = ckpt_dir / "transformer_rational_1_97_L128.pt"
            shutil.copy2(src_dir / "best_model.pt", dst)
            ckpt_meta.append({"model": "DigitTransformer", "dataset": "rational_1_97", "period": 96, "window_size": 128, "size_tier": "medium", "radix": 10, "d_model": 96, "n_heads": 4, "n_layers": 3, "param_count": rows_map[rat_key]["params_count"], "test_acc": rows_map[rat_key]["test_acc"], "source_run_dir": str(src_dir), "checkpoint_file": dst.name})
    (ckpt_dir / "checkpoints_metadata.json").write_text(json.dumps(ckpt_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    num_workers = int(config["training"].get("num_workers", 4))
    seed = int(config.get("seed", 42))
    aligned_preds: dict[tuple[str, int], np.ndarray] = {}
    for arch, m_name in (("mlp", "DigitMLP"), ("gru", "DigitGRU"), ("transformer", "DigitTransformer")):
        for L in (32, 64, 128):
            r = rows_map.get(("Block_B_PhaseTransition_H1", "rational_1_97", L, "medium", m_name))
            if not r:
                continue
            rd = resolve_run_dir(r["source"], ROOT_DIR)
            if rd is not None:
                corr, _, _ = predict_and_benchmark(arch, rd, rat97_split, 10, L, "medium", _pick_batch_size("medium", L), num_workers, device, amp_dtype="bfloat16", seed=seed)
                aligned_preds[(m_name, L)] = corr[128 - L :]

    ppm97 = PPMCompressor(radix=10, max_order=8).fit(rat97_split.train_digits)
    aligned_preds[("PPM_o8", 128)] = np.argmax(ppm97.predict_proba(rat97_split.test_digits, window_size=128), axis=1) == rat97_split.test_digits[128:]
    mcnemar_audit = {}
    for label, ka, kb in (("DigitTransformer_L128_vs_L64", ("DigitTransformer", 128), ("DigitTransformer", 64)), ("DigitMLP_L128_vs_L64", ("DigitMLP", 128), ("DigitMLP", 64)), ("DigitTransformer_L128_vs_PPM_o8", ("DigitTransformer", 128), ("PPM_o8", 128)), ("DigitGRU_L32_vs_DigitMLP_L32", ("DigitGRU", 32), ("DigitMLP", 32)), ("DigitGRU_L64_vs_DigitTransformer_L64", ("DigitGRU", 64), ("DigitTransformer", 64))):
        if ka in aligned_preds and kb in aligned_preds:
            mcnemar_audit[label] = compute_mcnemar_exact(aligned_preds[ka], aligned_preds[kb]) | {"model_a": f"{ka[0]}_L{ka[1]}", "model_b": f"{kb[0]}_L{kb[1]}"}

    z_h0 = df_metrics[df_metrics["dataset"] == "true_pi"][["block", "model", "window_size", "size_tier", "params_count", "test_acc", "wilson_ci_low", "wilson_ci_high", "z_score", "z_p_value", "bits_per_symbol"]].to_dict(orient="records")
    stat_audit = {"rational_1_97_phase_transition_mcnemar": mcnemar_audit, "true_pi_h0_z_audit": z_h0}
    (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    mem_df = pd.read_csv(summary_dir / "memorization_curve.csv") if (summary_dir / "memorization_curve.csv").exists() else None
    (summary_dir / "plot_data.json").write_text(json.dumps({"metrics": df_metrics.to_dict(orient="records"), "memorization_summary": {"epochs": int(len(mem_df)) if mem_df is not None else 0, "final_epoch": mem_df.iloc[-1].to_dict() if mem_df is not None and not mem_df.empty else None}, "stat_audit": stat_audit}, ensure_ascii=False, indent=2), encoding="utf-8")

    plots = [*plot_exp4_window_phase_transition(df_metrics, summary_dir), *plot_exp4_capacity_scaling(df_metrics, summary_dir)]
    if mem_df is not None:
        plots.extend(plot_exp4_memorization_scissors(mem_df, summary_dir))
    plots.extend(plot_exp4_compute_frontier(df_metrics, summary_dir))
    return plots


def run_experiment_4(blocks: list[str], reuse_dir: Path | None = None, finalize: bool = True, mem_epochs: int = 200) -> Path:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed = int(config["seed"])
    seed_everything(seed)
    runs_root, split_cfg = ROOT_DIR / config["paths"]["runs_dir"], config["data"]["split"]
    device = resolve_device(str(config["training"].get("device", "cuda")))

    summary_dir = reuse_dir if reuse_dir is not None else create_run_dir(runs_root, "exp4_scaling")
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    rows_map = _load_existing_partial(summary_dir)
    exp2_cache = {}
    exp2_csv = find_latest_csv(runs_root, "exp2_constants_final_") or (ROOT_DIR / "results" / "04_exp2_constants" / "metrics.csv")
    if exp2_csv.exists():
        for _, r in pd.read_csv(exp2_csv).iterrows():
            ds, m = str(r["dataset"]), canonical_model_name(str(r["model"]))
            if ds in ("true_pi", "champernowne") and m in CANONICAL_TO_ARCH:
                exp2_cache[(ds, m)] = r.to_dict()

    pi_art = get_or_create_dataset("true_pi", config=config)
    pi_split = split_sequence_chronologically(pi_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))

    if "A" in blocks:
        for L in (32, 64, 128):
            verify_zero_leakage(pi_split, window_size=L)
            for arch in ("mlp", "gru", "transformer"):
                _run_or_reuse_neural("Block_A_Window_H0", "true_pi", "H0", arch, "medium", L, pi_split, 10, config, exp2_cache, rows_map, summary_dir, device, seed, 10, 1, None)

    rat97_art = get_or_create_dataset("rational_1_97", config=config, length=1_000_050)
    rat97_split = split_sequence_chronologically(rat97_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
    if "B" in blocks:
        _run_block_b(rat97_split, config, exp2_cache, rows_map, summary_dir, device, seed)

    if "C" in blocks:
        champ_art = get_or_create_dataset("champernowne", config=config)
        champ_split = split_sequence_chronologically(champ_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
        for ds_name, ds_split, hyp, ep, pat, sched in (("true_pi", pi_split, "H0", 10, 1, None), ("champernowne", champ_split, "H1", 12, 3, "cosine")):
            for size_tier in ("light", "medium", "heavy"):
                for arch in ("mlp", "cnn", "gru", "transformer"):
                    _run_or_reuse_neural("Block_C_Capacity_Scaling", ds_name, hyp, arch, size_tier, 64, ds_split, 10, config, exp2_cache, rows_map, summary_dir, device, seed, ep, pat, sched)

    if "D" in blocks:
        run_block_d_memorization(pi_art.digits, summary_dir, config, seed=seed, epochs=mem_epochs, batch_size=128)
    if not finalize:
        return summary_dir

    plots = _finalize_exp4(summary_dir, rows_map, rat97_split, config, device)
    print(f"\nЭксперимент 4 завершён: {summary_dir} ({len(plots)} файлов графиков)")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 4: Масштабирование, 1/97 и стресс-тест")
    parser.add_argument("--blocks", type=str, default="A,B,C,D", help="Какие блоки запускать: A,B,C,D")
    parser.add_argument("--reuse-dir", type=str, default=None, help="Папка runs/exp4_scaling_* для продолжения")
    parser.add_argument("--no-finalize", action="store_true", help="Не запускать финальный аудит и графики")
    parser.add_argument("--mem-epochs", type=int, default=200, help="Число эпох стресс-теста Блока D")
    args = parser.parse_args()
    run_experiment_4(blocks=[b.strip().upper() for b in args.blocks.split(",") if b.strip()], reuse_dir=Path(args.reuse_dir) if args.reuse_dir else None, finalize=not args.no_finalize, mem_epochs=args.mem_epochs)


if __name__ == "__main__":
    main()
