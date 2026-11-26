"""Фаза 4 — Эксперимент 4: Масштабирование окна L, ёмкости параметров и стресс-тест на запоминание."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baselines import PPMCompressor, fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import ChronologicalSplit, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import compute_binomial_z_score, compute_mcnemar_exact, evaluate_model, wilson_ci
from src.models import build_model, count_parameters
from src.trainer import create_run_dir, resolve_device, train_model
from src.visualizer import (
    plot_exp4_capacity_scaling,
    plot_exp4_compute_frontier,
    plot_exp4_memorization_scissors,
    plot_exp4_window_phase_transition,
)

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
    for cand in (ROOT_DIR / source_str, ROOT_DIR / "runs" / source_str):
        if cand.exists():
            return cand
    return None


def _pick_batch_size(size_tier: str, window_size: int, arch: str = "") -> int:
    """Подбирает быстрый и безопасный batch_size под реальный запас свободной VRAM на RTX 3060 Ti."""
    if window_size >= 128:
        return 512 if size_tier != "heavy" else 256
    if size_tier == "heavy":
        return 256 if arch == "transformer" else 512
    return 1024 if size_tier == "medium" else 2048


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _measure_throughput_and_preds(
    arch: str,
    run_dir: Path,
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
    size_tier: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Загружает best_model.pt, замеряет GPU throughput и возвращает вектор верных ответов на Test."""
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
    sd = {
        k.replace(".attn.in_proj_weight", ".qkv_proj.weight")
        .replace(".attn.in_proj_bias", ".qkv_proj.bias")
        .replace(".attn.out_proj.", ".out_proj."): v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(sd)
    model = model.to(device).eval()

    amp_enabled = device.type == "cuda"
    test_loader = loaders["test"]
    for contexts, _, _ in test_loader:
        contexts = contexts.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            _ = model(contexts)
        break

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for contexts, targets, _ in test_loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(contexts)
        preds = torch.argmax(logits.float(), dim=-1)
        chunks.append((preds == targets).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-5)

    corr = np.concatenate(chunks, axis=0)
    sps = round(float(len(corr) / elapsed), 1)
    del model
    _free_gpu()
    return sps, corr


def _build_metric_row(
    block: str,
    dataset: str,
    hypothesis_group: str,
    radix: int,
    window_size: int,
    size_tier: str,
    model_name: str,
    params_count: int | None,
    test_acc: float,
    test_ce: float,
    test_ppl: float,
    pred_entropy: float | None,
    num_eval_samples: int,
    sec_per_epoch: float | None,
    samples_per_sec: float | None,
    epochs_ran: int | None,
    best_epoch: int | None,
    train_time_sec: float | None,
    source: str,
) -> dict[str, Any]:
    k_corr = int(round(test_acc * num_eval_samples))
    ci_lo, ci_hi = wilson_ci(k_corr, num_eval_samples)
    z_info = compute_binomial_z_score(test_acc, radix=radix, n_samples=num_eval_samples, ci_low=ci_lo, ci_high=ci_hi)
    shannon_bits = math.log2(radix)
    bits = test_ce / math.log(2.0)
    return {
        "block": block,
        "dataset": dataset,
        "hypothesis_group": hypothesis_group,
        "radix": radix,
        "window_size": window_size,
        "size_tier": size_tier,
        "model": model_name,
        "params_count": params_count,
        "test_acc": round(test_acc, 6),
        "wilson_ci_low": ci_lo,
        "wilson_ci_high": ci_hi,
        "z_score": z_info["z_score"],
        "z_p_value": z_info["z_p_value"],
        "test_ce_nats": round(test_ce, 6),
        "test_ppl": round(test_ppl, 4),
        "bits_per_symbol": round(bits, 6),
        "shannon_limit_bits": round(shannon_bits, 6),
        "normalized_bits": round(bits / shannon_bits, 6),
        "sec_per_epoch": sec_per_epoch,
        "samples_per_sec": samples_per_sec,
        "pred_entropy": round(pred_entropy, 6) if pred_entropy is not None else None,
        "num_eval_samples": num_eval_samples,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "train_time_sec": train_time_sec,
        "source": source,
    }


def _save_partial(rows_map: dict[tuple[str, str, int, str, str], dict[str, Any]], summary_dir: Path) -> None:
    df = pd.DataFrame(list(rows_map.values()))
    df.to_csv(summary_dir / "partial_metrics.csv", index=False)


def _load_existing_partial(summary_dir: Path) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    rows_map: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    part_p = summary_dir / "partial_metrics.csv"
    if not part_p.exists():
        return rows_map
    df = pd.read_csv(part_p)
    for _, row in df.iterrows():
        key = (
            str(row["block"]),
            str(row["dataset"]),
            int(row["window_size"]),
            str(row["size_tier"]),
            str(row["model"]),
        )
        rows_map[key] = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    return rows_map


def _run_or_reuse_neural(
    block: str,
    ds_name: str,
    hyp_group: str,
    arch: str,
    size_tier: str,
    window_size: int,
    split: ChronologicalSplit,
    radix: int,
    config: dict[str, Any],
    exp2_cache: dict[tuple[str, str], dict[str, Any]],
    rows_map: dict[tuple[str, str, int, str, str], dict[str, Any]],
    summary_dir: Path,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    scheduler_type: str | None,
) -> dict[str, Any]:
    """Обучает модель или подхватывает готовый результат (partial_metrics / Exp 2 cache)."""
    m_name = _canonical_model(arch.upper())
    key = (block, ds_name, int(window_size), size_tier, m_name)
    if key in rows_map:
        print(f"[{block} | {ds_name} | L={window_size} | {size_tier} | {m_name}] Из partial_metrics.csv")
        return rows_map[key]

    # Подхват L=64, medium из Эксперимента 2 для true_pi и champernowne
    if window_size == 64 and size_tier == "medium" and (ds_name, m_name) in exp2_cache:
        c = exp2_cache[(ds_name, m_name)]
        row = _build_metric_row(
            block=block,
            dataset=ds_name,
            hypothesis_group=hyp_group,
            radix=radix,
            window_size=64,
            size_tier="medium",
            model_name=m_name,
            params_count=int(c["params_count"]) if c.get("params_count") is not None else None,
            test_acc=float(c["test_acc"]),
            test_ce=float(c["test_ce_nats"]),
            test_ppl=float(c["test_ppl"]),
            pred_entropy=float(c["pred_entropy"]) if c.get("pred_entropy") is not None else None,
            num_eval_samples=int(c["num_eval_samples"]),
            sec_per_epoch=float(c["sec_per_epoch"]) if c.get("sec_per_epoch") is not None else None,
            samples_per_sec=float(c["samples_per_sec"]) if c.get("samples_per_sec") is not None else None,
            epochs_ran=int(c["epochs_ran"]) if c.get("epochs_ran") is not None else None,
            best_epoch=int(c["best_epoch"]) if c.get("best_epoch") is not None else None,
            train_time_sec=float(c["train_time_sec"]) if c.get("train_time_sec") is not None else None,
            source=str(c["source"]),
        )
        rows_map[key] = row
        _save_partial(rows_map, summary_dir)
        print(f"[{block} | {ds_name} | L=64 | medium | {m_name}] Подхвачено из Exp 2 ({c['source']})")
        return row

    seed_everything(seed)
    bs = _pick_batch_size(size_tier, window_size, arch=arch)
    num_workers = int(config["training"].get("num_workers", 4))
    loaders = create_dataloaders(
        split=split,
        window_size=window_size,
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    p_cnt = count_parameters(model)
    print(f"\n=== [Exp 4 | {block} | {ds_name} | {m_name} | {size_tier} ({p_cnt:,}) | L={window_size} | bs={bs}] ===")

    res = train_model(
        model=model,
        loaders=loaders,
        config=config,
        exp_name=f"exp4_{block.lower()}_{arch}_{ds_name}_L{window_size}_{size_tier}",
        radix=radix,
        epochs=max_epochs,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=1e-4,
        scheduler_type=scheduler_type,
        amp_dtype="bfloat16",
        early_stopping_patience=patience,
        enable_early_stopping=True,
    )
    del model, loaders
    _free_gpu()

    s = res["summary"]
    tm = res["test_metrics"]
    hist = res["history"]
    sec_ep = round(float(np.mean([h["epoch_time_sec"] for h in hist])), 3) if hist else None

    # Точный замер GPU throughput для Pareto-фронта
    run_dir = Path(s["run_dir"])
    sps, _ = _measure_throughput_and_preds(
        arch=arch,
        run_dir=run_dir,
        split=split,
        radix=radix,
        window_size=window_size,
        size_tier=size_tier,
        batch_size=bs,
        num_workers=num_workers,
        device=device,
    )

    row = _build_metric_row(
        block=block,
        dataset=ds_name,
        hypothesis_group=hyp_group,
        radix=radix,
        window_size=window_size,
        size_tier=size_tier,
        model_name=m_name,
        params_count=int(s["param_count"]),
        test_acc=float(tm["overall_acc"]),
        test_ce=float(tm["cross_entropy"]),
        test_ppl=float(tm["perplexity"]),
        pred_entropy=float(tm["pred_entropy"]),
        num_eval_samples=int(tm["num_eval_samples"]),
        sec_per_epoch=sec_ep,
        samples_per_sec=sps,
        epochs_ran=int(s["epochs_ran"]),
        best_epoch=int(s["best_epoch"]),
        train_time_sec=float(s["total_train_time_sec"]),
        source=s["run_dir"],
    )
    rows_map[key] = row
    _save_partial(rows_map, summary_dir)
    return row


def run_block_d_memorization(
    pi_digits: np.ndarray,
    summary_dir: Path,
    config: dict[str, Any],
    seed: int = 42,
    epochs: int = 200,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Блок D: Стресс-тест на запоминание 10,000 цифр true_pi (200 эпох, heavy Transformer, без регуляризации)."""
    curve_path = summary_dir / "memorization_curve.csv"
    if curve_path.exists():
        existing = pd.read_csv(curve_path)
        if len(existing) >= epochs:
            print(f"[Block D] Найдена готовая кривая запоминания ({len(existing)} эпох) -> пропускаем")
            return existing

    seed_everything(seed)
    device = resolve_device(str(config["training"].get("device", "cuda")))
    window_size = 64

    slice_10k = pi_digits[:10_000]
    split = split_sequence_chronologically(slice_10k, mask=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    verify_zero_leakage(split, window_size=window_size)

    loaders = create_dataloaders(split=split, window_size=window_size, batch_size=batch_size, num_workers=2, pin_memory=True)
    model = build_model("transformer", radix=10, window_size=window_size, size_tier="heavy", dropout=0.0).to(device)
    p_cnt = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    use_amp = device.type == "cuda"
    ln2 = math.log(2.0)

    print(
        f"\n=== [Exp 4 | Block D: Memorization Stress-Test] DigitTransformer heavy ({p_cnt:,}), "
        f"N=10,000 (Train={len(split.train_digits)-64:,}, Test={len(split.test_digits)-64:,}), {epochs} эпох ==="
    )
    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_sum = 0.0
        tr_corr = 0
        tr_n = 0

        for contexts, targets, _ in loaders["train"]:
            contexts = contexts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(contexts)
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                bn = targets.size(0)
                tr_n += bn
                tr_loss_sum += float(loss.item()) * bn
                tr_corr += int((torch.argmax(logits.float(), dim=-1) == targets).sum().item())

        tr_ce = tr_loss_sum / max(tr_n, 1)
        tr_acc = tr_corr / max(tr_n, 1)

        val_m = evaluate_model(model, loaders["val"], device=device, radix=10, use_amp=use_amp, collect_transitions=False)
        test_m = evaluate_model(model, loaders["test"], device=device, radix=10, use_amp=use_amp, collect_transitions=False)

        history.append(
            {
                "epoch": ep,
                "train_loss": round(tr_ce, 6),
                "train_acc": round(tr_acc, 6),
                "train_bits": round(tr_ce / ln2, 6),
                "val_loss": round(float(val_m["cross_entropy"]), 6),
                "val_acc": round(float(val_m["overall_acc"]), 6),
                "val_bits": round(float(val_m["cross_entropy"]) / ln2, 6),
                "test_loss": round(float(test_m["cross_entropy"]), 6),
                "test_acc": round(float(test_m["overall_acc"]), 6),
                "test_bits": round(float(test_m["cross_entropy"]) / ln2, 6),
                "epoch_time_sec": round(time.perf_counter() - ep_t0, 3),
            }
        )
        if ep in (1, 10, 25, 50, 100, 150, epochs) or ep % 40 == 0:
            last = history[-1]
            print(
                f"  Эпоха {ep:3d}/{epochs} ({time.perf_counter()-t_start:.1f}с) | "
                f"Train: {last['train_acc']*100:6.2f}% ({last['train_bits']:.3f} bit) | "
                f"Test: {last['test_acc']*100:5.2f}% ({last['test_bits']:.3f} bit)"
            )

    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "param_count": p_cnt,
            "train_acc": history[-1]["train_acc"],
            "test_acc": history[-1]["test_acc"],
        },
        ckpt_dir / "transformer_memorization_10k_heavy.pt",
    )
    del model, loaders, optimizer
    _free_gpu()

    mem_df = pd.DataFrame(history)
    mem_df.to_csv(curve_path, index=False)
    return mem_df


def run_experiment_4(
    blocks: list[str],
    reuse_dir: Path | None = None,
    finalize: bool = True,
    mem_epochs: int = 200,
) -> Path:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed = int(config["seed"])
    seed_everything(seed)

    runs_root = ROOT_DIR / config["paths"]["runs_dir"]
    device = resolve_device(str(config["training"].get("device", "cuda")))
    split_cfg = config["data"]["split"]

    if reuse_dir is not None:
        summary_dir = reuse_dir
        summary_dir.mkdir(parents=True, exist_ok=True)
    else:
        summary_dir = create_run_dir(runs_root, "exp4_scaling")
    (summary_dir / "config_snapshot.yaml").write_text(
        (ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    rows_map = _load_existing_partial(summary_dir)

    # Загружаем кэш Эксперимента 2 (L=64, medium для true_pi и champernowne)
    exp2_cache: dict[tuple[str, str], dict[str, Any]] = {}
    exp2_csv = _find_latest_exp2_final(runs_root)
    if exp2_csv is not None:
        df_e2 = pd.read_csv(exp2_csv)
        for _, r in df_e2.iterrows():
            ds = str(r["dataset"])
            m = _canonical_model(str(r["model"]))
            if ds in ("true_pi", "champernowne") and m in CANONICAL_TO_ARCH:
                exp2_cache[(ds, m)] = r.to_dict()

    # Датасеты
    pi_art = get_or_create_dataset("true_pi", config=config)
    pi_split = split_sequence_chronologically(
        pi_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )

    # Блок A: Масштабирование окна L in {32, 64, 128} на true_pi (H0)
    if "A" in blocks:
        print("\n================ БЛОК A: Масштабирование окна L={32,64,128} на true_pi (H0) ================")
        for L in (32, 64, 128):
            verify_zero_leakage(pi_split, window_size=L)
            for arch in ("mlp", "gru", "transformer"):
                _run_or_reuse_neural(
                    block="Block_A_Window_H0",
                    ds_name="true_pi",
                    hyp_group="H0",
                    arch=arch,
                    size_tier="medium",
                    window_size=L,
                    split=pi_split,
                    radix=10,
                    config=config,
                    exp2_cache=exp2_cache,
                    rows_map=rows_map,
                    summary_dir=summary_dir,
                    device=device,
                    seed=seed,
                    max_epochs=10,
                    patience=1,  # Шум выходит на плато на 1-й эпохе
                    scheduler_type=None,
                )

    # Блок B: Фазовый переход окна L in {32, 64, 128} на rational_1_97 (P=96, H1)
    if "B" in blocks:
        print("\n================ БЛОК B: Фазовый переход окна L={32,64,128} на 1/97 (P=96, H1) ================")
        rat97_art = get_or_create_dataset("rational_1_97", config=config, length=1_000_050)
        rat97_split = split_sequence_chronologically(
            rat97_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
        )
        # Контрольный табличный бейзлайн PPM_o8 (L=64 и L=128)
        for L in (64, 128):
            ppm_key = ("Block_B_PhaseTransition_H1", "rational_1_97", L, "tabular", "PPM_o8")
            if ppm_key not in rows_map:
                verify_zero_leakage(rat97_split, window_size=L)
                ppm_res, _ = fit_and_evaluate_ppm(rat97_split, radix=10, window_size=L, max_order=8)
                sps_ppm = round(ppm_res["num_eval_samples"] / max(float(ppm_res["eval_time_sec"]), 1e-4), 1)
                rows_map[ppm_key] = _build_metric_row(
                    block="Block_B_PhaseTransition_H1",
                    dataset="rational_1_97",
                    hypothesis_group="H1",
                    radix=10,
                    window_size=L,
                    size_tier="tabular",
                    model_name="PPM_o8",
                    params_count=None,
                    test_acc=ppm_res["overall_acc"],
                    test_ce=ppm_res["cross_entropy"],
                    test_ppl=ppm_res["perplexity"],
                    pred_entropy=ppm_res["pred_entropy"],
                    num_eval_samples=ppm_res["num_eval_samples"],
                    sec_per_epoch=None,
                    samples_per_sec=sps_ppm,
                    epochs_ran=None,
                    best_epoch=None,
                    train_time_sec=ppm_res["train_time_sec"],
                    source="exp4_ppm_o8",
                )
                _save_partial(rows_map, summary_dir)

        for L in (32, 64, 128):
            verify_zero_leakage(rat97_split, window_size=L)
            for arch in ("mlp", "gru", "transformer"):
                _run_or_reuse_neural(
                    block="Block_B_PhaseTransition_H1",
                    ds_name="rational_1_97",
                    hyp_group="H1",
                    arch=arch,
                    size_tier="medium",
                    window_size=L,
                    split=rat97_split,
                    radix=10,
                    config=config,
                    exp2_cache=exp2_cache,
                    rows_map=rows_map,
                    summary_dir=summary_dir,
                    device=device,
                    seed=seed,
                    max_epochs=12,
                    patience=3,
                    scheduler_type="cosine",
                )

    # Блок C: Лестница ёмкости light (~50K) -> medium (~250K) -> heavy (~1.5M) при L=64
    if "C" in blocks:
        print("\n================ БЛОК C: Лестница ёмкости 50K -> 250K -> 1.5M (true_pi и champernowne) ================")
        champ_art = get_or_create_dataset("champernowne", config=config)
        champ_split = split_sequence_chronologically(
            champ_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
        )
        verify_zero_leakage(pi_split, window_size=64)
        verify_zero_leakage(champ_split, window_size=64)

        for ds_name, ds_split, hyp_group, max_ep, pat, sched in (
            ("true_pi", pi_split, "H0", 10, 1, None),
            ("champernowne", champ_split, "H1", 12, 3, "cosine"),
        ):
            for size_tier in ("light", "medium", "heavy"):
                for arch in ("mlp", "cnn", "gru", "transformer"):
                    _run_or_reuse_neural(
                        block="Block_C_Capacity_Scaling",
                        ds_name=ds_name,
                        hyp_group=hyp_group,
                        arch=arch,
                        size_tier=size_tier,
                        window_size=64,
                        split=ds_split,
                        radix=10,
                        config=config,
                        exp2_cache=exp2_cache,
                        rows_map=rows_map,
                        summary_dir=summary_dir,
                        device=device,
                        seed=seed,
                        max_epochs=max_ep,
                        patience=pat,
                        scheduler_type=sched,
                    )

    # Блок D: Стресс-тест на запоминание 10K true_pi
    if "D" in blocks:
        run_block_d_memorization(
            pi_digits=pi_art.digits,
            summary_dir=summary_dir,
            config=config,
            seed=seed,
            epochs=mem_epochs,
            batch_size=128,
        )

    if not finalize:
        print(f"\nПромежуточный этап ({','.join(blocks)}) сохранён в: {summary_dir}")
        return summary_dir

    # Финализация: итоговый metrics.csv, чекпойнт 1/97 L=128, McNemar на выровненном срезе [128:] и 4 графика
    print("\n================ Финализация: McNemar-аудит, чекпойнт 1/97 (L=128) и 4 графика ================")
    df_metrics = pd.DataFrame(list(rows_map.values()))
    df_metrics.to_csv(summary_dir / "metrics.csv", index=False)

    # Копируем чекпойнт DigitTransformer (medium, L=128) на rational_1_97 для Exp 5
    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rat_tr_key = ("Block_B_PhaseTransition_H1", "rational_1_97", 128, "medium", "DigitTransformer")
    ckpt_meta: list[dict[str, Any]] = []
    if rat_tr_key in rows_map:
        src_dir = _resolve_run_dir(rows_map[rat_tr_key]["source"])
        if src_dir and (src_dir / "best_model.pt").exists():
            dst_ckpt = ckpt_dir / "transformer_rational_1_97_L128.pt"
            shutil.copy2(src_dir / "best_model.pt", dst_ckpt)
            ckpt_meta.append(
                {
                    "model": "DigitTransformer",
                    "dataset": "rational_1_97",
                    "period": 96,
                    "window_size": 128,
                    "size_tier": "medium",
                    "radix": 10,
                    "d_model": 96,
                    "n_heads": 4,
                    "n_layers": 3,
                    "param_count": rows_map[rat_tr_key]["params_count"],
                    "test_acc": rows_map[rat_tr_key]["test_acc"],
                    "source_run_dir": str(src_dir),
                    "checkpoint_file": dst_ckpt.name,
                }
            )
    (ckpt_dir / "checkpoints_metadata.json").write_text(json.dumps(ckpt_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Парный тест Макнемара на фазовом переходе rational_1_97 (выровненный срез последних N_test - 128 позиций!)
    rat97_art = get_or_create_dataset("rational_1_97", config=config, length=1_000_050)
    rat97_split = split_sequence_chronologically(
        rat97_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )
    num_workers = int(config["training"].get("num_workers", 4))
    aligned_preds: dict[tuple[str, int], np.ndarray] = {}
    for arch, m_name in (("mlp", "DigitMLP"), ("gru", "DigitGRU"), ("transformer", "DigitTransformer")):
        for L in (32, 64, 128):
            r = rows_map.get(("Block_B_PhaseTransition_H1", "rational_1_97", L, "medium", m_name))
            if r is None:
                continue
            rd = _resolve_run_dir(r["source"])
            if rd is None:
                continue
            _, corr = _measure_throughput_and_preds(
                arch=arch,
                run_dir=rd,
                split=rat97_split,
                radix=10,
                window_size=L,
                size_tier="medium",
                batch_size=_pick_batch_size("medium", L),
                num_workers=num_workers,
                device=device,
            )
            # Выравниваем хвост к позициям >= 128 на test_digits
            offset = 128 - L
            aligned_preds[(m_name, L)] = corr[offset:]

    # PPM_o8 предсказания на позициях >= 128
    ppm97 = PPMCompressor(radix=10, max_order=8).fit(rat97_split.train_digits)
    ppm97_preds = np.argmax(ppm97.predict_proba(rat97_split.test_digits, window_size=128), axis=1)
    aligned_preds[("PPM_o8", 128)] = ppm97_preds == rat97_split.test_digits[128:]

    mcnemar_audit: dict[str, Any] = {}
    phase_pairs = [
        ("DigitTransformer_L128_vs_L64", ("DigitTransformer", 128), ("DigitTransformer", 64)),
        ("DigitMLP_L128_vs_L64", ("DigitMLP", 128), ("DigitMLP", 64)),
        ("DigitTransformer_L128_vs_PPM_o8", ("DigitTransformer", 128), ("PPM_o8", 128)),
        ("DigitGRU_L32_vs_DigitMLP_L32", ("DigitGRU", 32), ("DigitMLP", 32)),
        ("DigitGRU_L64_vs_DigitTransformer_L64", ("DigitGRU", 64), ("DigitTransformer", 64)),
    ]
    for label, ka, kb in phase_pairs:
        if ka in aligned_preds and kb in aligned_preds:
            res_mc = compute_mcnemar_exact(aligned_preds[ka], aligned_preds[kb])
            res_mc["model_a"] = f"{ka[0]}_L{ka[1]}"
            res_mc["model_b"] = f"{kb[0]}_L{kb[1]}"
            mcnemar_audit[label] = res_mc

    z_h0_rows = df_metrics[df_metrics["dataset"] == "true_pi"][
        ["block", "model", "window_size", "size_tier", "params_count", "test_acc", "wilson_ci_low", "wilson_ci_high", "z_score", "z_p_value", "bits_per_symbol"]
    ].to_dict(orient="records")

    stat_audit = {
        "rational_1_97_phase_transition_mcnemar": mcnemar_audit,
        "true_pi_h0_z_audit": z_h0_rows,
    }
    (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    mem_df = pd.read_csv(summary_dir / "memorization_curve.csv") if (summary_dir / "memorization_curve.csv").exists() else None
    (summary_dir / "plot_data.json").write_text(
        json.dumps(
            {
                "metrics": df_metrics.to_dict(orient="records"),
                "memorization_summary": {
                    "epochs": int(len(mem_df)) if mem_df is not None else 0,
                    "final_epoch": mem_df.iloc[-1].to_dict() if mem_df is not None and not mem_df.empty else None,
                },
                "stat_audit": stat_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    saved_plots: list[Path] = []
    saved_plots.extend(plot_exp4_window_phase_transition(df_metrics, summary_dir))
    saved_plots.extend(plot_exp4_capacity_scaling(df_metrics, summary_dir))
    if mem_df is not None:
        saved_plots.extend(plot_exp4_memorization_scissors(mem_df, summary_dir))
    saved_plots.extend(plot_exp4_compute_frontier(df_metrics, summary_dir))

    print(f"\nЭксперимент 4 полностью завершён! Итоговая папка: {summary_dir}")
    print("\nГрафики:")
    for p in saved_plots:
        print(f"- {p}")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 4: Масштабирование, фазовый переход 1/97 и стресс-тест")
    parser.add_argument("--blocks", type=str, default="A,B,C,D", help="Какие блоки запускать: A,B,C,D")
    parser.add_argument("--reuse-dir", type=str, default=None, help="Путь к папке runs/exp4_scaling_* для продолжения")
    parser.add_argument("--no-finalize", action="store_true", help="Не запускать финальный аудит и графики")
    parser.add_argument("--mem-epochs", type=int, default=200, help="Число эпох стресс-теста Блока D")
    args = parser.parse_args()

    blocks = [b.strip().upper() for b in args.blocks.split(",") if b.strip()]
    reuse_p = Path(args.reuse_dir) if args.reuse_dir else None
    run_experiment_4(
        blocks=blocks,
        reuse_dir=reuse_p,
        finalize=not args.no_finalize,
        mem_epochs=args.mem_epochs,
    )


if __name__ == "__main__":
    main()
