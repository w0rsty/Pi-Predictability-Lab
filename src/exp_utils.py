# Общие утилиты экспериментов: резолв путей, инференс-бенчмарк и оценка бейзлайнов
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.baselines import PPMCompressor, _encode_markov_states
from src.dataset import ChronologicalSplit, create_dataloaders
from src.models import build_model, count_parameters
from src.trainer import remap_legacy_state_dict

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


def canonical_model_name(name: str) -> str:
    return CANONICAL_MODEL_MAP.get(name, name)


def find_latest_csv(runs_root: Path, prefixes: str | tuple[str, ...], filename: str = "metrics.csv") -> Path | None:
    pfx = (prefixes,) if isinstance(prefixes, str) else prefixes
    if not runs_root.exists():
        return None
    cands = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and any(p.name.startswith(k) for k in pfx)],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in cands:
        if (d / filename).exists():
            return d / filename
    return None


def resolve_run_dir(source_str: str | None, root_dir: Path) -> Path | None:
    if not source_str:
        return None
    p = Path(source_str)
    for cand in (p if p.is_absolute() else root_dir / source_str, root_dir / "runs" / source_str):
        if cand.exists():
            return cand
    return None


def read_sec_per_epoch(run_dir: Path | None, fallback_total: float | None, epochs_ran: int | None) -> float | None:
    if run_dir is not None and (run_dir / "plot_data.json").exists():
        data = json.loads((run_dir / "plot_data.json").read_text(encoding="utf-8"))
        ep_times = [float(h["epoch_time_sec"]) for h in data.get("history", []) if h.get("epoch_time_sec") is not None]
        if ep_times:
            return round(float(np.mean(ep_times)), 3)
    if fallback_total is not None and epochs_ran and epochs_ran > 0:
        return round(float(fallback_total) / float(epochs_ran), 3)
    return None


def eval_markov_preds_and_speed(split: ChronologicalSplit, radix: int, order: int, window_size: int) -> tuple[np.ndarray, float]:
    num_states = radix**order
    tr_states = _encode_markov_states(split.train_digits, order=order, radix=radix)
    tr_targets = split.train_digits[order:].astype(np.int64)
    counts = np.bincount(tr_states * radix + tr_targets, minlength=num_states * radix).reshape(num_states, radix).astype(np.float64)
    probs_table = (counts + 1e-3) / (counts.sum(axis=1, keepdims=True) + 1e-3 * radix)

    test_targets = split.test_digits[window_size:]
    t0 = time.perf_counter()
    test_states = _encode_markov_states(split.test_digits[window_size - order :], order=order, radix=radix)
    preds = np.argmax(probs_table[test_states], axis=1)
    elapsed = max(time.perf_counter() - t0, 1e-5)
    return (preds == test_targets), round(len(test_targets) / elapsed, 1)


def eval_ppm_preds_and_speed(split: ChronologicalSplit, radix: int, window_size: int, max_order: int = 8) -> tuple[np.ndarray, float]:
    ppm = PPMCompressor(radix=radix, max_order=max_order).fit(split.train_digits)
    test_targets = split.test_digits[window_size:]
    t0 = time.perf_counter()
    preds = np.argmax(ppm.predict_proba(split.test_digits, window_size=window_size), axis=1)
    elapsed = max(time.perf_counter() - t0, 1e-5)
    return (preds == test_targets), round(len(test_targets) / elapsed, 1)


def baseline_record(ds_name: str, radix: int, model_name: str, res: dict[str, Any], n_eval: int, source: str, train_time: float | None = None) -> dict[str, Any]:
    return {
        "dataset": ds_name,
        "radix": radix,
        "model": model_name,
        "test_acc": res["overall_acc"],
        "test_ce": res["cross_entropy"],
        "test_ppl": res["perplexity"],
        "pred_entropy": res["pred_entropy"],
        "num_eval_samples": n_eval,
        "train_time_sec": res["train_time_sec"] if train_time is None else train_time,
        "samples_per_sec": round(n_eval / max(float(res["eval_time_sec"]), 1e-4), 1),
        "source": source,
    }


def free_gpu() -> None:
    """Освобождает временные тензоры Python и кэш CUDA между прогонами."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def predict_and_benchmark(
    arch: str,
    run_dir: Path,
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
    size_tier: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: str = "float16",
    seed: int = 42,
    collect_meta: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Загружает best_model.pt, замеряет скорость инференса на тесте и возвращает маску верных ответов."""
    ckpt_path = run_dir / "best_model.pt"
    loaders = create_dataloaders(
        split=split,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        seed=seed,
    )
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    raw_sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd, remapped = remap_legacy_state_dict(raw_sd)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    amp_enabled = device.type == "cuda"
    use_bf16 = amp_dtype == "bfloat16" and amp_enabled and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    test_loader = loaders["test"]

    # Короткий прогрев ядра CUDA до замера throughput
    for ctx, _, _ in test_loader:
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=amp_enabled):
            _ = model(ctx.to(device, non_blocking=True))
        break
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for ctx, tgt, _ in test_loader:
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=amp_enabled):
            logits = model(ctx.to(device, non_blocking=True))
        chunks.append((torch.argmax(logits.float(), dim=-1) == tgt.to(device, non_blocking=True)).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()

    correct_bool = np.concatenate(chunks, axis=0)
    sps = round(float(len(correct_bool) / max(time.perf_counter() - t0, 1e-5)), 1)

    meta: dict[str, Any] = {}
    if collect_meta:
        meta = {
            "arch": arch,
            "radix": int(radix),
            "window_size": int(window_size),
            "size_tier": size_tier,
            "param_count": int(count_parameters(model)),
            "best_epoch": int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0,
            "val_loss": float(ckpt.get("val_loss", float("nan"))) if isinstance(ckpt, dict) else float("nan"),
            "ckpt_path": ckpt_path,
            "was_remapped_legacy": remapped,
        }
        if arch == "transformer":
            meta.update({
                "d_model": int(model.d_model),
                "n_heads": int(model.n_heads),
                "n_layers": int(model.n_layers),
                "token_emb": model.token_emb.weight.detach().cpu().numpy().copy(),
            })
        elif arch == "gru":
            meta.update({
                "hidden_size": int(model.hidden_size),
                "num_layers": int(model.num_layers),
                "emb_dim": int(model.emb_dim),
            })
        elif arch == "mlp":
            meta.update({"emb_dim": int(model.emb_dim)})
        elif arch == "cnn":
            meta.update({"channels": int(model.channels), "head_hidden": int(model.head_hidden)})

    del model, loaders
    free_gpu()
    return correct_bool, sps, meta
