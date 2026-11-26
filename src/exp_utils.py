# Общие хелперы для экспериментов: кэш запусков, оценка скорости и сопоставление моделей
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines import PPMCompressor, _encode_markov_states
from src.dataset import ChronologicalSplit

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
