# Сервис песочницы: безопасная генерация, stats_tests (Bonferroni ACF), PPM-C и PyTorch-инференс
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines import PPMCompressor
from src.data_generator import apply_csprng_backdoor, generate_csprng_digits, generate_rational_digits
from src.dataset import split_sequence_chronologically
from src.evaluator import compute_binomial_z_score
from src.stats_tests import run_full_statistical_suite
from web.services.model_registry import ModelRegistry
from web.services.safe_math import evaluate_formula_digits


def generate_sequence_sync(
    mode: str,
    length: int,
    radix: int,
    expression: str | None = None,
    numerator: int = 1,
    denominator: int = 97,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if not (100 <= int(length) <= 50_000):
        raise ValueError("length должен быть в диапазоне [100, 50000]")

    if mode == "formula":
        if not expression:
            raise ValueError("Не передано выражение для mode='formula'")
        digits = evaluate_formula_digits(expression, length=length, radix=radix)
        mask = None
        label = f"mpmath: {expression.strip()} (Base-{radix})"
    elif mode == "rational":
        if not (2 <= int(radix) <= 256):
            raise ValueError("radix должен быть в [2, 256]")
        if not (1 <= int(denominator) <= 500_000):
            raise ValueError("denominator должен быть в [1, 500000]")
        digits = generate_rational_digits(int(numerator), int(denominator), int(length), radix=int(radix))
        mask = None
        label = f"{numerator}/{denominator} (Base-{radix})"
    elif mode == "backdoor":
        if int(radix) != 10:
            raise ValueError("Режим backdoor поддерживается для radix=10")
        base = generate_csprng_digits(int(length), radix=10, deterministic=True, seed=42)
        digits, mask = apply_csprng_backdoor(base, lag1=24, lag2=7)
        label = "csprng_backdoor (t-24, t-7)"
    else:
        raise ValueError(f"Неизвестный режим генерации: {mode}")

    return {
        "digits": digits.tolist(),
        "mask": mask.tolist() if mask is not None else None,
        "radix": int(radix),
        "length": int(len(digits)),
        "label": label,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _eval_ppm_sync(digits: np.ndarray, radix: int, max_order: int = 8) -> dict[str, Any] | None:
    if len(digits) < 250:
        return None
    win_size = min(64, max(8, len(digits) // 12))
    split = split_sequence_chronologically(digits, None, 0.8, 0.1, 0.1)
    if len(split.test_digits) <= win_size:
        return None
    ppm = PPMCompressor(radix=radix, max_order=min(max_order, max(1, win_size - 1))).fit(split.train_digits)
    probs = ppm.predict_proba(split.test_digits, window_size=win_size)
    targets = split.test_digits[win_size:]
    probs_safe = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    probs_safe /= probs_safe.sum(axis=1, keepdims=True)
    acc = float(np.mean(np.argmax(probs_safe, axis=1) == targets))
    ce = float(-np.mean(np.log(probs_safe[np.arange(len(targets)), targets])))
    pred_entropy = float(-np.mean(np.sum(probs_safe * np.log(probs_safe), axis=1)))
    return {
        "overall_acc": round(acc, 6),
        "cross_entropy": round(ce, 6),
        "bits_per_symbol": round(ce / math.log(2.0), 6),
        "perplexity": round(math.exp(min(ce, 50.0)), 4),
        "pred_entropy": round(pred_entropy, 6),
        "num_eval_samples": int(len(targets)),
    }


@torch.no_grad()
def eval_loaded_models_sync(
    registry: ModelRegistry,
    digits: np.ndarray,
    mask: np.ndarray | None,
    radix: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    matching = [e for e in registry.models.values() if e.radix == radix]
    if not matching:
        return results

    for entry in matching:
        win = entry.window_size
        if len(digits) <= win + 10:
            continue
        # Берём хвост до 2000 окон для быстрого отклика на RTX 3060 Ti
        n_win = min(len(digits) - win, 2000)
        start = len(digits) - win - n_win
        ctx_np = np.lib.stride_tricks.sliding_window_view(digits[start : start + n_win + win - 1], win)
        tgt_np = digits[start + win : start + win + n_win]
        mask_np = mask[start + win : start + win + n_win] if mask is not None else None

        ctx_t = torch.as_tensor(ctx_np, dtype=torch.long, device=registry.device)
        tgt_t = torch.as_tensor(tgt_np, dtype=torch.long, device=registry.device)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=(registry.device.type == "cuda")):
                logits = entry.model(ctx_t)
            logits_f32 = logits.float()
            log_probs = F.log_softmax(logits_f32, dim=-1)
            probs = log_probs.exp()
            preds = torch.argmax(logits_f32, dim=-1)
            correct = (preds == tgt_t).cpu().numpy()
            ce = float(F.nll_loss(log_probs, tgt_t, reduction="mean").item())
            target_probs = probs.gather(1, tgt_t.unsqueeze(1)).squeeze(1).cpu().numpy()

        acc = float(np.mean(correct))
        tgt_acc = float(np.mean(correct[mask_np == 1])) if (mask_np is not None and np.any(mask_np == 1)) else None
        z_res = compute_binomial_z_score(acc, radix=radix, n_samples=len(correct))
        results.append({
            "model_key": entry.key,
            "arch": entry.arch,
            "dataset_trained": entry.dataset,
            "overall_acc": round(acc, 6),
            "targeted_acc": round(tgt_acc, 6) if tgt_acc is not None else None,
            "cross_entropy": round(ce, 6),
            "bits_per_symbol": round(ce / math.log(2.0), 6),
            "z_score": float(z_res["z_score"]),
            "z_p_value": float(z_res["z_p_value"]),
            "num_eval_samples": int(len(correct)),
            "position_probs": [round(float(x), 4) for x in target_probs[:200].tolist()],
        })
    return results


def analyze_cpu_part_sync(
    name: str,
    digits: np.ndarray,
    radix: int,
    run_ppm: bool,
) -> dict[str, Any]:
    stats = run_full_statistical_suite(name=name, digits=digits, radix=radix, max_lag=64)
    ppm_res = _eval_ppm_sync(digits, radix=radix, max_order=8) if run_ppm else None
    return {"stats": stats, "ppm": ppm_res}


def build_verdict(radix: int, stats: dict[str, Any], ppm: dict[str, Any] | None, neural: list[dict[str, Any]]) -> dict[str, str]:
    shannon = math.log2(radix)
    best_neural = min((m["bits_per_symbol"] for m in neural), default=shannon)
    best_acc = max((m["overall_acc"] for m in neural), default=1.0 / radix)
    max_z = max((abs(m["z_score"]) for m in neural), default=0.0)
    ppm_bits = ppm["bits_per_symbol"] if ppm else shannon

    if (
        not stats["chi2_passed"]
        or not stats["ks_passed"]
        or stats["acf_significant_count"] > 0
        or ppm_bits < shannon * 0.85
        or best_neural < shannon * 0.92
        or max_z >= 4.0
    ):
        return {
            "hypothesis": "H1",
            "summary_ru": (
                f"Обнаружена структура H₁: лучший нейросетевой битрейт {best_neural:.4f} бит/симв "
                f"(макс. точность {best_acc * 100:.2f}%, max|Z|={max_z:.2f}σ), "
                f"статистики χ² p={stats['chi2_p_value']:.4f}, ACF sig={stats['acf_significant_count']}."
            ),
        }
    return {
        "hypothesis": "H0",
        "summary_ru": (
            f"Последовательность согласуется с H₀ (равномерный шум): χ² p={stats['chi2_p_value']:.4f}, "
            f"KS p={stats['ks_p_value']:.4f}, лучший битрейт {min(best_neural, shannon):.4f} при пределе {shannon:.4f} бит."
        ),
    }
