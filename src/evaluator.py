"""Оценка PyTorch-моделей: Overall/Targeted Accuracy, Targeted CE, PPL, энтропия и матрицы переходов."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    radix: int,
    use_amp: bool = True,
    collect_transitions: bool = False,
) -> dict[str, Any]:
    """Прогоняет модель по DataLoader без расчёта градиентов и собирает честные метрики."""
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    total_samples = 0
    total_correct = 0
    total_ce_sum = 0.0
    total_entropy_sum = 0.0

    masked_samples = 0
    masked_correct = 0
    masked_ce_sum = 0.0
    masked_conf_sum = 0.0

    unmasked_samples = 0
    unmasked_conf_sum = 0.0

    if collect_transitions:
        # Накопитель условных вероятностей модели P_model(x_{t+1} | x_t) для рентгена 10x10
        pred_trans_sum = np.zeros((radix, radix), dtype=np.float64)
        pred_trans_counts = np.zeros(radix, dtype=np.int64)

    for contexts, targets, mask_flags in loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask_flags = mask_flags.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(contexts)
        logits_f32 = logits.float()

        log_probs = F.log_softmax(logits_f32, dim=-1)
        probs = log_probs.exp()

        batch_n = targets.size(0)
        total_samples += batch_n

        preds = torch.argmax(logits_f32, dim=-1)
        correct = preds == targets
        total_correct += int(correct.sum().item())

        ce_per_sample = F.nll_loss(log_probs, targets, reduction="none")
        total_ce_sum += float(ce_per_sample.sum().item())

        # Энтропия предсказания H(P) = -sum(p * log(p)) в натах
        entropy_per_sample = -(probs * log_probs).sum(dim=-1)
        total_entropy_sum += float(entropy_per_sample.sum().item())

        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mask_bool = mask_flags == 1
        m_count = int(mask_bool.sum().item())
        if m_count > 0:
            masked_samples += m_count
            masked_correct += int(correct[mask_bool].sum().item())
            masked_ce_sum += float(ce_per_sample[mask_bool].sum().item())
            masked_conf_sum += float(target_probs[mask_bool].sum().item())

        unmask_bool = ~mask_bool
        u_count = int(unmask_bool.sum().item())
        if u_count > 0:
            unmasked_samples += u_count
            unmasked_conf_sum += float(target_probs[unmask_bool].sum().item())

        if collect_transitions:
            last_digits = contexts[:, -1].cpu().numpy()
            probs_np = probs.cpu().numpy()
            np.add.at(pred_trans_sum, last_digits, probs_np)
            np.add.at(pred_trans_counts, last_digits, 1)

    overall_acc = total_correct / max(total_samples, 1)
    cross_entropy = total_ce_sum / max(total_samples, 1)
    perplexity = math.exp(min(cross_entropy, 50.0))
    pred_entropy = total_entropy_sum / max(total_samples, 1)

    targeted_acc = (masked_correct / masked_samples) if masked_samples > 0 else None
    targeted_ce = (masked_ce_sum / masked_samples) if masked_samples > 0 else None
    targeted_conf = (masked_conf_sum / masked_samples) if masked_samples > 0 else None
    unmasked_conf = (unmasked_conf_sum / unmasked_samples) if unmasked_samples > 0 else None

    result: dict[str, Any] = {
        "overall_acc": float(overall_acc),
        "targeted_acc": float(targeted_acc) if targeted_acc is not None else None,
        "cross_entropy": float(cross_entropy),
        "targeted_ce": float(targeted_ce) if targeted_ce is not None else None,
        "perplexity": float(perplexity),
        "pred_entropy": float(pred_entropy),
        "targeted_confidence": float(targeted_conf) if targeted_conf is not None else None,
        "unmasked_confidence": float(unmasked_conf) if unmasked_conf is not None else None,
        "num_eval_samples": int(total_samples),
        "num_masked_samples": int(masked_samples),
    }

    if collect_transitions:
        denom = np.maximum(pred_trans_counts[:, None], 1)
        result["predicted_transition_matrix"] = (pred_trans_sum / denom).tolist()

    return result


@torch.no_grad()
def probe_gru_hidden_dynamics(
    model: nn.Module,
    test_digits: np.ndarray,
    test_mask: np.ndarray,
    window_size: int,
    target_digit: int,
    device: torch.device,
) -> dict[str, Any]:
    """Снимает пошаговую норму ||h_t||_2 и P_t(target_digit) на тестовом окне с триггером."""
    if not hasattr(model, "forward_sequence"):
        raise ValueError("Модель должна поддерживать forward_sequence() для пошаговой диагностики")

    model.eval()
    # Ищем срез длиной window_size + 1, чтобы триггер (mask == 1) был и внутри окна, и ближе к его концу
    chosen_start = 0
    best_score = -1
    limit = min(len(test_digits) - window_size - 1, 5_000)
    for start in range(0, limit):
        win_mask = test_mask[start : start + window_size + 1]
        # Хотим, чтобы в середине/конце окна было несколько срабатываний, включая срабатывание на последнем шаге
        score = int(win_mask[10:window_size].sum()) + (5 if win_mask[window_size] == 1 else 0)
        if score > best_score:
            best_score = score
            chosen_start = start
            if win_mask[window_size] == 1 and win_mask[10:window_size].sum() >= 2:
                break

    ctx_np = test_digits[chosen_start : chosen_start + window_size]
    next_mask_np = test_mask[chosen_start + 1 : chosen_start + window_size + 1]
    ctx_mask_np = test_mask[chosen_start : chosen_start + window_size]

    ctx_t = torch.as_tensor(ctx_np, dtype=torch.long, device=device).unsqueeze(0)
    logits_seq, states_seq = model.forward_sequence(ctx_t)
    probs_seq = F.softmax(logits_seq.squeeze(0), dim=-1).cpu().numpy()
    hidden_norms = torch.linalg.vector_norm(states_seq.squeeze(0), dim=-1).cpu().numpy()

    return {
        "window_start_idx": int(chosen_start),
        "target_digit": int(target_digit),
        "digits": ctx_np.tolist(),
        "mask_at_t": ctx_mask_np.tolist(),
        "mask_next_target": next_mask_np.tolist(),
        "hidden_l2_norm": [round(float(v), 4) for v in hidden_norms],
        "next_target_prob": [round(float(p), 5) for p in probs_seq[:, target_digit]],
    }


@torch.no_grad()
def probe_transformer_attention(
    model: nn.Module,
    test_digits: np.ndarray,
    test_mask: np.ndarray | None,
    window_size: int,
    ds_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """Извлекает по-головные Attention Maps трансформера на характерном примере из Test."""
    model.eval()
    chosen_start = 0
    expected_keys: list[int] = []

    if test_mask is not None and ds_name == "hacked_pi_tier1":
        # Ищем 1-ю семёрку маркера [7, 7, 7]: цель на mask==1, а предыдущая позиция mask==0.
        # Тогда предыдущий маркер сидит ровно на индексах L-50, L-49, L-48 (14, 15, 16 для L=64).
        limit = min(len(test_digits) - window_size - 1, 5_000)
        for start in range(1, limit):
            t_idx = start + window_size
            if test_mask[t_idx] == 1 and test_mask[t_idx - 1] == 0:
                chosen_start = start
                break
        expected_keys = [p for p in (window_size - 50, window_size - 49, window_size - 48) if p >= 0]
    elif test_mask is not None and ds_name == "hacked_pi_tier2":
        # Ищем триггер (x_{t-2}+x_{t-1}+x_t) % 7 == 0 -> x_{t+2} = 9 (ключи L-4, L-3, L-2)
        limit = min(len(test_digits) - window_size - 1, 5_000)
        for start in range(0, limit):
            if test_mask[start + window_size] == 1:
                chosen_start = start
                break
        expected_keys = [p for p in (window_size - 4, window_size - 3, window_size - 2) if p >= 0]

    ctx_np = test_digits[chosen_start : chosen_start + window_size]
    mask_np = test_mask[chosen_start : chosen_start + window_size] if test_mask is not None else np.zeros(window_size, dtype=np.uint8)
    target_digit = int(test_digits[chosen_start + window_size])

    ctx_t = torch.as_tensor(ctx_np, dtype=torch.long, device=device).unsqueeze(0)
    logits, attn_weights = model(ctx_t, return_attn=True)
    probs = F.softmax(logits.float().squeeze(0), dim=-1).cpu().numpy()
    attn_np = attn_weights.squeeze(0).cpu().numpy()  # (n_layers, n_heads, L, L)

    # Находим слой и голову, которые сильнее всего фокусируются на ожидаемых триггерах с последнего шага t=L-1
    n_layers, n_heads, _, _ = attn_np.shape
    focus_layer = n_layers - 1
    focus_head = 0
    best_focus_mass = -1.0
    if expected_keys:
        for l_idx in range(n_layers):
            for h_idx in range(n_heads):
                mass = float(attn_np[l_idx, h_idx, -1, expected_keys].sum())
                if mass > best_focus_mass:
                    best_focus_mass = mass
                    focus_layer = l_idx
                    focus_head = h_idx

    return {
        "dataset": ds_name,
        "window_start_idx": int(chosen_start),
        "target_digit": target_digit,
        "digits": ctx_np.tolist(),
        "mask_at_t": mask_np.tolist(),
        "expected_key_positions": expected_keys,
        "focus_layer": int(focus_layer),
        "focus_head": int(focus_head),
        "focus_mass_on_expected": round(float(best_focus_mass), 4) if expected_keys else None,
        "predicted_probs": [round(float(p), 5) for p in probs],
        "attn_weights": np.round(attn_np, 5).tolist(),
    }


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Точный 95% биномиальный доверительный интервал Уилсона для доли k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)


def compute_binomial_z_score(
    acc: float,
    radix: int,
    n_samples: int,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> dict[str, float]:
    """Считает Z-score отклонения точности от случайного угадывания 1/radix и его ДИ в σ."""
    from scipy import stats

    p0 = 1.0 / float(radix)
    se0 = math.sqrt((p0 * (1.0 - p0)) / max(n_samples, 1))
    z_val = (acc - p0) / se0 if se0 > 0 else 0.0
    z_lo = ((ci_low - p0) / se0) if ci_low is not None and se0 > 0 else z_val
    z_hi = ((ci_high - p0) / se0) if ci_high is not None and se0 > 0 else z_val
    p_two = float(2.0 * stats.norm.sf(abs(z_val)))
    return {
        "random_chance": round(p0, 6),
        "se_random": round(se0, 6),
        "z_score": round(z_val, 4),
        "z_ci_low": round(z_lo, 4),
        "z_ci_high": round(z_hi, 4),
        "z_p_value": round(p_two, 6),
    }


def compute_mcnemar_exact(corr_a: np.ndarray, corr_b: np.ndarray) -> dict[str, Any]:
    """Считает точный парный тест Макнемара (биномиальный + Эдвардс chi2) между двумя моделями."""
    from scipy import stats

    if len(corr_a) != len(corr_b):
        raise ValueError(f"Длины массивов не совпадают: {len(corr_a)} vs {len(corr_b)}")
    n = int(len(corr_a))
    both_correct = int(np.sum(corr_a & corr_b))
    b_wins = int(np.sum(corr_a & ~corr_b))
    c_losses = int(np.sum(~corr_a & corr_b))
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
            "p_value_str": "1.000000",
            "significant_0_05": False,
            "significant_1e_100": False,
        }

    chi2_cc = ((abs(b_wins - c_losses) - 1.0) ** 2) / n_disc
    k_min = min(b_wins, c_losses)
    log_one_side = float(stats.binom.logcdf(k_min, n_disc, 0.5))
    if not math.isfinite(log_one_side):
        log_one_side = float(stats.norm.logsf(math.sqrt(max(chi2_cc, 1e-12))))
    log10_p = min(0.0, (log_one_side + math.log(2.0)) / math.log(10.0))
    p_val = float(stats.binomtest(k_min, n=n_disc, p=0.5, alternative="two-sided").pvalue)

    if log10_p < -300.0:
        p_str = f"< 1e-300 (log10_p ≈ {log10_p:.1f})"
    elif p_val == 0.0:
        # binomtest может уйти в float64 underflow около 1e-308, восстанавливаем мантиссу из log10_p
        exp_i = int(math.floor(log10_p))
        mant = 10.0 ** (log10_p - exp_i)
        p_str = f"{mant:.4f}e{exp_i:+03d}"
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


mcnemar_test = compute_mcnemar_exact


def compute_bitrate_metrics(
    cross_entropy_nats: float,
    radix: int,
    correct: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    """Считает битрейт (bits/symbol), предел Шеннона, нормированный битрейт, PPL и Wilson CI."""
    bits_per_symbol = float(cross_entropy_nats / math.log(2.0))
    shannon_limit_bits = float(math.log2(radix))
    normalized_bits = float(bits_per_symbol / shannon_limit_bits)
    perplexity = float(math.exp(min(cross_entropy_nats, 50.0)))
    out: dict[str, Any] = {
        "cross_entropy_nats": float(cross_entropy_nats),
        "bits_per_symbol": bits_per_symbol,
        "shannon_limit_bits": shannon_limit_bits,
        "normalized_bits": normalized_bits,
        "perplexity": perplexity,
    }
    if correct is not None and total is not None and total > 0:
        acc = float(correct / total)
        ci_lo, ci_hi = wilson_ci(int(correct), int(total))
        z_res = compute_binomial_z_score(acc=acc, radix=radix, n_samples=int(total), ci_low=ci_lo, ci_high=ci_hi)
        out.update(
            {
                "acc": acc,
                "wilson_ci_low": ci_lo,
                "wilson_ci_high": ci_hi,
                **z_res,
            }
        )
    return out
