# Сервис Арены «Человек против нейросети»: реальный инференс Transformer/GRU/PPM-C на тестовых окнах
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines import PPMCompressor
from src.data_generator import get_or_create_dataset, load_config
from src.dataset import split_sequence_chronologically
from src.models import DigitTransformer
from web.services.model_registry import ModelRegistry, ROOT_DIR

ARENA_DATASETS = ("true_pi", "csprng_backdoor", "champernowne", "e_cf", "rational_1_97")
BLIND_AUDIT_POOL = ("true_pi", "rational_1_97", "champernowne", "csprng_backdoor")
DISPLAY_CONTEXT_LEN = 64


def _round_probs(probs: np.ndarray, decimals: int = 4) -> list[float]:
    arr = np.clip(probs.astype(np.float64), 0.0, 1.0)
    total = float(arr.sum())
    if total > 0:
        arr /= total
    return [round(float(x), decimals) for x in arr.tolist()]


def _extract_top_lags(
    attn_sample: torch.Tensor,
    dataset: str,
    win_len: int,
    display_len: int = DISPLAY_CONTEXT_LEN,
) -> list[dict[str, float | int]]:
    """Извлекает топ-3 лага внимания (lag in 1..display_len) с реальными весами головы."""
    # attn_sample: (n_layers, n_heads, L, L)
    n_layers, n_heads, _, _ = attn_sample.shape
    if dataset == "csprng_backdoor" and n_heads > 1:
        focus_l, focus_h = 0, 1
    elif dataset == "e_cf" and n_heads > 2:
        focus_l, focus_h = 0, 2
    else:
        # Выбираем голову с наибольшей пиковой концентрацией на последнем запросе
        last_q = attn_sample[:, :, -1, :]  # (L_layers, H, win_len)
        peak_vals = last_q.max(dim=-1).values
        flat_best = int(torch.argmax(peak_vals).item())
        focus_l, focus_h = divmod(flat_best, n_heads)

    vec = attn_sample[focus_l, focus_h, -1, :].float().cpu().numpy()
    # Берём только лаги, видимые в отображаемом 64-токенном контексте
    tail_offset = max(0, win_len - display_len)
    visible_vec = vec[tail_offset:]
    eff_len = len(visible_vec)
    top_indices = np.argsort(visible_vec)[::-1][:3]
    out: list[dict[str, float | int]] = []
    for pos in top_indices:
        lag = int(eff_len - int(pos))
        w = float(visible_vec[int(pos)])
        if w >= 0.01:
            out.append({"lag": lag, "weight": round(w, 4)})
    return out


def _get_or_fit_ppm(
    registry: ModelRegistry,
    dataset: str,
    radix: int,
    train_digits: np.ndarray,
) -> PPMCompressor:
    cached = registry.ppm_cache.get(dataset)
    if cached is not None:
        return cached
    ppm = PPMCompressor(radix=radix, max_order=8).fit(train_digits)
    registry.ppm_cache[dataset] = ppm
    return ppm


def _pick_test_positions(
    dataset: str,
    test_digits: np.ndarray,
    test_mask: np.ndarray | None,
    min_ctx: int,
    n_rounds: int,
    rng: np.random.Generator,
    min_gap: int = 1,
) -> list[int]:
    n_test = len(test_digits)
    if n_test <= min_ctx + 10:
        raise ValueError(f"Тестовый сплит датасета '{dataset}' слишком мал ({n_test})")

    if dataset == "csprng_backdoor" and test_mask is not None and np.any(test_mask[min_ctx:] == 1):
        trig_target = max(1, (n_rounds + 1) // 2) if n_rounds != 30 else 16
        bg_target = n_rounds - trig_target

        valid_mask_idx = np.where(test_mask[min_ctx:] == 1)[0] + min_ctx
        valid_bg_idx = np.where(test_mask[min_ctx:] == 0)[0] + min_ctx

        chosen_trigs = rng.choice(
            valid_mask_idx, size=min(trig_target, len(valid_mask_idx)), replace=False
        ).tolist()
        chosen_bg = rng.choice(
            valid_bg_idx, size=min(bg_target, len(valid_bg_idx)), replace=False
        ).tolist()
        combined = [int(i) for i in (chosen_trigs + chosen_bg)]
        rng.shuffle(combined)
        return combined[:n_rounds]

    # Для остальных датасетов набираем раунды без пересечения целевых позиций (и с шагом при 30 раундах)
    candidates = np.arange(min_ctx, n_test, max(1, min_gap), dtype=np.int64)
    if len(candidates) < n_rounds:
        candidates = np.arange(min_ctx, n_test, 1, dtype=np.int64)
    picked = rng.choice(candidates, size=min(n_rounds, len(candidates)), replace=False)
    return [int(x) for x in picked.tolist()]


@torch.no_grad()
def build_arena_rounds_sync(
    registry: ModelRegistry,
    dataset: str,
    n_rounds: int = 15,
    seed: int | None = None,
    min_gap: int = 1,
) -> dict[str, Any]:
    """Формирует набор реальных тестовых раундов с прогнозами Transformer, GRU и PPM-C."""
    if dataset not in ARENA_DATASETS:
        raise ValueError(f"Неизвестный датасет для Арены: '{dataset}'")

    tf_key = f"transformer_{dataset}"
    if tf_key not in registry.models:
        raise RuntimeError(
            f"Чекпойнт '{tf_key}' не найден в runs/. Подгрузите чекпойнты или используйте статический банк."
        )
    tf_entry = registry.models[tf_key]
    tf_model = tf_entry.model
    assert isinstance(tf_model, DigitTransformer)
    gru_entry = registry.models.get(f"gru_{dataset}")

    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    req_len = 1_000_050 if dataset == "rational_1_97" else None
    art = get_or_create_dataset(dataset, config=config, length=req_len)
    split_cfg = config["data"]["split"]
    split = split_sequence_chronologically(
        art.digits,
        art.mask,
        float(split_cfg["train"]),
        float(split_cfg["val"]),
        float(split_cfg["test"]),
    )

    min_ctx = max(DISPLAY_CONTEXT_LEN, tf_entry.window_size, gru_entry.window_size if gru_entry else 0)
    rng = np.random.default_rng(seed)
    positions = _pick_test_positions(
        dataset=dataset,
        test_digits=split.test_digits,
        test_mask=split.test_mask,
        min_ctx=min_ctx,
        n_rounds=n_rounds,
        rng=rng,
        min_gap=min_gap,
    )

    # Инференс DigitTransformer
    tf_win = tf_entry.window_size
    tf_ctx_np = np.stack([split.test_digits[pos - tf_win : pos] for pos in positions], axis=0)
    tf_tgt_np = np.asarray([int(split.test_digits[pos]) for pos in positions], dtype=np.int64)
    tf_ctx_t = torch.as_tensor(tf_ctx_np, dtype=torch.long, device=registry.device)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(registry.device.type == "cuda")):
            tf_logits, tf_attn = tf_model.forward_with_head_ablation(
                tf_ctx_t, ablate_heads=None, return_attn=True
            )
        tf_probs_t = F.softmax(tf_logits.float(), dim=-1).cpu().numpy()

    # Инференс DigitGRU (если доступен чекпойнт)
    gru_probs_t: np.ndarray | None = None
    if gru_entry is not None:
        g_win = gru_entry.window_size
        g_ctx_np = np.stack([split.test_digits[pos - g_win : pos] for pos in positions], axis=0)
        g_ctx_t = torch.as_tensor(g_ctx_np, dtype=torch.long, device=registry.device)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=(registry.device.type == "cuda")):
                g_logits = gru_entry.model(g_ctx_t)
            gru_probs_t = F.softmax(g_logits.float(), dim=-1).cpu().numpy()

    # Инференс PPM-C (order=8) на обученном по Train компрессоре
    ppm = _get_or_fit_ppm(registry, dataset, int(art.radix), split.train_digits)
    ppm_probs_list: list[np.ndarray] = []
    for pos in positions:
        slice_65 = split.test_digits[pos - DISPLAY_CONTEXT_LEN : pos + 1]
        p_row = ppm.predict_proba(slice_65, window_size=DISPLAY_CONTEXT_LEN)[0]
        ppm_probs_list.append(p_row)

    rounds: list[dict[str, Any]] = []
    test_offset = split.test_range[0]

    for i, pos in enumerate(positions):
        target = int(tf_tgt_np[i])
        ctx_64 = [int(x) for x in split.test_digits[pos - DISPLAY_CONTEXT_LEN : pos].tolist()]
        is_trig = bool(split.test_mask is not None and int(split.test_mask[pos]) == 1)

        trig_info = None
        if dataset == "csprng_backdoor" and is_trig and pos >= 24:
            a = int(split.test_digits[pos - 24])
            b = int(split.test_digits[pos - 7])
            c_prev = int(split.test_digits[pos - 1])
            trig_info = {
                "lag_a": 24,
                "lag_b": 7,
                "a": a,
                "b": b,
                "c_prev": c_prev,
                "formula": f"(4·{a} + {b} + 5) mod 10 = {target}",
            }

        tf_p = np.clip(tf_probs_t[i].astype(np.float64), 1e-12, 1.0)
        tf_bits = float(-math.log2(float(tf_p[target])))
        top_lags = _extract_top_lags(tf_attn[i], dataset=dataset, win_len=tf_win)

        ppm_p = np.clip(ppm_probs_list[i].astype(np.float64), 1e-12, 1.0)
        ppm_p /= ppm_p.sum()
        ppm_bits = float(-math.log2(float(ppm_p[target])))

        gru_block = None
        if gru_probs_t is not None:
            gp = np.clip(gru_probs_t[i].astype(np.float64), 1e-12, 1.0)
            gru_block = {
                "probs": _round_probs(gp),
                "pred": int(np.argmax(gp)),
                "bits": round(float(-math.log2(float(gp[target]))), 4),
            }

        rounds.append({
            "id": f"{dataset}_{i + 1:02d}",
            "global_pos": int(test_offset + pos),
            "context": ctx_64,
            "target": target,
            "is_trigger": is_trig,
            "trigger_info": trig_info,
            "transformer": {
                "probs": _round_probs(tf_p),
                "pred": int(np.argmax(tf_p)),
                "bits": round(tf_bits, 4),
                "top_lags": top_lags,
            },
            "gru": gru_block,
            "ppm": {
                "probs": _round_probs(ppm_p),
                "pred": int(np.argmax(ppm_p)),
                "bits": round(ppm_bits, 4),
            },
        })

    return {
        "dataset": dataset,
        "radix": int(art.radix),
        "rounds": rounds,
    }


def generate_live_arena_session(
    registry: ModelRegistry,
    dataset: str,
    n_rounds: int = 15,
    seed: int | None = None,
) -> dict[str, Any]:
    """Точка входа для FastAPI /api/arena/session (поддержка blind_audit и live GPU)."""
    t0 = time.perf_counter()
    if not (5 <= int(n_rounds) <= 25):
        raise ValueError("n_rounds должен быть в диапазоне [5, 25]")

    actual_seed = int(seed) if seed is not None else int(np.random.default_rng().integers(1, 2_000_000_000))
    rng = np.random.default_rng(actual_seed)

    true_dataset = dataset
    if dataset == "blind_audit":
        avail = [ds for ds in BLIND_AUDIT_POOL if f"transformer_{ds}" in registry.models]
        if not avail:
            raise RuntimeError("Нет загруженных чекпойнтов для слепого теста")
        true_dataset = str(rng.choice(avail))

    payload = build_arena_rounds_sync(
        registry=registry,
        dataset=true_dataset,
        n_rounds=int(n_rounds),
        seed=actual_seed,
        min_gap=1,
    )
    return {
        "mode": dataset,
        "true_dataset": true_dataset,
        "radix": payload["radix"],
        "rounds": payload["rounds"],
        "source": "live_gpu",
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }
