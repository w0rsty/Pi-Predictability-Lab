# Сервис живой многоголовой абляции и извлечения Attention Maps на GPU
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data_generator import get_or_create_dataset, load_config
from src.dataset import split_sequence_chronologically
from src.evaluator import wilson_ci
from src.models import DigitTransformer
from web.services.model_registry import ModelRegistry, ROOT_DIR


@torch.no_grad()
def run_multi_head_ablation_sync(
    registry: ModelRegistry,
    dataset: str,
    ablate_heads: list[tuple[int, int]],
    max_windows: int = 4000,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model_key = f"transformer_{dataset}"
    if model_key not in registry.models:
        raise ValueError(f"Чекпойнт '{model_key}' не загружен. Доступны: {list(registry.models.keys())}")

    entry = registry.models[model_key]
    model = entry.model
    assert isinstance(model, DigitTransformer)

    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    req_len = 1_000_050 if dataset == "rational_1_97" else None
    art = get_or_create_dataset(dataset, config=config, length=req_len)
    split_cfg = config["data"]["split"]
    split = split_sequence_chronologically(
        art.digits, art.mask, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )

    win = entry.window_size
    n_win = min(int(max_windows), len(split.test_digits) - win)
    if n_win < 32:
        raise ValueError("Недостаточно тестовых окон для абляции")

    test_slice = split.test_digits[: n_win + win]
    ctx_np = np.lib.stride_tricks.sliding_window_view(test_slice[:-1], win)
    tgt_np = test_slice[win:]
    mask_np = split.test_mask[win : win + n_win] if split.test_mask is not None else None

    batch_size = 512
    base_corr_chunks: list[np.ndarray] = []
    base_nll_chunks: list[np.ndarray] = []
    abl_corr_chunks: list[np.ndarray] = []
    abl_nll_chunks: list[np.ndarray] = []
    attn_profile_sum: np.ndarray | None = None
    attn_count = 0

    with torch.inference_mode():
        for start in range(0, n_win, batch_size):
            end = min(n_win, start + batch_size)
            ctx_t = torch.as_tensor(ctx_np[start:end], dtype=torch.long, device=registry.device)
            tgt_t = torch.as_tensor(tgt_np[start:end], dtype=torch.long, device=registry.device)
            with torch.amp.autocast("cuda", enabled=(registry.device.type == "cuda")):
                base_logits = model(ctx_t)
                if start == 0:
                    abl_logits, attn_w = model.forward_with_head_ablation(
                        ctx_t, ablate_heads=ablate_heads, return_attn=True
                    )
                    # Усредняем профиль последнего запроса по головам первого слоя
                    attn_profile_sum = attn_w[:, 0, :, -1, :].float().mean(dim=1).sum(dim=0).cpu().numpy()
                    attn_count = int(attn_w.size(0))
                else:
                    abl_logits = model.forward_with_head_ablation(
                        ctx_t, ablate_heads=ablate_heads, return_attn=False
                    )

            bf32 = base_logits.float()
            af32 = abl_logits.float()
            base_corr_chunks.append((torch.argmax(bf32, dim=-1) == tgt_t).cpu().numpy())
            abl_corr_chunks.append((torch.argmax(af32, dim=-1) == tgt_t).cpu().numpy())
            base_nll_chunks.append(F.nll_loss(F.log_softmax(bf32, dim=-1), tgt_t, reduction="none").cpu().numpy())
            abl_nll_chunks.append(F.nll_loss(F.log_softmax(af32, dim=-1), tgt_t, reduction="none").cpu().numpy())

    base_corr = np.concatenate(base_corr_chunks)
    abl_corr = np.concatenate(abl_corr_chunks)
    base_nll = np.concatenate(base_nll_chunks)
    abl_nll = np.concatenate(abl_nll_chunks)

    b_acc = float(np.mean(base_corr))
    a_acc = float(np.mean(abl_corr))
    b_bits = float(np.mean(base_nll) / math.log(2.0))
    a_bits = float(np.mean(abl_nll) / math.log(2.0))
    ci_lo, ci_hi = wilson_ci(int(np.sum(abl_corr)), len(abl_corr))

    tgt_b_acc = tgt_a_acc = tgt_delta = None
    if mask_np is not None and np.any(mask_np == 1):
        m_bool = mask_np == 1
        tgt_b_acc = round(float(np.mean(base_corr[m_bool])), 6)
        tgt_a_acc = round(float(np.mean(abl_corr[m_bool])), 6)
        tgt_delta = round((tgt_a_acc - tgt_b_acc) * 100.0, 4)

    attn_prof = (attn_profile_sum / max(attn_count, 1)).tolist() if attn_profile_sum is not None else []
    return {
        "dataset": dataset,
        "ablate_heads": [[int(l), int(h)] for l, h in ablate_heads],
        "num_windows": int(n_win),
        "baseline_acc": round(b_acc, 6),
        "ablated_acc": round(a_acc, 6),
        "delta_acc_pp": round((a_acc - b_acc) * 100.0, 4),
        "baseline_bits": round(b_bits, 6),
        "ablated_bits": round(a_bits, 6),
        "delta_bits": round(a_bits - b_bits, 6),
        "wilson_ci": [ci_lo, ci_hi],
        "targeted_baseline_acc": tgt_b_acc,
        "targeted_ablated_acc": tgt_a_acc,
        "targeted_delta_acc_pp": tgt_delta,
        "attention_profile": [round(float(x), 5) for x in attn_prof],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }
