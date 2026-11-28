# Экспресс-обучение компактного DigitTransformer на кастомной 2-лаговой закладке (BYO-GPU)
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data_generator import generate_csprng_digits, seed_everything
from src.dataset import SequenceWindowDataset, split_sequence_chronologically
from src.evaluator import compute_binomial_z_score
from src.models import build_model


def apply_custom_backdoor(
    base_digits: np.ndarray,
    lag_a: int,
    lag_b: int,
    prev_threshold: int,
    coeff_a: int,
    coeff_b: int,
    coeff_const: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Внедряет параметризованную модульную закладку на произвольных лагах (lag_a, lag_b)."""
    digits = base_digits.copy()
    mask = np.zeros(len(digits), dtype=np.uint8)
    start_t = max(int(lag_a), int(lag_b), 1)
    for t in range(start_t, len(digits)):
        a = int(digits[t - lag_a])
        b = int(digits[t - lag_b])
        c = int(digits[t - 1])
        if (a + b) % 10 == 0 and c < prev_threshold:
            digits[t] = (coeff_a * a + coeff_b * b + coeff_const) % 10
            mask[t] = 1
    return digits, mask


def _analyze_bijectivity(coeff_a: int, coeff_b: int, coeff_const: int) -> dict[str, Any]:
    diff = abs(int(coeff_a) - int(coeff_b))
    gcd_val = math.gcd(diff, 10) if diff > 0 else 10
    reachable = sorted({(coeff_a * a + coeff_b * ((10 - a) % 10) + coeff_const) % 10 for a in range(10)})
    return {
        "gcd": int(gcd_val),
        "is_bijective": bool(gcd_val == 1 and len(reachable) == 10),
        "reachable_digits": reachable,
    }


def _to_device_tensors(ds: SequenceWindowDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.arange(len(ds), dtype=np.int64)
    offsets = idx[:, None] + np.arange(ds.window_size, dtype=np.int64)[None, :]
    x = torch.from_numpy(ds.digits[offsets].astype(np.int64)).to(device)
    target_pos = idx + ds.window_size
    y = torch.from_numpy(ds.digits[target_pos].astype(np.int64)).to(device)
    if ds.mask is not None:
        m = torch.from_numpy(ds.mask[target_pos].astype(np.uint8)).to(device)
    else:
        m = torch.zeros(len(ds), dtype=torch.uint8, device=device)
    return x, y, m


def train_custom_backdoor_sync(
    device: torch.device,
    lag_a: int = 31,
    lag_b: int = 11,
    prev_threshold: int = 4,
    coeff_a: int = 4,
    coeff_b: int = 1,
    coeff_const: int = 5,
    length: int = 32_000,
    epochs: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Обучает DigitTransformer (medium на GPU / light на CPU) на хронологическом сплите 75/5/20."""
    if lag_a == lag_b:
        raise ValueError("Лаги lag_a и lag_b должны различаться")
    if not (2 <= lag_a <= 48 and 2 <= lag_b <= 48):
        raise ValueError("Лаги должны лежать в диапазоне [2, 48]")

    t0 = time.perf_counter()
    seed_everything(seed)
    window_size = 64
    use_cuda = device.type == "cuda"
    eff_epochs = int(epochs) if epochs is not None else (18 if use_cuda else 8)
    batch_size = 1024 if use_cuda else 128
    size_tier = "medium" if use_cuda else "light"

    base_digits = generate_csprng_digits(length=length, radix=10, deterministic=True, seed=seed)
    digits, mask = apply_custom_backdoor(
        base_digits,
        lag_a=lag_a,
        lag_b=lag_b,
        prev_threshold=prev_threshold,
        coeff_a=coeff_a,
        coeff_b=coeff_b,
        coeff_const=coeff_const,
    )

    # Строгий хронологический сплит: окно не пересекает границы Train / Val / Test
    split = split_sequence_chronologically(digits, mask, train_ratio=0.75, val_ratio=0.05, test_ratio=0.20)
    train_ds = SequenceWindowDataset(split.train_digits, window_size=window_size, mask=split.train_mask)
    val_ds = SequenceWindowDataset(
        split.val_digits,
        window_size=window_size,
        mask=split.val_mask,
        global_offset=split.val_range[0],
    )
    test_ds = SequenceWindowDataset(
        split.test_digits,
        window_size=window_size,
        mask=split.test_mask,
        global_offset=split.test_range[0],
    )

    x_tr, y_tr, _ = _to_device_tensors(train_ds, device)
    x_va, y_va, _ = _to_device_tensors(val_ds, device)
    x_te, y_te, m_te = _to_device_tensors(test_ds, device)

    model = build_model("transformer", radix=10, window_size=window_size, size_tier=size_tier, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(eff_epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: list[dict[str, float | int]] = []
    n_train = x_tr.size(0)
    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, eff_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            b_idx = perm[start : start + batch_size]
            bx, by = x_tr[b_idx], y_tr[b_idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(bx)
                loss = F.cross_entropy(logits, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            steps += 1
        scheduler.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda):
            val_loss = float(F.cross_entropy(model(x_va), y_va).item())
        train_loss = loss_sum / max(steps, 1)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
            }
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits_te = model(x_te)
        preds_te = logits_te.argmax(dim=-1)
        correct_mask = preds_te == y_te
        test_n = int(y_te.numel())
        overall_acc = float(correct_mask.float().mean().item())

        trig_bool = m_te == 1
        trig_count = int(trig_bool.sum().item())
        targeted_acc = float(correct_mask[trig_bool].float().mean().item()) if trig_count > 0 else None
        unmasked_bool = m_te == 0
        unmasked_acc = float(correct_mask[unmasked_bool].float().mean().item()) if int(unmasked_bool.sum().item()) > 0 else overall_acc

        # Профиль внимания с триггерных (или первых тестовых) окон:
        # Берём максимум по головам и слоям после усреднения по выборке, чтобы чётко выделить специализированные лаги
        probe_x = x_te[trig_bool][:128] if trig_count > 0 else x_te[:128]
        _, stacked_attn = model(probe_x, return_attn=True)
        # stacked_attn: (B, n_layers, n_heads, 64, 64) -> last_row: (B, n_layers, n_heads, 64)
        last_row = stacked_attn[:, :, :, -1, :].mean(dim=0)  # (n_layers, n_heads, 64)
        mean_profile = last_row.mean(dim=(0, 1))
        peak_profile = last_row.max(dim=0).values.max(dim=0).values
        # Нормируем peak_profile в сумму=1 для единого графика дуг/лагов
        peak_norm = peak_profile / torch.clamp(peak_profile.sum(), min=1e-9)

        top_vals, top_pos = torch.topk(peak_norm, k=min(6, window_size))
        top_lags = [
            {"lag": int(window_size - int(p.item())), "weight": round(float(v.item()), 4)}
            for v, p in zip(top_vals, top_pos)
        ]

    z_overall = compute_binomial_z_score(overall_acc, 10, test_n)
    z_targeted = (
        compute_binomial_z_score(targeted_acc, 10, trig_count)
        if targeted_acc is not None and trig_count > 0
        else None
    )

    return {
        "params_count": int(sum(p.numel() for p in model.parameters())),
        "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "epochs_ran": eff_epochs,
        "train_samples": int(n_train),
        "test_samples": test_n,
        "total_triggers": int(mask.sum()),
        "targeted_count": trig_count,
        "overall_acc": round(overall_acc, 5),
        "targeted_acc": round(targeted_acc, 5) if targeted_acc is not None else None,
        "unmasked_acc": round(unmasked_acc, 5),
        "z_score": float(z_overall["z_score"]),
        "targeted_z_score": float(z_targeted["z_score"]) if z_targeted is not None else None,
        "attention_profile": [round(float(v), 5) for v in peak_norm.cpu().tolist()],
        "mean_attention_profile": [round(float(v), 5) for v in mean_profile.cpu().tolist()],
        "top_lags": top_lags,
        "history": history,
        "bijectivity": _analyze_bijectivity(coeff_a, coeff_b, coeff_const),
    }
