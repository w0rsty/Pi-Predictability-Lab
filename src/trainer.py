# Цикл обучения на GPU/CPU с CUDA AMP, прогресс-баром, EarlyStopping и снимком конфига
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_generator import make_dataloader_generator, seed_worker
from src.evaluator import evaluate_model
from src.models import count_parameters

# Коллбэк эпохи: (номер_эпохи, всего_эпох, словарь_метрик)
EpochCallback = Callable[[int, int, dict[str, Any]], None]


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    """Конвертирует ключи ранних чекпойнтов MultiheadAttention в ключи нативного SDPA."""
    remapped: dict[str, torch.Tensor] = {}
    changed = False
    for k, v in state_dict.items():
        nk = k.replace(".attn.in_proj_weight", ".qkv_proj.weight").replace(".attn.in_proj_bias", ".qkv_proj.bias").replace(".attn.out_proj.", ".out_proj.")
        if nk != k:
            changed = True
        remapped[nk] = v
    return remapped, changed


def create_run_dir(runs_root: str | Path, exp_name: str) -> Path:
    """Создаёт изолированную папку runs/<exp_name>_<YYYY-MM-DD_HH-MM-SS>/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(runs_root) / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_amp: bool,
    use_scaler: bool,
    epoch: int,
    num_epochs: int,
    show_progress: bool,
    debug_grad_norm: bool,
) -> tuple[float, float, float]:
    model.train()
    loss_sum, correct, entropy_sum, samples = 0.0, 0, 0.0, 0
    batch_iter = tqdm(loader, desc=f"Эпоха {epoch}/{num_epochs}", leave=False, unit="batch", disable=not show_progress)
    for contexts, targets, _ in batch_iter:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
            logits = model(contexts)
            loss = F.cross_entropy(logits, targets)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if debug_grad_norm and samples == 0:
            print(f"[debug] ep={epoch} grad_norm={float(grad_norm):.4f}")
        with torch.no_grad():
            logits_f32 = logits.float()
            log_probs = F.log_softmax(logits_f32, dim=-1)
            probs = log_probs.exp()
            batch_n = targets.size(0)
            samples += batch_n
            loss_sum += float(loss.item()) * batch_n
            correct += int((torch.argmax(logits_f32, dim=-1) == targets).sum().item())
            entropy_sum += float(-(probs * log_probs).sum(dim=-1).sum().item())

    denom = max(samples, 1)
    return loss_sum / denom, correct / denom, entropy_sum / denom


def _save_run_outputs(
    run_dir: Path,
    exp_name: str,
    device: torch.device,
    use_amp: bool,
    param_count: int,
    best_epoch: int,
    best_val_loss: float,
    early_stopped: bool,
    total_train_sec: float,
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    best_ckpt_path: Path,
) -> dict[str, Any]:
    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
    summary = {
        "exp_name": exp_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "use_amp": use_amp,
        "param_count": param_count,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": early_stopped,
        "total_train_time_sec": round(total_train_sec, 3),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "test_metrics": test_metrics,
        "run_dir": str(run_dir),
        "checkpoint_path": str(best_ckpt_path),
    }
    (run_dir / "plot_data.json").write_text(
        json.dumps({"summary": summary, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _build_adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Делит обучаемые параметры на decay (ndim >= 2) и no_decay (bias, LayerNorm, BatchNorm)."""
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    exp_name: str,
    radix: int,
    epochs: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float = 1e-4,
    scheduler_type: str | None = None,
    amp_dtype: str = "float16",
    early_stopping_patience: int | None = None,
    enable_early_stopping: bool = True,
    restore_best_val: bool | None = None,
    run_dir: Path | None = None,
    show_progress: bool = True,
    epoch_cb: EpochCallback | None = None,
    debug_grad_norm: bool = False,
) -> dict[str, Any]:
    """Обучает модель с AMP на CUDA, сохраняет лучший чекпойнт по Val и оценивает на Test."""
    train_cfg = config.get("training", {})
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    use_amp = bool(train_cfg.get("use_amp", True) and device.type == "cuda")
    use_bf16 = amp_dtype == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = use_amp and not use_bf16
    should_restore = enable_early_stopping if restore_best_val is None else bool(restore_best_val)

    num_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 30))
    lr = float(learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 1e-3))
    patience = int(early_stopping_patience if early_stopping_patience is not None else train_cfg.get("early_stopping_patience", 5))

    run_dir = create_run_dir(config["paths"]["runs_dir"], exp_name) if run_dir is None else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    model = model.to(device)
    param_count = count_parameters(model)
    param_groups = _build_adamw_param_groups(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=lr * 0.05) if scheduler_type == "cosine" else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_loss, best_epoch, no_improve = float("inf"), 0, 0
    best_ckpt_path = run_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    train_start = time.perf_counter()

    epoch_iter = tqdm(range(1, num_epochs + 1), desc=exp_name, unit="ep", disable=not show_progress)
    for epoch in epoch_iter:
        ep_t0 = time.perf_counter()
        tr_loss, tr_acc, tr_ent = _train_one_epoch(
            model, loaders["train"], optimizer, scaler, device, autocast_dtype, use_amp, use_scaler, epoch, num_epochs, show_progress, debug_grad_norm
        )
        val_metrics = evaluate_model(model=model, loader=loaders["val"], device=device, radix=radix, use_amp=use_amp, amp_dtype=amp_dtype)
        val_loss = val_metrics["cross_entropy"]
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_entropy": tr_ent,
            "val_loss": val_loss,
            "val_overall_acc": val_metrics["overall_acc"],
            "val_targeted_acc": val_metrics["targeted_acc"],
            "val_targeted_ce": val_metrics["targeted_ce"],
            "val_perplexity": val_metrics["perplexity"],
            "val_pred_entropy": val_metrics["pred_entropy"],
            "epoch_time_sec": round(time.perf_counter() - ep_t0, 3),
        }
        history.append(row)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch, no_improve = val_loss, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "param_count": param_count}, best_ckpt_path)
        else:
            no_improve += 1

        if show_progress:
            tgt_ce_str = f"{val_metrics['targeted_ce']:.3f}" if val_metrics["targeted_ce"] is not None else "-"
            epoch_iter.set_postfix(tr_loss=f"{tr_loss:.4f}", val_loss=f"{val_loss:.4f}", acc=f"{val_metrics['overall_acc']*100:.1f}%", tgt_ce=tgt_ce_str, ent=f"{val_metrics['pred_entropy']:.3f}")
        if epoch_cb is not None:
            epoch_cb(epoch, num_epochs, row)
        if scheduler is not None:
            scheduler.step()
        if enable_early_stopping and no_improve >= patience:
            break

    if should_restore and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_model(model=model, loader=loaders["test"], device=device, radix=radix, use_amp=use_amp, amp_dtype=amp_dtype, collect_transitions=(radix == 10))
    summary = _save_run_outputs(
        run_dir, exp_name, device, use_amp, param_count, best_epoch, best_val_loss,
        bool(enable_early_stopping and no_improve >= patience), time.perf_counter() - train_start, history, test_metrics, best_ckpt_path,
    )
    return {"summary": summary, "history": history, "test_metrics": test_metrics, "run_dir": run_dir}
