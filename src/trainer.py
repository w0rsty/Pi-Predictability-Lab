"""GPU-тренер с CUDA AMP, прогресс-баром, EarlyStopping и автосохранением в runs/."""

from __future__ import annotations

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

from src.evaluator import evaluate_model
from src.models import count_parameters

# Коллбэк эпохи для подключения к FastAPI WebSockets в Фазе 5: (номер_эпохи, всего_эпох, словарь_метрик)
EpochCallback = Callable[[int, int, dict[str, Any]], None]


def resolve_device(requested: str = "cuda") -> torch.device:
    """Выбирает GPU (CUDA), если она доступна и запрошена в конфиге."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_run_dir(runs_root: str | Path, exp_name: str) -> Path:
    """Создаёт изолированную папку runs/<exp_name>_<YYYY-MM-DD_HH-MM-SS>/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(runs_root) / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    run_dir: Path | None = None,
    show_progress: bool = True,
    epoch_cb: EpochCallback | None = None,
) -> dict[str, Any]:
    """Обучает PyTorch-модель с AMP на CUDA, оценивает на Test и сохраняет все артефакты."""
    train_cfg = config.get("training", {})
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    use_amp = bool(train_cfg.get("use_amp", True) and device.type == "cuda")
    if amp_dtype == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
        use_scaler = False
    else:
        autocast_dtype = torch.float16
        use_scaler = use_amp

    num_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 30))
    lr = float(learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 1e-3))
    patience = int(
        early_stopping_patience
        if early_stopping_patience is not None
        else train_cfg.get("early_stopping_patience", 5)
    )

    if run_dir is None:
        run_dir = create_run_dir(config["paths"]["runs_dir"], exp_name)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем снимок конфига, чтобы эксперимент можно было воспроизвести через год
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    model = model.to(device)
    param_count = count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=lr * 0.05)
        if scheduler_type == "cosine"
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    best_ckpt_path = run_dir / "best_model.pt"

    history: list[dict[str, Any]] = []
    train_start = time.perf_counter()

    epoch_iter = tqdm(range(1, num_epochs + 1), desc=exp_name, unit="ep", disable=not show_progress)
    for epoch in epoch_iter:
        ep_t0 = time.perf_counter()
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_entropy_sum = 0.0
        train_samples = 0

        batch_iter = tqdm(
            loaders["train"],
            desc=f"Эпоха {epoch}/{num_epochs}",
            leave=False,
            unit="batch",
            disable=not show_progress,
        )
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                logits_f32 = logits.float()
                log_probs = F.log_softmax(logits_f32, dim=-1)
                probs = log_probs.exp()
                batch_n = targets.size(0)
                train_samples += batch_n
                train_loss_sum += float(loss.item()) * batch_n
                train_correct += int((torch.argmax(logits_f32, dim=-1) == targets).sum().item())
                train_entropy_sum += float((-(probs * log_probs).sum(dim=-1)).sum().item())

        train_loss = train_loss_sum / max(train_samples, 1)
        train_acc = train_correct / max(train_samples, 1)
        train_entropy = train_entropy_sum / max(train_samples, 1)

        val_metrics = evaluate_model(
            model=model,
            loader=loaders["val"],
            device=device,
            radix=radix,
            use_amp=use_amp,
            collect_transitions=False,
        )

        ep_sec = time.perf_counter() - ep_t0
        val_loss = val_metrics["cross_entropy"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_entropy": train_entropy,
            "val_loss": val_loss,
            "val_overall_acc": val_metrics["overall_acc"],
            "val_targeted_acc": val_metrics["targeted_acc"],
            "val_targeted_ce": val_metrics["targeted_ce"],
            "val_perplexity": val_metrics["perplexity"],
            "val_pred_entropy": val_metrics["pred_entropy"],
            "epoch_time_sec": round(ep_sec, 3),
        }
        history.append(row)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "param_count": param_count,
                },
                best_ckpt_path,
            )
        else:
            no_improve_epochs += 1

        if show_progress:
            tgt_ce_str = f"{val_metrics['targeted_ce']:.3f}" if val_metrics["targeted_ce"] is not None else "-"
            epoch_iter.set_postfix(
                tr_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                acc=f"{val_metrics['overall_acc']*100:.1f}%",
                tgt_ce=tgt_ce_str,
                ent=f"{val_metrics['pred_entropy']:.3f}",
            )

        if epoch_cb is not None:
            epoch_cb(epoch, num_epochs, row)

        if scheduler is not None:
            scheduler.step()

        if enable_early_stopping and no_improve_epochs >= patience:
            break

    total_train_sec = time.perf_counter() - train_start

    # Загружаем лучший чекпойнт по Val Loss перед финальным экзаменом на Test
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        loader=loaders["test"],
        device=device,
        radix=radix,
        use_amp=use_amp,
        collect_transitions=(radix == 10),
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "metrics.csv", index=False)

    summary = {
        "exp_name": exp_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "use_amp": use_amp,
        "param_count": param_count,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": bool(enable_early_stopping and no_improve_epochs >= patience),
        "total_train_time_sec": round(total_train_sec, 3),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "test_metrics": test_metrics,
        "run_dir": str(run_dir),
        "checkpoint_path": str(best_ckpt_path),
    }

    with (run_dir / "plot_data.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "history": history}, f, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "history": history,
        "test_metrics": test_metrics,
        "run_dir": run_dir,
    }
