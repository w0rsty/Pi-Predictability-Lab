# Проверка пайплайна (Exp 1): true_pi против 3 уровней синтетических закладок hacked_pi
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import SequenceWindowDataset, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import probe_gru_hidden_dynamics, probe_transformer_attention
from src.models import build_model, count_parameters
from src.trainer import create_run_dir, resolve_device, train_model
from src.viz import (
    plot_arch_battle_comparison,
    plot_attention_maps,
    plot_exp1_test_xray,
    plot_exp1_training_curves,
    plot_gru_hidden_probe,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
EXP1_DATASETS = ["true_pi", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"]


def _find_completed_run(runs_root: Path, arch: str, ds_name: str, expected_params: int) -> dict[str, Any] | None:
    if not runs_root.exists():
        return None
    prefix = f"exp1_{arch}_{ds_name}_"
    candidates = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.name, reverse=True)
    for run_dir in candidates:
        pdata_path = run_dir / "plot_data.json"
        if not pdata_path.exists():
            continue
        try:
            payload = json.loads(pdata_path.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            if summary.get("param_count") == expected_params and "test_metrics" in summary:
                return {"summary": summary, "history": payload["history"], "test_metrics": summary["test_metrics"], "run_dir": run_dir}
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return None


def _collect_arch_comparison_df(runs_root: Path, current_arch: str, current_rows: list[dict[str, Any]]) -> pd.DataFrame:
    arch_dfs: list[pd.DataFrame] = []
    for arch_name in ("mlp", "cnn", "gru", "transformer"):
        if arch_name == current_arch:
            arch_dfs.append(pd.DataFrame(current_rows))
            continue
        summaries = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(f"exp1_{arch_name}_summary_")], key=lambda p: p.name, reverse=True)
        for cand in summaries:
            csv_p = cand / "metrics.csv"
            if csv_p.exists():
                arch_dfs.append(pd.read_csv(csv_p))
                break
    return pd.concat(arch_dfs, ignore_index=True) if arch_dfs else pd.DataFrame(current_rows)


def _format_summary_row(ds_name: str, arch: str, size_tier: str, res: dict[str, Any]) -> dict[str, Any]:
    s, tm = res["summary"], res["test_metrics"]
    return {
        "dataset": ds_name,
        "arch": arch,
        "size_tier": size_tier,
        "param_count": s["param_count"],
        "epochs_ran": s["epochs_ran"],
        "best_epoch": s["best_epoch"],
        "early_stopped": s["early_stopped"],
        "train_time_sec": s["total_train_time_sec"],
        "final_train_loss": round(s["final_train_loss"], 6),
        "best_val_loss": round(s["best_val_loss"], 6),
        "test_overall_acc": round(tm["overall_acc"], 6),
        "test_targeted_acc": round(tm["targeted_acc"], 6) if tm["targeted_acc"] is not None else None,
        "test_cross_entropy": round(tm["cross_entropy"], 6),
        "test_targeted_ce": round(tm["targeted_ce"], 6) if tm["targeted_ce"] is not None else None,
        "test_perplexity": round(tm["perplexity"], 4),
        "test_pred_entropy": round(tm["pred_entropy"], 6),
        "targeted_confidence": round(tm["targeted_confidence"], 4) if tm["targeted_confidence"] is not None else None,
        "unmasked_confidence": round(tm["unmasked_confidence"], 4) if tm["unmasked_confidence"] is not None else None,
        "run_dir": s["run_dir"],
    }


def _run_dataset_for_arch(
    ds_name: str,
    arch: str,
    size_tier: str,
    config: dict[str, Any],
    device: torch.device,
    epochs_override: int | None,
    reuse_completed: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    seed = int(config["seed"])
    window_size = int(config["data"]["default_window_size"])
    split_cfg, train_cfg = config["data"]["split"], config["training"]
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]

    seed_everything(seed)
    art = get_or_create_dataset(name=ds_name, config=config)
    model = build_model(arch=arch, radix=art.radix, window_size=window_size, size_tier=size_tier)
    expected_params = count_parameters(model)

    split = split_sequence_chronologically(
        digits=art.digits, mask=art.mask,
        train_ratio=float(split_cfg["train"]), val_ratio=float(split_cfg["val"]), test_ratio=float(split_cfg["test"]),
    )
    verify_zero_leakage(
        split,
        window_size=window_size,
        train_ds=SequenceWindowDataset(split.train_digits, window_size, split.train_mask, split.train_range[0]),
        val_ds=SequenceWindowDataset(split.val_digits, window_size, split.val_mask, split.val_range[0]),
        test_ds=SequenceWindowDataset(split.test_digits, window_size, split.test_mask, split.test_range[0]),
    )

    reused = _find_completed_run(runs_root, arch, ds_name, expected_params) if reuse_completed else None
    if reused is not None:
        print(f"\n=== [{arch.upper()} | {size_tier}] Датасет: {ds_name} | Используем прогон: {reused['run_dir'].name} ===")
        res = reused
        ckpt_path = Path(res["summary"]["checkpoint_path"])
        ckpt_path = ckpt_path if ckpt_path.is_absolute() else ROOT_DIR / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Чекпойнт завершённого прогона не найден: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["model_state_dict"])
        model.to(device)
    else:
        loaders = create_dataloaders(split=split, window_size=window_size, batch_size=int(train_cfg["batch_size"]), num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=True)
        print(f"\n=== [{arch.upper()} | {size_tier}] Датасет: {ds_name} | Параметров: {expected_params:,} ===")
        res = train_model(model=model, loaders=loaders, config=config, exp_name=f"exp1_{arch}_{ds_name}", radix=art.radix, epochs=epochs_override, enable_early_stopping=True)

    gru_probe = probe_gru_hidden_dynamics(model, split.test_digits, split.test_mask, window_size, 7 if ds_name == "hacked_pi_tier1" else 9, device) if arch == "gru" and ds_name in ("hacked_pi_tier1", "hacked_pi_tier2") and split.test_mask is not None else None
    attn_probe = probe_transformer_attention(model, split.test_digits, split.test_mask, window_size, ds_name, device) if arch == "transformer" and ds_name in ("true_pi", "hacked_pi_tier1", "hacked_pi_tier2") else None
    return res, gru_probe, attn_probe


def run_sanity_check_for_arch(arch: str, size_tier: str = "medium", epochs_override: int | None = None, reuse_completed: bool = False) -> Path:
    """Прогоняет одну архитектуру на true_pi и 3 уровнях hacked_pi и сохраняет сводный отчёт."""
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config["seed"]))
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]
    device = resolve_device(str(config["training"].get("device", "cuda")))

    runs_by_ds, probes_by_ds, attn_by_ds, summary_rows = {}, {}, {}, []
    for ds_name in EXP1_DATASETS:
        res, gp, ap = _run_dataset_for_arch(ds_name, arch, size_tier, config, device, epochs_override, reuse_completed)
        runs_by_ds[ds_name] = res
        if gp is not None:
            probes_by_ds[ds_name] = gp
        if ap is not None:
            attn_by_ds[ds_name] = ap
        summary_rows.append(_format_summary_row(ds_name, arch, size_tier, res))

    summary_dir = create_run_dir(runs_root, f"exp1_{arch}_summary")
    (summary_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_dir / "metrics.csv", index=False)
    (summary_dir / "plot_data.json").write_text(
        json.dumps({"summary_table": summary_rows, "histories": {k: v["history"] for k, v in runs_by_ds.items()}, "test_metrics": {k: v["test_metrics"] for k, v in runs_by_ds.items()}, "probes": probes_by_ds, "attention_maps": attn_by_ds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved_plots = [*plot_exp1_training_curves(arch, runs_by_ds, summary_dir), *plot_exp1_test_xray(arch, runs_by_ds, summary_dir)]
    if probes_by_ds and (p_paths := plot_gru_hidden_probe(probes_by_ds, summary_dir)) is not None:
        saved_plots.extend(p_paths)
    if attn_by_ds and (a_paths := plot_attention_maps(attn_by_ds, summary_dir)) is not None:
        saved_plots.extend(a_paths)
    combined_df = _collect_arch_comparison_df(runs_root, arch, summary_rows)
    if combined_df["arch"].nunique() >= 2:
        combined_df.to_csv(summary_dir / "arch_comparison_metrics.csv", index=False)
        saved_plots.extend(plot_arch_battle_comparison(combined_df, summary_dir, EXP1_DATASETS))

    print(f"\nСводка Эксперимента 1 ({arch.upper()}) сохранена в: {summary_dir}")
    print(df.drop(columns=["run_dir"]).to_string(index=False))
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 1: Sanity Check на True Pi и Hacked Pi")
    parser.add_argument("--arch", type=str, default="mlp", help="Архитектура: mlp, cnn, gru, transformer")
    parser.add_argument("--size", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--epochs", type=int, default=None, help="Переопределить число эпох из config.yaml")
    parser.add_argument("--reuse-completed", action="store_true", help="Подхватить уже завершённые прогоны из runs/")
    args = parser.parse_args()
    run_sanity_check_for_arch(arch=args.arch, size_tier=args.size, epochs_override=args.epochs, reuse_completed=args.reuse_completed)


if __name__ == "__main__":
    main()
