# Сравнение суффиксного компрессора PPM-C (max_order=8) с четырьмя нейросетями на Exp 1
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines import fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import SequenceWindowDataset, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import wilson_ci
from src.trainer import create_run_dir
from src.viz import plot_ppm_vs_neural_bits

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS = ["true_pi", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"]


def load_latest_neural_comparison(runs_root: Path) -> pd.DataFrame:
    """Подхватывает итоговую таблицу 4 нейросетей из runs/ или results/."""
    if runs_root.exists():
        summaries = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("exp1_transformer_summary_")], key=lambda p: p.name, reverse=True)
        for cand in summaries:
            csv_p = cand / "arch_comparison_metrics.csv"
            if csv_p.exists():
                return pd.read_csv(csv_p)
    fallback = ROOT_DIR / "results" / "03_exp1_sanity" / "arch_comparison_metrics.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError("Не найден arch_comparison_metrics.csv в runs/ или results/")


def _evaluate_ppm_dataset(ds_name: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    window_size = int(config["data"]["default_window_size"])
    split_cfg = config["data"]["split"]
    art = get_or_create_dataset(name=ds_name, config=config)
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

    metrics, pred_probs = fit_and_evaluate_ppm(split=split, radix=art.radix, window_size=window_size, max_order=8)
    targets = split.test_digits[window_size:]
    mask = split.test_mask[window_size:] if split.test_mask is not None else None
    correct = np.argmax(pred_probs, axis=1) == targets
    target_probs = pred_probs[np.arange(len(targets)), targets]

    audit_entry: dict[str, Any] = {
        "max_prob_norm_error": float(np.max(np.abs(pred_probs.sum(axis=1) - 1.0))),
        "overall_wilson_95": list(wilson_ci(int(np.sum(correct)), len(correct))),
    }
    if mask is not None and np.any(mask == 1):
        m_bool = mask == 1
        audit_entry["targeted_wilson_95"] = list(wilson_ci(int(np.sum(correct[m_bool])), int(np.sum(m_bool))))
        targeted_conf, unmasked_conf = float(np.mean(target_probs[m_bool])), float(np.mean(target_probs[~m_bool]))
    else:
        targeted_conf, unmasked_conf = None, float(np.mean(target_probs))

    row = {
        "dataset": ds_name,
        "model": metrics["model"],
        "overall_acc": round(metrics["overall_acc"], 6),
        "targeted_acc": round(metrics["targeted_acc"], 6) if metrics["targeted_acc"] is not None else None,
        "cross_entropy": round(metrics["cross_entropy"], 6),
        "targeted_ce": round(metrics["targeted_ce"], 6) if metrics["targeted_ce"] is not None else None,
        "bits_per_symbol": round(metrics["bits_per_symbol"], 6),
        "targeted_bits_per_symbol": round(metrics["targeted_bits_per_symbol"], 6) if metrics["targeted_bits_per_symbol"] is not None else None,
        "perplexity": round(metrics["perplexity"], 4),
        "pred_entropy": round(metrics["pred_entropy"], 6),
        "targeted_confidence": round(targeted_conf, 4) if targeted_conf is not None else None,
        "unmasked_confidence": round(unmasked_conf, 4),
        "train_time_sec": metrics["train_time_sec"],
        "eval_time_sec": metrics["eval_time_sec"],
    }
    return row, audit_entry


def main() -> None:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config["seed"]))
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]

    summary_dir = create_run_dir(runs_root, "ppm_summary")
    (summary_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    ppm_rows, ppm_audit = [], {}
    for ds_name in DATASETS:
        row, audit_entry = _evaluate_ppm_dataset(ds_name, config)
        ppm_rows.append(row)
        ppm_audit[ds_name] = audit_entry

    ppm_df = pd.DataFrame(ppm_rows)
    ppm_df.to_csv(summary_dir / "metrics.csv", index=False)

    neural_df = load_latest_neural_comparison(runs_root)
    neural_norm = pd.DataFrame({
        "dataset": neural_df["dataset"],
        "model": neural_df["arch"].str.upper(),
        "overall_acc": neural_df["test_overall_acc"],
        "targeted_acc": neural_df["test_targeted_acc"],
        "cross_entropy": neural_df["test_cross_entropy"],
        "targeted_ce": neural_df["test_targeted_ce"],
        "bits_per_symbol": (neural_df["test_cross_entropy"] / math.log(2.0)).round(6),
        "targeted_bits_per_symbol": (neural_df["test_targeted_ce"] / math.log(2.0)).round(6),
        "perplexity": neural_df["test_perplexity"],
        "pred_entropy": neural_df["test_pred_entropy"],
        "targeted_confidence": neural_df["targeted_confidence"],
        "unmasked_confidence": neural_df["unmasked_confidence"],
        "train_time_sec": neural_df["train_time_sec"],
        "eval_time_sec": None,
    })

    combined_df = pd.concat([ppm_df, neural_norm], ignore_index=True)
    combined_df.to_csv(summary_dir / "ppm_vs_neural_comparison.csv", index=False)
    (summary_dir / "plot_data.json").write_text(
        json.dumps(
            {
                "ppm_metrics": ppm_rows,
                "ppm_vs_neural": combined_df.where(pd.notna(combined_df), None).to_dict(orient="records"),
                "statistical_audit": ppm_audit,
                "shannon_limit_bits_base10": math.log2(10.0),
                "bayes_tier3_bits_p05": -math.log2(0.55),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    html_p, png_p = plot_ppm_vs_neural_bits(combined_df, summary_dir, DATASETS)
    print(f"\nРезультаты PPM-C (order=8) сохранены в: {summary_dir}")
    print(combined_df.pivot(index="dataset", columns="model", values="bits_per_symbol").reindex(DATASETS)[["PPM_o8", "MLP", "CNN", "GRU", "TRANSFORMER"]].to_string())
    print(f"\nГрафики:\n- {html_p}\n- {png_p}")


if __name__ == "__main__":
    main()
