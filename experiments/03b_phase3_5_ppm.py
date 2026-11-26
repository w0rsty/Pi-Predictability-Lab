# Теоретико-информационный эталон PPM-C (max_order=8) против 4 нейросетей.
# Изначально забыл добавить сильный суффиксный компрессор рядом с нейросетями, поэтому вынес этот этап как 03b между Exp 1 и Exp 2.
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baselines import fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import split_sequence_chronologically, verify_zero_leakage
from src.trainer import create_run_dir
from src.visualizer import plot_ppm_vs_neural_bits

DATASETS = ["true_pi", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"]


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    # TODO: перенести этот локальный хелпер в общий src/evaluator.py, когда дойдут руки
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def load_latest_neural_comparison(runs_root: Path) -> pd.DataFrame:
    """Подхватывает итоговую таблицу 4 нейросетей из последнего сводного запуска Фазы 3."""
    summaries = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("exp1_transformer_summary_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for cand in summaries:
        csv_p = cand / "arch_comparison_metrics.csv"
        if csv_p.exists():
            return pd.read_csv(csv_p)
    raise FileNotFoundError("Не найден файл arch_comparison_metrics.csv из Фазы 3")


def main() -> None:
    """Оценивает PPM-C (order=8), сравнивает в битах/символ с 4 нейросетями и сохраняет в runs/."""
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config["seed"]))

    window_size = int(config["data"]["default_window_size"])
    split_cfg = config["data"]["split"]
    runs_root = ROOT_DIR / config["paths"]["runs_dir"]

    summary_dir = create_run_dir(runs_root, "phase3_5_ppm_summary")
    (summary_dir / "config_snapshot.yaml").write_text(
        (ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    ppm_rows: list[dict[str, Any]] = []
    ppm_audit: dict[str, Any] = {}

    for ds_name in DATASETS:
        art = get_or_create_dataset(name=ds_name, config=config)
        split = split_sequence_chronologically(
            digits=art.digits,
            mask=art.mask,
            train_ratio=float(split_cfg["train"]),
            val_ratio=float(split_cfg["val"]),
            test_ratio=float(split_cfg["test"]),
        )
        verify_zero_leakage(split, window_size=window_size)

        metrics, pred_probs = fit_and_evaluate_ppm(
            split=split,
            radix=art.radix,
            window_size=window_size,
            max_order=8,
        )
        targets = split.test_digits[window_size:]
        mask = split.test_mask[window_size:] if split.test_mask is not None else None
        preds = np.argmax(pred_probs, axis=1)
        correct = preds == targets

        # Проверяем нормировку вероятностей и считаем доверительные интервалы Уилсона
        max_prob_err = float(np.max(np.abs(pred_probs.sum(axis=1) - 1.0)))
        overall_k = int(np.sum(correct))
        overall_n = len(correct)
        audit_entry: dict[str, Any] = {
            "max_prob_norm_error": max_prob_err,
            "overall_wilson_95": wilson_ci(overall_k, overall_n),
        }
        if mask is not None and np.any(mask == 1):
            m_bool = mask == 1
            audit_entry["targeted_wilson_95"] = wilson_ci(int(np.sum(correct[m_bool])), int(np.sum(m_bool)))
            target_probs = pred_probs[np.arange(overall_n), targets]
            targeted_conf = float(np.mean(target_probs[m_bool]))
            unmasked_conf = float(np.mean(target_probs[~m_bool]))
        else:
            targeted_conf = None
            unmasked_conf = float(np.mean(pred_probs[np.arange(overall_n), targets]))

        ppm_audit[ds_name] = audit_entry

        ppm_rows.append(
            {
                "dataset": ds_name,
                "model": metrics["model"],
                "overall_acc": round(metrics["overall_acc"], 6),
                "targeted_acc": round(metrics["targeted_acc"], 6) if metrics["targeted_acc"] is not None else None,
                "cross_entropy": round(metrics["cross_entropy"], 6),
                "targeted_ce": round(metrics["targeted_ce"], 6) if metrics["targeted_ce"] is not None else None,
                "bits_per_symbol": round(metrics["bits_per_symbol"], 6),
                "targeted_bits_per_symbol": (
                    round(metrics["targeted_bits_per_symbol"], 6)
                    if metrics["targeted_bits_per_symbol"] is not None
                    else None
                ),
                "perplexity": round(metrics["perplexity"], 4),
                "pred_entropy": round(metrics["pred_entropy"], 6),
                "targeted_confidence": round(targeted_conf, 4) if targeted_conf is not None else None,
                "unmasked_confidence": round(unmasked_conf, 4),
                "train_time_sec": metrics["train_time_sec"],
                "eval_time_sec": metrics["eval_time_sec"],
            }
        )

    ppm_df = pd.DataFrame(ppm_rows)
    ppm_df.to_csv(summary_dir / "metrics.csv", index=False)

    neural_df = load_latest_neural_comparison(runs_root)
    neural_norm = pd.DataFrame(
        {
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
        }
    )

    combined_df = pd.concat([ppm_df, neural_norm], ignore_index=True)
    combined_df.to_csv(summary_dir / "ppm_vs_neural_comparison.csv", index=False)

    with (summary_dir / "plot_data.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ppm_metrics": ppm_rows,
                "ppm_vs_neural": combined_df.where(pd.notna(combined_df), None).to_dict(orient="records"),
                "statistical_audit": ppm_audit,
                "shannon_limit_bits_base10": math.log2(10.0),
                "bayes_tier3_bits_p05": -math.log2(0.55),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    html_p, png_p = plot_ppm_vs_neural_bits(combined_df, summary_dir, DATASETS)

    print(f"\nРезультаты Фазы 3.5 (PPM-C order=8) сохранены в: {summary_dir}")
    print("\nPPM-C (order=8):")
    print(ppm_df.to_string(index=False))
    print("\nСравнение PPM-C и 4 нейросетей (bits_per_symbol):")
    piv = combined_df.pivot(index="dataset", columns="model", values="bits_per_symbol").reindex(DATASETS)
    print(piv[["PPM_o8", "MLP", "CNN", "GRU", "TRANSFORMER"]].to_string())
    print("\nTargeted bits_per_symbol (mask == 1):")
    piv_t = combined_df.pivot(index="dataset", columns="model", values="targeted_bits_per_symbol").reindex(DATASETS[1:])
    print(piv_t[["PPM_o8", "MLP", "CNN", "GRU", "TRANSFORMER"]].to_string())
    print(f"\nГрафики:\n- {html_p}\n- {png_p}")


if __name__ == "__main__":
    main()
