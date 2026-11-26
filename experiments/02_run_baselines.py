"""Фаза 2: проверка zero-leakage сплита, прогон классических/ML-бейзлайнов и тепловые карты 10x10."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baselines import (
    compute_empirical_transition_matrix,
    evaluate_uniform_baseline,
    fit_and_evaluate_lightgbm,
    fit_and_evaluate_markov,
)
from src.data_generator import get_or_create_dataset, list_available_datasets, load_config, seed_everything
from src.dataset import SequenceWindowDataset, split_sequence_chronologically, verify_zero_leakage


def save_transition_heatmaps(
    matrices: dict[str, np.ndarray],
    run_dir: Path,
) -> tuple[Path, Path]:
    """Строит сетку тепловых карт 10x10 P(x_{t+1} | x_t) для ключевых Base-10 датасетов."""
    ordered_names = [
        "true_pi",
        "csprng",
        "rational_1_999983",
        "hacked_pi_tier1",
        "hacked_pi_tier2",
        "hacked_pi_tier3",
    ]
    present = [name for name in ordered_names if name in matrices]
    titles = [f"P(x[t+1] | x[t]): {name}" for name in present]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    digits_labels = [str(d) for d in range(10)]
    for idx, name in enumerate(present):
        row = idx // 3 + 1
        col = idx % 3 + 1
        mat = matrices[name]
        text_grid = [[f"{val:.3f}" for val in r] for r in mat]
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=digits_labels,
                y=digits_labels,
                text=text_grid,
                texttemplate="%{text}",
                textfont={"size": 9},
                colorscale="Viridis",
                zmin=0.0,
                zmax=0.28,
                showscale=(idx == len(present) - 1),
                colorbar={"title": "P"} if idx == len(present) - 1 else None,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="x[t+1]", row=row, col=col)
        fig.update_yaxes(title_text="x[t]", autorange="reversed", row=row, col=col)

    fig.update_layout(
        template="plotly_dark",
        title="Фаза 2: Тепловые карты условных переходов 10×10 на тестовой выборке",
        height=850,
        width=1450,
    )

    html_path = run_dir / "transition_heatmaps_10x10.html"
    png_path = run_dir / "transition_heatmaps_10x10.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)
    return html_path, png_path


def save_baselines_comparison_plot(df: pd.DataFrame, run_dir: Path) -> tuple[Path, Path]:
    """Строит сравнительный график точности (Overall vs Targeted) и Cross-Entropy по бейзлайнам."""
    focus_datasets = [
        "true_pi",
        "rational_1_999983",
        "hacked_pi_tier1",
        "hacked_pi_tier2",
        "hacked_pi_tier3",
    ]
    sub = df[df["dataset"].isin(focus_datasets)].copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Overall Test Accuracy (%) по бейзлайнам",
            "Targeted Rule Accuracy (%) на mask == 1 (Hacked Pi)",
        ),
    )

    models = ["Uniform", "Markov_k1", "Markov_k2", "Markov_k3", "LightGBM"]
    for model_name in models:
        m_sub = sub[sub["model"] == model_name]
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=m_sub["dataset"],
                y=m_sub["overall_acc"] * 100.0,
                text=[f"{v*100:.1f}%" for v in m_sub["overall_acc"]],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

    hacked_sub = sub[sub["dataset"].str.startswith("hacked_pi")]
    for model_name in models:
        m_sub = hacked_sub[hacked_sub["model"] == model_name]
        vals = m_sub["targeted_acc"].fillna(0.0) * 100.0
        fig.add_trace(
            go.Bar(
                name=f"{model_name} (mask=1)",
                x=m_sub["dataset"],
                y=vals,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.7, row=1, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.7, row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Фаза 2: Сравнение классических и ML-бейзлайнов (L=64)",
        height=560,
        width=1380,
        barmode="group",
    )
    fig.update_yaxes(title_text="Точность (%)", range=[0, 108], row=1, col=1)
    fig.update_yaxes(title_text="Точность на маске (%)", range=[0, 108], row=1, col=2)

    html_path = run_dir / "baselines_comparison.html"
    png_path = run_dir / "baselines_comparison.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)
    return html_path, png_path


def main() -> None:
    """Проверяет сплит без утечек и прогоняет Uniform, Markov k=1..3 и LightGBM."""
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed = int(config["seed"])
    seed_everything(seed)

    window_size = int(config["data"]["default_window_size"])
    split_cfg = config["data"]["split"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT_DIR / config["paths"]["runs_dir"] / f"phase2_baselines_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(
        (ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    dataset_names = list_available_datasets(config)
    all_rows: list[dict] = []
    leakage_reports: dict[str, dict] = {}
    transition_matrices: dict[str, np.ndarray] = {}

    pbar = tqdm(dataset_names, desc="Фаза 2: бейзлайны", unit="dataset")
    for name in pbar:
        pbar.set_postfix_str(f"загрузка {name}")
        art = get_or_create_dataset(name=name, config=config)

        split = split_sequence_chronologically(
            digits=art.digits,
            mask=art.mask,
            train_ratio=float(split_cfg["train"]),
            val_ratio=float(split_cfg["val"]),
            test_ratio=float(split_cfg["test"]),
        )
        leak_info = verify_zero_leakage(split, window_size=window_size)
        leakage_reports[name] = leak_info

        # Быстрая проверка контракта ленивого PyTorch Dataset (context, target, mask_flag)
        test_ds = SequenceWindowDataset(split.test_digits, window_size=window_size, mask=split.test_mask)
        sample_ctx, sample_tgt, sample_flag = test_ds[0]
        assert sample_ctx.shape == (window_size,)
        assert sample_tgt.ndim == 0 and sample_flag.ndim == 0

        # Эмпирическая матрица переходов 10x10 на тестовом срезе для визуализации
        if art.radix == 10:
            transition_matrices[name] = compute_empirical_transition_matrix(split.test_digits, radix=10)

        pbar.set_postfix_str(f"{name}: Uniform & Markov")
        u_res = evaluate_uniform_baseline(split, radix=art.radix, window_size=window_size)
        u_res.update({"dataset": name, "radix": art.radix, "window_size": window_size})
        all_rows.append(u_res)

        for k in (1, 2, 3):
            mk_res, _ = fit_and_evaluate_markov(split, radix=art.radix, order=k, window_size=window_size)
            mk_res.update({"dataset": name, "radix": art.radix, "window_size": window_size})
            all_rows.append(mk_res)

        pbar.set_postfix_str(f"{name}: LightGBM")
        lgbm_res = fit_and_evaluate_lightgbm(
            split,
            radix=art.radix,
            window_size=window_size,
            seed=seed,
            max_train_samples=200_000,
            n_estimators=150,
        )
        lgbm_res.update({"dataset": name, "radix": art.radix, "window_size": window_size})
        all_rows.append(lgbm_res)

    df = pd.DataFrame(all_rows)
    cols_order = [
        "dataset",
        "radix",
        "model",
        "overall_acc",
        "targeted_acc",
        "cross_entropy",
        "targeted_ce",
        "perplexity",
        "pred_entropy",
        "num_eval_samples",
        "num_masked_samples",
        "train_time_sec",
        "eval_time_sec",
    ]
    extra_cols = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + extra_cols]
    csv_path = run_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    plot_payload = {
        "metrics": all_rows,
        "zero_leakage_checks": leakage_reports,
        "transition_matrices_test": {k: v.tolist() for k, v in transition_matrices.items()},
    }
    with (run_dir / "plot_data.json").open("w", encoding="utf-8") as f:
        json.dump(plot_payload, f, ensure_ascii=False, indent=2)

    hm_html, hm_png = save_transition_heatmaps(transition_matrices, run_dir)
    cmp_html, cmp_png = save_baselines_comparison_plot(df, run_dir)

    print(f"\nРезультаты Фазы 2 сохранены в: {run_dir}")
    print("\nПроверка нулевой утечки (true_pi):", leakage_reports["true_pi"])
    print("\nСводка по ключевым датасетам:")
    summary = df[df["dataset"].isin(["true_pi", "csprng", "rational_1_999983", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"])]
    print(summary[["dataset", "model", "overall_acc", "targeted_acc", "cross_entropy", "perplexity", "pred_entropy"]].to_string(index=False))
    print(f"\nАртефакты графиков:\n- {hm_html}\n- {hm_png}\n- {cmp_html}\n- {cmp_png}")


if __name__ == "__main__":
    main()
