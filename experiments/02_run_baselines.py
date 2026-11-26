# Проверка zero-leakage сплита и классические бейзлайны (Uniform, цепи Маркова k=1..3, LightGBM)
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.baselines import (
    compute_empirical_transition_matrix,
    evaluate_uniform_baseline,
    fit_and_evaluate_lightgbm,
    fit_and_evaluate_markov,
)
from src.data_generator import get_or_create_dataset, list_available_datasets, load_config, seed_everything
from src.dataset import SequenceWindowDataset, split_sequence_chronologically, verify_zero_leakage

ROOT_DIR = Path(__file__).resolve().parents[1]


def save_transition_heatmaps(matrices: dict[str, np.ndarray], run_dir: Path) -> tuple[Path, Path]:
    """Сетка тепловых карт 10x10 P(x_{t+1} | x_t) для Base-10 датасетов."""
    ordered = ["true_pi", "csprng", "rational_1_999983", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"]
    present = [name for name in ordered if name in matrices]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"P(x[t+1] | x[t]): {n}" for n in present], horizontal_spacing=0.06, vertical_spacing=0.14)

    digits_labels = [str(d) for d in range(10)]
    for idx, name in enumerate(present):
        r, c = idx // 3 + 1, idx % 3 + 1
        mat = matrices[name]
        fig.add_trace(
            go.Heatmap(
                z=mat, x=digits_labels, y=digits_labels, text=[[f"{v:.3f}" for v in row] for row in mat],
                texttemplate="%{text}", textfont={"size": 9}, colorscale="Viridis", zmin=0.0, zmax=0.28,
                showscale=(idx == len(present) - 1), colorbar={"title": "P"} if idx == len(present) - 1 else None,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="x[t+1]", row=r, col=c)
        fig.update_yaxes(title_text="x[t]", autorange="reversed", row=r, col=c)

    fig.update_layout(template="plotly_dark", title="Тепловые карты условных переходов 10×10 на тестовой выборке", height=850, width=1450)
    html_path, png_path = run_dir / "transition_heatmaps_10x10.html", run_dir / "transition_heatmaps_10x10.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)
    return html_path, png_path


def save_baselines_comparison_plot(df: pd.DataFrame, run_dir: Path) -> tuple[Path, Path]:
    """Сравнение Overall vs Targeted Accuracy и Cross-Entropy по классическим бейзлайнам."""
    focus = ["true_pi", "rational_1_999983", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"]
    sub = df[df["dataset"].isin(focus)].copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Overall Test Accuracy (%) по бейзлайнам", "Targeted Rule Accuracy (%) на mask == 1 (Hacked Pi)"),
    )
    models = ["Uniform", "Markov_k1", "Markov_k2", "Markov_k3", "LightGBM"]
    for m_name in models:
        m_sub = sub[sub["model"] == m_name]
        fig.add_trace(go.Bar(name=m_name, x=m_sub["dataset"], y=m_sub["overall_acc"] * 100.0, text=[f"{v*100:.1f}%" for v in m_sub["overall_acc"]], textposition="outside"), row=1, col=1)
        m_hk = m_sub[m_sub["dataset"].str.startswith("hacked_pi")]
        vals = m_hk["targeted_acc"].fillna(0.0) * 100.0
        fig.add_trace(go.Bar(name=f"{m_name} (mask=1)", x=m_hk["dataset"], y=vals, text=[f"{v:.1f}%" for v in vals], textposition="outside", showlegend=False), row=1, col=2)

    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.7, row=1, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.7, row=1, col=2)
    fig.update_layout(template="plotly_dark", title="Сравнение классических и ML-бейзлайнов (L=64)", height=560, width=1380, barmode="group")
    fig.update_yaxes(title_text="Точность (%)", range=[0, 108], row=1, col=1)
    fig.update_yaxes(title_text="Точность на маске (%)", range=[0, 108], row=1, col=2)

    html_path, png_path = run_dir / "baselines_comparison.html", run_dir / "baselines_comparison.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)
    return html_path, png_path


def _evaluate_dataset_baselines(name: str, config: dict, window_size: int, seed: int) -> tuple[list[dict], dict, np.ndarray | None]:
    split_cfg = config["data"]["split"]
    art = get_or_create_dataset(name=name, config=config)
    split = split_sequence_chronologically(
        digits=art.digits,
        mask=art.mask,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
    )
    leak_info = verify_zero_leakage(split, window_size=window_size)
    sample_ctx, sample_tgt, sample_flag = SequenceWindowDataset(split.test_digits, window_size=window_size, mask=split.test_mask)[0]
    assert sample_ctx.shape == (window_size,) and sample_tgt.ndim == 0 and sample_flag.ndim == 0

    tmat = compute_empirical_transition_matrix(split.test_digits, radix=10) if art.radix == 10 else None
    rows: list[dict] = []
    for res in [
        evaluate_uniform_baseline(split, radix=art.radix, window_size=window_size),
        *[fit_and_evaluate_markov(split, radix=art.radix, order=k, window_size=window_size)[0] for k in (1, 2, 3)],
        fit_and_evaluate_lightgbm(split, radix=art.radix, window_size=window_size, seed=seed, max_train_samples=200_000, n_estimators=150),
    ]:
        res.update({"dataset": name, "radix": art.radix, "window_size": window_size})
        rows.append(res)
    return rows, leak_info, tmat


def main() -> None:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed = int(config["seed"])
    seed_everything(seed)
    window_size = int(config["data"]["default_window_size"])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT_DIR / config["paths"]["runs_dir"] / f"baselines_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    all_rows, leakage_reports, transition_matrices = [], {}, {}
    pbar = tqdm(list_available_datasets(config), desc="Бейзлайны", unit="dataset")
    for name in pbar:
        pbar.set_postfix_str(name)
        rows, leak_info, tmat = _evaluate_dataset_baselines(name, config, window_size, seed)
        all_rows.extend(rows)
        leakage_reports[name] = leak_info
        if tmat is not None:
            transition_matrices[name] = tmat

    df = pd.DataFrame(all_rows)
    cols = ["dataset", "radix", "model", "overall_acc", "targeted_acc", "cross_entropy", "targeted_ce", "perplexity", "pred_entropy", "num_eval_samples", "num_masked_samples", "train_time_sec", "eval_time_sec"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(run_dir / "metrics.csv", index=False)

    payload = {"metrics": all_rows, "zero_leakage_checks": leakage_reports, "transition_matrices_test": {k: v.tolist() for k, v in transition_matrices.items()}}
    (run_dir / "plot_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hm_html, hm_png = save_transition_heatmaps(transition_matrices, run_dir)
    cmp_html, cmp_png = save_baselines_comparison_plot(df, run_dir)

    print(f"\nБейзлайны готовы, папка: {run_dir}")
    print("\nПроверка нулевой утечки (true_pi):", leakage_reports["true_pi"])
    pi_u = df[(df["dataset"] == "true_pi") & (df["model"] == "Uniform")].iloc[0]
    print(f"[check] true_pi Uniform: acc={pi_u['overall_acc']:.4f}, bits={pi_u['cross_entropy'] / math.log(2):.4f}")
    print(f"\nАртефакты графиков:\n- {hm_html}\n- {hm_png}\n- {cmp_html}\n- {cmp_png}")


if __name__ == "__main__":
    main()
