# Графики Exp 4: длина окна, масштабирование 50K–1.5M, кривая переобучения и throughput
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import MODEL_COLORS, apply_lab_layout, legend_bottom, save_plotly_figure


def plot_exp4_window_phase_transition(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Масштабирование окна L={32, 64, 128} на true_pi (H0) и 1/97 (P=96)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(
            "true_pi (H0: белый шум) — плато ~10% на всех окнах L",
            "rational_1_97 (H1: период P=96) — решается уже при L=32",
        ),
        horizontal_spacing=0.07,
    )
    dashes = {"DigitMLP": "solid", "DigitGRU": "dash", "DigitTransformer": "dot"}
    for col_idx, (block_key, ds_key) in enumerate([("Block_A_Window_H0", "true_pi"), ("Block_B_PhaseTransition_H1", "rational_1_97")], start=1):
        sub_b = df[(df["block"] == block_key) & (df["dataset"] == ds_key)]
        for m_name, dash in dashes.items():
            sub_m = sub_b[sub_b["model"] == m_name].sort_values("window_size")
            if sub_m.empty:
                continue
            color = MODEL_COLORS[m_name]
            fig.add_trace(
                go.Scatter(
                    name=m_name,
                    x=sub_m["window_size"].astype(int).tolist(),
                    y=(sub_m["test_acc"].astype(float) * 100.0).tolist(),
                    mode="lines+markers",
                    marker={"size": 10, "color": color},
                    line={"width": 3, "color": color, "dash": dash},
                    hovertemplate=f"{m_name}<br>L=%{{x}}<br>Acc=%{{y:.3f}}%<extra></extra>",
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.45)", row=1, col=col_idx)
        fig.update_xaxes(title_text="Длина контекстного окна L (цифр)", tickvals=[32, 64, 128], range=[22, 138], row=1, col=col_idx)

    fig.add_annotation(x=80, y=15.5, text="Диапазон H0: 9.73%–10.09%", showarrow=False, font={"color": "#CBD5E1", "size": 12}, row=1, col=1)
    fig.add_vline(x=96, line_dash="dot", line_color="#00CC96", line_width=2, row=1, col=2)
    fig.add_annotation(x=96, y=55, text="Период P = 96", showarrow=False, xshift=-48, font={"color": "#00CC96", "size": 12}, row=1, col=2)
    ppm_rows = df[(df["block"] == "Block_B_PhaseTransition_H1") & (df["model"] == "PPM_o8")]
    if not ppm_rows.empty:
        ppm_acc = float(ppm_rows["test_acc"].mean()) * 100.0
        fig.add_hline(y=ppm_acc, line_dash="dot", line_color="#F7B924", opacity=0.8, row=1, col=2)
        fig.add_annotation(
            x=64,
            y=91.5,
            text=f"Все модели + PPM_o8: {ppm_acc:.2f}% (10^2=100 > 97: 2-граммы внутри цикла уникальны)",
            showarrow=False,
            font={"color": "#F7B924", "size": 11.5},
            row=1,
            col=2,
        )

    fig.update_yaxes(title_text="Test Accuracy (%)", range=[-2, 110], row=1, col=1)
    apply_lab_layout(
        fig,
        "Эксперимент 4 (Блоки A и B): Инвариантность шума true_pi vs 1/97 (P=96)",
        height=640,
        width=1580,
        legend=legend_bottom(y=-0.16, font_size=13),
        margin={"l": 80, "r": 50, "t": 95, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_window_phase_transition")


def plot_exp4_capacity_scaling(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Масштабирование параметров 50K -> 250K -> 1.5M (true_pi vs champernowne)."""
    sub = df[df["block"] == "Block_C_Capacity_Scaling"]
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=(
            "true_pi (H0): Test Accuracy (%)",
            "champernowne (H1): Test Accuracy (zoom 98.5%–100.1%)",
            "true_pi (H0): Битрейт (bits/symbol, потолок 3.3219)",
            "champernowne (H1): Битрейт (лог. шкала, меньше = лучше)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
    )
    models = ["DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    for col_idx, ds_name in enumerate(("true_pi", "champernowne"), start=1):
        sub_d = sub[sub["dataset"] == ds_name]
        for m_name in models:
            sub_m = sub_d[sub_d["model"] == m_name].sort_values("params_count")
            if sub_m.empty:
                continue
            color = MODEL_COLORS[m_name]
            xs = sub_m["params_count"].astype(float).tolist()
            fig.add_trace(
                go.Scatter(name=m_name, x=xs, y=(sub_m["test_acc"].astype(float) * 100.0).tolist(), mode="lines+markers", marker={"size": 10, "color": color}, line={"width": 3, "color": color}, hovertemplate=f"{m_name}<br>params=%{{x:,.0f}}<br>Acc=%{{y:.4f}}%<extra></extra>", showlegend=(col_idx == 1)),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(name=m_name, x=xs, y=sub_m["bits_per_symbol"].astype(float).tolist(), mode="lines+markers", marker={"size": 10, "color": color}, line={"width": 3, "color": color}, hovertemplate=f"{m_name}<br>params=%{{x:,.0f}}<br>bits=%{{y:.6f}}<extra></extra>", showlegend=False),
                row=2,
                col=col_idx,
            )

    fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.5)", row=1, col=1)
    fig.add_hline(y=3.321928, line_dash="dash", line_color="rgba(255,255,255,0.5)", row=2, col=1)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(type="log", tickvals=[50_000, 250_000, 1_500_000], ticktext=["50K", "250K", "1.5M"], row=r, col=c)
    fig.update_xaxes(title_text="Число обучаемых параметров (лог. шкала)", row=2, col=1)
    fig.update_xaxes(title_text="Число обучаемых параметров (лог. шкала)", row=2, col=2)
    fig.update_yaxes(title_text="Test Accuracy (%)", range=[8.5, 11.5], row=1, col=1)
    fig.update_yaxes(title_text="Test Accuracy (%)", range=[98.7, 100.12], row=1, col=2)
    fig.update_yaxes(title_text="bits/symbol", range=[3.318, 3.332], row=2, col=1)
    fig.update_yaxes(title_text="bits/symbol (log)", type="log", row=2, col=2)
    apply_lab_layout(
        fig,
        "Эксперимент 4 (Блок C): Масштабирование параметров (50K → 250K → 1.5M, L=64)",
        height=860,
        width=1580,
        legend=legend_bottom(y=-0.09, font_size=13),
        margin={"l": 80, "r": 50, "t": 95, "b": 100},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_capacity_scaling")


def plot_exp4_memorization_scissors(mem_df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Стресс-тест на 10K знаках π за 200 эпох: запоминание без обобщения."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Точность (%): Train зубрится к 99.3%, Test застывает на ~10%",
            "Битрейт (bits/symbol): Train → 0.038, Test взрывается к 27.4 бит",
        ),
        horizontal_spacing=0.08,
    )
    epochs = mem_df["epoch"].tolist()
    splits = [
        ("Train (7,936 окон)", "train_acc", "train_bits", "#00CC96"),
        ("Val (936 окон)", "val_acc", "val_bits", "#FFA15A"),
        ("Test (936 окон)", "test_acc", "test_bits", "#EF553B"),
    ]
    for label, acc_col, bits_col, color in splits:
        fig.add_trace(go.Scatter(name=label, x=epochs, y=(mem_df[acc_col] * 100.0).tolist(), mode="lines", line={"color": color, "width": 2.5}), row=1, col=1)
        fig.add_trace(go.Scatter(name=label, x=epochs, y=mem_df[bits_col].tolist(), mode="lines", line={"color": color, "width": 2.5}, showlegend=False), row=1, col=2)

    fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=1)
    fig.add_hline(y=3.321928, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=2)
    fig.add_annotation(x=135, y=4.5, xanchor="center", text="Потолок Шеннона (3.3219 бит)", showarrow=False, font={"color": "#E2E8F0", "size": 12}, row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(title_text="Эпоха обучения (без EarlyStopping и dropout)", row=1, col=col)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 106], row=1, col=1)
    fig.update_yaxes(title_text="Битрейт (bits/symbol)", range=[0, 30], row=1, col=2)
    apply_lab_layout(
        fig,
        "Эксперимент 4 (Блок D): Стресс-тест на 10,000 знаках π — Memorization vs Generalization",
        height=620,
        width=1580,
        legend=legend_bottom(y=-0.16, font_size=13),
        margin={"l": 80, "r": 50, "t": 95, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_memorization_scissors")


def plot_exp4_compute_frontier(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Парето-фронт на RTX 3060 Ti: Throughput vs Битрейт на алгоритмических задачах."""
    sub = df[(df["hypothesis_group"] == "H1") & (df["params_count"].notna()) & (df["samples_per_sec"].notna())].copy()
    fig = go.Figure()
    symbols = {"champernowne": "circle", "rational_1_97": "diamond"}
    positions = ["top center", "bottom center", "top right", "bottom right"]
    idx_counter = 0

    for ds_name in ("champernowne", "rational_1_97"):
        sub_d = sub[sub["dataset"] == ds_name]
        for m_name in ("DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"):
            sub_m = sub_d[sub_d["model"] == m_name]
            if sub_m.empty:
                continue
            labels, sizes, y_vals, text_pos = [], [], [], []
            for _, row in sub_m.iterrows():
                p_k = float(row["params_count"]) / 1000.0
                p_lbl = f"{p_k/1000:.1f}M" if p_k >= 900 else f"{p_k:.0f}K"
                labels.append(f"{m_name.replace('Digit', '')} {p_lbl} L{int(row['window_size'])}")
                sizes.append(max(12.0, min(34.0, math.sqrt(float(row["params_count"]) / 1_800.0))))
                y_vals.append(max(float(row["bits_per_symbol"]), 3e-5))
                text_pos.append(positions[idx_counter % len(positions)])
                idx_counter += 1

            fig.add_trace(
                go.Scatter(
                    name=f"{m_name} ({ds_name})",
                    x=sub_m["samples_per_sec"].astype(float).tolist(),
                    y=y_vals,
                    mode="markers+text",
                    marker={"size": sizes, "color": MODEL_COLORS[m_name], "symbol": symbols[ds_name], "opacity": 0.88, "line": {"width": 1.5, "color": "#0F172A"}},
                    text=labels,
                    textposition=text_pos,
                    textfont={"size": 10},
                    hovertemplate=f"<b>%{{text}}</b><br>Датасет: {ds_name}<br>Throughput: %{{x:,.0f}} samp/s<br>Bits/Symbol: %{{y:.6f}}<extra></extra>",
                )
            )

    apply_lab_layout(
        fig,
        "Эксперимент 4: Compute Frontier на RTX 3060 Ti (Throughput vs Битрейт; оптимум — внизу справа)",
        height=700,
        width=1620,
        xaxis={"title": "Пропускная способность инференса (samples/sec, лог. шкала)", "type": "log"},
        yaxis={"title": "Test Битрейт (bits/symbol, лог. шкала, меньше = лучше)", "type": "log", "range": [-4.7, -0.8]},
        legend=legend_bottom(y=-0.17, font_size=11),
        margin={"l": 85, "r": 50, "t": 90, "b": 135},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_compute_frontier")
