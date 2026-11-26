# Графики Exp 2 (константы и радар), Exp 3 (системы счисления) и Exp 4 (масштабирование)
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import MODEL_COLORS, apply_lab_layout, legend_bottom, save_plotly_figure


def plot_exp2_constants_comparison(
    df: pd.DataFrame,
    summary_dir: Path,
    datasets: list[str],
    models_to_show: list[str],
    title_suffix: str = "Exp 2",
) -> tuple[Path, Path]:
    """Сравнение констант, ГПСЧ и цепных дробей по битрейту и выигрышу над Марковым."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Битрейт (bits_per_symbol = CE / ln 2, меньше = предсказуемее)",
            "Выигрыш над лучшей цепью Маркова (gain_over_markov_bits, больше = лучше)",
        ),
        horizontal_spacing=0.08,
    )
    for m_name in models_to_show:
        sub = df[df["model"] == m_name].set_index("dataset").reindex(datasets)
        if sub["bits_per_symbol"].isna().all():
            continue
        color = MODEL_COLORS.get(m_name, "#FFA15A")
        bits_vals = sub["bits_per_symbol"].fillna(0.0).tolist()
        gain_vals = sub["gain_over_markov_bits"].fillna(0.0).tolist()
        fig.add_trace(
            go.Bar(name=m_name, x=datasets, y=bits_vals, marker_color=color, text=[f"{v:.2f}" for v in bits_vals], textposition="outside"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(name=m_name, x=datasets, y=gain_vals, marker_color=color, text=[f"{v:+.2f}" for v in gain_vals], textposition="outside", showlegend=False),
            row=1,
            col=2,
        )

    fig.add_hline(y=3.321928, line_dash="dash", line_color="white", opacity=0.55, row=1, col=1)
    fig.add_hline(y=3.459432, line_dash="dot", line_color="#FFA15A", opacity=0.55, row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="white", opacity=0.55, row=1, col=2)
    apply_lab_layout(fig, f"Эксперимент 2 ({title_suffix}): Битва констант, ГПСЧ и цепных дробей (L=64)", height=580, width=1640, barmode="group")
    fig.update_yaxes(title_text="Бит на символ", range=[0, 4.35], row=1, col=1)
    fig.update_yaxes(title_text="Выигрыш над Марковым (бит)", row=1, col=2)
    return save_plotly_figure(fig, summary_dir, "exp2_constants_bits_comparison")


def plot_exp2_heatmap_bits(
    df: pd.DataFrame,
    summary_dir: Path,
    datasets: list[str],
    models: list[str],
) -> tuple[Path, Path]:
    """Тепловая карта нормированного битрейта (bits / log2(radix)) с аннотациями."""
    z_mat, annotations = [], []
    sub_df = df.set_index(["model", "dataset"])
    for m_name in models:
        z_row = []
        for ds_name in datasets:
            row = sub_df.loc[(m_name, ds_name)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            bits, shannon, acc_pct = float(row["bits_per_symbol"]), float(row["shannon_limit_bits"]), float(row["test_acc"]) * 100.0
            norm_bits = bits / shannon if shannon > 0 else 1.0
            z_row.append(norm_bits)
            annotations.append({
                "x": ds_name,
                "y": m_name,
                "text": f"{bits:.4f} bit<br>{acc_pct:.2f}%",
                "showarrow": False,
                "font": {"color": "#0E1117" if norm_bits < 0.45 else "#FFFFFF", "size": 12, "family": "Arial, sans-serif"},
            })
        z_mat.append(z_row)

    colorscale = [[0.0, "#00CC96"], [0.39, "#19D3F3"], [0.72, "#2A3B8F"], [0.87, "#4C2A85"], [1.0, "#EF553B"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z_mat,
            x=datasets,
            y=models,
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.15,
            xgap=3,
            ygap=3,
            colorbar={
                "title": "bits / log₂(K)",
                "tickvals": [0.0, 0.25, 0.5, 0.75, 1.0, 1.15],
                "ticktext": ["0.00 (Детерм.)", "0.25", "0.50", "0.75", "1.00 (Шум)", "1.15 (Overfit)"],
            },
            hovertemplate="Модель: %{y}<br>Датасет: %{x}<br>Норм. битрейт: %{z:.4f}<extra></extra>",
        )
    )
    apply_lab_layout(
        fig,
        "Эксперимент 2: Матрица предсказуемости (bits / log₂(K); 1.00 = потолок Шеннона, >1.00 = переобучение)",
        height=760,
        width=1560,
        annotations=annotations,
        xaxis={"title": "Последовательность (H0 слева, H1 справа)", "tickangle": -15, "side": "bottom"},
        yaxis={"title": "Модель", "autorange": "reversed"},
        margin={"l": 140, "r": 120, "t": 90, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp2_heatmap_bits")


def plot_exp2_cf_and_champernowne_contrast(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Детальный контраст алгоритмического обобщения на 4 ключевых тестах с 95% Wilson CI."""
    key_datasets = ["pi_cf", "e_cf", "champernowne", "rational_1_999983"]
    key_labels = {
        "pi_cf": "pi_cf (H0, Гаусс-Кузьмин)",
        "e_cf": "e_cf (H1, ритм [1, 2k, 1])",
        "champernowne": "champernowne (H1, счёт +1)",
        "rational_1_999983": "1/999983 (H1, период 499991)",
    }
    models_order = ["Markov_k3", "PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Точность на Test (%) с 95% доверительными интервалами Уилсона",
            "Нормированный битрейт (bits_per_symbol / log₂(K)): 1.000 = случайный потолок, 0.000 = полное сжатие",
        ),
        vertical_spacing=0.16,
    )
    x_labels = [key_labels[d] for d in key_datasets]
    for m_name in models_order:
        sub = df[df["model"] == m_name].set_index("dataset").reindex(key_datasets)
        if sub["test_acc"].isna().all():
            continue
        color = MODEL_COLORS.get(m_name, "#FFA15A")
        acc_pct = (sub["test_acc"] * 100.0).tolist()
        err_up = ((sub["wilson_ci_high"] - sub["test_acc"]) * 100.0).clip(lower=0.0).tolist()
        err_lo = ((sub["test_acc"] - sub["wilson_ci_low"]) * 100.0).clip(lower=0.0).tolist()
        norm_bits = (sub["bits_per_symbol"] / sub["shannon_limit_bits"]).tolist()

        fig.add_trace(
            go.Bar(
                name=m_name,
                x=x_labels,
                y=acc_pct,
                marker_color=color,
                error_y={"type": "data", "symmetric": False, "array": err_up, "arrayminus": err_lo, "thickness": 1.5},
                text=[f"{v:.1f}%" for v in acc_pct],
                textposition="outside",
                textfont={"size": 10},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=m_name,
                x=x_labels,
                y=norm_bits,
                marker_color=color,
                text=[f"{v:.3f}" for v in norm_bits],
                textposition="outside",
                textfont={"size": 10},
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.65, row=2, col=1)
    apply_lab_layout(
        fig,
        "Эксперимент 2: Крах табличной памяти (PPM_o8 / Markov) перед нейросетевым обобщением",
        height=960,
        width=1720,
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        legend=legend_bottom(y=-0.08, font_size=13),
        margin={"l": 80, "r": 50, "t": 95, "b": 105},
    )
    fig.update_yaxes(title_text="Test Accuracy (%)", range=[0, 115], row=1, col=1)
    fig.update_yaxes(title_text="bits / log₂(K)", range=[0, 1.24], row=2, col=1)
    return save_plotly_figure(fig, summary_dir, "exp2_cf_and_champernowne_contrast")


def _render_radar_png(html_path: Path, png_path: Path) -> None:
    html_text = html_path.read_text(encoding="utf-8").replace(
        "<body >",
        "<body style='margin:0; padding:0; background:#111111; overflow:hidden;'>",
    )
    html_path.write_text(html_text, encoding="utf-8")
    subprocess.run(
        [
            "chromium", "--headless", "--no-sandbox", "--disable-gpu",
            "--virtual-time-budget=4000", "--window-size=1160,800",
            f"--screenshot={png_path}", f"file://{html_path.resolve()}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def plot_exp2_radar_pyecharts(
    radar_scores: dict[str, dict[str, float]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """5-осевой радар эффективности семейств моделей на PyEcharts."""
    from pyecharts import options as opts
    from pyecharts.charts import Radar
    from pyecharts.globals import ThemeType

    summary_dir.mkdir(parents=True, exist_ok=True)
    html_path, png_path = summary_dir / "exp2_radar_pyecharts.html", summary_dir / "exp2_radar_pyecharts.png"
    indicators = [
        opts.RadarIndicatorItem(name="Шумоустойчивость\n(H0 Calibration)", max_=100),
        opts.RadarIndicatorItem(name="Период. память\n(1 / 999983)", max_=100),
        opts.RadarIndicatorItem(name="Многоразрядный счёт\n(Champernowne)", max_=100),
        opts.RadarIndicatorItem(name="Арифметика дробей\n(e_cf vs pi_cf)", max_=100),
        opts.RadarIndicatorItem(name="Throughput на GPU\n(RTX 3060 Ti)", max_=100),
    ]
    radar = Radar(init_opts=opts.InitOpts(width="1160px", height="800px", theme=ThemeType.DARK, bg_color="#111111", animation_opts=opts.AnimationOpts(animation=False)))
    radar.add_schema(
        schema=indicators,
        shape="polygon",
        center=["50%", "54%"],
        radius="62%",
        splitarea_opt=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.12)),
        textstyle_opts=opts.TextStyleOpts(color="#E2E8F0", font_size=13),
    )
    axis_keys = ("noise_calibration", "periodic_memory", "multi_digit_counting", "cf_arithmetic", "throughput")
    for m_name in ("PPM_o8", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"):
        if m_name not in radar_scores:
            continue
        vec = [round(float(radar_scores[m_name][k]), 2) for k in axis_keys]
        radar.add(
            series_name=m_name,
            data=[vec],
            color=MODEL_COLORS.get(m_name, "#19D3F3"),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.12),
            linestyle_opts=opts.LineStyleOpts(width=3),
            label_opts=opts.LabelOpts(is_show=False),
        )
    radar.set_global_opts(
        title_opts=opts.TitleOpts(
            title="Эксперимент 2: 5-осевой радар эффективности семейств моделей (шкала 0–100)",
            subtitle="Сравнение табличной памяти PPM_o8 и 4 нейроархитектур (~250K параметров, L=64)",
            pos_left="center",
            pos_top="14px",
            title_textstyle_opts=opts.TextStyleOpts(color="#F8FAFC", font_size=20),
            subtitle_textstyle_opts=opts.TextStyleOpts(color="#94A3B8", font_size=13),
        ),
        legend_opts=opts.LegendOpts(pos_bottom="14px", pos_left="center", textstyle_opts=opts.TextStyleOpts(color="#F8FAFC", font_size=14)),
    )
    radar.render(str(html_path))
    _render_radar_png(html_path, png_path)
    return html_path, png_path


def plot_exp2_embedding_pca_preview(
    pca_data: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """2D PCA эмбеддингов цифр DigitTransformer: шум true_pi vs структура e_cf и champernowne."""
    import numpy as np
    from src.viz.theme import DIGIT_PALETTE

    ordered = [d for d in ("true_pi", "e_cf", "champernowne") if d in pca_data]
    titles = [
        f"{'Контроль (H0 шум)' if d == 'true_pi' else f'Арифметика ({d})'} — PC1+PC2: {pca_data[d]['explained_variance_sum'] * 100:.1f}% дисперсии"
        for d in ordered
    ]
    fig = make_subplots(rows=1, cols=len(ordered), subplot_titles=titles, horizontal_spacing=0.07)

    for col_idx, ds_name in enumerate(ordered, start=1):
        coords = np.asarray(pca_data[ds_name]["coords"], dtype=np.float64)
        digits = coords[:10]
        fig.add_trace(
            go.Scatter(x=digits[:, 0].tolist(), y=digits[:, 1].tolist(), mode="lines", line={"color": "rgba(148, 163, 184, 0.45)", "width": 1.5, "dash": "dot"}, showlegend=False, hoverinfo="skip"),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=digits[:, 0].tolist(),
                y=digits[:, 1].tolist(),
                mode="markers+text",
                text=[str(d) for d in range(10)],
                textposition="top center",
                textfont={"size": 14, "color": "#F8FAFC"},
                marker={"size": 14, "color": DIGIT_PALETTE, "line": {"width": 1.5, "color": "#0F172A"}},
                name=f"{ds_name}: цифры 0..9",
                showlegend=False,
                hovertemplate="Цифра %{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        if len(coords) > 10:
            fig.add_trace(
                go.Scatter(
                    x=[float(coords[10, 0])],
                    y=[float(coords[10, 1])],
                    mode="markers+text",
                    text=["SEP(10)"],
                    textposition="bottom center",
                    textfont={"size": 12, "color": "#F7B924"},
                    marker={"size": 15, "symbol": "diamond", "color": "#F7B924", "line": {"width": 1.5, "color": "#0F172A"}},
                    name="Разделитель ';'",
                    showlegend=(col_idx == 2),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text="PC1", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", row=1, col=col_idx)
        fig.update_yaxes(title_text="PC2", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", row=1, col=col_idx)

    apply_lab_layout(
        fig,
        "Эксперимент 2: 2D PCA эмбеддингов цифр DigitTransformer (token_emb.weight)",
        height=540,
        width=1580,
        margin={"l": 70, "r": 50, "t": 90, "b": 70},
        legend=legend_bottom(y=-0.14, font_size=12),
    )
    return save_plotly_figure(fig, summary_dir, "exp2_embedding_pca_preview")


def plot_exp3_bases_normalized_bits(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Сравнение нормированного битрейта bits_per_symbol / log2(K) по системам счисления."""
    models_order = ["PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    bases_order = [
        ("Base-2", "Base-2 (K=2, окно 64 бит)", "#19D3F3"),
        ("Base-10", "Base-10 (K=10, окно ≈213 бит)", "#636EFA"),
        ("Base-16", "Base-16 BBP (K=16, окно 256 бит)", "#AB63FA"),
    ]
    fig, all_vals = go.Figure(), []
    for base_key, label, color in bases_order:
        vals = df[df["base_label"] == base_key].set_index("model").reindex(models_order)["normalized_bits"].astype(float).tolist()
        all_vals.extend(vals)
        fig.add_trace(
            go.Bar(name=label, x=models_order, y=vals, marker_color=color, text=[f"{v:.4f}" for v in vals], textposition="outside", textfont={"size": 11})
        )

    y_max = max(1.18, max(all_vals) + 0.04) if all_vals else 1.18
    y_min = min(0.95, min(all_vals) - 0.02) if all_vals else 0.95
    fig.add_hline(y=1.0, line_dash="dash", line_color="#EF553B", line_width=2)
    fig.add_annotation(
        x="DigitTransformer",
        y=1.032,  # выше баров DigitTransformer, чтобы подпись не налезала на цифры
        xanchor="right",
        yanchor="bottom",
        text="↓ Потолок Шеннона (1.0000 = случайный шум)",
        showarrow=False,
        font={"color": "#EF553B", "size": 13},
    )
    apply_lab_layout(
        fig,
        "Эксперимент 3 (BBP vs Base-2 / Base-10): Нормированный битрейт bits_per_symbol / log₂(K)",
        height=640,
        width=1540,
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        xaxis={"title": "Архитектура / Модель"},
        yaxis={"title": "Нормированный битрейт (Efficiency Ratio)", "range": [y_min, y_max]},
        legend=legend_bottom(y=-0.16, font_size=13),
        margin={"l": 85, "r": 50, "t": 90, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp3_bases_normalized_bits")


def plot_exp3_bases_zscore_ci(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Forest-plot Z-score отклонений точности с 95% ДИ Уилсона относительно 1/K."""
    models_order = ["Markov_k1", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    bases_info = [
        ("Base-2", "Base-2 (K=2, 1/K = 50.00%)", "#19D3F3"),
        ("Base-10", "Base-10 (K=10, 1/K = 10.00%)", "#636EFA"),
        ("Base-16", "Base-16 BBP (K=16, 1/K = 6.25%)", "#AB63FA"),
    ]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[b[1] for b in bases_info], horizontal_spacing=0.06)

    max_abs_z = 3.2
    for col_idx, (base_key, _, color) in enumerate(bases_info, start=1):
        sub = df[df["base_label"] == base_key].set_index("model").reindex(models_order)
        z_vals = sub["z_score"].astype(float).tolist()
        err_hi = (sub["z_ci_high"].astype(float) - sub["z_score"].astype(float)).clip(lower=0.0).tolist()
        err_lo = (sub["z_score"].astype(float) - sub["z_ci_low"].astype(float)).clip(lower=0.0).tolist()
        max_abs_z = max(max_abs_z, max(abs(z) for z in sub["z_ci_high"].tolist() + sub["z_ci_low"].tolist()) + 0.6)

        fig.add_vrect(x0=-1.96, x1=1.96, fillcolor="rgba(0, 204, 150, 0.10)", line_width=0, row=1, col=col_idx)
        fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=col_idx)
        for z_bound in (-1.96, 1.96):
            fig.add_vline(x=z_bound, line_dash="dot", line_color="#00CC96", opacity=0.65, row=1, col=col_idx)
        fig.add_trace(
            go.Scatter(
                name=base_key,
                x=z_vals,
                y=models_order,
                mode="markers+text",
                marker={"size": 11, "color": color, "line": {"width": 1.5, "color": "#0F172A"}},
                error_x={"type": "data", "symmetric": False, "array": err_hi, "arrayminus": err_lo, "thickness": 2, "width": 5},
                text=[f"{v:+.2f}σ" for v in z_vals],
                textposition="top center",
                textfont={"size": 11, "color": "#F8FAFC"},
                showlegend=False,
                hovertemplate="Модель: %{y}<br>Z-score: %{x:+.3f}σ<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Z-score отклонения от 1/K (σ)", range=[-max_abs_z, max_abs_z], row=1, col=col_idx)

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    apply_lab_layout(
        fig,
        "Эксперимент 3 (95% Wilson CI): Статистическая неотличимость авторегрессии π от шума (зелёная зона = ±1.96σ, p > 0.05)",
        height=600,
        width=1580,
        margin={"l": 145, "r": 45, "t": 100, "b": 70},
    )
    return save_plotly_figure(fig, summary_dir, "exp3_bases_zscore_ci")


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
    """Стресс-тест на 10K знаках π за 200 эпох: «ножницы» запоминания."""
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
        fig.add_trace(
            go.Scatter(name=label, x=epochs, y=(mem_df[acc_col] * 100.0).tolist(), mode="lines", line={"color": color, "width": 2.5}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(name=label, x=epochs, y=mem_df[bits_col].tolist(), mode="lines", line={"color": color, "width": 2.5}, showlegend=False),
            row=1,
            col=2,
        )

    fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=1)
    fig.add_hline(y=3.321928, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=2)
    fig.add_annotation(x=135, y=4.5, xanchor="center", text="Потолок Шеннона (3.3219 бит)", showarrow=False, font={"color": "#E2E8F0", "size": 12}, row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(title_text="Эпоха обучения (без EarlyStopping и dropout)", row=1, col=col)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 106], row=1, col=1)
    fig.update_yaxes(title_text="Битрейт (bits/symbol)", range=[0, 30], row=1, col=2)
    apply_lab_layout(
        fig,
        "Эксперимент 4 (Блок D): Стресс-тест на 10,000 знаках π — «ножницы» Memorization vs Generalization",
        height=620,
        width=1580,
        legend=legend_bottom(y=-0.16, font_size=13),
        margin={"l": 80, "r": 50, "t": 95, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_memorization_scissors")


def plot_exp4_compute_frontier(df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Pareto-фронт на RTX 3060 Ti: Throughput vs Битрейт на алгоритмических задачах."""
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
        "Эксперимент 4: Compute Frontier на RTX 3060 Ti (Throughput vs Битрейт; Парето-оптимум — внизу справа)",
        height=700,
        width=1620,
        xaxis={"title": "Пропускная способность инференса (samples/sec, лог. шкала)", "type": "log"},
        yaxis={"title": "Test Битрейт (bits/symbol, лог. шкала, меньше = лучше)", "type": "log", "range": [-4.7, -0.8]},
        legend=legend_bottom(y=-0.17, font_size=11),
        margin={"l": 85, "r": 50, "t": 90, "b": 135},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_compute_frontier")
