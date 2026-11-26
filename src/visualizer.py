# Графики проекта на Plotly (HTML + PNG) и PyEcharts (радар)
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_plotly_figure(
    fig: go.Figure,
    out_dir: Path,
    stem: str,
    scale: int = 2,
) -> tuple[Path, Path]:
    # Сохраняем сразу в HTML (для интерактива) и PNG 2x (чтобы не мылилось в README на GitHub)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{stem}.html"
    png_path = out_dir / f"{stem}.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=scale)
    return html_path, png_path


def plot_exp1_training_curves(
    arch: str,
    runs_by_dataset: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит графики обучения по эпохам: Val Loss, Val Acc, Targeted CE и Prediction Entropy."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Validation Cross-Entropy Loss (меньше = лучше)",
            "Validation Overall Accuracy (%)",
            "Validation Targeted CE на mask == 1 (меньше = лучше)",
            "Validation Prediction Entropy (наты, потолок ln 10 ≈ 2.303)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    colors = {
        "true_pi": "#636EFA",
        "hacked_pi_tier1": "#00CC96",
        "hacked_pi_tier2": "#EF553B",
        "hacked_pi_tier3": "#FFA15A",
    }

    for ds_name, res in runs_by_dataset.items():
        hist = res["history"]
        epochs = [h["epoch"] for h in hist]
        color = colors.get(ds_name, "#AB63FA")

        fig.add_trace(
            go.Scatter(
                name=ds_name,
                x=epochs,
                y=[h["val_loss"] for h in hist],
                mode="lines+markers",
                line={"color": color},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name=ds_name,
                x=epochs,
                y=[h["val_overall_acc"] * 100.0 for h in hist],
                mode="lines+markers",
                line={"color": color},
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        tgt_ce = [h["val_targeted_ce"] for h in hist if h["val_targeted_ce"] is not None]
        if tgt_ce:
            fig.add_trace(
                go.Scatter(
                    name=ds_name,
                    x=epochs[: len(tgt_ce)],
                    y=tgt_ce,
                    mode="lines+markers",
                    line={"color": color},
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                name=ds_name,
                x=epochs,
                y=[h["val_pred_entropy"] for h in hist],
                mode="lines+markers",
                line={"color": color},
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.add_hline(y=2.302585, line_dash="dash", line_color="white", opacity=0.6, row=1, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.6, row=1, col=2)
    fig.add_hline(y=2.302585, line_dash="dash", line_color="white", opacity=0.6, row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        title=f"Эксперимент 1 ({arch.upper()}): Кривые обучения на True Pi vs 3 уровнях Hacked Pi",
        height=760,
        width=1360,
    )
    return save_plotly_figure(fig, summary_dir, f"exp1_{arch}_training_curves")


def plot_exp1_test_xray(
    arch: str,
    runs_by_dataset: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит сравнение итоговых Test-метрик и уверенности на маске vs фоне."""
    ds_names = list(runs_by_dataset.keys())
    overall_ce = [runs_by_dataset[d]["test_metrics"]["cross_entropy"] for d in ds_names]
    targeted_ce = [runs_by_dataset[d]["test_metrics"]["targeted_ce"] or 0.0 for d in ds_names]
    tgt_conf = [(runs_by_dataset[d]["test_metrics"]["targeted_confidence"] or 0.0) * 100.0 for d in ds_names]
    unm_conf = [(runs_by_dataset[d]["test_metrics"]["unmasked_confidence"] or 0.0) * 100.0 for d in ds_names]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Test Cross-Entropy: общая vs на триггерах (mask==1)",
            "Средняя уверенность P_model(истинная цифра): триггер vs фон (%)",
        ),
    )

    fig.add_trace(
        go.Bar(
            name="Overall CE",
            x=ds_names,
            y=overall_ce,
            text=[f"{v:.3f}" for v in overall_ce],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Targeted CE (mask=1)",
            x=ds_names,
            y=targeted_ce,
            text=[f"{v:.3f}" for v in targeted_ce],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="На триггере (mask=1)",
            x=ds_names,
            y=tgt_conf,
            text=[f"{v:.1f}%" for v in tgt_conf],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="На фоне true_pi (mask=0)",
            x=ds_names,
            y=unm_conf,
            text=[f"{v:.1f}%" for v in unm_conf],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.add_hline(y=2.302585, line_dash="dash", line_color="white", opacity=0.6, row=1, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.6, row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title=f"Эксперимент 1 ({arch.upper()}): Рентген обнаружения скрытых правил на Test",
        height=540,
        width=1360,
        barmode="group",
    )
    fig.update_yaxes(title_text="Cross-Entropy (наты)", range=[0, 2.7], row=1, col=1)
    fig.update_yaxes(title_text="Уверенность P (%)", range=[0, 112], row=1, col=2)
    return save_plotly_figure(fig, summary_dir, f"exp1_{arch}_test_xray")


def plot_gru_hidden_probe(
    probes: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path] | None:
    """Строит диагностический график ||h_t||_2 и P_t(target_digit) по шагам окна t=0..63."""
    ds_list = [k for k in ("hacked_pi_tier1", "hacked_pi_tier2") if k in probes]
    if not ds_list:
        return None

    titles = []
    for ds_name in ds_list:
        td = probes[ds_name]["target_digit"]
        titles.append(f"{ds_name}: норма скрытого состояния ||h_t||_2 (t=0..63)")
        titles.append(f"{ds_name}: вероятность следующей цифры P_t({td})")

    fig = make_subplots(
        rows=len(ds_list),
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    for row_idx, ds_name in enumerate(ds_list, start=1):
        p = probes[ds_name]
        steps = list(range(len(p["digits"])))
        digits_txt = [str(d) for d in p["digits"]]

        fig.add_trace(
            go.Scatter(
                name=f"{ds_name} ||h_t||_2",
                x=steps,
                y=p["hidden_l2_norm"],
                mode="lines+markers",
                marker={"size": 4},
                text=digits_txt,
                hovertemplate="t=%{x}, digit=%{text}, ||h_t||=%{y:.3f}",
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name=f"{ds_name} P_t({p['target_digit']})",
                x=steps,
                y=[v * 100.0 for v in p["next_target_prob"]],
                mode="lines+markers",
                marker={"size": 4},
                text=digits_txt,
                hovertemplate="t=%{x}, digit=%{text}, P=%{y:.1f}%",
            ),
            row=row_idx,
            col=2,
        )

        trigger_steps = [t for t, flg in enumerate(p["mask_next_target"]) if flg == 1]
        for t in trigger_steps:
            fig.add_vline(x=t, line_dash="dot", line_color="#EF553B", opacity=0.65, row=row_idx, col=1)
            fig.add_vline(x=t, line_dash="dot", line_color="#EF553B", opacity=0.65, row=row_idx, col=2)

        fig.add_hline(y=10.0, line_dash="dash", line_color="gray", opacity=0.6, row=row_idx, col=2)
        fig.update_yaxes(title_text="||h_t||_2", row=row_idx, col=1)
        fig.update_yaxes(title_text="P (%)", range=[0, 108], row=row_idx, col=2)
        fig.update_xaxes(title_text="Шаг в окне t", row=row_idx, col=1)
        fig.update_xaxes(title_text="Шаг в окне t", row=row_idx, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="GRU Probe: динамика скрытого состояния и вероятности на триггерах (красный пунктир: mask_next == 1)",
        height=360 * len(ds_list),
        width=1380,
        showlegend=False,
    )
    return save_plotly_figure(fig, summary_dir, "gru_hidden_probe")


def plot_arch_battle_comparison(
    combined_df: pd.DataFrame,
    summary_dir: Path,
    all_datasets: list[str],
) -> tuple[Path, Path]:
    """Строит сводный график битвы архитектур (MLP vs 1D-CNN vs GRU vs Transformer)."""
    hacked_ds = [d for d in all_datasets if d.startswith("hacked_pi")]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Targeted CE на mask == 1 (меньше = лучше)",
            "Уверенность на триггерах mask == 1 (%)",
            "Overall Perplexity (меньше = лучше)",
        ),
        horizontal_spacing=0.07,
    )

    arch_colors = {"mlp": "#636EFA", "cnn": "#00CC96", "gru": "#EF553B", "transformer": "#AB63FA"}
    present_archs = list(dict.fromkeys(combined_df["arch"].tolist()))

    for a in present_archs:
        sub_all = combined_df[combined_df["arch"] == a].set_index("dataset")
        sub_hacked = sub_all.reindex(hacked_ds)
        sub_full = sub_all.reindex(all_datasets)
        color = arch_colors.get(a, "#FFA15A")

        tgt_ce_vals = sub_hacked["test_targeted_ce"].fillna(0.0).tolist()
        fig.add_trace(
            go.Bar(
                name=a.upper(),
                x=hacked_ds,
                y=tgt_ce_vals,
                marker_color=color,
                text=[f"{v:.3f}" for v in tgt_ce_vals],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        conf_vals = (sub_hacked["targeted_confidence"].fillna(0.0) * 100.0).tolist()
        fig.add_trace(
            go.Bar(
                name=a.upper(),
                x=hacked_ds,
                y=conf_vals,
                marker_color=color,
                text=[f"{v:.1f}%" for v in conf_vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        ppl_vals = sub_full["test_perplexity"].fillna(10.0).tolist()
        fig.add_trace(
            go.Bar(
                name=a.upper(),
                x=all_datasets,
                y=ppl_vals,
                marker_color=color,
                text=[f"{v:.2f}" for v in ppl_vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=3)
    archs_label = " vs ".join(a.upper() for a in present_archs)
    fig.update_layout(
        template="plotly_dark",
        title=f"Эксперимент 1: Битва архитектур ({archs_label}, ~250K параметров, L=64)",
        height=560,
        width=1620,
        barmode="group",
    )
    fig.update_yaxes(title_text="Targeted CE (наты)", row=1, col=1)
    fig.update_yaxes(title_text="Уверенность (%)", range=[0, 114], row=1, col=2)
    fig.update_yaxes(title_text="Perplexity", range=[0, 11.4], row=1, col=3)

    tag = "_".join(present_archs)
    return save_plotly_figure(fig, summary_dir, f"exp1_arch_battle_{tag}")


def plot_attention_maps(
    attn_probes: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path] | None:
    """Строит сравнение карт внимания Transformer для true_pi и hacked_pi (Tier 1 и Tier 2)."""
    ordered = [d for d in ("true_pi", "hacked_pi_tier1", "hacked_pi_tier2") if d in attn_probes]
    if not ordered:
        return None

    titles_row1 = []
    titles_row2 = []
    for ds_name in ordered:
        probe = attn_probes[ds_name]
        best_l = probe["focus_layer"]
        best_h = probe["focus_head"]
        titles_row1.append(f"{ds_name}: L×L (Слой {best_l}, Голова {best_h})")
        titles_row2.append(f"{ds_name}: фокус t=63 (Слой {best_l}, все головы)")

    fig = make_subplots(
        rows=2,
        cols=len(ordered),
        subplot_titles=titles_row1 + titles_row2,
        vertical_spacing=0.16,
        horizontal_spacing=0.07,
    )

    head_colors = ["#636EFA", "#00CC96", "#EF553B", "#FFA15A", "#AB63FA", "#19D3F3"]
    for col_idx, ds_name in enumerate(ordered, start=1):
        probe = attn_probes[ds_name]
        attn = np.asarray(probe["attn_weights"], dtype=np.float32)  # (n_layers, n_heads, L, L)
        best_l = int(probe["focus_layer"])
        best_h = int(probe["focus_head"])
        seq_len = attn.shape[-1]
        positions = list(range(seq_len))

        # Верхний ряд: полная нижнетреугольная карта внимания лучшей головы
        mat = attn[best_l, best_h]
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=positions,
                y=positions,
                colorscale="Viridis",
                zmin=0.0,
                zmax=float(max(0.25, np.percentile(mat, 99.5))),
                showscale=(col_idx == len(ordered)),
                colorbar={"title": "Weight", "len": 0.42, "y": 0.78} if col_idx == len(ordered) else None,
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Key позиция (прошлое)", row=1, col=col_idx)
        fig.update_yaxes(title_text="Query позиция", autorange="reversed", row=1, col=col_idx)

        # Нижний ряд: распределение внимания из последнего токена t=L-1 по всем головам слоя best_l
        n_heads = attn.shape[1]
        for h_idx in range(n_heads):
            fig.add_trace(
                go.Scatter(
                    name=f"Head {h_idx}",
                    x=positions,
                    y=attn[best_l, h_idx, -1, :].tolist(),
                    mode="lines",
                    line={"color": head_colors[h_idx % len(head_colors)], "width": 2},
                    showlegend=(col_idx == 1),
                ),
                row=2,
                col=col_idx,
            )

        # Отмечаем красным пунктиром целевые ключевые позиции в окне (например, -50 или t-4..t-2)
        for key_pos in probe.get("expected_key_positions", []):
            fig.add_vline(x=key_pos, line_dash="dot", line_color="#EF553B", opacity=0.8, row=2, col=col_idx)

        fig.update_xaxes(title_text="Key позиция k (t=0..63)", row=2, col=col_idx)
        fig.update_yaxes(title_text="Вес внимания α", row=2, col=col_idx)

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 1 (Transformer): Рентген карт внимания — True Pi против Hacked Pi (красный пунктир = триггеры)",
        height=860,
        width=1580,
        legend={"orientation": "h", "y": -0.07, "x": 0.5, "xanchor": "center"},
    )
    return save_plotly_figure(fig, summary_dir, "transformer_attention_maps")


def plot_ppm_vs_neural_bits(
    comparison_df: pd.DataFrame,
    summary_dir: Path,
    datasets: list[str],
    shannon_limit_bits: float = 3.321928,
) -> tuple[Path, Path]:
    """Сравнение PPM-C (суффиксная память) против 4 нейросетей в битах на символ."""
    hacked_ds = [d for d in datasets if d.startswith("hacked_pi")]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Overall Bits per Symbol (CE / ln 2, порог Шеннона 3.322 бит)",
            "Targeted Bits per Symbol на триггерах mask == 1 (меньше = лучше)",
        ),
        horizontal_spacing=0.08,
    )

    model_colors = {
        "PPM_o8": "#F7B924",
        "MLP": "#636EFA",
        "CNN": "#00CC96",
        "GRU": "#EF553B",
        "TRANSFORMER": "#AB63FA",
    }
    present_models = list(dict.fromkeys(comparison_df["model"].tolist()))

    for m_name in present_models:
        sub = comparison_df[comparison_df["model"] == m_name].set_index("dataset")
        sub_all = sub.reindex(datasets)
        sub_hacked = sub.reindex(hacked_ds)
        color = model_colors.get(m_name, "#19D3F3")

        overall_bits = sub_all["bits_per_symbol"].tolist()
        fig.add_trace(
            go.Bar(
                name=m_name,
                x=datasets,
                y=overall_bits,
                marker_color=color,
                text=[f"{v:.3f}" for v in overall_bits],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        tgt_bits = sub_hacked["targeted_bits_per_symbol"].fillna(0.0).tolist()
        fig.add_trace(
            go.Bar(
                name=m_name,
                x=hacked_ds,
                y=tgt_bits,
                marker_color=color,
                text=[f"{v:.3f}" for v in tgt_bits],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=shannon_limit_bits, line_dash="dash", line_color="white", opacity=0.6, row=1, col=1)
    # Байесовский оптимум на Tier 3 (p=0.5): -log2(0.55) ≈ 0.8625 бит/цифру
    fig.add_hline(y=0.862496, line_dash="dot", line_color="gray", opacity=0.6, row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Фаза 3.5: PPM-компрессор (табличная память суффиксов, order=8) против 4 нейросетей (~250K)",
        height=560,
        width=1580,
        barmode="group",
    )
    fig.update_yaxes(title_text="Бит на символ (CE / ln 2)", range=[0, 4.25], row=1, col=1)
    fig.update_yaxes(title_text="Бит на символ на mask==1", range=[0, 2.75], row=1, col=2)
    return save_plotly_figure(fig, summary_dir, "phase3_5_ppm_vs_neural_bits")


def plot_exp2_constants_comparison(
    df: pd.DataFrame,
    summary_dir: Path,
    datasets: list[str],
    models_to_show: list[str],
    title_suffix: str = "Этап 4A",
) -> tuple[Path, Path]:
    """Строит сравнение констант, ГПСЧ и цепных дробей в битах на символ и по выигрышу над Марковым."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Битрейт (bits_per_symbol = CE / ln 2, меньше = предсказуемее)",
            "Выигрыш над лучшей цепью Маркова (gain_over_markov_bits, больше = лучше)",
        ),
        horizontal_spacing=0.08,
    )

    palette = {
        "PPM_o8": "#F7B924",
        "LightGBM": "#19D3F3",
        "MLP": "#636EFA",
        "CNN": "#00CC96",
        "GRU": "#EF553B",
        "TRANSFORMER": "#AB63FA",
    }

    for m_name in models_to_show:
        sub = df[df["model"] == m_name].set_index("dataset").reindex(datasets)
        if sub["bits_per_symbol"].isna().all():
            continue
        color = palette.get(m_name, "#FFA15A")
        bits_vals = sub["bits_per_symbol"].fillna(0.0).tolist()
        gain_vals = sub["gain_over_markov_bits"].fillna(0.0).tolist()

        fig.add_trace(
            go.Bar(
                name=m_name,
                x=datasets,
                y=bits_vals,
                marker_color=color,
                text=[f"{v:.2f}" for v in bits_vals],
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=m_name,
                x=datasets,
                y=gain_vals,
                marker_color=color,
                text=[f"{v:+.2f}" for v in gain_vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=3.321928, line_dash="dash", line_color="white", opacity=0.55, row=1, col=1)
    fig.add_hline(y=3.459432, line_dash="dot", line_color="#FFA15A", opacity=0.55, row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="white", opacity=0.55, row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title=f"Эксперимент 2 ({title_suffix}): Битва констант, ГПСЧ и цепных дробей (L=64)",
        height=580,
        width=1640,
        barmode="group",
    )
    fig.update_yaxes(title_text="Бит на символ", range=[0, 4.35], row=1, col=1)
    fig.update_yaxes(title_text="Выигрыш над Марковым (бит)", row=1, col=2)
    return save_plotly_figure(fig, summary_dir, "exp2_constants_bits_comparison")


def plot_exp2_heatmap_bits(
    df: pd.DataFrame,
    summary_dir: Path,
    datasets: list[str],
    models: list[str],
) -> tuple[Path, Path]:
    """Строит тепловую карту нормированного битрейта (bits / log2(radix)) с аннотациями битрейта и точности."""
    z_matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []
    font_colors: list[list[str]] = []

    sub_df = df.set_index(["model", "dataset"])
    for m_name in models:
        z_row: list[float] = []
        t_row: list[str] = []
        c_row: list[str] = []
        for ds_name in datasets:
            row = sub_df.loc[(m_name, ds_name)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            bits = float(row["bits_per_symbol"])
            shannon = float(row["shannon_limit_bits"])
            acc_pct = float(row["test_acc"]) * 100.0
            norm_bits = bits / shannon if shannon > 0 else 1.0
            z_row.append(norm_bits)
            t_row.append(f"{bits:.4f} bit<br>{acc_pct:.2f}%")
            # На светло-зелёной/бирюзовой части шкалы (< 0.45) делаем тёмный текст, на тёмной — белый
            c_row.append("#0E1117" if norm_bits < 0.45 else "#FFFFFF")
        z_matrix.append(z_row)
        text_matrix.append(t_row)
        font_colors.append(c_row)

    colorscale = [
        [0.0, "#00CC96"],
        [0.39, "#19D3F3"],
        [0.72, "#2A3B8F"],
        [0.87, "#4C2A85"],
        [1.0, "#EF553B"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
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

    annotations = []
    for r_idx, m_name in enumerate(models):
        for c_idx, ds_name in enumerate(datasets):
            annotations.append(
                {
                    "x": ds_name,
                    "y": m_name,
                    "text": text_matrix[r_idx][c_idx],
                    "showarrow": False,
                    "font": {"color": font_colors[r_idx][c_idx], "size": 12, "family": "Arial, sans-serif"},
                }
            )

    fig.update_layout(
        template="plotly_dark",
        title=(
            "Эксперимент 2: Матрица предсказуемости (нормированный битрейт bits / log₂(K), "
            "1.00 = потолок Шеннона, >1.00 = переобучение)"
        ),
        annotations=annotations,
        xaxis={"title": "Последовательность (H0 слева, H1 справа)", "tickangle": -15, "side": "bottom"},
        yaxis={"title": "Модель", "autorange": "reversed"},
        width=1560,
        height=760,
        margin={"l": 140, "r": 120, "t": 90, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp2_heatmap_bits")


def plot_exp2_cf_and_champernowne_contrast(
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит детальный контраст алгоритмического обобщения на 4 ключевых тестах с 95% Wilson CI."""
    key_datasets = ["pi_cf", "e_cf", "champernowne", "rational_1_999983"]
    key_labels = {
        "pi_cf": "pi_cf (H0, Гаусс-Кузьмин)",
        "e_cf": "e_cf (H1, ритм [1, 2k, 1])",
        "champernowne": "champernowne (H1, счёт +1)",
        "rational_1_999983": "1/999983 (H1, период 499991)",
    }
    models_order = ["Markov_k3", "PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    palette = {
        "Markov_k3": "#94A3B8",
        "PPM_o8": "#F7B924",
        "LightGBM": "#19D3F3",
        "DigitMLP": "#636EFA",
        "DigitCNN": "#00CC96",
        "DigitGRU": "#EF553B",
        "DigitTransformer": "#AB63FA",
    }

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
        color = palette.get(m_name, "#FFA15A")
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
    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 2: Крах табличной памяти (PPM_o8 / Markov) перед нейросетевым обобщением",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        height=960,
        width=1720,
        legend={"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
        margin={"l": 80, "r": 50, "t": 95, "b": 105},
    )
    fig.update_yaxes(title_text="Test Accuracy (%)", range=[0, 115], row=1, col=1)
    fig.update_yaxes(title_text="bits / log₂(K)", range=[0, 1.24], row=2, col=1)
    return save_plotly_figure(fig, summary_dir, "exp2_cf_and_champernowne_contrast")


def plot_exp2_radar_pyecharts(
    radar_scores: dict[str, dict[str, float]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит 5-осевой радар эффективности семейств моделей (PyEcharts) в .html и .png."""
    import subprocess
    from pyecharts import options as opts
    from pyecharts.charts import Radar
    from pyecharts.globals import ThemeType

    summary_dir.mkdir(parents=True, exist_ok=True)
    html_path = summary_dir / "exp2_radar_pyecharts.html"
    png_path = summary_dir / "exp2_radar_pyecharts.png"

    indicators = [
        opts.RadarIndicatorItem(name="Шумоустойчивость\n(H0 Calibration)", max_=100),
        opts.RadarIndicatorItem(name="Период. память\n(1 / 999983)", max_=100),
        opts.RadarIndicatorItem(name="Многоразрядный счёт\n(Champernowne)", max_=100),
        opts.RadarIndicatorItem(name="Арифметика дробей\n(e_cf vs pi_cf)", max_=100),
        opts.RadarIndicatorItem(name="Throughput на GPU\n(RTX 3060 Ti)", max_=100),
    ]
    colors = {
        "PPM_o8": "#F7B924",
        "DigitMLP": "#636EFA",
        "DigitCNN": "#00CC96",
        "DigitGRU": "#EF553B",
        "DigitTransformer": "#AB63FA",
    }

    radar = Radar(
        init_opts=opts.InitOpts(
            width="1160px",
            height="800px",
            theme=ThemeType.DARK,
            bg_color="#111111",
            animation_opts=opts.AnimationOpts(animation=False),
        )
    )
    radar.add_schema(
        schema=indicators,
        shape="polygon",
        center=["50%", "54%"],
        radius="62%",
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.12),
        ),
        textstyle_opts=opts.TextStyleOpts(color="#E2E8F0", font_size=13),
    )

    for m_name in ("PPM_o8", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"):
        if m_name not in radar_scores:
            continue
        sc = radar_scores[m_name]
        vec = [
            round(float(sc["noise_calibration"]), 2),
            round(float(sc["periodic_memory"]), 2),
            round(float(sc["multi_digit_counting"]), 2),
            round(float(sc["cf_arithmetic"]), 2),
            round(float(sc["throughput"]), 2),
        ]
        radar.add(
            series_name=m_name,
            data=[vec],
            color=colors.get(m_name, "#19D3F3"),
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
        legend_opts=opts.LegendOpts(
            pos_bottom="14px",
            pos_left="center",
            textstyle_opts=opts.TextStyleOpts(color="#F8FAFC", font_size=14),
        ),
    )
    radar.render(str(html_path))

    # Убираем стандартные отступы body, чтобы скриншот точно совпадал с тёмным фоном графика
    html_text = html_path.read_text(encoding="utf-8")
    html_text = html_text.replace(
        "<body >",
        "<body style='margin:0; padding:0; background:#111111; overflow:hidden;'>",
    )
    html_path.write_text(html_text, encoding="utf-8")

    subprocess.run(
        [
            "chromium",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--virtual-time-budget=4000",
            "--window-size=1160,800",
            f"--screenshot={png_path}",
            f"file://{html_path.resolve()}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return html_path, png_path


def plot_exp2_embedding_pca_preview(
    pca_data: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит 2D PCA эмбеддингов цифр DigitTransformer: хаос на true_pi vs геометрия на e_cf и champernowne."""
    ordered = [d for d in ("true_pi", "e_cf", "champernowne") if d in pca_data]
    titles = []
    for ds_name in ordered:
        var_pct = pca_data[ds_name]["explained_variance_sum"] * 100.0
        role = "Контроль (H0 шум)" if ds_name == "true_pi" else f"Арифметика ({ds_name})"
        titles.append(f"{role} — PC1+PC2: {var_pct:.1f}% дисперсии")

    fig = make_subplots(rows=1, cols=len(ordered), subplot_titles=titles, horizontal_spacing=0.07)

    digit_palette = [
        "#636EFA", "#19D3F3", "#00CC96", "#B6E880", "#FECB52",
        "#FFA15A", "#EF553B", "#FF6692", "#AB63FA", "#FFFFFF",
    ]
    for col_idx, ds_name in enumerate(ordered, start=1):
        coords = np.asarray(pca_data[ds_name]["coords"], dtype=np.float64)
        digits_coords = coords[:10]

        # Соединяем цифры 0..9 тонкой линией порядка, чтобы сразу видеть геометрию счёта
        fig.add_trace(
            go.Scatter(
                x=digits_coords[:, 0].tolist(),
                y=digits_coords[:, 1].tolist(),
                mode="lines",
                line={"color": "rgba(148, 163, 184, 0.45)", "width": 1.5, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=digits_coords[:, 0].tolist(),
                y=digits_coords[:, 1].tolist(),
                mode="markers+text",
                text=[str(d) for d in range(10)],
                textposition="top center",
                textfont={"size": 14, "color": "#F8FAFC"},
                marker={
                    "size": 14,
                    "color": digit_palette,
                    "line": {"width": 1.5, "color": "#0F172A"},
                },
                name=f"{ds_name}: цифры 0..9",
                showlegend=False,
                hovertemplate="Цифра %{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        # Для e_cf отдельно отмечаем токен-разделитель 10 (';')
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

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 2 (Предрентген для Exp 5): 2D PCA эмбеддингов цифр DigitTransformer (token_emb.weight)",
        height=540,
        width=1580,
        margin={"l": 70, "r": 50, "t": 90, "b": 70},
        legend={"orientation": "h", "y": -0.14, "x": 0.5, "xanchor": "center"},
    )
    return save_plotly_figure(fig, summary_dir, "exp2_embedding_pca_preview")


def plot_exp3_bases_normalized_bits(
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Сравнение нормированного битрейта bits_per_symbol / log2(K) по системам счисления (Base-2, 10, 16)."""
    models_order = ["PPM_o8", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    bases_order = [
        ("Base-2", "Base-2 (K=2, окно 64 бит)", "#19D3F3"),
        ("Base-10", "Base-10 (K=10, окно ≈213 бит)", "#636EFA"),
        ("Base-16", "Base-16 BBP (K=16, окно 256 бит)", "#AB63FA"),
    ]

    fig = go.Figure()
    all_vals: list[float] = []
    for base_key, label, color in bases_order:
        sub = df[df["base_label"] == base_key].set_index("model").reindex(models_order)
        vals = sub["normalized_bits"].astype(float).tolist()
        all_vals.extend(vals)
        fig.add_trace(
            go.Bar(
                name=label,
                x=models_order,
                y=vals,
                marker_color=color,
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
                textfont={"size": 11},
            )
        )

    y_max = max(1.18, max(all_vals) + 0.04) if all_vals else 1.18
    y_min = min(0.95, min(all_vals) - 0.02) if all_vals else 0.95

    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#EF553B",
        line_width=2,
    )
    fig.add_annotation(
        x="DigitTransformer",
        y=1.032,  # вынес повыше: иначе наезжало на цифры над столбцами DigitTransformer
        xanchor="right",
        yanchor="bottom",
        text="↓ Потолок Шеннона (1.0000 = случайный шум)",
        showarrow=False,
        font={"color": "#EF553B", "size": 13},
    )

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 3 (BBP vs Base-2 / Base-10): Нормированный битрейт bits_per_symbol / log₂(K)",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        height=640,
        width=1540,
        xaxis={"title": "Архитектура / Модель"},
        yaxis={"title": "Нормированный битрейт (Efficiency Ratio)", "range": [y_min, y_max]},
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
        margin={"l": 85, "r": 50, "t": 90, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp3_bases_normalized_bits")


def plot_exp3_bases_zscore_ci(
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Forest-plot отклонений точности (Z-score в σ с 95% ДИ Уилсона) относительно случайного 1/K."""
    models_order = ["Markov_k1", "LightGBM", "DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"]
    bases_info = [
        ("Base-2", "Base-2 (K=2, 1/K = 50.00%)", "#19D3F3"),
        ("Base-10", "Base-10 (K=10, 1/K = 10.00%)", "#636EFA"),
        ("Base-16", "Base-16 BBP (K=16, 1/K = 6.25%)", "#AB63FA"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[b[1] for b in bases_info],
        horizontal_spacing=0.06,
    )

    max_abs_z = 3.2
    for col_idx, (base_key, _, color) in enumerate(bases_info, start=1):
        sub = df[df["base_label"] == base_key].set_index("model").reindex(models_order)
        z_vals = sub["z_score"].astype(float).tolist()
        err_hi = (sub["z_ci_high"].astype(float) - sub["z_score"].astype(float)).clip(lower=0.0).tolist()
        err_lo = (sub["z_score"].astype(float) - sub["z_ci_low"].astype(float)).clip(lower=0.0).tolist()
        max_abs_z = max(max_abs_z, max(abs(z) for z in sub["z_ci_high"].tolist() + sub["z_ci_low"].tolist()) + 0.6)

        # Коридор 95% ([-1.96σ, +1.96σ]) для H0: случайное угадывание
        fig.add_vrect(
            x0=-1.96,
            x1=1.96,
            fillcolor="rgba(0, 204, 150, 0.10)",
            line_width=0,
            row=1,
            col=col_idx,
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=col_idx)
        fig.add_vline(x=-1.96, line_dash="dot", line_color="#00CC96", opacity=0.65, row=1, col=col_idx)
        fig.add_vline(x=1.96, line_dash="dot", line_color="#00CC96", opacity=0.65, row=1, col=col_idx)

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
    fig.update_layout(
        template="plotly_dark",
        title=(
            "Эксперимент 3 (Forest-Plot 95% Wilson CI): Статистическая неотличимость авторегрессии π "
            "от шума (зелёная зона = ±1.96σ, p > 0.05)"
        ),
        height=600,
        width=1580,
        margin={"l": 145, "r": 45, "t": 100, "b": 70},
    )
    return save_plotly_figure(fig, summary_dir, "exp3_bases_zscore_ci")


def plot_exp4_window_phase_transition(
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит сравнение масштабирования окна L={32,64,128} на true_pi (H0) и на 1/97 (P=96)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(
            "true_pi (H0: белый шум) — плато ~10% на всех окнах L",
            "rational_1_97 (H1: период P=96) — локально решается уже при L=32",
        ),
        horizontal_spacing=0.07,
    )

    palette = {
        "DigitMLP": "#636EFA",
        "DigitGRU": "#EF553B",
        "DigitTransformer": "#AB63FA",
    }
    windows = [32, 64, 128]
    dashes = {"DigitMLP": "solid", "DigitGRU": "dash", "DigitTransformer": "dot"}

    for col_idx, (block_key, ds_key) in enumerate(
        [("Block_A_Window_H0", "true_pi"), ("Block_B_PhaseTransition_H1", "rational_1_97")],
        start=1,
    ):
        sub_b = df[(df["block"] == block_key) & (df["dataset"] == ds_key)]
        for m_name, color in palette.items():
            sub_m = sub_b[sub_b["model"] == m_name].sort_values("window_size")
            if sub_m.empty:
                continue
            xs = sub_m["window_size"].astype(int).tolist()
            ys = (sub_m["test_acc"].astype(float) * 100.0).tolist()
            fig.add_trace(
                go.Scatter(
                    name=m_name,
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    marker={"size": 10, "color": color},
                    line={"width": 3, "color": color, "dash": dashes[m_name]},
                    hovertemplate=f"{m_name}<br>L=%{{x}}<br>Acc=%{{y:.3f}}%<extra></extra>",
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )

        fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.45)", row=1, col=col_idx)
        fig.update_xaxes(
            title_text="Длина контекстного окна L (цифр)",
            tickvals=windows,
            range=[22, 138],
            row=1,
            col=col_idx,
        )

    fig.add_annotation(
        x=80,
        y=15.5,
        text="Диапазон H0: 9.73%–10.09%",
        showarrow=False,
        font={"color": "#CBD5E1", "size": 12},
        row=1,
        col=1,
    )
    fig.add_vline(x=96, line_dash="dot", line_color="#00CC96", line_width=2, row=1, col=2)
    fig.add_annotation(
        x=96,
        y=55,
        text="Период P = 96",
        showarrow=False,
        xshift=-48,
        font={"color": "#00CC96", "size": 12},
        row=1,
        col=2,
    )
    ppm_rows = df[(df["block"] == "Block_B_PhaseTransition_H1") & (df["model"] == "PPM_o8")]
    if not ppm_rows.empty:
        ppm_acc = float(ppm_rows["test_acc"].mean()) * 100.0
        fig.add_hline(y=ppm_acc, line_dash="dot", line_color="#F7B924", opacity=0.8, row=1, col=2)
        fig.add_annotation(
            x=64,
            y=91.5,
            text=f"Все модели + PPM_o8: {ppm_acc:.2f}% (уникальные 3–8-граммы внутри цикла 96)",
            showarrow=False,
            font={"color": "#F7B924", "size": 11.5},
            row=1,
            col=2,
        )

    fig.update_yaxes(title_text="Test Accuracy (%)", range=[-2, 110], row=1, col=1)
    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 4 (Блоки A и B): Инвариантность шума true_pi vs 1/97 (P=96)",
        height=640,
        width=1580,
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
        margin={"l": 80, "r": 50, "t": 95, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_window_phase_transition")


def plot_exp4_capacity_scaling(
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит закон масштабирования 50K -> 250K -> 1.5M по accuracy и bits_per_symbol (true_pi vs champernowne)."""
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

    palette = {
        "DigitMLP": "#636EFA",
        "DigitCNN": "#00CC96",
        "DigitGRU": "#EF553B",
        "DigitTransformer": "#AB63FA",
    }

    for col_idx, ds_name in enumerate(("true_pi", "champernowne"), start=1):
        sub_d = sub[sub["dataset"] == ds_name]
        for m_name, color in palette.items():
            sub_m = sub_d[sub_d["model"] == m_name].sort_values("params_count")
            if sub_m.empty:
                continue
            xs = sub_m["params_count"].astype(float).tolist()
            accs = (sub_m["test_acc"].astype(float) * 100.0).tolist()
            bits = sub_m["bits_per_symbol"].astype(float).tolist()

            fig.add_trace(
                go.Scatter(
                    name=m_name,
                    x=xs,
                    y=accs,
                    mode="lines+markers",
                    marker={"size": 10, "color": color},
                    line={"width": 3, "color": color},
                    hovertemplate=f"{m_name}<br>params=%{{x:,.0f}}<br>Acc=%{{y:.4f}}%<extra></extra>",
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    name=m_name,
                    x=xs,
                    y=bits,
                    mode="lines+markers",
                    marker={"size": 10, "color": color},
                    line={"width": 3, "color": color},
                    hovertemplate=f"{m_name}<br>params=%{{x:,.0f}}<br>bits=%{{y:.6f}}<extra></extra>",
                    showlegend=False,
                ),
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

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 4 (Блок C): Масштабирование параметров (50K → 250K → 1.5M, L=64)",
        height=860,
        width=1580,
        legend={"orientation": "h", "y": -0.09, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
        margin={"l": 80, "r": 50, "t": 95, "b": 100},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_capacity_scaling")


def plot_exp4_memorization_scissors(
    mem_df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит «ножницы запоминания» за 200 эпох: Train зубрится до 100%, Test взрывается без обобщения."""
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
    series_acc = [
        ("Train (7,936 окон)", (mem_df["train_acc"] * 100.0).tolist(), "#00CC96", True),
        ("Val (936 окон)", (mem_df["val_acc"] * 100.0).tolist(), "#FFA15A", True),
        ("Test (936 окон)", (mem_df["test_acc"] * 100.0).tolist(), "#EF553B", True),
    ]
    series_bits = [
        ("Train (bits/symbol)", mem_df["train_bits"].tolist(), "#00CC96", False),
        ("Val (bits/symbol)", mem_df["val_bits"].tolist(), "#FFA15A", False),
        ("Test (bits/symbol)", mem_df["test_bits"].tolist(), "#EF553B", False),
    ]

    for name, ys, color, show_leg in series_acc:
        fig.add_trace(
            go.Scatter(name=name, x=epochs, y=ys, mode="lines", line={"color": color, "width": 2.5}, showlegend=show_leg),
            row=1,
            col=1,
        )
    for name, ys, color, _ in series_bits:
        fig.add_trace(
            go.Scatter(name=name, x=epochs, y=ys, mode="lines", line={"color": color, "width": 2.5}, showlegend=False),
            row=1,
            col=2,
        )

    fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=1)
    fig.add_hline(y=3.321928, line_dash="dash", line_color="rgba(255,255,255,0.6)", row=1, col=2)
    fig.add_annotation(
        x=135,
        y=4.5,
        xanchor="center",
        text="Потолок Шеннона (3.3219 бит)",
        showarrow=False,
        font={"color": "#E2E8F0", "size": 12},
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Эпоха обучения (без EarlyStopping и dropout)", row=1, col=1)
    fig.update_xaxes(title_text="Эпоха обучения (без EarlyStopping и dropout)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 106], row=1, col=1)
    fig.update_yaxes(title_text="Битрейт (bits/symbol)", range=[0, 30], row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 4 (Блок D): Стресс-тест на 10,000 знаках π — «ножницы» Memorization vs Generalization",
        height=620,
        width=1580,
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
        margin={"l": 80, "r": 50, "t": 95, "b": 110},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_memorization_scissors")


def plot_exp4_compute_frontier(
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит Pareto-фронт: Throughput на RTX 3060 Ti vs Битрейт (log) на алгоритмических задачах."""
    sub = df[(df["hypothesis_group"] == "H1") & (df["params_count"].notna()) & (df["samples_per_sec"].notna())].copy()
    fig = go.Figure()

    palette = {
        "DigitMLP": "#636EFA",
        "DigitCNN": "#00CC96",
        "DigitGRU": "#EF553B",
        "DigitTransformer": "#AB63FA",
    }
    symbols = {"champernowne": "circle", "rational_1_97": "diamond"}
    positions = ["top center", "bottom center", "top right", "bottom right"]
    idx_counter = 0

    for ds_name in ("champernowne", "rational_1_97"):
        sub_d = sub[sub["dataset"] == ds_name]
        for m_name, color in palette.items():
            sub_m = sub_d[sub_d["model"] == m_name]
            if sub_m.empty:
                continue
            labels = []
            sizes = []
            y_vals = []
            text_pos = []
            for _, row in sub_m.iterrows():
                p_k = float(row["params_count"]) / 1000.0
                p_lbl = f"{p_k/1000:.1f}M" if p_k >= 900 else f"{p_k:.0f}K"
                win_lbl = f"L{int(row['window_size'])}"
                labels.append(f"{m_name.replace('Digit', '')} {p_lbl} {win_lbl}")
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
                    marker={
                        "size": sizes,
                        "color": color,
                        "symbol": symbols.get(ds_name, "circle"),
                        "opacity": 0.88,
                        "line": {"width": 1.5, "color": "#0F172A"},
                    },
                    text=labels,
                    textposition=text_pos,
                    textfont={"size": 10},
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Датасет: {ds_name}<br>"
                        "Throughput: %{x:,.0f} samp/s<br>"
                        "Bits/Symbol: %{y:.6f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 4: Compute Frontier на RTX 3060 Ti (Throughput vs Битрейт; Парето-оптимум — внизу справа)",
        height=700,
        width=1620,
        xaxis={"title": "Пропускная способность инференса (samples/sec, лог. шкала)", "type": "log"},
        yaxis={"title": "Test Битрейт (bits/symbol, лог. шкала, меньше = лучше)", "type": "log", "range": [-4.7, -0.8]},
        legend={"orientation": "h", "y": -0.17, "x": 0.5, "xanchor": "center", "font": {"size": 11}},
        margin={"l": 85, "r": 50, "t": 90, "b": 135},
    )
    return save_plotly_figure(fig, summary_dir, "exp4_compute_frontier")


def plot_exp5_embedding_clock_pca(
    pca_data: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит 2D PCA проекцию эмбеддингов цифр 0..9 с векторами переходов 0 -> 1 -> ... -> 9."""
    ordered = [d for d in ("true_pi", "champernowne", "e_cf", "rational_1_97") if d in pca_data]
    role_names = {
        "true_pi": "true_pi (Контроль H0: изотропный шум)",
        "champernowne": "champernowne (H1: арифметика +1 и перенос)",
        "e_cf": "e_cf (H1: тройки (1, 2k, 1) + разделитель)",
        "rational_1_97": "rational_1_97 (H1: 96-шаговый цикл P=96)",
    }
    titles = []
    for ds_name in ordered:
        info = pca_data[ds_name]
        var2 = info["explained_variance_pc1_pc2"] * 100.0
        cos_adj = info["adjacent_cosine_0_to_8"]
        titles.append(f"{role_names.get(ds_name, ds_name)}<br>PC1+PC2={var2:.1f}% | cos(d,d+1)={cos_adj:+.3f}")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )

    digit_palette = [
        "#636EFA", "#19D3F3", "#00CC96", "#B6E880", "#FECB52",
        "#FFA15A", "#EF553B", "#FF6692", "#AB63FA", "#F8FAFC",
    ]
    positions_map = ["top center", "top right", "bottom right", "bottom center", "bottom left", "top left"]

    for idx, ds_name in enumerate(ordered):
        r = idx // 2 + 1
        c = idx % 2 + 1
        coords = np.asarray(pca_data[ds_name]["coords"], dtype=np.float64)
        digits_coords = coords[:10]

        # Линия переходов 0 -> 1 -> ... -> 9
        fig.add_trace(
            go.Scatter(
                x=digits_coords[:, 0].tolist(),
                y=digits_coords[:, 1].tolist(),
                mode="lines",
                line={"color": "rgba(148, 163, 184, 0.55)", "width": 2.0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )
        # Стрелка замыкания 9 -> 0 (перенос разряда) пунктиром
        fig.add_trace(
            go.Scatter(
                x=[float(digits_coords[9, 0]), float(digits_coords[0, 0])],
                y=[float(digits_coords[9, 1]), float(digits_coords[0, 1])],
                mode="lines",
                line={"color": "rgba(247, 185, 36, 0.65)", "width": 1.8, "dash": "dot"},
                name="Переход 9 → 0" if idx == 0 else None,
                showlegend=(idx == 0),
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )
        # Точки цифр 0..9 с аккуратными подписями (отталкивание от центра + от ближайшего соседа)
        cx, cy = float(digits_coords[:, 0].mean()), float(digits_coords[:, 1].mean())
        text_pos = []
        for d in range(10):
            dx, dy = float(digits_coords[d, 0]) - cx, float(digits_coords[d, 1]) - cy
            # Если рядом есть другая цифра ближе 1.2 ед., отталкиваем подпись в противоположную сторону
            dists = [
                (math.hypot(digits_coords[d, 0] - digits_coords[j, 0], digits_coords[d, 1] - digits_coords[j, 1]), j)
                for j in range(10)
                if j != d
            ]
            min_dist, nearest_j = min(dists)
            if min_dist < 1.4:
                dx = float(digits_coords[d, 0] - digits_coords[nearest_j, 0])
                dy = float(digits_coords[d, 1] - digits_coords[nearest_j, 1])
            horiz = "right" if dx >= 0 else "left"
            vert = "top" if dy >= 0 else "bottom"
            text_pos.append(f"{vert} {horiz}")

        fig.add_trace(
            go.Scatter(
                x=digits_coords[:, 0].tolist(),
                y=digits_coords[:, 1].tolist(),
                mode="markers+text",
                text=[str(d) for d in range(10)],
                textposition=text_pos,
                textfont={"size": 13, "color": "#F8FAFC"},
                marker={
                    "size": 14,
                    "color": digit_palette,
                    "line": {"width": 1.5, "color": "#0F172A"},
                },
                showlegend=False,
                hovertemplate="Цифра %{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        # Направленные наконечники вдоль переходов d -> d+1
        axis_suffix = "" if idx == 0 else str(idx + 1)
        for d in range(9):
            x0, y0 = float(digits_coords[d, 0]), float(digits_coords[d, 1])
            x1, y1 = float(digits_coords[d + 1, 0]), float(digits_coords[d + 1, 1])
            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref=f"x{axis_suffix}",
                yref=f"y{axis_suffix}",
                axref=f"x{axis_suffix}",
                ayref=f"y{axis_suffix}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=1.4,
                arrowcolor="rgba(226, 232, 240, 0.55)",
                standoff=8,
                startstandoff=8,
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
                    showlegend=True,
                ),
                row=r,
                col=c,
            )

        # Запас по осям 22%, чтобы крайние подписи не обрезались
        all_x = coords[:, 0]
        all_y = coords[:, 1]
        x_pad = max(0.15, 0.25 * float(all_x.max() - all_x.min()))
        y_pad = max(0.15, 0.25 * float(all_y.max() - all_y.min()))
        fig.update_xaxes(
            title_text="PC1",
            range=[float(all_x.min()) - x_pad, float(all_x.max()) + x_pad],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.18)",
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text="PC2",
            range=[float(all_y.min()) - y_pad, float(all_y.max()) + y_pad],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.18)",
            row=r,
            col=c,
        )

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 5 (Модуль 1): Геометрия эмбеддингов (2D PCA) — «арифметический циферблат» против шума true_pi",
        height=940,
        width=1540,
        margin={"l": 70, "r": 50, "t": 110, "b": 85},
        legend={"orientation": "h", "y": -0.06, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_embedding_clock_pca")


def plot_exp5_head_ablation_heatmap(
    ablation_df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит тепловую карту Causal Head Ablation (Слои x Головы) по режимам и позициям."""
    has_fine_champ = not ablation_df[
        (ablation_df["dataset"] == "champernowne") & (ablation_df["subset"] == "carry_digit")
    ].empty
    slices = [
        (
            ("champernowne", "carry_digit", "champernowne: цифра переноса (199→200)")
            if has_fine_champ
            else ("champernowne", "carry_over", "champernowne: перенос (199→200)")
        ),
        (
            ("champernowne", "inc_digit", "champernowne: шаг +1 (123→124)")
            if has_fine_champ
            else ("champernowne", "non_carry", "champernowne: обычный шаг (123→124)")
        ),
        ("e_cf", "even_2k", "e_cf: чётные позиции 2k"),
        ("e_cf", "const_1_sep", "e_cf: маркеры (1, ;)"),
        ("rational_1_97", "all", "rational_1_97 (L=128, P=96)"),
        ("true_pi", "all", "true_pi (Контроль H0)"),
    ]
    active_slices = [
        s for s in slices if not ablation_df[(ablation_df["dataset"] == s[0]) & (ablation_df["subset"] == s[1])].empty
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[s[2] for s in active_slices],
        horizontal_spacing=0.07,
        vertical_spacing=0.18,
    )

    active_mask = pd.Series(False, index=ablation_df.index)
    for ds_n, sub_n, _ in active_slices:
        active_mask |= (ablation_df["dataset"] == ds_n) & (ablation_df["subset"] == sub_n)
    max_drop = max(1.0, float((ablation_df.loc[active_mask, "delta_acc_pp"].abs()).max()))

    for idx, (ds_name, subset, _) in enumerate(active_slices):
        r = idx // 3 + 1
        c = idx % 3 + 1
        sub = ablation_df[(ablation_df["dataset"] == ds_name) & (ablation_df["subset"] == subset)]
        n_layers = int(sub["layer"].max()) + 1
        n_heads = int(sub["head"].max()) + 1

        z_mat = np.zeros((n_layers, n_heads), dtype=np.float64)
        text_mat = [["" for _ in range(n_heads)] for _ in range(n_layers)]
        for _, row in sub.iterrows():
            l_idx, h_idx = int(row["layer"]), int(row["head"])
            drop_pp = float(-row["delta_acc_pp"])  # положительное число = падение точности при абляции
            d_bits = float(row["delta_bits"])
            z_mat[l_idx, h_idx] = drop_pp
            text_mat[l_idx][h_idx] = f"-{drop_pp:.1f}%\n(+{d_bits:.2f}b)" if drop_pp >= 0.05 else f"{row['delta_acc_pp']:+.1f}%\n({d_bits:+.2f}b)"

        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=[f"H{h}" for h in range(n_heads)],
                y=[f"Layer {l}" for l in range(n_layers)],
                text=text_mat,
                texttemplate="%{text}",
                textfont={"size": 11},
                colorscale="Inferno",
                zmin=0.0,
                zmax=max_drop,
                showscale=(idx == 2),
                colorbar={"title": "Падение Acc (п.п.)", "len": 0.85} if idx == 2 else None,
                hovertemplate=(
                    f"{ds_name} [{subset}]<br>"
                    "%{y}, %{x}<br>"
                    "Падение Acc: %{z:.2f} п.п.<extra></extra>"
                ),
            ),
            row=r,
            col=c,
        )
        fig.update_yaxes(autorange="reversed", row=r, col=c)

    fig.update_layout(
        template="plotly_dark",
        title="Эксперимент 5 (Модуль 1): Causal Head Ablation — специализация голов переноса и индукции (ΔAcc / ΔBits)",
        height=760,
        width=1620,
        margin={"l": 80, "r": 95, "t": 100, "b": 60},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_head_ablation_heatmap")


def plot_exp5_linear_probing_bars(
    probing_df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Строит сравнительную диаграмму Linear Probing (Trained vs Untrained Random) для GRU и Transformer."""
    task_order = [
        ("e_cf_phase", "e_cf: фаза<br>(1, 2k, 1)<br>[3 кл., Acc]"),
        ("champ_pos", "champernowne:<br>позиция в числе<br>[6 кл., Acc]"),
        ("champ_carry", "champernowne:<br>перенос (=9)<br>[Bal. Acc]"),
        ("rat97_phase", "rational 1/97:<br>фаза t mod 96<br>[96 кл., Acc]"),
    ]
    labels = [lbl for _, lbl in task_order]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "DigitGRU (скрытое состояние последнего слоя)",
            "DigitTransformer (скрытое состояние последнего слоя)",
        ),
        horizontal_spacing=0.08,
    )

    for col_idx, arch_name in enumerate(("DigitGRU", "DigitTransformer"), start=1):
        sub_arch = probing_df[probing_df["model"] == arch_name]
        trained_vals = []
        untrained_vals = []
        chance_vals = []
        for task_key, _ in task_order:
            tr_row = sub_arch[(sub_arch["task"] == task_key) & (sub_arch["state"] == "Trained")]
            un_row = sub_arch[(sub_arch["task"] == task_key) & (sub_arch["state"] == "Untrained")]
            metric_col = "balanced_acc" if task_key == "champ_carry" else "probe_acc"
            trained_vals.append(float(tr_row[metric_col].iloc[0]) * 100.0 if not tr_row.empty else 0.0)
            untrained_vals.append(float(un_row[metric_col].iloc[0]) * 100.0 if not un_row.empty else 0.0)
            chance_vals.append(float(tr_row["chance_baseline"].iloc[0]) * 100.0 if not tr_row.empty else 0.0)

        fig.add_trace(
            go.Bar(
                name="Trained (обученная модель)",
                x=labels,
                y=trained_vals,
                marker_color="#00CC96",
                text=[f"{v:.1f}%" for v in trained_vals],
                textposition="outside",
                textfont={"size": 12},
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Bar(
                name="Untrained (случайные веса)",
                x=labels,
                y=untrained_vals,
                marker_color="#636EFA",
                text=[f"{v:.1f}%" for v in untrained_vals],
                textposition="outside",
                textfont={"size": 12},
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                name="Случайное угадывание (Chance)",
                x=labels,
                y=chance_vals,
                mode="markers+lines",
                marker={"size": 9, "color": "#F7B924", "symbol": "diamond"},
                line={"color": "#F7B924", "dash": "dash", "width": 2},
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(tickangle=0, tickfont={"size": 11.5}, row=1, col=col_idx)
        fig.update_yaxes(title_text="Точность зонда (%)", range=[0, 116], row=1, col=col_idx)

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Эксперимент 5 (Модуль 1): Linear Probing скрытых состояний (Trained vs Untrained Random)",
        height=680,
        width=1580,
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center", "font": {"size": 13}},
        margin={"l": 75, "r": 45, "t": 95, "b": 150},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_linear_probing_bars")


def plot_exp5_crypto_backdoor_xray(
    audit_data: dict[str, Any],
    attn_profile: dict[str, Any],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Двухпанельный рентген крипто-закладки csprng_backdoor: слепота классики vs Attention пики t-24, t-7."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Детекция крипто-закладки: слепота PPM_o8/статистики vs нейросети",
            "Профиль внимания DigitTransformer на триггере (пики на t-24 и t-7)",
        ),
        horizontal_spacing=0.09,
    )

    models_data = audit_data["models"]
    names = [m["label"] for m in models_data]
    overall_vals = [float(m["overall_acc"]) * 100.0 for m in models_data]
    targeted_vals = [float(m["targeted_acc"]) * 100.0 for m in models_data]

    fig.add_trace(
        go.Bar(
            name="Overall Test Acc (%)",
            x=names,
            y=overall_vals,
            marker_color="#636EFA",
            text=[f"{v:.2f}%" for v in overall_vals],
            textposition="outside",
            textfont={"size": 11.5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Targeted Acc на закладке (mask==1, %)",
            x=names,
            y=targeted_vals,
            marker_color="#00CC96",
            text=[f"{v:.1f}%" for v in targeted_vals],
            textposition="outside",
            textfont={"size": 11.5},
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.55)", row=1, col=1)

    stat_box = audit_data["classical_stats"]
    fig.add_annotation(
        x=0.02,
        y=0.96,
        xref="x domain",
        yref="y domain",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(148, 163, 184, 0.5)",
        borderwidth=1,
        font={"size": 11.5, "color": "#E2E8F0"},
        text=(
            "<b>Классический аудит слеп (N=1M):</b><br>"
            f"• χ² p-value = {stat_box['chi2_p_value']:.4f} (&gt; 0.05)<br>"
            f"• KS p-value = {stat_box['ks_p_value']:.4f} (&gt; 0.05)<br>"
            f"• ACF max = {stat_box['acf_max_abs']:.5f} (&lt; 0.005)<br>"
            f"• LZMA = {stat_box['lzma_bits_per_digit']:.4f} бит/цифру"
        ),
        row=1,
        col=1,
    )

    lags = attn_profile["lags"]  # [-64, ..., -1]
    trig_mean = attn_profile["trigger_mean_attn"]
    nontrig_mean = attn_profile["nontrigger_mean_attn"]
    focus_label = attn_profile["focus_head_label"]

    fig.add_trace(
        go.Scatter(
            name=f"Триггер mask==1 ({focus_label})",
            x=lags,
            y=trig_mean,
            mode="lines+markers",
            line={"color": "#EF553B", "width": 2.8},
            marker={"size": 5},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name="Фоновый шум mask==0",
            x=lags,
            y=nontrig_mean,
            mode="lines",
            line={"color": "#94A3B8", "width": 2.0, "dash": "dot"},
        ),
        row=1,
        col=2,
    )

    max_y = max(max(trig_mean), max(nontrig_mean), 0.1)
    idx_24 = lags.index(-24)
    idx_7 = lags.index(-7)
    y_24 = float(trig_mean[idx_24])
    y_7 = float(trig_mean[idx_7])

    fig.add_vline(x=-24, line_dash="dash", line_color="rgba(247, 185, 36, 0.55)", row=1, col=2)
    fig.add_vline(x=-7, line_dash="dash", line_color="rgba(247, 185, 36, 0.55)", row=1, col=2)

    fig.add_annotation(
        x=-24,
        y=y_24,
        text=f"Скрытый лаг t-24<br>(вес {y_24:.3f})",
        showarrow=True,
        arrowhead=2,
        ax=-55,
        ay=-15,
        xanchor="right",
        bgcolor="rgba(15, 23, 42, 0.88)",
        bordercolor="#F7B924",
        borderwidth=1,
        font={"color": "#F7B924", "size": 11.5},
        arrowcolor="#F7B924",
        row=1,
        col=2,
    )
    fig.add_annotation(
        x=-7,
        y=y_7,
        text=f"Скрытый лаг t-7<br>(вес {y_7:.3f})",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-35,
        xanchor="right",
        bgcolor="rgba(15, 23, 42, 0.88)",
        bordercolor="#F7B924",
        borderwidth=1,
        font={"color": "#F7B924", "size": 11.5},
        arrowcolor="#F7B924",
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 118], row=1, col=1)
    fig.update_yaxes(title_text="Средний вес внимания из t-1", range=[0, max_y * 1.28], row=1, col=2)
    fig.update_xaxes(title_text="Позиция ключа относительно цели t (лаги -64 .. -1)", range=[-65, 1], row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Эксперимент 5 (Модуль 2): Крипто-аудитор ГПСЧ — обнаружение скрытой закладки (t-24, t-7) в os.urandom",
        height=640,
        width=1620,
        legend={"orientation": "h", "y": -0.17, "x": 0.5, "xanchor": "center", "font": {"size": 12}},
        margin={"l": 75, "r": 50, "t": 95, "b": 115},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_crypto_backdoor_xray")
