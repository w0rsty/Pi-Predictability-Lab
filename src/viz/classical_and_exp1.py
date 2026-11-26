# Графики Exp 1 (MLP, CNN, GRU, Transformer) и PPM-C бенчмарка
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import (
    ARCH_COLORS,
    EXP1_DS_COLORS,
    HEAD_COLORS,
    MODEL_COLORS,
    apply_lab_layout,
    legend_bottom,
    save_plotly_figure,
)


def plot_exp1_training_curves(
    arch: str,
    runs_by_dataset: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Кривые Val Loss, Acc, Targeted CE и Prediction Entropy по эпохам."""
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
    for ds_name, res in runs_by_dataset.items():
        hist = res["history"]
        epochs = [h["epoch"] for h in hist]
        color = EXP1_DS_COLORS.get(ds_name, "#AB63FA")
        traces = [
            (1, 1, [h["val_loss"] for h in hist], epochs, True),
            (1, 2, [h["val_overall_acc"] * 100.0 for h in hist], epochs, False),
            (2, 2, [h["val_pred_entropy"] for h in hist], epochs, False),
        ]
        tgt_ce = [h["val_targeted_ce"] for h in hist if h["val_targeted_ce"] is not None]
        if tgt_ce:
            traces.append((2, 1, tgt_ce, epochs[: len(tgt_ce)], False))
        for r, c, ys, xs, show_leg in traces:
            fig.add_trace(
                go.Scatter(name=ds_name, x=xs, y=ys, mode="lines+markers", line={"color": color}, showlegend=show_leg),
                row=r,
                col=c,
            )

    for r, c, y_val in ((1, 1, 2.302585), (1, 2, 10.0), (2, 2, 2.302585)):
        fig.add_hline(y=y_val, line_dash="dash", line_color="white", opacity=0.6, row=r, col=c)

    apply_lab_layout(
        fig,
        f"Эксперимент 1 ({arch.upper()}): Кривые обучения на True Pi vs 3 уровнях Hacked Pi",
        height=760,
        width=1360,
    )
    return save_plotly_figure(fig, summary_dir, f"exp1_{arch}_training_curves")


def plot_exp1_test_xray(
    arch: str,
    runs_by_dataset: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Итоговые Test-метрики и уверенность модели на триггерах против фона."""
    ds_names = list(runs_by_dataset.keys())
    overall_ce = [runs_by_dataset[d]["test_metrics"]["cross_entropy"] for d in ds_names]
    targeted_ce = [runs_by_dataset[d]["test_metrics"]["targeted_ce"] or 0.0 for d in ds_names]
    tgt_conf = [(runs_by_dataset[d]["test_metrics"]["targeted_confidence"] or 0.0) * 100.0 for d in ds_names]
    unm_conf = [(runs_by_dataset[d]["test_metrics"]["unmasked_confidence"] or 0.0) * 100.0 for d in ds_names]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Test Loss: общий vs на позициях mask == 1",
            "Средняя уверенность на правильной цифре: триггер vs фон",
        ),
        horizontal_spacing=0.08,
    )
    bars = [
        (1, "Overall CE", overall_ce, "{:.3f}"),
        (1, "Targeted CE (mask=1)", targeted_ce, "{:.3f}"),
        (2, "На триггере (mask=1)", tgt_conf, "{:.1f}%"),
        (2, "На фоне true_pi (mask=0)", unm_conf, "{:.1f}%"),
    ]
    for col, label, ys, fmt in bars:
        fig.add_trace(
            go.Bar(name=label, x=ds_names, y=ys, text=[fmt.format(v) for v in ys], textposition="outside"),
            row=1,
            col=col,
        )

    fig.add_hline(y=2.302585, line_dash="dash", line_color="white", opacity=0.6, row=1, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.6, row=1, col=2)
    apply_lab_layout(
        fig,
        f"Эксперимент 1 ({arch.upper()}): Рентген обнаружения скрытых правил на Test",
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
    """Норма скрытого состояния ||h_t||_2 и P_t(target_digit) внутри окна."""
    ds_list = [k for k in ("hacked_pi_tier1", "hacked_pi_tier2") if k in probes]
    if not ds_list:
        return None

    titles = []
    for ds_name in ds_list:
        td = probes[ds_name]["target_digit"]
        titles.extend([
            f"{ds_name}: норма скрытого состояния ||h_t||_2 (t=0..63)",
            f"{ds_name}: вероятность следующей цифры P_t({td})",
        ])

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
        for t in (t for t, flg in enumerate(p["mask_next_target"]) if flg == 1):
            fig.add_vline(x=t, line_dash="dot", line_color="#EF553B", opacity=0.65, row=row_idx, col=1)
            fig.add_vline(x=t, line_dash="dot", line_color="#EF553B", opacity=0.65, row=row_idx, col=2)
        fig.add_hline(y=10.0, line_dash="dash", line_color="gray", opacity=0.6, row=row_idx, col=2)
        fig.update_yaxes(title_text="||h_t||_2", row=row_idx, col=1)
        fig.update_yaxes(title_text="P (%)", range=[0, 108], row=row_idx, col=2)
        fig.update_xaxes(title_text="Шаг в окне t", row=row_idx, col=1)
        fig.update_xaxes(title_text="Шаг в окне t", row=row_idx, col=2)

    apply_lab_layout(
        fig,
        "GRU Probe: динамика скрытого состояния и вероятности на триггерах (красный пунктир: mask_next == 1)",
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
    """Сравнение архитектур MLP vs 1D-CNN vs GRU vs Transformer по Targeted CE, уверенности и PPL."""
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
    present_archs = list(dict.fromkeys(combined_df["arch"].tolist()))
    for a in present_archs:
        sub_all = combined_df[combined_df["arch"] == a].set_index("dataset")
        sub_hacked = sub_all.reindex(hacked_ds)
        sub_full = sub_all.reindex(all_datasets)
        color = ARCH_COLORS.get(a, "#FFA15A")

        specs = [
            (1, hacked_ds, sub_hacked["test_targeted_ce"].fillna(0.0).tolist(), "{:.3f}", True),
            (2, hacked_ds, (sub_hacked["targeted_confidence"].fillna(0.0) * 100.0).tolist(), "{:.1f}%", False),
            (3, all_datasets, sub_full["test_perplexity"].fillna(10.0).tolist(), "{:.2f}", False),
        ]
        for col, xs, ys, fmt, show_leg in specs:
            fig.add_trace(
                go.Bar(
                    name=a.upper(),
                    x=xs,
                    y=ys,
                    marker_color=color,
                    text=[fmt.format(v) for v in ys],
                    textposition="outside",
                    showlegend=show_leg,
                ),
                row=1,
                col=col,
            )

    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=3)
    archs_label = " vs ".join(a.upper() for a in present_archs)
    apply_lab_layout(
        fig,
        f"Эксперимент 1: Битва архитектур ({archs_label}, ~250K параметров, L=64)",
        height=560,
        width=1620,
        barmode="group",
    )
    fig.update_yaxes(title_text="Targeted CE (наты)", row=1, col=1)
    fig.update_yaxes(title_text="Уверенность (%)", range=[0, 114], row=1, col=2)
    fig.update_yaxes(title_text="Perplexity", range=[0, 11.4], row=1, col=3)
    return save_plotly_figure(fig, summary_dir, f"exp1_arch_battle_{'_'.join(present_archs)}")


def plot_attention_maps(
    attn_probes: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path] | None:
    """Карты внимания Transformer для true_pi и hacked_pi (Tier 1 и Tier 2)."""
    ordered = [d for d in ("true_pi", "hacked_pi_tier1", "hacked_pi_tier2") if d in attn_probes]
    if not ordered:
        return None

    t1 = [f"{d}: L×L (Слой {attn_probes[d]['focus_layer']}, Голова {attn_probes[d]['focus_head']})" for d in ordered]
    t2 = [f"{d}: фокус t=63 (Слой {attn_probes[d]['focus_layer']}, все головы)" for d in ordered]
    fig = make_subplots(rows=2, cols=len(ordered), subplot_titles=t1 + t2, vertical_spacing=0.16, horizontal_spacing=0.07)

    for col_idx, ds_name in enumerate(ordered, start=1):
        probe = attn_probes[ds_name]
        attn = np.asarray(probe["attn_weights"], dtype=np.float32)  # (n_layers, n_heads, L, L)
        best_l, best_h = int(probe["focus_layer"]), int(probe["focus_head"])
        positions = list(range(attn.shape[-1]))
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

        for h_idx in range(attn.shape[1]):
            fig.add_trace(
                go.Scatter(
                    name=f"Head {h_idx}",
                    x=positions,
                    y=attn[best_l, h_idx, -1, :].tolist(),
                    mode="lines",
                    line={"color": HEAD_COLORS[h_idx % len(HEAD_COLORS)], "width": 2},
                    showlegend=(col_idx == 1),
                ),
                row=2,
                col=col_idx,
            )
        for key_pos in probe.get("expected_key_positions", []):
            fig.add_vline(x=key_pos, line_dash="dot", line_color="#EF553B", opacity=0.8, row=2, col=col_idx)
        fig.update_xaxes(title_text="Key позиция k (t=0..63)", row=2, col=col_idx)
        fig.update_yaxes(title_text="Вес внимания α", row=2, col=col_idx)

    apply_lab_layout(
        fig,
        "Эксперимент 1 (Transformer): Рентген карт внимания — True Pi против Hacked Pi (красный пунктир = триггеры)",
        height=860,
        width=1580,
        legend=legend_bottom(y=-0.07, font_size=12),
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
    for m_name in dict.fromkeys(comparison_df["model"].tolist()):
        sub = comparison_df[comparison_df["model"] == m_name].set_index("dataset")
        overall_bits = sub.reindex(datasets)["bits_per_symbol"].tolist()
        tgt_bits = sub.reindex(hacked_ds)["targeted_bits_per_symbol"].fillna(0.0).tolist()
        color = MODEL_COLORS.get(m_name, "#19D3F3")
        fig.add_trace(
            go.Bar(name=m_name, x=datasets, y=overall_bits, marker_color=color, text=[f"{v:.3f}" for v in overall_bits], textposition="outside"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(name=m_name, x=hacked_ds, y=tgt_bits, marker_color=color, text=[f"{v:.3f}" for v in tgt_bits], textposition="outside", showlegend=False),
            row=1,
            col=2,
        )

    fig.add_hline(y=shannon_limit_bits, line_dash="dash", line_color="white", opacity=0.6, row=1, col=1)
    # Байесовский предел на Tier 3 при p=0.5: -log2(0.55) ≈ 0.8625 бит/цифру
    fig.add_hline(y=0.862496, line_dash="dot", line_color="gray", opacity=0.6, row=1, col=2)
    apply_lab_layout(
        fig,
        "PPM-C (суффиксное дерево, order=8) против 4 нейросетей (~250K параметров)",
        height=560,
        width=1580,
        barmode="group",
    )
    fig.update_yaxes(title_text="Бит на символ (CE / ln 2)", range=[0, 4.25], row=1, col=1)
    fig.update_yaxes(title_text="Бит на символ на mask==1", range=[0, 2.75], row=1, col=2)
    return save_plotly_figure(fig, summary_dir, "ppm_vs_neural_bits")
