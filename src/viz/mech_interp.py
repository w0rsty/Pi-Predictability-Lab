# Построение графиков для Эксперимента 5 (PCA, абляция голов, probing и детектор закладок)
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import DIGIT_PALETTE, apply_lab_layout, legend_bottom, save_plotly_figure


def _compute_repelled_positions(digits_coords: np.ndarray) -> list[str]:
    cx, cy = float(digits_coords[:, 0].mean()), float(digits_coords[:, 1].mean())
    text_pos = []
    for d in range(10):
        dx, dy = float(digits_coords[d, 0]) - cx, float(digits_coords[d, 1]) - cy
        dists = [
            (math.hypot(digits_coords[d, 0] - digits_coords[j, 0], digits_coords[d, 1] - digits_coords[j, 1]), j)
            for j in range(10)
            if j != d
        ]
        min_dist, nearest_j = min(dists)
        if min_dist < 1.4:
            dx = float(digits_coords[d, 0] - digits_coords[nearest_j, 0])
            dy = float(digits_coords[d, 1] - digits_coords[nearest_j, 1])
        text_pos.append(f"{'top' if dy >= 0 else 'bottom'} {'right' if dx >= 0 else 'left'}")
    return text_pos


def _add_clock_pca_panel(
    fig: go.Figure,
    coords: np.ndarray,
    idx: int,
) -> None:
    r, c = idx // 2 + 1, idx % 2 + 1
    digits = coords[:10]
    fig.add_trace(
        go.Scatter(x=digits[:, 0].tolist(), y=digits[:, 1].tolist(), mode="lines", line={"color": "rgba(148, 163, 184, 0.55)", "width": 2.0}, showlegend=False, hoverinfo="skip"),
        row=r,
        col=c,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(digits[9, 0]), float(digits[0, 0])],
            y=[float(digits[9, 1]), float(digits[0, 1])],
            mode="lines",
            line={"color": "rgba(247, 185, 36, 0.65)", "width": 1.8, "dash": "dot"},
            name="Переход 9 → 0" if idx == 0 else None,
            showlegend=(idx == 0),
            hoverinfo="skip",
        ),
        row=r,
        col=c,
    )
    fig.add_trace(
        go.Scatter(
            x=digits[:, 0].tolist(),
            y=digits[:, 1].tolist(),
            mode="markers+text",
            text=[str(d) for d in range(10)],
            textposition=_compute_repelled_positions(digits),
            textfont={"size": 13, "color": "#F8FAFC"},
            marker={"size": 14, "color": DIGIT_PALETTE, "line": {"width": 1.5, "color": "#0F172A"}},
            showlegend=False,
            hovertemplate="Цифра %{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
        ),
        row=r,
        col=c,
    )
    axis_sfx = "" if idx == 0 else str(idx + 1)
    for d in range(9):
        fig.add_annotation(
            x=float(digits[d + 1, 0]), y=float(digits[d + 1, 1]),
            ax=float(digits[d, 0]), ay=float(digits[d, 1]),
            xref=f"x{axis_sfx}", yref=f"y{axis_sfx}",
            axref=f"x{axis_sfx}", ayref=f"y{axis_sfx}",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.4,
            arrowcolor="rgba(226, 232, 240, 0.55)", standoff=8, startstandoff=8,
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
    x_pad = max(0.15, 0.25 * float(coords[:, 0].max() - coords[:, 0].min()))
    y_pad = max(0.15, 0.25 * float(coords[:, 1].max() - coords[:, 1].min()))
    fig.update_xaxes(title_text="PC1", range=[float(coords[:, 0].min()) - x_pad, float(coords[:, 0].max()) + x_pad], zeroline=True, zerolinecolor="rgba(255,255,255,0.18)", row=r, col=c)
    fig.update_yaxes(title_text="PC2", range=[float(coords[:, 1].min()) - y_pad, float(coords[:, 1].max()) + y_pad], zeroline=True, zerolinecolor="rgba(255,255,255,0.18)", row=r, col=c)


def plot_exp5_embedding_clock_pca(
    pca_data: dict[str, dict[str, Any]],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """2D PCA проекция эмбеддингов цифр 0..9 с векторами переходов 0 -> 1 -> ... -> 9."""
    ordered = [d for d in ("true_pi", "champernowne", "e_cf", "rational_1_97") if d in pca_data]
    role_names = {
        "true_pi": "true_pi (Контроль H0: изотропный шум)",
        "champernowne": "champernowne (H1: арифметика +1 и перенос)",
        "e_cf": "e_cf (H1: тройки (1, 2k, 1) + разделитель)",
        "rational_1_97": "rational_1_97 (H1: 96-шаговый цикл P=96)",
    }
    titles = [
        f"{role_names.get(d, d)}<br>PC1+PC2={pca_data[d]['explained_variance_pc1_pc2'] * 100:.1f}% | cos(d,d+1)={pca_data[d]['adjacent_cosine_0_to_8']:+.3f}"
        for d in ordered
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.09, vertical_spacing=0.16)
    for idx, ds_name in enumerate(ordered):
        _add_clock_pca_panel(fig, np.asarray(pca_data[ds_name]["coords"], dtype=np.float64), idx)

    apply_lab_layout(
        fig,
        "Эксперимент 5 (Модуль 1): Геометрия эмбеддингов (2D PCA) на структурированных рядах и контроле true_pi",
        height=940,
        width=1540,
        margin={"l": 70, "r": 50, "t": 110, "b": 85},
        legend=legend_bottom(y=-0.06, font_size=13),
    )
    return save_plotly_figure(fig, summary_dir, "exp5_embedding_clock_pca")


def plot_exp5_head_ablation_heatmap(ablation_df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Тепловая карта Causal Head Ablation (Слои x Головы) по режимам и позициям."""
    has_fine = not ablation_df[(ablation_df["dataset"] == "champernowne") & (ablation_df["subset"] == "carry_digit")].empty
    slices = [
        ("champernowne", "carry_digit" if has_fine else "carry_over", "champernowne: перенос (199→200)"),
        ("champernowne", "inc_digit" if has_fine else "non_carry", "champernowne: шаг +1 (123→124)"),
        ("e_cf", "even_2k", "e_cf: чётные позиции 2k"),
        ("e_cf", "const_1_sep", "e_cf: маркеры (1, ;)"),
        ("rational_1_97", "all", "rational_1_97 (L=128, P=96)"),
        ("true_pi", "all", "true_pi (Контроль H0)"),
    ]
    active = [s for s in slices if not ablation_df[(ablation_df["dataset"] == s[0]) & (ablation_df["subset"] == s[1])].empty]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[s[2] for s in active], horizontal_spacing=0.07, vertical_spacing=0.18)

    active_mask = pd.Series(False, index=ablation_df.index)
    for ds_n, sub_n, _ in active:
        active_mask |= (ablation_df["dataset"] == ds_n) & (ablation_df["subset"] == sub_n)
    max_drop = max(1.0, float(ablation_df.loc[active_mask, "delta_acc_pp"].abs().max()))

    for idx, (ds_name, subset, _) in enumerate(active):
        r, c = idx // 3 + 1, idx % 3 + 1
        sub = ablation_df[(ablation_df["dataset"] == ds_name) & (ablation_df["subset"] == subset)]
        n_layers, n_heads = int(sub["layer"].max()) + 1, int(sub["head"].max()) + 1
        z_mat = np.zeros((n_layers, n_heads), dtype=np.float64)
        text_mat = [["" for _ in range(n_heads)] for _ in range(n_layers)]
        for _, row in sub.iterrows():
            l_i, h_i = int(row["layer"]), int(row["head"])
            drop_pp, d_bits = float(-row["delta_acc_pp"]), float(row["delta_bits"])
            z_mat[l_i, h_i] = drop_pp
            text_mat[l_i][h_i] = f"-{drop_pp:.1f}%\n(+{d_bits:.2f}b)" if drop_pp >= 0.05 else f"{row['delta_acc_pp']:+.1f}%\n({d_bits:+.2f}b)"

        fig.add_trace(
            go.Heatmap(
                z=z_mat, x=[f"H{h}" for h in range(n_heads)], y=[f"Layer {l}" for l in range(n_layers)],
                text=text_mat, texttemplate="%{text}", textfont={"size": 11}, colorscale="Inferno",
                zmin=0.0, zmax=max_drop, showscale=(idx == 2),
                colorbar={"title": "Падение Acc (п.п.)", "len": 0.85} if idx == 2 else None,
                hovertemplate=f"{ds_name} [{subset}]<br>%{{y}}, %{{x}}<br>Падение Acc: %{{z:.2f}} п.п.<extra></extra>",
            ),
            row=r,
            col=c,
        )
        fig.update_yaxes(autorange="reversed", row=r, col=c)

    apply_lab_layout(
        fig,
        "Эксперимент 5 (Модуль 1): Causal Head Ablation — специализация голов переноса и индукции (ΔAcc / ΔBits)",
        height=760,
        width=1620,
        margin={"l": 80, "r": 95, "t": 100, "b": 60},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_head_ablation_heatmap")


def plot_exp5_linear_probing_bars(probing_df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    """Сравнение Linear Probing скрытых состояний (Trained vs Untrained Random)."""
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
        subplot_titles=("DigitGRU (скрытое состояние последнего слоя)", "DigitTransformer (скрытое состояние последнего слоя)"),
        horizontal_spacing=0.08,
    )
    for col_idx, arch_name in enumerate(("DigitGRU", "DigitTransformer"), start=1):
        sub_arch = probing_df[probing_df["model"] == arch_name]
        tr_vals, un_vals, ch_vals = [], [], []
        for task_key, _ in task_order:
            metric_col = "balanced_acc" if task_key == "champ_carry" else "probe_acc"
            tr = sub_arch[(sub_arch["task"] == task_key) & (sub_arch["state"] == "Trained")]
            un = sub_arch[(sub_arch["task"] == task_key) & (sub_arch["state"] == "Untrained")]
            tr_vals.append(float(tr[metric_col].iloc[0]) * 100.0 if not tr.empty else 0.0)
            un_vals.append(float(un[metric_col].iloc[0]) * 100.0 if not un.empty else 0.0)
            ch_vals.append(float(tr["chance_baseline"].iloc[0]) * 100.0 if not tr.empty else 0.0)

        for name, ys, color in (("Trained (обученная модель)", tr_vals, "#00CC96"), ("Untrained (случайные веса)", un_vals, "#636EFA")):
            fig.add_trace(
                go.Bar(name=name, x=labels, y=ys, marker_color=color, text=[f"{v:.1f}%" for v in ys], textposition="outside", textfont={"size": 12}, showlegend=(col_idx == 1)),
                row=1,
                col=col_idx,
            )
        fig.add_trace(
            go.Scatter(name="Случайное угадывание (Chance)", x=labels, y=ch_vals, mode="markers+lines", marker={"size": 9, "color": "#F7B924", "symbol": "diamond"}, line={"color": "#F7B924", "dash": "dash", "width": 2}, showlegend=(col_idx == 1)),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(tickangle=0, tickfont={"size": 11.5}, row=1, col=col_idx)
        fig.update_yaxes(title_text="Точность зонда (%)", range=[0, 116], row=1, col=col_idx)

    apply_lab_layout(
        fig,
        "Эксперимент 5 (Модуль 1): Linear Probing скрытых состояний (Trained vs Untrained Random)",
        height=680,
        width=1580,
        barmode="group",
        legend=legend_bottom(y=-0.25, font_size=13),
        margin={"l": 75, "r": 45, "t": 95, "b": 150},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_linear_probing_bars")


def _add_crypto_audit_bars(fig: go.Figure, audit_data: dict[str, Any]) -> None:
    models = audit_data["models"]
    names = [m["label"] for m in models]
    overall = [float(m["overall_acc"]) * 100.0 for m in models]
    targeted = [float(m["targeted_acc"]) * 100.0 for m in models]
    fig.add_trace(go.Bar(name="Overall Test Acc (%)", x=names, y=overall, marker_color="#636EFA", text=[f"{v:.2f}%" for v in overall], textposition="outside", textfont={"size": 11.5}), row=1, col=1)
    fig.add_trace(go.Bar(name="Targeted Acc на закладке (mask==1, %)", x=names, y=targeted, marker_color="#00CC96", text=[f"{v:.1f}%" for v in targeted], textposition="outside", textfont={"size": 11.5}), row=1, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.55)", row=1, col=1)

    stat_box = audit_data["classical_stats"]
    fig.add_annotation(
        x=0.02, y=0.96, xref="x domain", yref="y domain", xanchor="left", yanchor="top", align="left", showarrow=False,
        bgcolor="rgba(15, 23, 42, 0.85)", bordercolor="rgba(148, 163, 184, 0.5)", borderwidth=1, font={"size": 11.5, "color": "#E2E8F0"},
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
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 118], row=1, col=1)


def _add_crypto_attn_profile(fig: go.Figure, attn_profile: dict[str, Any]) -> None:
    lags = attn_profile["lags"]
    trig = attn_profile["trigger_mean_attn"]
    nontrig = attn_profile["nontrigger_mean_attn"]
    fig.add_trace(go.Scatter(name=f"Триггер mask==1 ({attn_profile['focus_head_label']})", x=lags, y=trig, mode="lines+markers", line={"color": "#EF553B", "width": 2.8}, marker={"size": 5}), row=1, col=2)
    fig.add_trace(go.Scatter(name="Фоновый шум mask==0", x=lags, y=nontrig, mode="lines", line={"color": "#94A3B8", "width": 2.0, "dash": "dot"}), row=1, col=2)

    for lag, ay in ((-24, -15), (-7, -35)):
        y_val = float(trig[lags.index(lag)])
        fig.add_vline(x=lag, line_dash="dash", line_color="rgba(247, 185, 36, 0.55)", row=1, col=2)
        fig.add_annotation(
            x=lag, y=y_val, text=f"Скрытый лаг t{lag}<br>(вес {y_val:.3f})", showarrow=True, arrowhead=2,
            ax=-50, ay=ay, xanchor="right", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#F7B924",
            borderwidth=1, font={"color": "#F7B924", "size": 11.5}, arrowcolor="#F7B924", row=1, col=2,
        )
    max_y = max(max(trig), max(nontrig), 0.1)
    fig.update_yaxes(title_text="Средний вес внимания из t-1", range=[0, max_y * 1.28], row=1, col=2)
    fig.update_xaxes(title_text="Позиция ключа относительно цели t (лаги -64 .. -1)", range=[-65, 1], row=1, col=2)


def plot_exp5_crypto_backdoor_xray(
    audit_data: dict[str, Any],
    attn_profile: dict[str, Any],
    summary_dir: Path,
) -> tuple[Path, Path]:
    """Двухпанельный разбор детектора крипто-закладки и профиля внимания по лагам t-24, t-7."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Точность на закладке (PPM-C / LightGBM слепы vs Neural Detectors)",
            "Attention Map локализует скрытые лаги (t-24, t-7) в os.urandom",
        ),
        horizontal_spacing=0.09,
    )
    _add_crypto_audit_bars(fig, audit_data)
    _add_crypto_attn_profile(fig, attn_profile)
    apply_lab_layout(
        fig,
        "Эксперимент 5 (Модуль 2): Крипто-аудитор ГПСЧ — обнаружение скрытой закладки (t-24, t-7) в os.urandom",
        height=640,
        width=1620,
        barmode="group",
        legend=legend_bottom(y=-0.17, font_size=12),
        margin={"l": 75, "r": 50, "t": 95, "b": 115},
    )
    return save_plotly_figure(fig, summary_dir, "exp5_crypto_backdoor_xray")
