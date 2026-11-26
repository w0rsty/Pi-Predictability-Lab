# Единая тема оформления, палитры и экспорт графиков (Plotly + PyEcharts)
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

EXP1_DS_COLORS = {
    "true_pi": "#636EFA",
    "hacked_pi_tier1": "#00CC96",
    "hacked_pi_tier2": "#EF553B",
    "hacked_pi_tier3": "#FFA15A",
}

MODEL_COLORS = {
    "Markov_k3": "#94A3B8",
    "PPM_o8": "#F7B924",
    "LightGBM": "#19D3F3",
    "MLP": "#636EFA",
    "DigitMLP": "#636EFA",
    "CNN": "#00CC96",
    "DigitCNN": "#00CC96",
    "GRU": "#EF553B",
    "DigitGRU": "#EF553B",
    "TRANSFORMER": "#AB63FA",
    "DigitTransformer": "#AB63FA",
}

ARCH_COLORS = {
    "mlp": "#636EFA",
    "cnn": "#00CC96",
    "gru": "#EF553B",
    "transformer": "#AB63FA",
}

HEAD_COLORS = ["#636EFA", "#00CC96", "#EF553B", "#FFA15A", "#AB63FA", "#19D3F3"]

DIGIT_PALETTE = [
    "#636EFA", "#19D3F3", "#00CC96", "#B6E880", "#FECB52",
    "#FFA15A", "#EF553B", "#FF6692", "#AB63FA", "#F8FAFC",
]


def legend_bottom(y: float = -0.16, font_size: int = 13) -> dict[str, Any]:
    return {"orientation": "h", "y": y, "x": 0.5, "xanchor": "center", "font": {"size": font_size}}


def apply_lab_layout(
    fig: go.Figure,
    title: str,
    height: int = 600,
    width: int = 1580,
    **kwargs: Any,
) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=height,
        width=width,
        **kwargs,
    )
    return fig


def save_plotly_figure(
    fig: go.Figure,
    out_dir: Path,
    stem: str,
    scale: int = 2,
) -> tuple[Path, Path]:
    # Сохраняем и интерактивный HTML, и 2x PNG для README/отчётов
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{stem}.html"
    png_path = out_dir / f"{stem}.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=scale)
    return html_path, png_path


save_fig = save_plotly_figure
