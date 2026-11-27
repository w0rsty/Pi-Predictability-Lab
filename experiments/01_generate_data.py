# Скрипт генерации последовательностей (N=1M) и запуска классических статтестов (chi2, KS, ACF, gzip/lzma)
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.data_generator import get_or_create_dataset, list_available_datasets, load_config, seed_everything
from src.stats_tests import run_full_statistical_suite

ROOT_DIR = Path(__file__).resolve().parents[1]


def save_stats_plots(results: list[dict], run_dir: Path) -> tuple[Path, Path]:
    """Строит интерактивный HTML и PNG с частотами цифр и автокорреляцией."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Отклонение частот цифр (Base-10) от идеала 10%",
            "Автокорреляция (ACF) по лагам 1..64",
        ),
    )
    by_name = {r["name"]: r for r in results}

    for ds_name in ("true_pi", "csprng", "rational_1_999983", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"):
        if ds_name in by_name:
            fig.add_trace(
                go.Bar(name=ds_name, x=list(range(10)), y=[f * 100.0 for f in by_name[ds_name]["freqs"]]),
                row=1,
                col=1,
            )
    fig.add_hline(y=10.0, line_dash="dash", line_color="white", opacity=0.7, row=1, col=1)

    for ds_name in ("true_pi", "rational_1_999983", "hacked_pi_tier1", "hacked_pi_tier2", "hacked_pi_tier3"):
        if ds_name in by_name:
            r = by_name[ds_name]
            fig.add_trace(
                go.Scatter(name=f"{ds_name} ACF", x=r["acf_lags"], y=r["acf_values"], mode="lines+markers", marker={"size": 4}),
                row=1,
                col=2,
            )

    if "true_pi" in by_name:
        bound = by_name["true_pi"]["acf_conf_bound_95"]
        fig.add_hline(y=bound, line_dash="dot", line_color="gray", row=1, col=2)
        fig.add_hline(y=-bound, line_dash="dot", line_color="gray", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Статистический аудит констант, ГПСЧ и 3 уровней Hacked Pi (N=1,000,000)",
        height=520,
        width=1300,
        barmode="group",
    )
    fig.update_xaxes(title_text="Цифра (0–9)", row=1, col=1)
    fig.update_yaxes(title_text="Частота (%)", row=1, col=1)
    fig.update_xaxes(title_text="Лаг k", row=1, col=2)
    fig.update_yaxes(title_text="Автокорреляция", row=1, col=2)

    html_path = run_dir / "stats_overview.html"
    png_path = run_dir / "stats_overview.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)
    return html_path, png_path


def _to_summary_row(r: dict) -> dict:
    return {
        "name": r["name"],
        "radix": r["radix"],
        "length": r["length"],
        "mask_count": r["mask_count"],
        "mask_fraction": round(r["mask_fraction"], 6),
        "chi2_stat": round(r["chi2_stat"], 4),
        "chi2_p_value": r["chi2_p_value"],
        "chi2_passed": r["chi2_passed"],
        "ks_stat": round(r["ks_stat"], 6),
        "ks_p_value": r["ks_p_value"],
        "ks_passed": r["ks_passed"],
        "acf_max_abs": round(r["acf_max_abs"], 6),
        "acf_max_lag": r["acf_max_lag"],
        "acf_significant_count": r["acf_significant_count"],
        "gzip_ratio": round(r["gzip_ratio"], 5),
        "lzma_ratio": round(r["lzma_ratio"], 5),
        "lzma_bits_per_digit": round(r["lzma_bits_per_digit"], 4),
        "shannon_limit_bits": round(r["shannon_limit_bits_per_digit"], 4),
        "empirical_entropy_nats": round(r["empirical_entropy_nats"], 6),
    }


def main() -> None:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config["seed"]))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT_DIR / config["paths"]["runs_dir"] / f"classical_stats_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    all_stats: list[dict] = []
    pbar = tqdm(list_available_datasets(config), desc="Генерация и классическая статистика", unit="dataset")
    for name in pbar:
        pbar.set_postfix_str(f"генерация {name}")
        artifact = get_or_create_dataset(name=name, config=config)
        pbar.set_postfix_str(f"тесты {name}")
        stat_res = run_full_statistical_suite(name=name, digits=artifact.digits, radix=artifact.radix, max_lag=64)
        stat_res["from_cache"] = artifact.from_cache
        stat_res["mask_count"] = artifact.metadata.get("mask_count", 0)
        stat_res["mask_fraction"] = artifact.metadata.get("mask_fraction", 0.0)
        all_stats.append(stat_res)

    summary_rows = [_to_summary_row(r) for r in all_stats]
    df = pd.DataFrame(summary_rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "plot_data.json").write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path, png_path = save_stats_plots(all_stats, run_dir)

    print(f"\nГотово, всё сохранил в: {run_dir}")
    print(df.to_string(index=False))
    for r in summary_rows[:3]:
        print(f"[check] {r['name']}: lzma_excess={r['lzma_bits_per_digit'] - r['shannon_limit_bits']:+.4f} bits, chi2_p={r['chi2_p_value']:.4f}")
    print(f"\nГрафики: {html_path} и {png_path}")


if __name__ == "__main__":
    main()
