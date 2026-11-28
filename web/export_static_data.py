# Сборка канонического lab_data.json для статической витрины из results/01..07 и data/
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_generator import get_or_create_dataset, load_config

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
RUNS_DIR = ROOT_DIR / "runs"
OUT_PATH = ROOT_DIR / "web" / "frontend" / "public" / "data" / "lab_data.json"


def _clean_value(v: Any) -> Any:
    """Рекурсивно очищает NaN/Inf и NumPy-скаляры для строгого JSON (RFC 8259)."""
    if isinstance(v, dict):
        return {str(k): _clean_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if v is None or pd.isna(v):
        return None
    return v


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    return [_clean_value(rec) for rec in df.to_dict(orient="records")]


def _read_json(path: Path) -> Any:
    return _clean_value(json.loads(path.read_text(encoding="utf-8")))


def _load_crypto_attn_profile() -> dict[str, Any]:
    """Читает реальный профиль внимания по лагам из последнего запуска Exp 5."""
    if RUNS_DIR.exists():
        cands = sorted(
            [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("exp5_mech_interp_")],
            key=lambda p: p.name,
            reverse=True,
        )
        for d in cands:
            pd_path = d / "plot_data.json"
            if pd_path.exists():
                payload = _read_json(pd_path)
                prof = payload.get("crypto_attn_profile")
                if isinstance(prof, dict) and prof.get("trigger_mean_attn"):
                    return prof
    return {}


def _build_real_samples(config: dict[str, Any], sample_len: int = 2500) -> dict[str, Any]:
    """Извлекает точные фрагменты цифр и триггеров из кэша data/ или генераторов."""
    samples: dict[str, Any] = {}
    for name in ("true_pi", "rational_1_97", "champernowne", "e_cf"):
        art = get_or_create_dataset(name, config=config, length=None)
        sl = art.digits[:sample_len]
        samples[name] = {
            "radix": int(art.radix),
            "length": int(len(sl)),
            "digits": [int(x) for x in sl.tolist()],
        }

    # Для крипто-закладки берём срез прямо из тестовой области (индекс 900_000), где модель не обучалась
    bd_art = get_or_create_dataset("csprng_backdoor", config=config, length=None)
    offset = 900_000 if len(bd_art.digits) >= 900_000 + sample_len else 0
    bd_digits = bd_art.digits[offset : offset + sample_len]
    bd_mask = bd_art.mask[offset : offset + sample_len] if bd_art.mask is not None else np.zeros(sample_len, dtype=np.uint8)

    triggers: list[dict[str, Any]] = []
    for t in range(24, len(bd_digits)):
        if int(bd_mask[t]) == 1:
            a = int(bd_digits[t - 24])
            b = int(bd_digits[t - 7])
            c = int(bd_digits[t - 1])
            target = int(bd_digits[t])
            rhs = (4 * a + b + 5) % 10
            triggers.append({
                "t": int(t),
                "lag24_idx": int(t - 24),
                "lag7_idx": int(t - 7),
                "lag1_idx": int(t - 1),
                "a": a,
                "b": b,
                "c": c,
                "target": target,
                "rhs": int(rhs),
                "bijection_rhs": int((3 * a + 5) % 10),
            })

    samples["csprng_backdoor"] = {
        "radix": 10,
        "length": int(len(bd_digits)),
        "global_offset": int(offset),
        "digits": [int(x) for x in bd_digits.tolist()],
        "mask": [int(x) for x in bd_mask.tolist()],
        "triggers": triggers,
    }
    return samples


def build_lab_data() -> dict[str, Any]:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")

    exp1_dir = RESULTS_DIR / "03_exp1_sanity"
    exp2_dir = RESULTS_DIR / "04_exp2_constants"
    exp3_dir = RESULTS_DIR / "05_exp3_bases"
    exp4_dir = RESULTS_DIR / "06_exp4_scaling"
    exp5_dir = RESULTS_DIR / "07_exp5_mech_interp"

    payload = {
        "meta": {
            "project": "Pi-Predictability-Lab",
            "n_digits": int(config["data"]["default_length"]),
            "window_size": int(config["data"]["default_window_size"]),
            "split": _clean_value(config["data"]["split"]),
            "seed": int(config.get("seed", 42)),
            "mde_pp": 0.27,
            "se_pp": 0.095,
            "cpu_tests_count": 17,
            "architectures": ["DigitMLP", "DigitCNN", "DigitGRU", "DigitTransformer"],
            "budget_params": "~250K",
        },
        "classical_stats": {
            "overview": _read_csv_records(RESULTS_DIR / "01_classical_stats" / "metrics.csv"),
            "new_constants": _read_csv_records(exp2_dir / "classical_stats_new_constants.csv"),
            "bases": _read_csv_records(exp3_dir / "classical_stats_bases.csv"),
        },
        "exp1_sanity": {
            "arch_comparison": _read_csv_records(exp1_dir / "arch_comparison_metrics.csv"),
            "ppm_vs_neural": _read_csv_records(exp1_dir / "ppm_vs_neural_comparison.csv"),
            "stat_audit": _read_json(exp1_dir / "stat_audit.json"),
        },
        "exp2_constants": {
            "metrics": _read_csv_records(exp2_dir / "metrics.csv"),
            "stat_audit": _read_json(exp2_dir / "stat_audit.json"),
        },
        "exp3_bases": {
            "metrics": _read_csv_records(exp3_dir / "metrics.csv"),
            "block_4bit_vs_nibble": _read_csv_records(exp3_dir / "block_4bit_vs_nibble.csv"),
            "stat_audit": _read_json(exp3_dir / "stat_audit.json"),
        },
        "exp4_scaling": {
            "metrics": _read_csv_records(exp4_dir / "metrics.csv"),
            "memorization_curve": _read_csv_records(exp4_dir / "memorization_curve.csv"),
            "stat_audit": _read_json(exp4_dir / "stat_audit.json"),
        },
        "exp5_mech_interp": {
            "metrics": _read_csv_records(exp5_dir / "metrics.csv"),
            "head_ablation": _read_csv_records(exp5_dir / "head_ablation.csv"),
            "head_ablation_backdoor": _read_csv_records(exp5_dir / "head_ablation_backdoor.csv"),
            "linear_probing": _read_csv_records(exp5_dir / "linear_probing.csv"),
            "embedding_geometry": _read_json(exp5_dir / "embedding_geometry.json"),
            "stat_audit": _read_json(exp5_dir / "stat_audit.json"),
            "crypto_attn_profile": _load_crypto_attn_profile(),
        },
        "samples": _build_real_samples(config, sample_len=2500),
    }
    return _clean_value(payload)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = build_lab_data()
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    OUT_PATH.write_text(serialized, encoding="utf-8")
    size_kb = len(serialized.encode("utf-8")) / 1024.0
    trig_count = len(data["samples"]["csprng_backdoor"]["triggers"])
    print(f"[export_static_data] OK -> {OUT_PATH} ({size_kb:.1f} KB, backdoor triggers in slice: {trig_count})")

    from web.export_arena_data import export_arena_rounds
    from web.export_dream_seeds import export_dream_seeds

    export_arena_rounds()
    export_dream_seeds()


if __name__ == "__main__":
    main()
