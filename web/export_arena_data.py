# Экспорт реального банка раундов Арены (30 тестовых окон × 5 датасетов) для статической витрины
import json
from pathlib import Path
from typing import Any

from web.services.arena_service import ARENA_DATASETS, build_arena_rounds_sync
from web.services.model_registry import ModelRegistry, ROOT_DIR

ARENA_OUT_PATH = ROOT_DIR / "web" / "frontend" / "public" / "data" / "arena_rounds.json"


def build_arena_rounds_bank(n_rounds_per_dataset: int = 30, seed: int = 42) -> dict[str, Any]:
    """Собирает банк раундов Арены из реальных чекпойнтов runs/ и тестовых сплитов data/."""
    registry = ModelRegistry(ROOT_DIR)
    registry.load_all()
    try:
        datasets_payload: dict[str, Any] = {}
        for idx, ds in enumerate(ARENA_DATASETS):
            bank = build_arena_rounds_sync(
                registry=registry,
                dataset=ds,
                n_rounds=n_rounds_per_dataset,
                seed=seed + idx * 101,
                min_gap=150,
            )
            datasets_payload[ds] = bank
        return {
            "version": 1,
            "seed": seed,
            "n_rounds_per_dataset": n_rounds_per_dataset,
            "datasets": datasets_payload,
        }
    finally:
        registry.clear()


def export_arena_rounds(n_rounds_per_dataset: int = 30, seed: int = 42) -> Path:
    ARENA_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_arena_rounds_bank(n_rounds_per_dataset=n_rounds_per_dataset, seed=seed)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    ARENA_OUT_PATH.write_text(serialized, encoding="utf-8")

    size_kb = len(serialized.encode("utf-8")) / 1024.0
    print(f"[export_arena_data] OK -> {ARENA_OUT_PATH} ({size_kb:.1f} KB)")
    for ds, bank in payload["datasets"].items():
        rounds = bank["rounds"]
        n = len(rounds)
        tf_acc = sum(1 for r in rounds if r["transformer"]["pred"] == r["target"]) / max(1, n)
        ppm_acc = sum(1 for r in rounds if r["ppm"]["pred"] == r["target"]) / max(1, n)
        gru_rounds = [r for r in rounds if r.get("gru") is not None]
        gru_str = (
            f"GRU={sum(1 for r in gru_rounds if r['gru']['pred'] == r['target']) / len(gru_rounds):.1%}"
            if gru_rounds
            else "GRU=n/a"
        )
        extra = ""
        if ds == "csprng_backdoor":
            trigs = [r for r in rounds if r["is_trigger"]]
            tf_trig = sum(1 for r in trigs if r["transformer"]["pred"] == r["target"]) / max(1, len(trigs))
            extra = f" | triggers({len(trigs)}): TF={tf_trig:.1%}"
        print(f"  - {ds:16s} (radix={bank['radix']:2d}, n={n}): TF={tf_acc:.1%}, {gru_str}, PPM_o8={ppm_acc:.1%}{extra}")
    return ARENA_OUT_PATH


def main() -> None:
    export_arena_rounds()


if __name__ == "__main__":
    main()
