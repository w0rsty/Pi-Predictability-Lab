# Экспорт канонических затравок и истинных продолжений (dream_seeds.json) для вкладки «Сны нейросети»
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data_generator import get_or_create_dataset, load_config
from src.dataset import split_sequence_chronologically

ROOT_DIR = Path(__file__).resolve().parents[1]
DREAM_SEEDS_PATH = ROOT_DIR / "web" / "frontend" / "public" / "data" / "dream_seeds.json"

SEED_LEN = 64
TRUTH_LEN = 500


def _make_entry(
    seed_id: str,
    dataset: str,
    label: str,
    description: str,
    vocab_size: int,
    global_offset: int,
    arr: np.ndarray,
    pos: int,
) -> dict[str, Any]:
    seed_tokens = [int(x) for x in arr[pos : pos + SEED_LEN].tolist()]
    ground_truth = [int(x) for x in arr[pos + SEED_LEN : pos + SEED_LEN + TRUTH_LEN].tolist()]
    return {
        "id": seed_id,
        "dataset": dataset,
        "label": label,
        "description": description,
        "vocab_size": int(vocab_size),
        "global_offset": int(global_offset + pos),
        "seed_tokens": seed_tokens,
        "ground_truth": ground_truth,
    }


def _find_champernowne_label(seq: list[int]) -> str:
    txt = "".join(str(d) for d in seq)
    return f"...{txt[:16]}..."


def build_dream_seeds() -> dict[str, Any]:
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    entries: list[dict[str, Any]] = []

    # 1. true_pi: узнаваемый префикс 3.(14159...) + 2 строгих среза из тестового хвоста
    pi_art = get_or_create_dataset("true_pi", config=config, length=None)
    pi_split = split_sequence_chronologically(pi_art.digits, None, 0.8, 0.1, 0.1)
    entries.append(
        _make_entry(
            seed_id="true_pi_start",
            dataset="true_pi",
            label="Начало π (3.14159265...)",
            description="Первые 64 знака π после запятой и следующие 500 истинных цифр",
            vocab_size=10,
            global_offset=0,
            arr=pi_art.digits,
            pos=0,
        )
    )
    entries.append(
        _make_entry(
            seed_id="true_pi_test_1",
            dataset="true_pi",
            label="Тест π (#900,000)",
            description="Срез строго из тестовой выборки (индекс 900,000)",
            vocab_size=10,
            global_offset=pi_split.test_range[0],
            arr=pi_split.test_digits,
            pos=0,
        )
    )
    entries.append(
        _make_entry(
            seed_id="true_pi_test_2",
            dataset="true_pi",
            label="Тест π (#942,000)",
            description="Второй контрольный тестовый срез π (индекс 942,000)",
            vocab_size=10,
            global_offset=pi_split.test_range[0],
            arr=pi_split.test_digits,
            pos=42_000,
        )
    )

    # 2. champernowne: 3 тестовых среза с 6-значными числами (наглядный автономный +1 счёт)
    ch_art = get_or_create_dataset("champernowne", config=config, length=None)
    ch_split = split_sequence_chronologically(ch_art.digits, None, 0.8, 0.1, 0.1)
    for idx, pos in enumerate((1_000, 28_000, 64_000), start=1):
        preview = _find_champernowne_label(ch_split.test_digits[pos : pos + 20].tolist())
        global_pos = ch_split.test_range[0] + pos
        entries.append(
            _make_entry(
                seed_id=f"champernowne_test_{idx}",
                dataset="champernowne",
                label=f"Чемперноун #{global_pos:,} ({preview})",
                description="Тестовый фрагмент константы Чемперноуна с 6-значными числами",
                vocab_size=10,
                global_offset=ch_split.test_range[0],
                arr=ch_split.test_digits,
                pos=pos,
            )
        )

    # 3. e_cf: 3 тестовых среза цепной дроби числа e (base-11, токен 10 = ';')
    ecf_art = get_or_create_dataset("e_cf", config=config, length=None)
    ecf_split = split_sequence_chronologically(ecf_art.digits, None, 0.8, 0.1, 0.1)
    for idx, pos in enumerate((1_200, 24_000, 58_000), start=1):
        global_pos = ecf_split.test_range[0] + pos
        entries.append(
            _make_entry(
                seed_id=f"e_cf_test_{idx}",
                dataset="e_cf",
                label=f"Цепная дробь e (#{global_pos:,})",
                description="Тестовый срез [2;1,2,1,1,4,...] с арифметической прогрессией 2k",
                vocab_size=11,
                global_offset=ecf_split.test_range[0],
                arr=ecf_split.test_digits,
                pos=pos,
            )
        )

    # 4. csprng_backdoor: 3 тестовых среза с максимальной плотностью триггеров в окне ground_truth
    bd_art = get_or_create_dataset("csprng_backdoor", config=config, length=None)
    bd_split = split_sequence_chronologically(bd_art.digits, bd_art.mask, 0.8, 0.1, 0.1)
    bd_mask = bd_split.test_mask if bd_split.test_mask is not None else np.zeros_like(bd_split.test_digits)
    candidate_offsets = list(range(0, len(bd_split.test_digits) - SEED_LEN - TRUTH_LEN - 1, 2_500))
    scored = sorted(
        candidate_offsets,
        key=lambda p: int(bd_mask[p + SEED_LEN : p + SEED_LEN + TRUTH_LEN].sum()),
        reverse=True,
    )
    chosen_bd: list[int] = []
    for p in scored:
        if all(abs(p - prev) >= 10_000 for prev in chosen_bd):
            chosen_bd.append(p)
        if len(chosen_bd) == 3:
            break
    for idx, pos in enumerate(chosen_bd, start=1):
        trig_cnt = int(bd_mask[pos + SEED_LEN : pos + SEED_LEN + TRUTH_LEN].sum())
        global_pos = bd_split.test_range[0] + pos
        entries.append(
            _make_entry(
                seed_id=f"csprng_backdoor_test_{idx}",
                dataset="csprng_backdoor",
                label=f"CSPRNG + Backdoor #{global_pos:,} ({trig_cnt} триг.)",
                description=f"Тестовый срез os.urandom с закладкой (t-24, t-7); {trig_cnt} триггеров в горизонте",
                vocab_size=10,
                global_offset=bd_split.test_range[0],
                arr=bd_split.test_digits,
                pos=pos,
            )
        )

    return {
        "version": 1,
        "seed_length": SEED_LEN,
        "truth_length": TRUTH_LEN,
        "seeds": entries,
    }


def export_dream_seeds() -> Path:
    DREAM_SEEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dream_seeds()
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    DREAM_SEEDS_PATH.write_text(serialized, encoding="utf-8")
    size_kb = len(serialized.encode("utf-8")) / 1024.0
    print(f"[export_dream_seeds] OK -> {DREAM_SEEDS_PATH} ({size_kb:.1f} KB, seeds={len(payload['seeds'])})")
    return DREAM_SEEDS_PATH


if __name__ == "__main__":
    export_dream_seeds()
