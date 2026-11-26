"""Генератор и кэш-менеджер последовательностей цифр (константы, ГПСЧ, дроби, Hacked Pi)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mpmath as mp
import numpy as np
import torch
import yaml

# Тип коллбэка прогресса: (имя_датасета, прогресс_от_0_до_1, сообщение)
ProgressCallback = Callable[[str, float, str], None]


# @dataclass автоматически создаёт __init__ и __repr__ для удобного хранения
# путей и метаданных одного датасета, чтобы не таскать длинные кортежи.
@dataclass
class DatasetArtifact:
    """Описание сгенерированного или загруженного из кэша датасета."""

    name: str
    radix: int
    length: int
    digits: np.ndarray
    mask: np.ndarray | None
    digits_path: Path
    mask_path: Path | None
    from_cache: bool
    metadata: dict[str, Any]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Загружает централизованный конфиг проекта из YAML."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Фиксирует сиды для random, numpy и torch ради воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _notify(
    progress_cb: ProgressCallback | None,
    name: str,
    fraction: float,
    message: str,
) -> None:
    if progress_cb is not None:
        progress_cb(name, min(max(fraction, 0.0), 1.0), message)


def _mpmath_constant_value(const_name: str):
    """Возвращает объект константы mpmath в текущей точности."""
    if const_name == "pi":
        return mp.pi
    if const_name == "e":
        return mp.e
    if const_name == "sqrt2":
        return mp.sqrt(2)
    raise ValueError(f"Неизвестная математическая константа: {const_name}")


def generate_constant_digits(
    const_name: str,
    length: int,
    radix: int = 10,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str | None = None,
) -> np.ndarray:
    """Генерирует дробные цифры константы (pi, e, sqrt2) в системе счисления radix."""
    if radix not in (2, 10, 16):
        raise ValueError(f"Поддерживаются только системы счисления 2, 10, 16; получено {radix}")
    if length <= 0:
        raise ValueError("Длина последовательности должна быть положительной")

    label = dataset_name or f"{const_name}_base{radix}"
    _notify(progress_cb, label, 0.05, f"Вычисляем {const_name} (radix={radix}, N={length:,}) через mpmath")

    # Берём небольшой запас точности (guard digits/bits), чтобы на границе не было ошибок округления
    if radix == 10:
        mp.mp.dps = length + 32
        val = _mpmath_constant_value(const_name)
        frac = val - int(val)
        _notify(progress_cb, label, 0.65, "Переводим дробную часть в строку цифр")
        # Просим length + 10 знаков после запятой в научной/обычной записи, убираем '0.'
        s = mp.nstr(frac, length + 10, strip_zeros=False)
        if "." not in s:
            raise RuntimeError(f"Неожиданный формат строки mpmath для {const_name}: {s[:30]}")
        frac_str = s.split(".", 1)[1]
        # Если mpmath выдал экспоненциальную запись вида '0.xxxxe-1' (для малых дробей — не наш случай, но проверим)
        if "e" in frac_str or "E" in frac_str:
            raise RuntimeError(f"Экспоненциальная запись при разборе {const_name}")
        frac_str = frac_str[:length]
        if len(frac_str) < length:
            raise RuntimeError(f"Недостаточно знаков ({len(frac_str)} < {length}) для {const_name}")
        # Быстрое преобразование ASCII-символов '0'..'9' в uint8 без медленного цикла в Python
        digits = np.frombuffer(frac_str.encode("ascii"), dtype=np.uint8) - ord("0")
        digits = digits.copy()
    else:
        bits_per_digit = 1 if radix == 2 else 4
        mp.mp.prec = length * bits_per_digit + 128
        val = _mpmath_constant_value(const_name)
        frac = val - int(val)
        _notify(progress_cb, label, 0.65, f"Извлекаем цифры в Base-{radix} из двоичной мантиссы")
        # Масштабируем дробную часть на radix^length через сдвиг степени двойки — это точно и быстро
        scaled = int(mp.ldexp(frac, length * bits_per_digit))
        if radix == 2:
            raw_str = f"{scaled:0{length}b}"[-length:]
            digits = np.frombuffer(raw_str.encode("ascii"), dtype=np.uint8) - ord("0")
            digits = digits.copy()
        else:
            raw_str = f"{scaled:0{length}x}"[-length:]
            raw_bytes = np.frombuffer(raw_str.encode("ascii"), dtype=np.uint8)
            # Переводим ASCII '0'..'9' и 'a'..'f' в числа 0..15 векторизованно
            digits = np.where(raw_bytes <= ord("9"), raw_bytes - ord("0"), raw_bytes - ord("a") + 10).astype(np.uint8)

    _notify(progress_cb, label, 1.0, "Готово")
    return digits


def generate_mt19937_digits(
    length: int,
    radix: int = 10,
    seed: int = 42,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "mt19937",
) -> np.ndarray:
    """Генерирует последовательность цифр с помощью ГПСЧ Mersenne Twister (MT19937)."""
    _notify(progress_cb, dataset_name, 0.1, f"Генерируем MT19937 (N={length:,}, seed={seed})")
    rng = np.random.Generator(np.random.MT19937(seed))
    digits = rng.integers(0, radix, size=length, dtype=np.uint8)
    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits


def generate_csprng_digits(
    length: int,
    radix: int = 10,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "csprng",
) -> np.ndarray:
    """Генерирует криптостойкие случайные цифры через os.urandom без modulo-bias."""
    _notify(progress_cb, dataset_name, 0.1, f"Читаем энтропию из os.urandom (N={length:,})")
    if 256 % radix == 0:
        # Для radix 2 и 16 остаток по модулю не даёт перекоса вероятностей
        raw = np.frombuffer(os.urandom(length), dtype=np.uint8)
        digits = (raw % radix).astype(np.uint8)
    else:
        # Для radix=10 отбрасываем байты >= 250 (rejection sampling), иначе цифры 0..5 будут встречаться чаще 6..9!
        limit = (256 // radix) * radix
        chunks: list[np.ndarray] = []
        collected = 0
        while collected < length:
            need = length - collected
            # Берём с небольшим запасом, так как ~2.3% байтов отсеется
            batch_size = max(int(need * 1.05) + 64, 4096)
            raw = np.frombuffer(os.urandom(batch_size), dtype=np.uint8)
            valid = raw[raw < limit] % radix
            if valid.size > need:
                valid = valid[:need]
            chunks.append(valid.astype(np.uint8))
            collected += valid.size
            _notify(progress_cb, dataset_name, 0.1 + 0.85 * (collected / length), "Фильтруем байты")
        digits = np.concatenate(chunks)
    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits


def generate_rational_digits(
    numerator: int,
    denominator: int,
    length: int,
    radix: int = 10,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "rational_1_999983",
) -> np.ndarray:
    """Генерирует цифры дробной части рационального числа numerator / denominator делением в столбик."""
    _notify(progress_cb, dataset_name, 0.05, f"Вычисляем {numerator}/{denominator} (N={length:,})")
    digits = np.empty(length, dtype=np.uint8)
    rem = numerator % denominator
    # Для ускорения пишем сразу крупными блоками по 18 цифр (влезает в int64), если radix == 10
    if radix == 10:
        block_size = 9
        scale = 10**block_size
        pos = 0
        report_every = max(length // 20, block_size)
        next_report = report_every
        while pos + block_size <= length:
            rem *= scale
            q, rem = divmod(rem, denominator)
            block_str = f"{q:0{block_size}d}"
            digits[pos : pos + block_size] = np.frombuffer(block_str.encode("ascii"), dtype=np.uint8) - ord("0")
            pos += block_size
            if pos >= next_report:
                _notify(progress_cb, dataset_name, 0.05 + 0.9 * (pos / length), "Деление в столбик")
                next_report += report_every
        while pos < length:
            rem *= 10
            q, rem = divmod(rem, denominator)
            digits[pos] = q
            pos += 1
    else:
        for pos in range(length):
            rem *= radix
            q, rem = divmod(rem, denominator)
            digits[pos] = q
    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits


def generate_rational_mpmath_digits(
    numerator: int,
    denominator: int,
    length: int,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "rational_1_97",
) -> np.ndarray:
    """Генерирует дробную часть rational через mpmath (dps >= length) для эталонной точности."""
    _notify(progress_cb, dataset_name, 0.1, f"Вычисляем {numerator}/{denominator} через mpmath (dps={length:,})")
    mp.mp.dps = int(length) + 64
    val = mp.mpf(numerator) / mp.mpf(denominator)
    s = mp.nstr(val, int(length) + 16, min_fixed=-(int(length) + 128), strip_zeros=False)
    if "." not in s or "e" in s or "E" in s:
        raise RuntimeError(f"Неожиданный формат строки mpmath для {numerator}/{denominator}: {s[:40]}")
    frac_str = s.split(".", 1)[1][:length]
    if len(frac_str) < length:
        raise RuntimeError(f"Недостаточно знаков ({len(frac_str)} < {length}) для {numerator}/{denominator}")
    digits = (np.frombuffer(frac_str.encode("ascii"), dtype=np.uint8) - ord("0")).copy()
    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits


def generate_champernowne_digits(
    length: int,
    radix: int = 10,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "champernowne",
) -> np.ndarray:
    """Генерирует константу Чемперноуна C_10 = 0.123456789101112... в Base-10."""
    if radix != 10:
        raise ValueError("Константа Чемперноуна поддерживается для radix=10")
    _notify(progress_cb, dataset_name, 0.1, f"Собираем числа 1, 2, 3... (N={length:,})")

    # Набираем числа кусками, пока суммарная строка не достигнет length цифр
    chunks: list[str] = []
    total_chars = 0
    k = 1
    step = 20_000
    while total_chars < length:
        block = "".join(str(n) for n in range(k, k + step))
        chunks.append(block)
        total_chars += len(block)
        k += step
        _notify(progress_cb, dataset_name, 0.1 + 0.7 * min(total_chars / length, 1.0), "Конкатенация чисел")

    full_str = "".join(chunks)[:length]
    digits = (np.frombuffer(full_str.encode("ascii"), dtype=np.uint8) - ord("0")).copy()
    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits


def generate_e_cf_tokens(
    length: int,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "e_cf",
) -> np.ndarray:
    """Цепная дробь числа e = [2; 1,2,1, 1,4,1, ...] в потоке десятичных цифр + разделитель 10 (radix=11)."""
    _notify(progress_cb, dataset_name, 0.1, f"Генерируем цепную дробь e (N={length:,}, radix=11)")
    tokens: list[int] = [2, 10]
    k = 1
    while len(tokens) < length:
        tokens.extend((1, 10))
        even_val = 2 * k
        if even_val < 10:
            tokens.append(even_val)
        else:
            for ch in str(even_val):
                tokens.append(ord(ch) - 48)
        tokens.extend((10, 1, 10))
        k += 1

    arr = np.asarray(tokens[:length], dtype=np.uint8)
    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return arr


def _extract_cf_lehmer(p: int, q: int, max_tokens: int) -> list[int]:
    """Быстрое точное извлечение цепной дроби из p/q через 60-битный интервал Лемера."""
    tokens: list[int] = []
    while q > 0 and len(tokens) < max_tokens:
        nb = p.bit_length()
        if nb <= 128 or (nb - q.bit_length()) > 28:
            a, r = divmod(p, q)
            p, q = q, r
            if a < 10:
                tokens.append(a)
            else:
                for ch in str(a):
                    tokens.append(ord(ch) - 48)
            tokens.append(10)
            continue

        # Вырезаем старшие 58 бит с интервалом неопределённости [p_lo, p_hi] / [q_lo, q_hi]
        s = nb - 58
        p_lo = p >> s
        p_hi = p_lo + 1
        q_lo = q >> s
        q_hi = q_lo + 1

        u0, v0, u1, v1 = 1, 0, 0, 1
        steps = 0
        while q_lo > 0 and len(tokens) < max_tokens:
            a_min = p_lo // q_hi
            a_max = p_hi // q_lo
            if a_min != a_max:
                break
            a = a_min
            r_lo = p_lo - a * q_hi
            r_hi = p_hi - a * q_lo
            p_lo, p_hi, q_lo, q_hi = q_lo, q_hi, r_lo, r_hi
            u0, v0, u1, v1 = u1, v1, u0 - a * u1, v0 - a * v1
            if a < 10:
                tokens.append(a)
            else:
                for ch in str(a):
                    tokens.append(ord(ch) - 48)
            tokens.append(10)
            steps += 1

        if steps == 0:
            a, r = divmod(p, q)
            p, q = q, r
            if a < 10:
                tokens.append(a)
            else:
                for ch in str(a):
                    tokens.append(ord(ch) - 48)
            tokens.append(10)
        else:
            p, q = u0 * p + v0 * q, u1 * p + v1 * q
    return tokens[:max_tokens]


def generate_pi_cf_tokens(
    length: int = 250_000,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "pi_cf",
) -> np.ndarray:
    """Цепная дробь числа pi (radix=11) через mpmath и алгоритм Лемера на длинных int."""
    _notify(progress_cb, dataset_name, 0.05, f"Вычисляем pi и извлекаем цепную дробь (N={length:,})")
    # В среднем ~2.15 токена на неполное частное и ~3.42 бита на частное + хороший запас точности
    precision_bits = max(int(length * 2.2), 128_000)
    t0 = time.perf_counter()
    mp.mp.prec = precision_bits + 256
    p0 = int(mp.ldexp(mp.pi, precision_bits))
    q0 = 1 << precision_bits
    _notify(progress_cb, dataset_name, 0.45, "Извлекаем неполные частные алгоритмом Лемера")

    tokens = _extract_cf_lehmer(p0, q0, max_tokens=length)
    if len(tokens) < length:
        raise RuntimeError(f"Не хватило точности для pi_cf: получено {len(tokens)} < {length}")
    elapsed = time.perf_counter() - t0
    _notify(progress_cb, dataset_name, 1.0, f"Готово за {elapsed:.2f}с")
    return np.asarray(tokens, dtype=np.uint8)


def apply_hacked_pi_tier1(
    base_digits: np.ndarray,
    period: int = 50,
    pattern: list[int] | tuple[int, ...] = (7, 7, 7),
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "hacked_pi_tier1",
) -> tuple[np.ndarray, np.ndarray]:
    """Уровень 1: вставляет периодический маркер pattern каждые period шагов и возвращает (digits, mask)."""
    _notify(progress_cb, dataset_name, 0.1, f"Внедряем маркер {list(pattern)} каждые {period} шагов")
    digits = base_digits.copy()
    mask = np.zeros(len(digits), dtype=np.uint8)
    pat = np.asarray(pattern, dtype=np.uint8)
    pat_len = len(pat)

    for start in range(period, len(digits) - pat_len + 1, period):
        digits[start : start + pat_len] = pat
        mask[start : start + pat_len] = 1

    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits, mask


def apply_hacked_pi_algebraic(
    base_digits: np.ndarray,
    mod_base: int = 7,
    trigger_rem: int = 0,
    offset: int = 2,
    target_digit: int = 9,
    fire_prob: float = 1.0,
    seed: int = 42,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "hacked_pi_tier2",
) -> tuple[np.ndarray, np.ndarray]:
    """Уровни 2 и 3: алгебраическая зависимость (x[t-2]+x[t-1]+x[t]) % mod_base == trigger_rem -> x[t+offset] = target_digit."""
    if not (0.0 <= fire_prob <= 1.0):
        raise ValueError(f"fire_prob должен быть в [0, 1], получено {fire_prob}")
    if offset < 1:
        raise ValueError("offset должен быть >= 1")

    _notify(progress_cb, dataset_name, 0.1, f"Применяем алгебраическое правило (p={fire_prob})")
    digits = base_digits.copy()
    mask = np.zeros(len(digits), dtype=np.uint8)
    n = len(digits)

    # Заранее генерируем случайные пороги для Уровня 3, чтобы не дёргать RNG внутри цикла по 1М раз
    use_noise = fire_prob < 1.0
    if use_noise:
        rng = np.random.default_rng(seed)
        coin_flips = rng.random(n) < fire_prob
    else:
        coin_flips = None

    # Идём строго слева направо по уже зафиксированным цифрам, чтобы каузальное окно всегда видело точный триггер
    for t in range(2, n - offset):
        if (int(digits[t - 2]) + int(digits[t - 1]) + int(digits[t])) % mod_base == trigger_rem:
            if coin_flips is None or coin_flips[t]:
                digits[t + offset] = target_digit
                mask[t + offset] = 1

    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits, mask


def apply_csprng_backdoor(
    base_digits: np.ndarray,
    lag1: int = 24,
    lag2: int = 7,
    progress_cb: ProgressCallback | None = None,
    dataset_name: str = "csprng_backdoor",
) -> tuple[np.ndarray, np.ndarray]:
    """Внедряет скрытую многопозиционную крипто-закладку (~4% шагов) с нулевой линейной корреляцией.

    Триггер при t >= max(lag1, lag2): (x[t-24] + x[t-7]) % 10 == 0 и x[t-1] < 4 ->
    x[t] = (4 * x[t-24] + x[t-7] + 5) % 10.
    На триггере x[t-7] = -x[t-24] (mod 10), поэтому выход (3*a + 5) % 10 строго равномерен на 0..9
    и имеет околонулевую корреляцию (-0.03) с x[t-24] и x[t-7], обманывая chi2, KS и ACF.
    """
    _notify(progress_cb, dataset_name, 0.1, f"Внедряем скрытую закладку на лагах t-{lag1}, t-{lag2}")
    digits = base_digits.copy()
    mask = np.zeros(len(digits), dtype=np.uint8)
    start_t = max(int(lag1), int(lag2), 1)

    for t in range(start_t, len(digits)):
        a = int(digits[t - lag1])
        b = int(digits[t - lag2])
        c = int(digits[t - 1])
        if (a + b) % 10 == 0 and c < 4:
            digits[t] = (4 * a + b + 5) % 10
            mask[t] = 1

    _notify(progress_cb, dataset_name, 1.0, "Готово")
    return digits, mask


def _build_registry(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Собирает реестр всех стандартных датасетов Фазы 1 на основе configs/config.yaml."""
    seed = int(config.get("seed", 42))
    hacked_cfg = config.get("hacked_pi", {})
    t1 = hacked_cfg.get("tier1", {})
    t2 = hacked_cfg.get("tier2", {})
    t3 = hacked_cfg.get("tier3", {})

    registry: dict[str, dict[str, Any]] = {
        "true_pi": {"kind": "constant", "const": "pi", "radix": 10},
        "e": {"kind": "constant", "const": "e", "radix": 10},
        "sqrt2": {"kind": "constant", "const": "sqrt2", "radix": 10},
        "pi_base2": {"kind": "constant", "const": "pi", "radix": 2},
        "pi_base16": {"kind": "constant", "const": "pi", "radix": 16},
        "mt19937": {"kind": "mt19937", "radix": 10, "seed": seed},
        "csprng": {"kind": "csprng", "radix": 10},
        "csprng_backdoor": {"kind": "csprng_backdoor", "radix": 10, "lag1": 24, "lag2": 7},
        "rational_1_999983": {"kind": "rational", "num": 1, "den": 999983, "radix": 10},
        "rational_1_97": {
            "kind": "rational_mpmath",
            "num": 1,
            "den": 97,
            "period": 96,
            "radix": 10,
            "default_length": 1_000_050,
        },
        "champernowne": {"kind": "champernowne", "radix": 10},
        "e_cf": {"kind": "e_cf", "radix": 11},
        "pi_cf": {"kind": "pi_cf", "radix": 11, "default_length": 250_000},
        "hacked_pi_tier1": {
            "kind": "hacked_tier1",
            "radix": 10,
            "period": int(t1.get("period", 50)),
            "pattern": list(t1.get("pattern", [7, 7, 7])),
        },
        "hacked_pi_tier2": {
            "kind": "hacked_algebraic",
            "radix": 10,
            "mod_base": int(t2.get("mod_base", 7)),
            "trigger_rem": int(t2.get("trigger_rem", 0)),
            "offset": int(t2.get("offset", 2)),
            "target_digit": int(t2.get("target_digit", 9)),
            "fire_prob": 1.0,
            "seed": seed,
        },
        "hacked_pi_tier3": {
            "kind": "hacked_algebraic",
            "radix": 10,
            "mod_base": int(t2.get("mod_base", 7)),
            "trigger_rem": int(t2.get("trigger_rem", 0)),
            "offset": int(t2.get("offset", 2)),
            "target_digit": int(t2.get("target_digit", 9)),
            "fire_prob": float(t3.get("default_prob", 0.5)),
            "seed": seed,
        },
    }
    return registry


def list_available_datasets(config: dict[str, Any]) -> list[str]:
    """Возвращает список имён датасетов, описанных в реестре."""
    return list(_build_registry(config).keys())


def get_or_create_dataset(
    name: str,
    config: dict[str, Any],
    length: int | None = None,
    force_regenerate: bool = False,
    progress_cb: ProgressCallback | None = None,
    extra_params: dict[str, Any] | None = None,
) -> DatasetArtifact:
    """Загружает датасет из кэша data/ или генерирует и сохраняет в uint8 .npy."""
    registry = _build_registry(config)
    if name not in registry:
        raise ValueError(f"Датасет '{name}' не найден в реестре. Доступны: {list(registry.keys())}")

    spec = dict(registry[name])
    if extra_params:
        spec.update(extra_params)

    default_len = int(spec.get("default_length", config["data"]["default_length"]))
    target_length = int(length if length is not None else default_len)
    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    digits_path = data_dir / f"{name}_digits.npy"
    mask_path = data_dir / f"{name}_mask.npy"
    meta_path = data_dir / f"{name}_meta.json"
    expects_mask = spec["kind"] in ("hacked_tier1", "hacked_algebraic", "csprng_backdoor")

    # Проверяем, есть ли валидный кэш с нужной длиной и параметрами
    if not force_regenerate and digits_path.exists() and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if (
                cached_meta.get("length") == target_length
                and cached_meta.get("spec") == spec
                and (not expects_mask or mask_path.exists())
            ):
                digits = np.load(digits_path)
                mask = np.load(mask_path) if expects_mask else None
                if len(digits) == target_length:
                    _notify(progress_cb, name, 1.0, "Загружено из кэша")
                    return DatasetArtifact(
                        name=name,
                        radix=int(spec["radix"]),
                        length=target_length,
                        digits=digits,
                        mask=mask,
                        digits_path=digits_path,
                        mask_path=mask_path if expects_mask else None,
                        from_cache=True,
                        metadata=cached_meta,
                    )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Кэша нет или параметры изменились — генерируем заново
    kind = spec["kind"]
    mask: np.ndarray | None = None
    if kind == "constant":
        digits = generate_constant_digits(
            const_name=spec["const"],
            length=target_length,
            radix=int(spec["radix"]),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "mt19937":
        digits = generate_mt19937_digits(
            length=target_length,
            radix=int(spec["radix"]),
            seed=int(spec["seed"]),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "csprng":
        digits = generate_csprng_digits(
            length=target_length,
            radix=int(spec["radix"]),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "rational":
        digits = generate_rational_digits(
            numerator=int(spec["num"]),
            denominator=int(spec["den"]),
            length=target_length,
            radix=int(spec["radix"]),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "rational_mpmath":
        digits = generate_rational_mpmath_digits(
            numerator=int(spec["num"]),
            denominator=int(spec["den"]),
            length=target_length,
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "champernowne":
        digits = generate_champernowne_digits(
            length=target_length,
            radix=int(spec["radix"]),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "e_cf":
        digits = generate_e_cf_tokens(
            length=target_length,
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "pi_cf":
        digits = generate_pi_cf_tokens(
            length=target_length,
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "hacked_tier1":
        # Для честного сравнения H0 и H1 всегда берём базу из true_pi той же длины
        pi_art = get_or_create_dataset("true_pi", config=config, length=target_length)
        digits, mask = apply_hacked_pi_tier1(
            base_digits=pi_art.digits,
            period=int(spec["period"]),
            pattern=spec["pattern"],
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "hacked_algebraic":
        pi_art = get_or_create_dataset("true_pi", config=config, length=target_length)
        digits, mask = apply_hacked_pi_algebraic(
            base_digits=pi_art.digits,
            mod_base=int(spec["mod_base"]),
            trigger_rem=int(spec["trigger_rem"]),
            offset=int(spec["offset"]),
            target_digit=int(spec["target_digit"]),
            fire_prob=float(spec["fire_prob"]),
            seed=int(spec["seed"]),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    elif kind == "csprng_backdoor":
        cs_art = get_or_create_dataset("csprng", config=config, length=target_length)
        digits, mask = apply_csprng_backdoor(
            base_digits=cs_art.digits,
            lag1=int(spec.get("lag1", 24)),
            lag2=int(spec.get("lag2", 7)),
            progress_cb=progress_cb,
            dataset_name=name,
        )
    else:
        raise ValueError(f"Неподдерживаемый тип генератора: {kind}")

    np.save(digits_path, digits.astype(np.uint8, copy=False))
    if mask is not None:
        np.save(mask_path, mask.astype(np.uint8, copy=False))

    meta = {
        "name": name,
        "radix": int(spec["radix"]),
        "length": target_length,
        "spec": spec,
        "hash": hashlib.sha256(digits.tobytes()).hexdigest(),
        "mask_count": int(mask.sum()) if mask is not None else 0,
        "mask_fraction": float(mask.mean()) if mask is not None else 0.0,
    }
    if "period" in spec:
        meta["period"] = int(spec["period"])
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return DatasetArtifact(
        name=name,
        radix=int(spec["radix"]),
        length=target_length,
        digits=digits,
        mask=mask,
        digits_path=digits_path,
        mask_path=mask_path if mask is not None else None,
        from_cache=False,
        metadata=meta,
    )
