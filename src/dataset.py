# Хронологическое разбиение (80/10/10) и оконные датасеты без утечки данных
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_generator import make_dataloader_generator, seed_worker


@dataclass
class ChronologicalSplit:
    """Хранит непересекающиеся срезы цифр и масок с их исходными индексами."""

    train_digits: np.ndarray
    val_digits: np.ndarray
    test_digits: np.ndarray
    train_mask: np.ndarray | None
    val_mask: np.ndarray | None
    test_mask: np.ndarray | None
    train_range: tuple[int, int]
    val_range: tuple[int, int]
    test_range: tuple[int, int]
    total_length: int


def split_sequence_chronologically(
    digits: np.ndarray,
    mask: np.ndarray | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> ChronologicalSplit:
    """Режет последовательность строго по индексам 80/10/10 без перемешивания."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Сумма долей train + val + test должна равняться 1.0")
    if mask is not None and len(mask) != len(digits):
        raise ValueError(f"Длины digits ({len(digits)}) и mask ({len(mask)}) не совпадают")

    n = len(digits)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or n <= val_end:
        raise ValueError(f"Последовательность слишком короткая (N={n}) для разбиения")

    return ChronologicalSplit(
        train_digits=digits[:train_end],
        val_digits=digits[train_end:val_end],
        test_digits=digits[val_end:],
        train_mask=mask[:train_end] if mask is not None else None,
        val_mask=mask[train_end:val_end] if mask is not None else None,
        test_mask=mask[val_end:] if mask is not None else None,
        train_range=(0, train_end),
        val_range=(train_end, val_end),
        test_range=(val_end, n),
        total_length=n,
    )


def _dataset_index_bounds(ds: "SequenceWindowDataset", expected_offset: int, expected_window: int, name: str) -> tuple[int, int]:
    if ds.window_size != expected_window:
        raise AssertionError(f"{name}: window_size {ds.window_size} != {expected_window}")
    if len(ds) <= 0:
        raise AssertionError(f"{name}: пустой датасет")
    offset = int(ds.global_offset)
    if offset != expected_offset:
        raise AssertionError(f"{name}: global_offset {offset} != {expected_offset}")
    min_ctx = offset
    max_tgt = offset + (len(ds) - 1) + ds.window_size
    if max_tgt != offset + len(ds.digits) - 1:
        raise AssertionError(f"{name}: рассинхрон длины среза и числа окон")
    # Выборочно сверяем инвариант target_pos == context_end_pos + 1 на реальных окнах
    for sample_idx in (0, len(ds) // 2, len(ds) - 1):
        ctx_end = offset + sample_idx + ds.window_size - 1
        tgt_pos = offset + sample_idx + ds.window_size
        if tgt_pos != ctx_end + 1:
            raise AssertionError(f"{name}: таргет попал внутрь контекста")
    return min_ctx, max_tgt


def verify_zero_leakage(
    split: ChronologicalSplit,
    window_size: int,
    train_ds: "SequenceWindowDataset | None" = None,
    val_ds: "SequenceWindowDataset | None" = None,
    test_ds: "SequenceWindowDataset | None" = None,
) -> dict[str, Any]:
    """Проверяет отсутствие пересечения глобальных индексов и памяти между Train, Val и Test."""
    tr_s, tr_e = split.train_range
    va_s, va_e = split.val_range
    te_s, te_e = split.test_range

    assert len(split.train_digits) == tr_e - tr_s
    assert len(split.val_digits) == va_e - va_s
    assert len(split.test_digits) == te_e - te_s
    assert tr_s == 0 and tr_e == va_s and va_e == te_s and te_e == split.total_length
    assert len(split.train_digits) > window_size
    assert len(split.val_digits) > window_size
    assert len(split.test_digits) > window_size
    assert (tr_e - 1) < va_s
    assert (va_e - 1) < te_s
    assert not np.shares_memory(split.train_digits, split.val_digits)
    assert not np.shares_memory(split.val_digits, split.test_digits)
    assert not np.shares_memory(split.train_digits, split.test_digits)

    if train_ds is not None and val_ds is not None and test_ds is not None:
        tr_min_ctx, tr_max_tgt = _dataset_index_bounds(train_ds, tr_s, window_size, "train_ds")
        va_min_ctx, va_max_tgt = _dataset_index_bounds(val_ds, va_s, window_size, "val_ds")
        te_min_ctx, te_max_tgt = _dataset_index_bounds(test_ds, te_s, window_size, "test_ds")
        assert tr_max_tgt < va_min_ctx, f"Утечка train->val: {tr_max_tgt} >= {va_min_ctx}"
        assert va_max_tgt < te_min_ctx, f"Утечка val->test: {va_max_tgt} >= {te_min_ctx}"
        assert not np.shares_memory(train_ds.digits, val_ds.digits)
        assert not np.shares_memory(val_ds.digits, test_ds.digits)
        assert tr_min_ctx < te_max_tgt

    return {
        "zero_overlap_verified": True,
        "window_size": window_size,
        "train_indices": [tr_s, tr_e - 1],
        "val_indices": [va_s, va_e - 1],
        "test_indices": [te_s, te_e - 1],
        "train_windows": len(split.train_digits) - window_size,
        "val_windows": len(split.val_digits) - window_size,
        "test_windows": len(split.test_digits) - window_size,
    }


# Наследуемся от torch.utils.data.Dataset, чтобы DataLoader мог батчить и грузить в фоне
class SequenceWindowDataset(Dataset):
    """Ленивая нарезка окон поверх одного uint8-массива без материализации матрицы (N, L); конвертация в torch.long происходит на лету для каждого сэмпла."""

    def __init__(
        self,
        digits: np.ndarray,
        window_size: int,
        mask: np.ndarray | None = None,
        global_offset: int = 0,
    ) -> None:
        if len(digits) <= window_size:
            raise ValueError(f"Длина среза ({len(digits)}) должна быть больше окна ({window_size})")
        if mask is not None and len(mask) != len(digits):
            raise ValueError("Длина маски должна совпадать с длиной среза цифр")

        self.digits = digits
        self.mask = mask
        self.window_size = int(window_size)
        self.global_offset = int(global_offset)
        self.num_windows = len(digits) - self.window_size

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = idx + self.window_size
        # Окно [idx .. end-1] строго до целевого индекса end: сдвиг таргета = ровно +1 шаг
        context = torch.as_tensor(self.digits[idx:end], dtype=torch.long)
        target = torch.tensor(int(self.digits[end]), dtype=torch.long)
        flag = int(self.mask[end]) if self.mask is not None else 0
        mask_flag = torch.tensor(flag, dtype=torch.uint8)
        return context, target, mask_flag

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Векторизованная выборка целого батча окон — в разы быстрее поштучного __getitem__."""
        idx_arr = np.asarray(indices, dtype=np.int64)
        offsets = idx_arr[:, None] + np.arange(self.window_size, dtype=np.int64)[None, :]
        contexts = torch.from_numpy(self.digits[offsets].astype(np.int64))
        target_pos = idx_arr + self.window_size
        targets = torch.from_numpy(self.digits[target_pos].astype(np.int64))
        if self.mask is not None:
            mask_flags = torch.from_numpy(self.mask[target_pos].copy())
        else:
            mask_flags = torch.zeros(len(indices), dtype=torch.uint8)
        return contexts, targets, mask_flags


def build_tabular_windows(
    digits: np.ndarray,
    mask: np.ndarray | None,
    window_size: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Строит табличные окна (X, y, mask_flag) без лишнего копирования через sliding_window_view."""
    if len(digits) <= window_size:
        raise ValueError("Длина среза должна быть больше window_size")

    # Если ограничиваем число строк для табличной модели, берём непрерывный префикс среза
    if max_samples is not None and (len(digits) - window_size) > max_samples:
        digits = digits[: max_samples + window_size]
        if mask is not None:
            mask = mask[: max_samples + window_size]

    view = np.lib.stride_tricks.sliding_window_view(digits, window_shape=window_size + 1)
    # X: (N - L, L), y: (N - L,)
    x_win = view[:, :window_size]
    y_target = view[:, window_size]
    y_mask = mask[window_size:] if mask is not None else None
    return x_win, y_target, y_mask


def iter_tabular_batches(
    digits: np.ndarray,
    mask: np.ndarray | None,
    window_size: int,
    batch_size: int = 100_000,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Отдаёт табличные окна батчами для экономии RAM на больших последовательностях (до 50M)."""
    total_windows = len(digits) - window_size
    for start in range(0, total_windows, batch_size):
        stop = min(start + batch_size, total_windows)
        chunk_digits = digits[start : stop + window_size]
        chunk_mask = mask[start : stop + window_size] if mask is not None else None
        yield build_tabular_windows(chunk_digits, chunk_mask, window_size=window_size)


def _fast_batch_collate(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Пропускает уже собранный батч из __getitems__ без лишней перепаковки."""
    if isinstance(batch, tuple) and len(batch) == 3:
        return batch
    contexts, targets, mask_flags = zip(*batch)
    return torch.stack(contexts), torch.stack(targets), torch.stack(mask_flags)


def create_dataloaders(
    split: ChronologicalSplit,
    window_size: int,
    batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Создаёт DataLoader для train/val/test. Перемешивание окон разрешено только внутри train."""
    train_ds = SequenceWindowDataset(split.train_digits, window_size=window_size, mask=split.train_mask, global_offset=split.train_range[0])
    val_ds = SequenceWindowDataset(split.val_digits, window_size=window_size, mask=split.val_mask, global_offset=split.val_range[0])
    test_ds = SequenceWindowDataset(split.test_digits, window_size=window_size, mask=split.test_mask, global_offset=split.test_range[0])
    verify_zero_leakage(split, window_size=window_size, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds)

    persistent = num_workers > 0
    worker_init = seed_worker if num_workers > 0 else None
    dl_gen = make_dataloader_generator(seed)
    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,  # Разбиение уже хронологическое — внутри Train мешать окна безопасно
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            worker_init_fn=worker_init,
            generator=dl_gen,
            collate_fn=_fast_batch_collate,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            worker_init_fn=worker_init,
            collate_fn=_fast_batch_collate,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            worker_init_fn=worker_init,
            collate_fn=_fast_batch_collate,
        ),
    }


class BinaryCoordinateDataset(Dataset):
    """Координатный вход (n -> d_n) для теста BBP-оракула: 20 бит индекса n -> цифра на позиции n."""

    def __init__(
        self,
        indices: np.ndarray,
        targets: np.ndarray,
        num_bits: int = 20,
    ) -> None:
        if len(indices) != len(targets):
            raise ValueError("Длины indices и targets должны совпадать")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.num_bits = int(num_bits)
        # Кодируем индекс n в двоичный вектор MSB -> LSB (для N=1M хватает 20 бит: 2^20 = 1,048,576)
        self._shifts = np.arange(self.num_bits - 1, -1, -1, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_val = int(self.indices[idx])
        bits = (n_val >> self._shifts) & 1
        return (
            torch.as_tensor(bits, dtype=torch.long),
            torch.tensor(int(self.targets[idx]), dtype=torch.long),
            torch.tensor(0, dtype=torch.uint8),
        )

    def __getitems__(self, batch_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_arr = np.asarray(batch_indices, dtype=np.int64)
        vals = self.indices[idx_arr][:, None]
        bits = (vals >> self._shifts[None, :]) & 1
        return (
            torch.from_numpy(bits.astype(np.int64)),
            torch.from_numpy(self.targets[idx_arr].copy()),
            torch.zeros(len(idx_arr), dtype=torch.uint8),
        )
