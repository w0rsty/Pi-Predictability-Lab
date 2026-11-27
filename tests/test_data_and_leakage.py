# Тесты точности генераторов, выравнивания окон и проверки утечки данных
import mpmath as mp
import numpy as np
import pytest
import torch

from src.data_generator import (
    apply_csprng_backdoor,
    generate_champernowne_digits,
    generate_constant_digits,
    generate_csprng_digits,
    generate_e_cf_tokens,
    generate_pi_cf_tokens,
    generate_rational_digits,
    generate_rational_mpmath_digits,
    get_or_create_dataset,
)
from src.dataset import SequenceWindowDataset, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.stats_tests import chi_square_uniformity


def _arr_to_str(arr: np.ndarray) -> str:
    return "".join(str(int(x)) for x in arr)


def test_mathematical_generators_exact_prefixes() -> None:
    pi_d = generate_constant_digits("pi", 25, radix=10)
    assert _arr_to_str(pi_d) == "1415926535897932384626433"

    e_d = generate_constant_digits("e", 25, radix=10)
    assert _arr_to_str(e_d) == "7182818284590452353602874"

    s2_d = generate_constant_digits("sqrt2", 25, radix=10)
    assert _arr_to_str(s2_d) == "4142135623730950488016887"

    rat97 = generate_rational_digits(1, 97, 120, radix=10)
    assert _arr_to_str(rat97[:25]) == "0103092783505154639175257"
    assert np.array_equal(rat97[:24], rat97[96:120])

    pi_cf = generate_pi_cf_tokens(30)
    expected_pi_cf = [3, 10, 7, 10, 1, 5, 10, 1, 10, 2, 9, 2, 10, 1, 10, 1, 10, 1, 10, 2, 10, 1, 10, 3, 10, 1, 10, 1, 4, 10]
    assert pi_cf.tolist() == expected_pi_cf

    e_cf = generate_e_cf_tokens(24)
    expected_e_cf = [2, 10, 1, 10, 2, 10, 1, 10, 1, 10, 4, 10, 1, 10, 1, 10, 6, 10, 1, 10, 1, 10, 8, 10]
    assert e_cf.tolist() == expected_e_cf

    champ = generate_champernowne_digits(25, radix=10)
    assert _arr_to_str(champ) == "1234567891011121314151617"


def test_sequence_window_dataset_alignment() -> None:
    digits = (np.arange(150) % 10).astype(np.uint8)
    mask = (np.arange(150) % 7 == 0).astype(np.uint8)
    window_size = 16
    ds = SequenceWindowDataset(digits, window_size=window_size, mask=mask, global_offset=1000)

    assert len(ds) == 150 - window_size
    for idx in (0, 1, 17, 63, len(ds) - 1):
        ctx, tgt, flag = ds[idx]
        expected_ctx = torch.as_tensor(digits[idx : idx + window_size], dtype=torch.long)
        assert torch.equal(ctx, expected_ctx)
        assert tgt.item() == int(digits[idx + window_size])
        assert flag.item() == int(mask[idx + window_size])

    batch_ctx, batch_tgt, batch_flag = ds.__getitems__([2, 19, 44])
    for row, idx in enumerate((2, 19, 44)):
        c_i, t_i, f_i = ds[idx]
        assert torch.equal(batch_ctx[row], c_i)
        assert batch_tgt[row].item() == t_i.item()
        assert batch_flag[row].item() == f_i.item()


def test_verify_zero_leakage_positive_and_negative() -> None:
    digits = (np.arange(500) % 10).astype(np.uint8)
    window_size = 16
    split = split_sequence_chronologically(digits, None, 0.8, 0.1, 0.1)
    loaders = create_dataloaders(split, window_size=window_size, batch_size=32, num_workers=0, pin_memory=False)
    res = verify_zero_leakage(
        split,
        window_size=window_size,
        train_ds=loaders["train"].dataset,
        val_ds=loaders["val"].dataset,
        test_ds=loaders["test"].dataset,
    )
    assert res["zero_overlap_verified"] is True

    # Смещаем val на 1 позицию влево, чтобы последнее train-окно перекрылось с первым val-окном
    tr_e = split.train_range[1]
    va_e = split.val_range[1]
    leaky_val_ds = SequenceWindowDataset(digits[tr_e - 1 : va_e], window_size=window_size, global_offset=tr_e - 1)
    with pytest.raises(AssertionError):
        verify_zero_leakage(
            split,
            window_size=window_size,
            train_ds=loaders["train"].dataset,
            val_ds=leaky_val_ds,
            test_ds=loaders["test"].dataset,
        )

    # Подмена содержимого val_ds.digits при валидном global_offset тоже должна ловиться
    tampered_val_digits = split.val_digits.copy()
    tampered_val_digits[0] = np.uint8((int(tampered_val_digits[0]) + 1) % 10)
    tampered_val_ds = SequenceWindowDataset(tampered_val_digits, window_size=window_size, global_offset=split.val_range[0])
    with pytest.raises(AssertionError):
        verify_zero_leakage(
            split,
            window_size=window_size,
            train_ds=loaders["train"].dataset,
            val_ds=tampered_val_ds,
            test_ds=loaders["test"].dataset,
        )


def test_mpmath_global_state_isolation() -> None:
    orig_dps, orig_prec = mp.mp.dps, mp.mp.prec
    try:
        mp.mp.dps = 17
        _ = generate_constant_digits("pi", 300, radix=10)
        assert mp.mp.dps == 17
        _ = generate_constant_digits("pi", 300, radix=16)
        assert mp.mp.dps == 17
        _ = generate_rational_mpmath_digits(1, 97, 250)
        assert mp.mp.dps == 17
        _ = generate_pi_cf_tokens(300)
        assert mp.mp.dps == 17
    finally:
        mp.mp.dps = orig_dps
        mp.mp.prec = orig_prec


def test_deterministic_csprng_and_cache_integrity(tmp_path) -> None:
    with pytest.raises(ValueError, match="radix"):
        generate_csprng_digits(16, radix=300)
    with pytest.raises(ValueError, match="radix"):
        generate_csprng_digits(16, radix=0)
    with pytest.raises(ValueError, match="Длина"):
        generate_csprng_digits(0, radix=10)

    s1 = generate_csprng_digits(4096, radix=10, deterministic=True, seed=123)
    s2 = generate_csprng_digits(4096, radix=10, deterministic=True, seed=123)
    s3 = generate_csprng_digits(4096, radix=10, deterministic=True, seed=124)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    assert int(s1.min()) >= 0 and int(s1.max()) <= 9

    base = generate_csprng_digits(50_000, radix=10, deterministic=True, seed=7)
    bd_digits, bd_mask = apply_csprng_backdoor(base, lag1=24, lag2=7)
    assert int(bd_mask.sum()) > 1000
    assert chi_square_uniformity(bd_digits, radix=10)["chi2_p_value"] > 0.01

    cfg = {
        "seed": 42,
        "paths": {"data_dir": str(tmp_path), "runs_dir": str(tmp_path / "runs")},
        "data": {"default_length": 2000, "split": {"train": 0.8, "val": 0.1, "test": 0.1}, "default_window_size": 16},
        "hacked_pi": {},
    }
    art_cs = get_or_create_dataset("csprng", config=cfg, length=2000)
    art_bd = get_or_create_dataset("csprng_backdoor", config=cfg, length=2000)
    assert art_bd.metadata["parent_sha256"] == art_cs.metadata["hash"]

    # Порча байта в кэше csprng_backdoor должна ловиться по SHA-256 и вызывать перегенерацию
    corrupted = art_bd.digits.copy()
    corrupted[10] = (corrupted[10] + 1) % 10
    np.save(art_bd.digits_path, corrupted)
    with pytest.warns(UserWarning, match="SHA-256"):
        reloaded = get_or_create_dataset("csprng_backdoor", config=cfg, length=2000)
    assert reloaded.from_cache is False
    assert np.array_equal(reloaded.digits, art_bd.digits)

    # Порча байта в mask.npy (даже без изменения mask_count) должна ловиться по mask_sha256 и вызывать перегенерацию
    assert reloaded.mask is not None and reloaded.mask_path is not None
    corrupted_mask = reloaded.mask.copy()
    zero_idx = int(np.flatnonzero(corrupted_mask == 0)[0])
    one_idx = int(np.flatnonzero(corrupted_mask == 1)[0])
    corrupted_mask[zero_idx] = 1
    corrupted_mask[one_idx] = 0
    np.save(reloaded.mask_path, corrupted_mask)
    with pytest.warns(UserWarning, match="SHA-256 для маски"):
        reloaded_mask = get_or_create_dataset("csprng_backdoor", config=cfg, length=2000)
    assert reloaded_mask.from_cache is False
    assert np.array_equal(reloaded_mask.mask, reloaded.mask)

    # Смена родительского csprng должна автоматически инвалидировать кэш csprng_backdoor
    art_cs_new = get_or_create_dataset("csprng", config=cfg, length=2000, force_regenerate=True)
    assert art_cs_new.metadata["hash"] != art_cs.metadata["hash"]
    with pytest.warns(UserWarning, match="родительского csprng"):
        bd_invalidated = get_or_create_dataset("csprng_backdoor", config=cfg, length=2000)
    assert bd_invalidated.from_cache is False
    assert bd_invalidated.metadata["parent_sha256"] == art_cs_new.metadata["hash"]
