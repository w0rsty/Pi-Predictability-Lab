# Тесты всех 4 архитектур, legacy-ремаппинга SDPA, бинарного датасета, статистики и пакета src/viz/
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import torch

import src.visualizer as viz_shim
import src.viz as viz_pkg
from src.dataset import BinaryCoordinateDataset
from src.models import DigitTransformer, build_model, count_parameters
from src.stats_tests import run_full_statistical_suite
from src.trainer import remap_legacy_state_dict


@pytest.mark.parametrize("arch", ["mlp", "cnn", "gru", "transformer"])
def test_all_architectures_build_and_forward(arch: str):
    torch.manual_seed(42)
    model = build_model(arch=arch, radix=10, window_size=32, size_tier="light", dropout=0.0)
    model.eval()
    n_params = count_parameters(model)
    assert 35_000 <= n_params <= 70_000

    x = torch.randint(0, 10, (4, 32), dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_legacy_mha_to_sdpa_state_dict_remapping():
    torch.manual_seed(42)
    src_model = DigitTransformer(radix=10, window_size=16, size_tier="light", dropout=0.0).eval()
    x = torch.randint(0, 10, (2, 16), dtype=torch.long)
    with torch.no_grad():
        expected_logits = src_model(x)

    legacy_sd = {}
    for k, v in src_model.state_dict().items():
        lk = (
            k.replace(".qkv_proj.weight", ".attn.in_proj_weight")
            .replace(".qkv_proj.bias", ".attn.in_proj_bias")
            .replace(".out_proj.", ".attn.out_proj.")
        )
        legacy_sd[lk] = v.clone()

    remapped_sd, changed = remap_legacy_state_dict(legacy_sd)
    assert changed is True

    dst_model = DigitTransformer(radix=10, window_size=16, size_tier="light", dropout=0.0).eval()
    dst_model.load_state_dict(remapped_sd)
    with torch.no_grad():
        actual_logits = dst_model(x)
    assert torch.allclose(expected_logits, actual_logits, atol=1e-6)


def test_binary_coordinate_dataset_20bit_encoding():
    indices = np.array([0, 5, 13, 1_048_575], dtype=np.int64)
    targets = np.array([3, 1, 4, 9], dtype=np.int64)
    ds = BinaryCoordinateDataset(indices=indices, targets=targets, num_bits=20)
    assert len(ds) == 4

    b0, t0, m0 = ds[0]
    b5, t5, _ = ds[1]
    bmax, tmax, _ = ds[3]

    assert b0.shape == (20,) and b0.dtype == torch.long
    assert int(b0.sum()) == 0 and int(t0) == 3 and int(m0) == 0
    # 5 = 0b...00101 -> биты в позициях -3 и -1 равны 1
    assert b5[-4:].tolist() == [0, 1, 0, 1] and int(t5) == 1
    # 2^20 - 1 -> все 20 бит равны 1
    assert int(bmax.sum()) == 20 and int(tmax) == 9

    batch_b, batch_t, batch_m = ds.__getitems__([0, 1, 2, 3])
    assert batch_b.shape == (4, 20)
    assert torch.equal(batch_b[1], b5)
    assert batch_t.tolist() == [3, 1, 4, 9]
    assert batch_m.tolist() == [0, 0, 0, 0]


def test_statistical_suite_uniform_vs_constant():
    rng = np.random.default_rng(42)
    uniform_seq = rng.integers(0, 10, size=20_000, dtype=np.uint8)
    u_res = run_full_statistical_suite("uniform_test", uniform_seq, radix=10, max_lag=16)

    assert u_res["chi2_passed"] is True
    assert u_res["ks_passed"] is True
    assert u_res["empirical_entropy_bits"] > 3.30
    assert u_res["lzma_bits_per_digit"] > 3.20

    const_seq = np.full(5_000, 7, dtype=np.uint8)
    c_res = run_full_statistical_suite("const_test", const_seq, radix=10, max_lag=16)
    assert c_res["chi2_passed"] is False
    assert c_res["ks_passed"] is False
    assert c_res["empirical_entropy_bits"] == pytest.approx(0.0, abs=1e-9)
    assert c_res["lzma_bits_per_digit"] < 0.2


def test_viz_package_exports_and_smoke_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert set(viz_pkg.__all__) == set(viz_shim.__all__)
    captured_figs: list[go.Figure] = []

    def fake_write_image(self: go.Figure, file: str, *args, **kwargs) -> None:
        captured_figs.append(self)

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)

    df_mem = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_acc": [0.1, 0.5, 0.9],
            "val_acc": [0.1, 0.1, 0.1],
            "test_acc": [0.1, 0.1, 0.1],
            "train_bits": [3.3, 1.8, 0.4],
            "val_bits": [3.3, 4.0, 5.5],
            "test_bits": [3.3, 4.1, 6.0],
        }
    )
    html_p, png_p = viz_pkg.plot_exp4_memorization_scissors(df_mem, tmp_path)
    assert html_p.exists() and html_p.suffix == ".html"
    assert png_p.name == "exp4_memorization_scissors.png"
    assert len(captured_figs) == 1
    assert len(captured_figs[0].data) == 6
