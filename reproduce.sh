#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---quick}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
if [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
fi

case "$MODE" in
  --quick)
    echo "[quick] 1/3 Синтаксическая проверка исходников..."
    "$PYTHON_BIN" -m py_compile src/*.py src/viz/*.py experiments/*.py tests/*.py
    echo "[quick] 2/3 Прогон автотестов..."
    "$PYTHON_BIN" -m pytest -v
    echo "[quick] 3/3 Двусторонний smoke-тест (H1: 1/97 vs H0: равномерный шум)..."
    "$PYTHON_BIN" - <<'PY'
import math
import numpy as np
from src.baselines import fit_and_evaluate_ppm
from src.data_generator import generate_rational_digits
from src.dataset import create_dataloaders, split_sequence_chronologically, verify_zero_leakage

# H1: периодическая дробь 1/97 (P=96)
digits_h1 = generate_rational_digits(1, 97, length=5000, radix=10)
split_h1 = split_sequence_chronologically(digits_h1)
ld_h1 = create_dataloaders(split_h1, window_size=32, batch_size=128, num_workers=0, pin_memory=False)
verify_zero_leakage(split_h1, window_size=32, train_ds=ld_h1["train"].dataset, val_ds=ld_h1["val"].dataset, test_ds=ld_h1["test"].dataset)
m_h1, _ = fit_and_evaluate_ppm(split_h1, radix=10, window_size=32, max_order=8)
bits_h1 = m_h1["cross_entropy"] / math.log(2.0)
print(f"[smoke H1] 1/97 PPM-C (o=8): acc={m_h1['overall_acc']*100:.2f}%, bits={bits_h1:.4f}")
assert m_h1["overall_acc"] > 0.99 and bits_h1 < 0.1

# H0: равномерный i.i.d. шум (order=2) — проверка потолка ~log2(10) ~ 3.3219 бит
digits_h0 = np.random.default_rng(42).integers(0, 10, size=40_000, dtype=np.uint8)
split_h0 = split_sequence_chronologically(digits_h0)
ld_h0 = create_dataloaders(split_h0, window_size=32, batch_size=128, num_workers=0, pin_memory=False)
verify_zero_leakage(split_h0, window_size=32, train_ds=ld_h0["train"].dataset, val_ds=ld_h0["val"].dataset, test_ds=ld_h0["test"].dataset)
m_h0, _ = fit_and_evaluate_ppm(split_h0, radix=10, window_size=32, max_order=2)
bits_h0 = m_h0["cross_entropy"] / math.log(2.0)
print(f"[smoke H0] Uniform PPM-C (o=2): acc={m_h0['overall_acc']*100:.2f}%, bits={bits_h0:.4f}")
assert 3.20 < bits_h0 < 3.45 and m_h0["overall_acc"] < 0.13
PY
    ;;
  --full)
    "$PYTHON_BIN" experiments/01_generate_data.py
    "$PYTHON_BIN" experiments/02_run_baselines.py
    for arch in mlp cnn gru transformer; do
      "$PYTHON_BIN" experiments/03_exp1_sanity_check.py --arch "$arch"
    done
    "$PYTHON_BIN" experiments/03b_ppm_baseline.py
    "$PYTHON_BIN" experiments/04_exp2_constants.py --arch all
    "$PYTHON_BIN" experiments/05_exp3_bases.py --arch all
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" experiments/06_exp4_scaling.py
    "$PYTHON_BIN" experiments/07_exp5_mech_interp_and_crypto.py
    ;;
  *)
    echo "Использование: $0 [--quick | --full]" >&2
    exit 1
    ;;
esac
