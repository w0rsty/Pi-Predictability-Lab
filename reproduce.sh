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
    echo "[quick] 3/3 Быстрый smoke-тест сплита и PPM-C на дроби 1/97..."
    "$PYTHON_BIN" - <<'PY'
import math
from src.baselines import fit_and_evaluate_ppm
from src.data_generator import generate_rational_digits
from src.dataset import split_sequence_chronologically, verify_zero_leakage

digits = generate_rational_digits(1, 97, length=5000, radix=10)
split = split_sequence_chronologically(digits)
verify_zero_leakage(split, window_size=32)
m, _ = fit_and_evaluate_ppm(split, radix=10, window_size=32, max_order=8)
bits = m["cross_entropy"] / math.log(2.0)
print(f"[smoke] 1/97 PPM-C: acc={m['overall_acc']*100:.2f}%, bits={bits:.4f}")
assert m["overall_acc"] > 0.99 and bits < 0.1
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
