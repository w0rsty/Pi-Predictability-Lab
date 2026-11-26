#!/usr/bin/env bash
set -euo pipefail

# Полное воспроизведение исследовательского ядра (Эксперименты 1–5) на RTX 3060 Ti (seed=42)
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" experiments/01_generate_data.py
"$PYTHON_BIN" experiments/02_run_baselines.py

for arch in mlp cnn gru transformer; do
  "$PYTHON_BIN" experiments/03_exp1_sanity_check.py --arch "$arch"
done
"$PYTHON_BIN" experiments/03b_phase3_5_ppm.py

"$PYTHON_BIN" experiments/04_exp2_constants.py --arch all
"$PYTHON_BIN" experiments/05_exp3_bases.py --arch all
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" experiments/06_exp4_scaling.py
"$PYTHON_BIN" experiments/07_exp5_mech_interp_and_crypto.py
