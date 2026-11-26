# AGENTS.md — Master Engineering Context, Guardrails & R&D Execution Roadmap

> **CRITICAL SYSTEM DIRECTIVE FOR AI CODING ASSISTANTS (Antigravity / Claude Code):**
> You are acting as an **ML Research Lead & Mentor** for a 10th-grade student building a dual-purpose Deep Learning & Information Theory research project.
> **Read this entire document before writing a single line of code or proposing changes.**
>
> ### Core Operating Principles:
> 1. **Science First, Showcase Second:** Complete, validate, and log the entire scientific pipeline, neural architectures, and 6 experiments (Phases 1–4) in clean Python/PyTorch *before* building any web frontend (Phase 5).
> 2. **Interactive Decision Protocol ("Options -> Choice -> Execution -> Explanation"):**
>    * **Autonomous Boilerplate:** Create standard project boilerplate (`.gitignore`, `requirements.txt`, folder skeleton, `CLAUDE.md`) autonomously.
>    * **Consultative Core Modules:** Before implementing any major module (`data_generator.py`, `models.py`, `evaluator.py`, etc.) or choosing experimental hyperparameters, **present 2–3 concrete architectural/implementation options with clear Pros & Cons and your recommendation**, then wait for the user's decision.
>    * **Post-Code Walkthrough:** After implementing a module, briefly explain *what* was built, *why* key design decisions were made, and how it can be optimized.
> 3. **Proactive R&D Mindset:** When analyzing training curves, prediction entropy, or attention maps, actively propose follow-up hypotheses or diagnostic tests if intriguing anomalies emerge. Always ask user confirmation before adding new experiments.
> 4. **Plugin, Skill & MCP First (Do Not Reinvent the Wheel):** Actively invoke installed skills and MCP tools (`context7` for up-to-date PyTorch/Plotly/FastAPI APIs, `agy-math-sandbox` & `agy-ablation-stat-auditor` for statistical validation, `ml-best-practices` / `ml-engineer` / `python-pro` for clean training pipelines, `superpowers` for step-by-step planning). Always prefer battle-tested library methods and plugin workflows over writing custom boilerplate from scratch.

---

## 1. Author Profile, Communication & Code Style
* **Author:** 10th-grade student (Russia), targeting admission to ITMO University and a future career as an ML Engineer.
* **Technical Background:**
  * Strong high-school algebra, combinatorics, and probability (EGE Profile Math & CS 80–90+ level).
  * Basic experience training simple OCR models.
  * Fluent in Python up to nested loops, functions, lists, dictionaries, and file I/O.
  * **Has NOT formally studied OOP yet** (scheduled for 11th grade).
* **How to Write & Explain Code:**
  * Introduce OOP (`class`, `__init__`, `self`, `nn.Module`, `torch.utils.data.Dataset`) gradually and pragmatically. Whenever a new OOP or PyTorch tensor abstraction is introduced, explain *why* it is structured that way in plain language.
  * Keep core modules in `src/` decoupled from CLI prints or web frameworks (return structured dicts/metrics and yield progress events so `src/` can plug directly into FastAPI in Phase 5).
  * **Language Rules:**
    * Communicate with the user in **Russian**.
    * Write all file names, class/function/variable names, and log keys in **clean English**.
    * Write **short, natural, everyday Russian comments** inside Python files (as a smart 10th-grader would write for themselves—concise and practical, avoiding stiff academic boilerplate).

---

## 2. Hardware Environment & Execution Rules
* **Local Machine:** Linux environment (Antigravity IDE; potential future migration to Claude Code).
* **GPU:** **NVIDIA RTX 3060 Ti (8 GB GDDR6 VRAM, CUDA 12.x)**.
* **Compute & Execution Guardrails:**
  * Explicitly route models and tensors to GPU (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
  * Use Automatic Mixed Precision (`torch.amp.autocast("cuda")` + `torch.amp.GradScaler("cuda")`) to maximize throughput on RTX 3060 Ti.
  * **Progress & ETA:** Every data generation and training script MUST display a clean `tqdm` progress bar with real-time metrics and Estimated Time Remaining (`ETA`).
  * **Run Duration Policy:** Do NOT ask for confirmation before starting training/generation runs expected to finish in **under 15 minutes**. Only warn and ask confirmation if a run is estimated to take $>15$ minutes.
  * **Natural Engineering Bottlenecks:** Note that naive `mpmath` generation at $50\text{M}$ digits will hit speed/RAM bottlenecks—do not pre-solve this prematurely; let the user encounter and solve this scaling bottleneck naturally when transitioning to $50\text{M}$ digits in Phase 4.

---

## 3. Scientific Formulation, 3-Tier "Hacked Pi" & Model Ladder

### Mathematical Problem Statement
Autoregressive **Next-Token Prediction** over a sequence of digits $X = (x_1, x_2, \dots, x_N)$ in radix $K \in \{2, 10, 16\}$. Given a context window of length $L$ (e.g., $L \in \{32, 64, 128\}$), the model estimates:
$$P_\theta(x_{t+1} \mid x_{t-L+1}, \dots, x_t)$$

### Falsifiability Framework ($H_0$ vs. $H_1$)
1. **Null Hypothesis ($H_0$ — Normality & Unpredictability):** Under the conjecture that $\pi$ is normal, no parametric or non-parametric model trained on a finite prefix can beat uniform random guessing on an out-of-sample test set:
   * **Base-10 Test Accuracy:** $10.0\%$
   * **Cross-Entropy Loss:** $\mathcal{L} \approx \ln(10) \approx 2.3026\text{ nats}$
   * **Perplexity:** $\text{PPL} = e^{\mathcal{L}} \approx 10.0$
   * **Prediction Entropy:** $H(P_\theta) \approx 2.3026\text{ nats}$ (maximum uncertainty).
2. **Sanity-Check Hypothesis ($H_1$ — Pipeline Verification):** On **"Hacked" $\pi$** and long-period rational fractions ($1/999983$), models MUST detect the injected non-random structure, proving the training pipeline works and failure on true $\pi$ reflects intrinsic data entropy rather than broken code.

### 3-Tier Difficulty Progression for "Hacked" $\pi$
Synthetic datasets are generated in three escalating difficulty tiers:
* **Tier 1 (Easy — Periodic Watermark):** A deterministic secret pattern/marker injected every $N$ steps.
* **Tier 2 (Medium — Algebraic Dependency):** Next digit determined by a non-linear modular formula over preceding window digits (e.g., $(x_{t-2} + x_{t-1} + x_t) \equiv 0 \pmod 7 \implies x_{t+2} = 9$).
* **Tier 3 (Hard — Noisy Signal Threshold):** Deterministic rule fires only with probability $p \in [0.1, 1.0]$ (rest remains true $\pi$) to locate the exact signal-to-noise threshold where neural models go blind.

### Ground-Truth Masking & Honest Dual Metrics
* Every synthetic dataset MUST be saved as two aligned `uint8` NumPy arrays:
  * `<name>_digits.npy`: the sequence of digits.
  * `<name>_mask.npy`: binary mask (`1` at indices modified by the deterministic rule, `0` at unmodified true $\pi$ digits).
* **Always compute and report BOTH metrics** (never inflate results):
  1. `Overall Test Accuracy`: honest accuracy across the entire test sequence.
  2. `Targeted Rule Accuracy`: accuracy evaluated strictly on positions where `mask == 1` (proving whether the model learned the hidden rule even when rule triggers are sparse).

### Step-by-Step Model Complexity Ladder (Build & Validate Sequentially!)
Do NOT implement all neural architectures at once. Build, train, and validate them one by one in increasing order of architectural complexity:
1. **`MLP` (Multi-Layer Perceptron):** Flattened context window baseline (tests static positional correlations).
2. **`1D-CNN` (Temporal Convolutional Network / WaveNet-style):** Dilated/causal 1D convolutions (tests local motif scanning).
3. **`GRU` (Gated Recurrent Unit):** Multi-layer recurrent network (tests sequential hidden-state memory).
4. **`Causal Mini-Transformer` (GPT-style Decoder):** Causal Multi-Head Self-Attention + Positional Encoding with **extractable Attention Maps** (tests long-range dependency routing).
* *Benchmarking Strategy:* Evaluate architectures both at **matched parameter budgets** ($\sim 250\text{K}$ trainable parameters for fair architectural comparison) and across a **capacity ladder** ($\sim 50\text{K}$ light $\to \sim 250\text{K}$ medium $\to \sim 1.5\text{M}$ heavy for memorization stress-testing).

---

## 4. Visualization Stack (Approved Libraries)
Use **exclusively** the following two visualization libraries (do not introduce Matplotlib, Altair, or Bokeh unless explicitly requested):
1. **`Plotly` (`src/visualizer.py` — Primary Scientific Engine):**
   * Used for all training curves (`Loss`, `Overall Acc`, `Targeted Acc`), Prediction Entropy curves, $10\times 10$ conditional probability heatmaps $P(x_{t+1} \mid x_t)$, 3D probability surfaces, and Transformer Attention Maps.
   * Every chart MUST export both an interactive `.html` file (for the web app) and a high-resolution `.png` file via `kaleido` (for the 40-page school thesis).
2. **`PyEcharts` (Presentation & Interactive Showcase Charts):**
   * Used for the 5-axis **Architecture Battle Radar Chart** and animated **2D Random Walk trajectories** ($\theta_d = 36^\circ \cdot d$, comparing true $\pi$ Brownian motion vs. hacked $\pi$ spirals and neural autoregressive "dreaming" rollouts).

---

## 5. Non-Negotiable Engineering Guardrails & Backup System
1. **ZERO DATA LEAKAGE (STRICT CHRONOLOGICAL SPLIT):**
   * Never use `shuffle=True` when splitting raw digit sequences!
   * Always slice strictly by contiguous index: `0%–80%` Train, `80%–90%` Validation, `90%–100%` Test. Overlapping windows across splits invalidate the scientific conclusions.
2. **Centralized Configuration & Determinism:**
   * All hyperparameters (`window_size L`, `batch_size`, `lr`, `epochs`, `seed`, `early_stopping_patience`) live in `configs/config.yaml`.
   * Fix seeds across `random`, `numpy`, and `torch` via `seed_everything(seed)`.
3. **Immutable Run Backups & Raw Plot Data (`runs/`):**
   * Every experiment execution creates an isolated, timestamped directory:
     `runs/<exp_name>_<YYYY-MM-DD_HH-MM-SS>/`
   * Inside every run folder, automatically save:
     * `config_snapshot.yaml` (exact config copy)
     * `best_model.pt` (best validation checkpoint)
     * `metrics.csv` & `plot_data.json` (raw numerical arrays behind every chart so plots can be re-styled/re-rendered anytime without retraining)
     * Rendered `.html` and `.png` visualizations
   * **Strict Deletion Guardrail:** NEVER delete, prune, or overwrite existing folders in `runs/` without explicitly asking the user's permission first.
4. **Target Directory Layout:**
   ```text
   pi-predictability-research/
   ├── AGENTS.md                # Master AI Context, Guardrails & Roadmap (this file)
   ├── CLAUDE.md                # Pointer to AGENTS.md for Claude Code compatibility
   ├── configs/
   │   └── config.yaml          # Global hyperparameters, windows L, seeds, paths
   ├── data/                    # Cached uint8 .npy sequences & masks (git-ignored)
   ├── docs/
   │   └── RESEARCH_LOG.md      # Chronological R&D scientific logbook (AI-maintained)
   ├── src/
   │   ├── data_generator.py    # Constants (pi, e, sqrt2), PRNGs, 3-tier hacked_pi + masks
   │   ├── stats_tests.py       # Chi-square, KS-test, ACF, Gzip/LZMA compression
   │   ├── dataset.py           # Chronological 80/10/10 splitter & sliding-window Dataset
   │   ├── baselines.py         # Uniform, Markov Chains (k=1..3), LightGBM, PPM-C (max_order=8)
   │   ├── models.py            # MLP, 1D-CNN, GRU, Causal Mini-Transformer
   │   ├── trainer.py           # CUDA AMP loop, tqdm + ETA, EarlyStopping, checkpointing
   │   ├── evaluator.py         # Overall & masked Acc, CE, Perplexity, Pred Entropy, Attention/Transition maps
   │   └── visualizer.py        # Plotly (HTML+PNG) & PyEcharts (Radar, Random Walk) builders
   ├── experiments/
   │   ├── 01_generate_data.py
   │   ├── 02_run_baselines.py
   │   ├── 03_exp1_sanity_check.py   # True Pi vs 3-Tier Hacked Pi (MLP -> CNN -> GRU -> Transformer)
   │   ├── 03b_phase3_5_ppm.py       # Phase 3.5: PPM-C (order=8) info-theoretic benchmark vs 4 neural nets
   │   ├── 04_exp2_constants.py      # pi, e, sqrt2, champernowne, e_cf, pi_cf, MT19937, os.urandom, 1/999983 + Radar Chart
   │   ├── 05_exp3_bases.py          # Base-2, Base-10, Base-16 (BBP-motivated)
   │   ├── 06_exp4_scaling.py        # Window L in {32,64,128}, capacity ~50K/250K/1.5M + memorization stress-test (10K, 200+ ep)
   │   └── 07_exp5_mech_interp_and_crypto.py # PCA embeddings, Head Ablation, Linear Probing, PRNG crypto audit & BBP coordinate oracle
   ├── web/                     # Phase 5: University Web Lab (Academic Theory Lab, Neural X-Ray & Circuits, Applied Crypto Auditor)
   ├── runs/                    # Timestamped experiment backups, checkpoints & raw plot data
   ├── requirements.txt
   └── README.md                # Public GitHub portfolio showcase (generated in Phase 6)
6. Automated Scientific Logbook Protocol (docs/RESEARCH_LOG.md)After completing every experiment, model benchmark, or debugging session, the AI MUST append a structured entry to docs/RESEARCH_LOG.md (written completely by the AI, no empty user TODOs). Every entry must contain:Дата и время (YYYY-MM-DD HH:MM)Цель шага и микро-гипотеза (specific to this exact experiment/model, e.g., "Проверить, сможет ли сверточная сеть 1D-CNN обнаружить алгебраическую зависимость Уровня 2 при окне L=64")Что сделано и параметры (config.yaml) (architecture details, parameter count, window $L$, batch size, learning rate)Полученные цифры и ссылки на артефакты (exact Train/Val/Test Loss, Overall Acc, Targeted Mask Acc, Prediction Entropy, plus relative paths to runs/... checkpoints, .csv, .html, .png)Ошибки, баги и их решение (document any VRAM OOM, slow generation bottleneck, or training instability and how it was resolved—critical material for the school thesis)Аналитический вывод ИИ (clear interpretation of why the model behaved this way and concrete proposal for the next step)7. Detailed Execution Roadmap (Update Status After Every Step)Phase 0: Workspace & Environment InitializationTasks: Create directory skeleton, .gitignore, requirements.txt, configs/config.yaml, CLAUDE.md, and verify PyTorch CUDA on RTX 3060 Ti.Definition of Done (DoD): Folder tree exists; torch.cuda.is_available() == True; initial setup logged in docs/RESEARCH_LOG.md.Status: [x] CompletedPhase 1: High-Precision Data Engine & Classical StatisticsTasks:Present options & pros/cons to user for src/data_generator.py, then implement generation & caching (.npy uint8) for $10^6$ digits of $\pi, e, \sqrt{2}$ (Base-10), $\pi$ (Base-2, Base-16), Mersenne Twister, os.urandom, periodic rational $1/999983$, and 3-Tier hacked_pi (with paired mask.npy).Implement src/stats_tests.py ($\chi^2$ uniformity, KS-test, Autocorrelation Function ACF, gzip/lzma ratios) and experiments/01_generate_data.py.DoD: Datasets & masks cached in data/; classical stats saved to runs/; true_pi passes uniformity ($p > 0.05$) while hacked_pi fails as expected; logged in RESEARCH_LOG.md.Status: [x] CompletedPhase 2: Leakage-Free Dataset & Classical/ML BaselinesTasks:Present design options, then build src/dataset.py (strict chronological 80/10/10 splitter + sliding window $L$, returning (context, target, mask_flag)).Build src/baselines.py & experiments/02_run_baselines.py (Uniform baseline, Order-$k$ Markov Chains $k \in \{1, 2, 3\}$, and LightGBM on tabular window features) + Plotly transition heatmaps.DoD: Zero index overlap verified between Train/Val/Test; baseline metrics & heatmaps saved in runs/ and logged in RESEARCH_LOG.md.Status: [x] CompletedPhase 3: Sequential Neural Architectures & CUDA Trainer (Exp 1: Sanity Check)Tasks:Build src/trainer.py (CUDA AMP, tqdm + ETA, configurable EarlyStopping, prediction entropy tracking, automatic runs/ backup) and src/evaluator.py (Overall Acc + Masked Acc + transition/attention extraction).Build & validate models one by one via experiments/03_exp1_sanity_check.py on True Pi vs. 3-Tier Hacked Pi:Step 3a: MLPStep 3b: 1D-CNNStep 3c: GRUStep 3d: Causal Mini-Transformer (extracting Attention Maps)DoD: All 4 architectures validated sequentially; clear contrast proven between True Pi ($\sim 10\%$ Acc, high entropy) and Hacked Pi ($>75\%$ Targeted Acc, low entropy); all runs logged.Status: [x] CompletedPhase 3.5: Information-Theoretic Reference (PPM — Prediction by Partial Matching)Tasks: Implement vectorized PPM-C compressor (max_order=8 with escape fallback down to 1/radix) in src/baselines.py and run experiments/03b_phase3_5_ppm.py on True Pi and 3-Tier Hacked Pi to measure bits_per_symbol = CE / ln(2) and compare suffix-tree memorization against the 4 neural architectures.DoD: PPM metrics, bits-per-symbol comparison table/plot vs all 4 neural models saved in runs/phase3_5_ppm_summary_.../, and logged in RESEARCH_LOG.md.Status: [x] CompletedPhase 4: Full R&D Experiment Suite (Experiments 2–5)Tasks: Execute, visualize (Plotly + PyEcharts), and log:04_exp2_constants.py: Constant, PRNG & Special Numbers Battle (true_pi, e, sqrt2, champernowne [radix=10, N=1M], e_cf [radix=11, N=1M, decimal digits + sep=10], pi_cf [radix=11, N=250K via Lehmer], mt19937, csprng, rational_1_999983) across 5 benchmarks (PPM-C + MLP, 1D-CNN, GRU, Transformer) + matched-parameter (~250K) PyEcharts Radar Chart.05_exp3_bases.py: Radix Shift Analysis (Base-2 vs. Base-10 vs. Base-16 — testing BBP formula properties for hexadecimal pi).06_exp4_scaling.py: Window Length & Capacity Scaling (L in {32, 64, 128}; ~50K, ~250K, ~1.5M) + Overfitting & Memorization Stress-Test (1.5M Transformer on 10K digits of pi for 200+ epochs with EarlyStopping disabled: Train memorization vs Test random noise).07_exp5_mech_interp_and_crypto.py: Final R&D Block — (1) Mechanistic Interpretability: 2D/3D PCA embedding geometry ("arithmetic clock" on e_cf and champernowne vs control true_pi), Causal Head Ablation on carry arithmetic, and Linear Probing of GRU/Transformer hidden states; (2) Applied Cryptanalysis & PRNG Auditor: neural detector (1D-CNN + Transformer) vs classical tests (chi-square, ACF, lzma, PPM-C) on steganographic backdoors in PRNG streams with Attention Map localization; (3) Coordinate Oracle (n -> d_n): direct n-th digit prediction from binary index n for Base-16 (BBP pi) vs Base-10.DoD: Complete CSV tables, raw plot JSONs, HTML/PNG figures in runs/, and detailed entries in docs/RESEARCH_LOG.md.Status: [x] Completed (Exp 2, 3, 4, 5 [x] Completed)Phase 5: University Web Laboratory (web/ — FastAPI + Plotly + PyEcharts)Tasks: Build an interactive scientific-educational web showcase (suitable for university courses in Information Theory, Cryptography, and Deep Learning) with local GPU inference on RTX 3060 Ti and three modes:Mode 1 — Academic Theory Lab: constants, radixes, Shannon entropy formulas, Wilson confidence intervals, and automated text result interpreter.Mode 2 — Neural X-Ray & Circuits: interactive Attention Maps viewer, embedding PCA geometry, and live Causal Head Ablation.Mode 3 — Applied Crypto & PRNG Auditor: user sandbox to enter a custom periodic fraction, upload a pi slice, or hide a secret steganographic backdoor in noise and watch GPU neural detection live (plus 2D Random Walk & Human vs. AI game).DoD: Local web laboratory runs smoothly offline at localhost:8000.Status: [ ] PendingPhase 6: Final Packaging & School Defense Synthesis (Track 2)Tasks: Clean repository, generate public English README.md with real benchmark tables and embedded plots, and synthesize docs/RESEARCH_LOG.md into the 40-page school thesis text and 5-minute defense pitch.DoD: Clean GitHub repository (git status clean, large files ignored) and complete school submission materials.Status: [ ] Not Started