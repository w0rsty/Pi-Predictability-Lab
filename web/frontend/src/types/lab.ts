export type ThemeMode = "dark" | "light";

export type TabId = "overview" | "atlas" | "mech-interp" | "sandbox" | "arena";

export interface ClassicalStatRow {
  name?: string;
  dataset?: string;
  base_label?: string;
  radix: number;
  length: number;
  mask_count?: number;
  mask_fraction?: number;
  chi2_stat: number;
  chi2_p_value: number;
  chi2_passed: boolean;
  ks_stat?: number;
  ks_p_value: number;
  ks_passed: boolean;
  acf_max_abs: number;
  acf_max_lag: number;
  acf_significant_count: number;
  gzip_ratio?: number;
  lzma_ratio?: number;
  lzma_bits_per_digit?: number;
  lzma_bits_per_symbol?: number;
  shannon_limit_bits: number;
  empirical_entropy_nats?: number;
  lzma_normalized?: number;
}

export interface Exp1Row {
  dataset: string;
  arch: string;
  size_tier: string;
  param_count: number;
  epochs_ran: number;
  best_epoch: number;
  early_stopped: boolean;
  train_time_sec: number;
  final_train_loss: number;
  best_val_loss: number;
  test_overall_acc: number;
  test_targeted_acc: number | null;
  test_cross_entropy: number;
  test_targeted_ce: number | null;
  test_perplexity: number;
  test_pred_entropy: number;
  targeted_confidence: number | null;
  unmasked_confidence: number;
  run_dir: string;
}

export interface PpmComparisonRow {
  dataset: string;
  model: string;
  overall_acc: number;
  targeted_acc: number | null;
  cross_entropy: number;
  targeted_ce: number | null;
  bits_per_symbol: number;
  targeted_bits_per_symbol: number | null;
  perplexity: number;
  pred_entropy: number;
  targeted_confidence: number | null;
  unmasked_confidence: number | null;
  train_time_sec: number;
  eval_time_sec: number | null;
}

export interface Exp2Row {
  dataset: string;
  radix: number;
  hypothesis_group: "H0" | "H1";
  model: string;
  params_count: number | null;
  test_acc: number;
  wilson_ci_low: number;
  wilson_ci_high: number;
  test_ce_nats: number;
  test_ppl: number;
  bits_per_symbol: number;
  shannon_limit_bits: number;
  gain_over_markov_bits: number;
  sec_per_epoch: number | null;
  samples_per_sec: number | null;
  pred_entropy: number;
  num_eval_samples: number;
  epochs_ran: number | null;
  best_epoch: number | null;
  train_time_sec: number;
  source: string;
}

export interface Exp3Row {
  dataset: string;
  base_label: string;
  radix: number;
  window_size: number;
  window_info_bits: number;
  model: string;
  params_count: number | null;
  test_acc: number;
  random_chance: number;
  wilson_ci_low: number;
  wilson_ci_high: number;
  z_score: number;
  z_ci_low: number;
  z_ci_high: number;
  z_p_value: number;
  test_ce_nats: number;
  test_ppl: number;
  bits_per_symbol: number;
  shannon_limit_bits: number;
  normalized_bits: number;
  excess_bits_per_bit: number;
  equiv_4bit_nibble_bits: number;
  gain_over_markov_bits: number;
  pred_entropy: number;
  num_eval_samples: number;
}

export interface Block4BitRow {
  model: string;
  base2_bit_acc_pct: number;
  base2_equiv_4bit_acc_pct: number;
  base16_nibble_acc_pct: number;
  base2_4bit_entropy_bits: number;
  base16_nibble_entropy_bits: number;
  delta_nibble_bits_base16_minus_base2: number;
}

export interface Exp4Row {
  block: string;
  dataset: string;
  hypothesis_group: "H0" | "H1";
  radix: number;
  window_size: number;
  size_tier: string;
  model: string;
  params_count: number | null;
  test_acc: number;
  wilson_ci_low: number;
  wilson_ci_high: number;
  z_score: number;
  z_p_value: number;
  test_ce_nats: number;
  test_ppl: number;
  bits_per_symbol: number;
  shannon_limit_bits: number;
  normalized_bits: number;
  sec_per_epoch: number | null;
  samples_per_sec: number | null;
  pred_entropy: number;
  num_eval_samples: number;
  epochs_ran: number | null;
  best_epoch: number | null;
  train_time_sec: number;
}

export interface MemorizationPoint {
  epoch: number;
  train_loss: number;
  train_acc: number;
  train_bits: number;
  val_loss: number;
  val_acc: number;
  val_bits: number;
  test_loss: number;
  test_acc: number;
  test_bits: number;
  epoch_time_sec: number;
}

export interface HeadAblationRow {
  dataset: string;
  window_size?: number;
  subset: string;
  n_subset: number;
  layer: number;
  head: number;
  base_acc: number;
  ablated_acc: number;
  delta_acc_pp: number;
  base_bits?: number;
  ablated_bits?: number;
  delta_bits: number;
  mcnemar_p: string;
  mcnemar_p_raw: number;
  mcnemar_log10_p: number;
  mcnemar_sig_0_05: boolean;
  mcnemar_p_holm: number;
  mcnemar_sig_holm: boolean;
  head_role?: string;
}

export interface LinearProbingRow {
  model: string;
  task: string;
  dataset: string;
  n_classes: number;
  state: "Trained" | "Untrained";
  n_train: number;
  n_test: number;
  probe_acc: number;
  wilson_ci_low: number;
  wilson_ci_high: number;
  balanced_acc: number;
  f1_macro: number;
  f1_pos: number | null;
  pr_auc: number | null;
  majority_baseline: number;
  chance_baseline: number;
  mcnemar_vs_untrained_p: string;
}

export interface EmbeddingGeometryEntry {
  d_model: number;
  vocab_size: number;
  explained_variance_ratio_pc1_pc2_pc3: number[];
  explained_variance_pc1_pc2: number;
  explained_variance_pc1_3: number;
  delta_from_init_l2: number;
  cos_with_init_mean: number;
  delta_explained_variance_pc1_pc2: number;
  adjacent_cosine_0_to_8: number;
  cosine_9_to_0: number;
  adjacent_cosine_mod10_mean: number;
  nonadjacent_cosine_mean: number;
  adj_0_to_8_minus_other: number;
  adj_0_to_8_perm_p: number;
  adj_mod10_minus_nonadj: number;
  adj_mod10_perm_p: number;
  perm_test_one_sided: boolean;
  perm_test_resamples: number;
  cyclic_order_consistency_10: number;
  arc_order_consistency_9: number;
  angles_rad: number[];
  coords: [number, number][];
}

export interface Exp5MetricRow {
  module: string;
  dataset: string;
  radix: number;
  model: string;
  params_count: number | null;
  overall_acc: number;
  overall_correct: number;
  wilson_ci_low: number;
  wilson_ci_high: number;
  targeted_acc: number | null;
  targeted_correct: number | null;
  targeted_ci_low: number | null;
  targeted_ci_high: number | null;
  cross_entropy_nats: number;
  targeted_ce_nats: number | null;
  bits_per_symbol: number;
  shannon_limit_bits: number;
  z_score: number;
  overall_z_score: number;
  z_p_value: number;
  targeted_z_score: number | null;
  targeted_z_p_value: number | null;
  targeted_z_log10_p: number | null;
  num_eval_samples: number;
  num_masked_samples: number;
  clean_csprng_acc: number | null;
  source: string;
}

export interface BackdoorTrigger {
  t: number;
  lag24_idx: number;
  lag7_idx: number;
  lag1_idx: number;
  a: number;
  b: number;
  c: number;
  target: number;
  rhs: number;
  bijection_rhs: number;
}

export interface CryptoAttnProfile {
  lags: number[];
  focus_layer: number;
  focus_head: number;
  focus_head_label: string;
  trigger_samples_averaged: number;
  nontrigger_samples_averaged: number;
  trigger_mean_attn: number[];
  nontrigger_mean_attn: number[];
  trigger_lag24_attn: number;
  trigger_lag7_attn: number;
  per_head_lag_mass: Array<{
    layer: number;
    head: number;
    mass_t_minus_24: number;
    mass_t_minus_7: number;
    mass_t_minus_1: number;
    sum_24_7: number;
  }>;
}

export interface SequenceSample {
  radix: number;
  length: number;
  digits: number[];
}

export interface BackdoorSample extends SequenceSample {
  global_offset: number;
  mask: number[];
  triggers: BackdoorTrigger[];
}

export interface LabData {
  meta: {
    project: string;
    n_digits: number;
    window_size: number;
    split: { train: number; val: number; test: number };
    seed: number;
    mde_pp: number;
    se_pp: number;
    cpu_tests_count: number;
    architectures: string[];
    budget_params: string;
  };
  classical_stats: {
    overview: ClassicalStatRow[];
    new_constants: ClassicalStatRow[];
    bases: ClassicalStatRow[];
  };
  exp1_sanity: {
    arch_comparison: Exp1Row[];
    ppm_vs_neural: PpmComparisonRow[];
    stat_audit: Record<string, unknown>;
  };
  exp2_constants: {
    metrics: Exp2Row[];
    stat_audit: Record<string, {
      hypothesis_group: string;
      best_neural_model: string;
      best_neural_bits: number;
      best_neural_acc: number;
      comparisons: Record<string, {
        n_samples: number;
        acc_a: number;
        acc_b: number;
        p_value: number;
        p_value_str: string;
        significant_0_05: boolean;
        model_a: string;
        model_b: string;
      }>;
    }>;
  };
  exp3_bases: {
    metrics: Exp3Row[];
    block_4bit_vs_nibble: Block4BitRow[];
    stat_audit: Record<string, unknown>;
  };
  exp4_scaling: {
    metrics: Exp4Row[];
    memorization_curve: MemorizationPoint[];
    stat_audit: Record<string, unknown>;
  };
  exp5_mech_interp: {
    metrics: Exp5MetricRow[];
    head_ablation: HeadAblationRow[];
    head_ablation_backdoor: HeadAblationRow[];
    linear_probing: LinearProbingRow[];
    embedding_geometry: Record<string, EmbeddingGeometryEntry>;
    stat_audit: Record<string, unknown>;
    crypto_attn_profile: CryptoAttnProfile;
  };
  samples: {
    true_pi: SequenceSample;
    rational_1_97: SequenceSample;
    champernowne: SequenceSample;
    e_cf: SequenceSample;
    csprng_backdoor: BackdoorSample;
  };
}

export interface HealthResponse {
  status: string;
  api_version: string;
  device: string;
  gpu_name: string;
  vram_gb: number | null;
  checkpoints_loaded: number;
  available_models: Array<{
    key: string;
    dataset: string;
    arch: string;
    radix: number;
    window_size: number;
    size_tier: string;
  }>;
  ablation_datasets: string[];
}

export interface SandboxGenerateResponse {
  digits: number[];
  mask: number[] | null;
  radix: number;
  length: number;
  label: string;
  elapsed_ms: number;
}

export interface SandboxAnalyzeResponse {
  name: string;
  radix: number;
  length: number;
  stats: {
    chi2_stat: number;
    chi2_p_value: number;
    chi2_passed: boolean;
    ks_stat: number;
    ks_p_value: number;
    ks_passed: boolean;
    acf_max_abs: number;
    acf_max_lag: number;
    acf_conf_bound_raw: number;
    acf_conf_bound_bonferroni: number;
    acf_significant_count: number;
    acf_lags: number[];
    acf_values: number[];
    gzip_bits_per_digit: number;
    lzma_bits_per_digit: number;
    empirical_entropy_bits: number;
    shannon_limit_bits: number;
    freqs: number[];
  };
  ppm: {
    overall_acc: number;
    cross_entropy: number;
    bits_per_symbol: number;
    perplexity: number;
    pred_entropy: number;
    num_eval_samples: number;
  } | null;
  neural_models: Array<{
    model_key: string;
    arch: string;
    dataset_trained: string;
    overall_acc: number;
    targeted_acc: number | null;
    cross_entropy: number;
    bits_per_symbol: number;
    z_score: number;
    z_p_value: number;
    num_eval_samples: number;
    position_probs: number[];
  }>;
  verdict: {
    hypothesis: "H0" | "H1";
    summary_ru: string;
  };
  elapsed_ms: number;
}

export interface AblationResponse {
  dataset: string;
  ablate_heads: [number, number][];
  num_windows: number;
  baseline_acc: number;
  ablated_acc: number;
  delta_acc_pp: number;
  baseline_bits: number;
  ablated_bits: number;
  delta_bits: number;
  wilson_ci: [number, number];
  targeted_baseline_acc: number | null;
  targeted_ablated_acc: number | null;
  targeted_delta_acc_pp: number | null;
  attention_profile: number[];
  elapsed_ms: number;
}

export type ArenaDatasetId =
  | "true_pi"
  | "csprng_backdoor"
  | "champernowne"
  | "e_cf"
  | "rational_1_97";

export type ArenaModeId = ArenaDatasetId | "blind_audit";

export interface ArenaLagWeight {
  lag: number;
  weight: number;
}

export interface ArenaModelPrediction {
  probs: number[];
  pred: number;
  bits: number;
}

export interface ArenaTransformerPrediction extends ArenaModelPrediction {
  top_lags: ArenaLagWeight[];
}

export interface ArenaTriggerInfo {
  lag_a: number;
  lag_b: number;
  a: number;
  b: number;
  c_prev: number;
  formula: string;
}

export interface ArenaRound {
  id: string;
  global_pos: number;
  context: number[];
  target: number;
  is_trigger: boolean;
  trigger_info: ArenaTriggerInfo | null;
  transformer: ArenaTransformerPrediction;
  gru: ArenaModelPrediction | null;
  ppm: ArenaModelPrediction;
}

export interface ArenaBankDataset {
  dataset: ArenaDatasetId;
  radix: number;
  rounds: ArenaRound[];
}

export interface ArenaRoundsBank {
  version: number;
  seed: number;
  n_rounds_per_dataset: number;
  datasets: Record<ArenaDatasetId, ArenaBankDataset>;
}

export interface ArenaSessionResponse {
  mode: ArenaModeId;
  true_dataset: ArenaDatasetId;
  radix: number;
  rounds: ArenaRound[];
  source: "live_gpu";
  latency_ms: number;
}
