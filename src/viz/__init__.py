# Пакет визуализации экспериментов проекта
from src.viz.classical_and_exp1 import (
    plot_arch_battle_comparison,
    plot_attention_maps,
    plot_exp1_test_xray,
    plot_exp1_training_curves,
    plot_gru_hidden_probe,
    plot_ppm_vs_neural_bits,
)
from src.viz.constants_and_scaling import (
    plot_exp2_cf_and_champernowne_contrast,
    plot_exp2_constants_comparison,
    plot_exp2_embedding_pca_preview,
    plot_exp2_heatmap_bits,
    plot_exp2_radar_pyecharts,
    plot_exp3_bases_normalized_bits,
    plot_exp3_bases_zscore_ci,
    plot_exp4_capacity_scaling,
    plot_exp4_compute_frontier,
    plot_exp4_memorization_scissors,
    plot_exp4_window_phase_transition,
)
from src.viz.mech_interp import (
    plot_exp5_crypto_backdoor_xray,
    plot_exp5_embedding_clock_pca,
    plot_exp5_head_ablation_heatmap,
    plot_exp5_linear_probing_bars,
)
from src.viz.theme import apply_lab_layout, save_fig, save_plotly_figure

__all__ = [
    "apply_lab_layout",
    "save_fig",
    "save_plotly_figure",
    "plot_exp1_training_curves",
    "plot_exp1_test_xray",
    "plot_gru_hidden_probe",
    "plot_arch_battle_comparison",
    "plot_attention_maps",
    "plot_ppm_vs_neural_bits",
    "plot_exp2_constants_comparison",
    "plot_exp2_heatmap_bits",
    "plot_exp2_cf_and_champernowne_contrast",
    "plot_exp2_radar_pyecharts",
    "plot_exp2_embedding_pca_preview",
    "plot_exp3_bases_normalized_bits",
    "plot_exp3_bases_zscore_ci",
    "plot_exp4_window_phase_transition",
    "plot_exp4_capacity_scaling",
    "plot_exp4_memorization_scissors",
    "plot_exp4_compute_frontier",
    "plot_exp5_embedding_clock_pca",
    "plot_exp5_head_ablation_heatmap",
    "plot_exp5_linear_probing_bars",
    "plot_exp5_crypto_backdoor_xray",
]
