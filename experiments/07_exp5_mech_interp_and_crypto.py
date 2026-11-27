# Скрипт Exp 5: анализ эмбеддингов, абляция голов, probing, крипто-детектор и BBP-тест
import argparse
import json
import math
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baselines import evaluate_uniform_baseline, fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, make_dataloader_generator, seed_everything, seed_worker
from src.dataset import BinaryCoordinateDataset, SequenceWindowDataset, _fast_batch_collate, create_dataloaders, split_sequence_chronologically, verify_zero_leakage
from src.evaluator import compute_binomial_z_score, compute_bitrate_metrics, evaluate_model, holm_bonferroni_correction, mcnemar_test, wilson_ci
from src.exp_utils import free_gpu
from src.models import DigitGRU, DigitTransformer, build_model, count_parameters
from src.stats_tests import run_classical_tests
from src.trainer import create_run_dir, remap_legacy_state_dict, resolve_device, train_model
from src.viz import (
    plot_exp5_crypto_backdoor_xray,
    plot_exp5_embedding_clock_pca,
    plot_exp5_head_ablation_heatmap,
    plot_exp5_linear_probing_bars,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_runs_root(config: dict[str, Any]) -> Path:
    raw = Path(config["paths"]["runs_dir"])
    return raw if raw.is_absolute() else ROOT_DIR / raw


def _find_latest_dir(runs_root: Path, prefix: str) -> Path | None:
    if not runs_root.exists():
        return None
    cands = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.name, reverse=True)
    return cands[0] if cands else None


def _load_checkpoint_model(arch: str, ckpt_path: Path, radix: int, window_size: int, size_tier: str, device: torch.device) -> torch.nn.Module:
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    remapped, _ = remap_legacy_state_dict(sd)
    model.load_state_dict(remapped)
    return model.to(device).eval()


def _build_champernowne_annotations(digits: np.ndarray) -> dict[str, np.ndarray]:
    n_len = len(digits)
    pos_in_num, num_len_arr = np.empty(n_len, dtype=np.int16), np.empty(n_len, dtype=np.int16)
    prev_ends_9, boundary_9 = np.zeros(n_len, dtype=np.uint8), np.zeros(n_len, dtype=np.uint8)
    carry_digit, inc_digit = np.zeros(n_len, dtype=np.uint8), np.zeros(n_len, dtype=np.uint8)

    pos, num = 0, 1
    while pos < n_len:
        s = str(num)
        dlen = len(s)
        end_pos = min(pos + dlen, n_len)
        count = end_pos - pos
        pos_in_num[pos:end_pos] = np.arange(count, dtype=np.int16)
        num_len_arr[pos:end_pos] = dlen
        if (num - 1) % 10 == 9:
            prev_ends_9[pos:end_pos] = 1
        if num % 10 == 9 and end_pos == pos + dlen:
            boundary_9[pos + dlen - 1] = 1
        if num > 1:
            sp = str(num - 1).zfill(dlen)
            is_carry = (num - 1) % 10 == 9
            for j in range(count):
                if s[j] != sp[j]:
                    (carry_digit if is_carry else inc_digit)[pos + j] = 1
        pos, num = end_pos, num + 1

    return {"pos_in_num": pos_in_num, "num_len": num_len_arr, "prev_ends_9": prev_ends_9, "boundary_9": boundary_9, "carry_digit": carry_digit, "inc_digit": inc_digit}


def _build_e_cf_annotations(digits: np.ndarray) -> dict[str, np.ndarray]:
    n_len = len(digits)
    triple_phase, is_even_2k = np.zeros(n_len, dtype=np.uint8), np.zeros(n_len, dtype=np.uint8)
    pos, k = min(2, n_len), 1
    while pos < n_len:
        for _ in (1, 10):
            if pos < n_len:
                triple_phase[pos] = 0
                pos += 1
        for _ in str(2 * k):
            if pos < n_len:
                triple_phase[pos], is_even_2k[pos] = 1, 1
                pos += 1
        if pos < n_len:
            triple_phase[pos] = 1
            pos += 1
        for _ in (1, 10):
            if pos < n_len:
                triple_phase[pos] = 2
                pos += 1
        k += 1
    return {"triple_phase": triple_phase, "is_even_2k": is_even_2k}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom >= 1e-12 else 0.0


def _embedding_adj_perm_test(emb_d: np.ndarray, n_perms: int = 10_000, seed: int = 42) -> dict[str, float]:
    """Односторонний перестановочный тест (diff >= obs_diff) для adjacent_cosine_0_to_8 и mod-10."""
    norms = np.linalg.norm(emb_d, axis=1, keepdims=True)
    unit = emb_d / np.maximum(norms, 1e-12)
    cos_mat = unit @ unit.T

    lin_adj = np.array([(d, d + 1) for d in range(9)], dtype=np.int64)
    lin_other = np.array([(i, j) for i in range(10) for j in range(i + 2, 10)], dtype=np.int64)
    mod_adj = np.array([(d, (d + 1) % 10) for d in range(10)], dtype=np.int64)
    mod_nonadj = np.array([(i, j) for i in range(10) for j in range(i + 1, 10) if (j - i) not in (1, 9)], dtype=np.int64)

    obs_lin = float(cos_mat[lin_adj[:, 0], lin_adj[:, 1]].mean() - cos_mat[lin_other[:, 0], lin_other[:, 1]].mean())
    obs_mod = float(cos_mat[mod_adj[:, 0], mod_adj[:, 1]].mean() - cos_mat[mod_nonadj[:, 0], mod_nonadj[:, 1]].mean())

    rng = np.random.default_rng(seed)
    ge_lin, ge_mod = 0, 0
    for _ in range(n_perms):
        perm = rng.permutation(10)
        p_la, p_lo = perm[lin_adj], perm[lin_other]
        p_ma, p_mn = perm[mod_adj], perm[mod_nonadj]
        if float(cos_mat[p_la[:, 0], p_la[:, 1]].mean() - cos_mat[p_lo[:, 0], p_lo[:, 1]].mean()) >= obs_lin:
            ge_lin += 1
        if float(cos_mat[p_ma[:, 0], p_ma[:, 1]].mean() - cos_mat[p_mn[:, 0], p_mn[:, 1]].mean()) >= obs_mod:
            ge_mod += 1
    return {
        "adj_0_to_8_minus_other": obs_lin,
        "adj_0_to_8_perm_p": float((1 + ge_lin) / (n_perms + 1)),
        "adj_mod10_minus_nonadj": obs_mod,
        "adj_mod10_perm_p": float((1 + ge_mod) / (n_perms + 1)),
    }


def _analyze_embeddings(checkpoints: dict[str, dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    results, init_cache = {}, {}
    for name, info in checkpoints.items():
        ckpt = torch.load(info["path"], map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        emb = sd["token_emb.weight"].float().numpy()
        emb_d, vocab_sz, d_model = emb[:10], int(emb.shape[0]), int(emb.shape[1])
        tier = "heavy" if d_model >= 192 else "medium"
        if (vocab_sz, tier) not in init_cache:
            seed_everything(seed)
            init_cache[(vocab_sz, tier)] = build_model("transformer", radix=vocab_sz, window_size=64, size_tier=tier).token_emb.weight.detach().float().numpy()
        w0 = init_cache[(vocab_sz, tier)]
        delta_pca = PCA(n_components=3, random_state=42).fit(emb_d - w0[:10])

        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(emb_d)
        adj_0_8 = [_cosine(emb_d[d], emb_d[d + 1]) for d in range(9)]
        cos_9_0 = _cosine(emb_d[9], emb_d[0])
        non_adj = [_cosine(emb_d[i], emb_d[j]) for i in range(10) for j in range(i + 1, 10) if (j - i) not in (1, 9)]
        perm_res = _embedding_adj_perm_test(emb_d, n_perms=10_000, seed=seed)

        pc2d = coords_3d[:, :2]
        centered = pc2d - pc2d.mean(axis=0, keepdims=True)
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        step_d = np.angle(np.exp(1j * (np.roll(angles, -1) - angles)))
        arc_d = np.angle(np.exp(1j * np.diff(angles)))

        results[name] = {
            "d_model": d_model, "vocab_size": vocab_sz,
            "explained_variance_ratio_pc1_pc2_pc3": [round(float(v), 6) for v in pca.explained_variance_ratio_],
            "explained_variance_pc1_pc2": round(float(sum(pca.explained_variance_ratio_[:2])), 6),
            "explained_variance_pc1_3": round(float(sum(pca.explained_variance_ratio_[:3])), 6),
            "delta_from_init_l2": round(float(np.linalg.norm(emb - w0)), 4),
            "cos_with_init_mean": round(float(np.mean([_cosine(emb_d[d], w0[d]) for d in range(10)])), 6),
            "delta_explained_variance_pc1_pc2": round(float(sum(delta_pca.explained_variance_ratio_[:2])), 6),
            "adjacent_cosine_0_to_8": round(float(np.mean(adj_0_8)), 6),
            "cosine_9_to_0": round(float(cos_9_0), 6),
            "adjacent_cosine_mod10_mean": round(float(np.mean(adj_0_8 + [cos_9_0])), 6),
            "nonadjacent_cosine_mean": round(float(np.mean(non_adj)), 6),
            "adj_0_to_8_minus_other": round(perm_res["adj_0_to_8_minus_other"], 6),
            "adj_0_to_8_perm_p": round(perm_res["adj_0_to_8_perm_p"], 6),
            "adj_mod10_minus_nonadj": round(perm_res["adj_mod10_minus_nonadj"], 6),
            "adj_mod10_perm_p": round(perm_res["adj_mod10_perm_p"], 6),
            "perm_test_one_sided": True,
            "perm_test_resamples": 10_000,
            "cyclic_order_consistency_10": round(max(int(np.sum(step_d > 0)), int(np.sum(step_d < 0))) / 10.0, 4),
            "arc_order_consistency_9": round(max(int(np.sum(arc_d > 0)), int(np.sum(arc_d < 0))) / 9.0, 4),
            "angles_rad": [round(float(a), 4) for a in angles],
            "coords": np.round(pca.transform(emb)[:, :2], 5).tolist(),
        }
    return results


@torch.no_grad()
def _eval_preds_and_nll(model: DigitTransformer, loader: DataLoader, device: torch.device, ablate_heads: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    amp_enabled, corr_chunks, nll_chunks = device.type == "cuda", [], []
    for contexts, targets, _ in loader:
        contexts, targets = contexts.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model.forward_with_head_ablation(contexts, ablate_heads=ablate_heads) if ablate_heads else model(contexts)
        logits_f32 = logits.float()
        corr_chunks.append((torch.argmax(logits_f32, dim=-1) == targets).cpu().numpy())
        nll_chunks.append(F.nll_loss(F.log_softmax(logits_f32, dim=-1), targets, reduction="none").cpu().numpy())
    return np.concatenate(corr_chunks), np.concatenate(nll_chunks)


def _classify_head_role(ds_name: str, sub_name: str, sig_holm: bool, delta_acc_pp: float, delta_bits: float) -> str:
    # Негативный контроль проходит тот же статистический фильтр без исключений по имени датасета
    if not sig_holm:
        return "redundant"
    if delta_acc_pp <= -1.0 or delta_bits >= 0.05:
        if sub_name in ("carry_over", "carry_digit"):
            return "carry_head"
        if sub_name == "inc_digit":
            return "increment_head"
        if sub_name == "even_2k":
            return "induction_2k_head"
        return "periodic_head" if ds_name == "rational_1_97" else "critical_head"
    return "supporting_head" if (delta_acc_pp <= -0.35 or delta_bits >= 0.015) else "redundant"


def _ablate_dataset_heads(ds_name: str, radix: int, win_size: int, size_tier: str, req_len: int | None, ckpt_path: Path, config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    split_cfg = config["data"]["split"]
    art = get_or_create_dataset(ds_name, config=config, length=req_len)
    split = split_sequence_chronologically(art.digits, art.mask, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
    verify_zero_leakage(
        split,
        window_size=win_size,
        train_ds=SequenceWindowDataset(split.train_digits, win_size, split.train_mask, split.train_range[0]),
        val_ds=SequenceWindowDataset(split.val_digits, win_size, split.val_mask, split.val_range[0]),
        test_ds=SequenceWindowDataset(split.test_digits, win_size, split.test_mask, split.test_range[0]),
    )
    loader = DataLoader(SequenceWindowDataset(split.test_digits, window_size=win_size, mask=split.test_mask, global_offset=split.test_range[0]), batch_size=512 if win_size >= 128 else 1024, shuffle=False, num_workers=int(config["training"].get("num_workers", 4)), pin_memory=(device.type == "cuda"), collate_fn=_fast_batch_collate)

    idx = np.arange(split.test_range[0] + win_size, split.test_range[0] + len(split.test_digits))
    subsets = {"all": np.ones(len(idx), dtype=bool)}
    if ds_name == "champernowne":
        ann = _build_champernowne_annotations(art.digits)
        subsets.update({"carry_over": ann["prev_ends_9"][idx] == 1, "non_carry": ann["prev_ends_9"][idx] != 1, "carry_digit": ann["carry_digit"][idx] == 1, "inc_digit": ann["inc_digit"][idx] == 1})
    elif ds_name == "e_cf":
        ann = _build_e_cf_annotations(art.digits)
        subsets.update({"even_2k": ann["is_even_2k"][idx] == 1, "const_1_sep": ann["is_even_2k"][idx] != 1})

    model = _load_checkpoint_model("transformer", ckpt_path, radix, win_size, size_tier, device)
    assert isinstance(model, DigitTransformer)
    base_corr, base_nll = _eval_preds_and_nll(model, loader, device=device)
    base_stats = {s: (float(np.mean(base_corr[m])), float(np.mean(base_nll[m]) / math.log(2.0))) for s, m in subsets.items()}

    rows: list[dict[str, Any]] = []
    for l_idx, h_idx in tqdm([(l, h) for l in range(model.n_layers) for h in range(model.n_heads)], desc=f"Head Ablation [{ds_name}]", unit="head"):
        abl_corr, abl_nll = _eval_preds_and_nll(model, loader, device=device, ablate_heads=[(l_idx, h_idx)])
        for sub_name, sub_mask in subsets.items():
            b_acc, b_bits = base_stats[sub_name]
            a_acc, a_bits = float(np.mean(abl_corr[sub_mask])), float(np.mean(abl_nll[sub_mask]) / math.log(2.0))
            d_acc, d_bits = (a_acc - b_acc) * 100.0, a_bits - b_bits
            mc = mcnemar_test(base_corr[sub_mask], abl_corr[sub_mask])
            rows.append({
                "dataset": ds_name, "window_size": win_size, "subset": sub_name, "n_subset": int(np.sum(sub_mask)),
                "layer": int(l_idx), "head": int(h_idx), "base_acc": round(b_acc, 6), "ablated_acc": round(a_acc, 6),
                "delta_acc_pp": round(d_acc, 4), "base_bits": round(b_bits, 6), "ablated_bits": round(a_bits, 6),
                "delta_bits": round(d_bits, 6), "mcnemar_p": mc["p_value_str"], "mcnemar_p_raw": float(mc["p_value"]),
                "mcnemar_log10_p": float(mc["log10_p_value"]), "mcnemar_sig_0_05": bool(mc["significant_0_05"]),
            })

    del model
    free_gpu()
    return rows


def _run_causal_head_ablation(config: dict[str, Any], checkpoints: dict[str, dict[str, Any]], device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ds_name, radix, win_size, size_tier, req_len in (("champernowne", 10, 64, "medium", None), ("e_cf", 11, 64, "medium", None), ("rational_1_97", 10, 128, "medium", 1_000_050), ("true_pi", 10, 64, "medium", None)):
        rows.extend(_ablate_dataset_heads(ds_name, radix, win_size, size_tier, req_len, checkpoints[ds_name]["path"], config, device))
    # Единая глобальная поправка Холма — Бонферрони по всей матрице абляции (все датасеты и срезы)
    p_adj, sig_flags = holm_bonferroni_correction([r["mcnemar_p_raw"] for r in rows], alpha=0.05)
    for row, pa, sig in zip(rows, p_adj, sig_flags):
        row["mcnemar_p_holm"] = round(pa, 8) if pa >= 1e-8 else pa
        row["mcnemar_sig_holm"] = bool(sig)
        row["head_role"] = _classify_head_role(str(row["dataset"]), str(row["subset"]), bool(sig), float(row["delta_acc_pp"]), float(row["delta_bits"]))
    return pd.DataFrame(rows)


@torch.no_grad()
def _extract_hidden_slice(model: DigitGRU | DigitTransformer, test_digits: np.ndarray, window_size: int, n_windows: int, device: torch.device, batch_size: int = 512) -> np.ndarray:
    loader = DataLoader(SequenceWindowDataset(test_digits[: n_windows + window_size], window_size=window_size), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_fast_batch_collate)
    model.eval()
    return np.concatenate([model.extract_hidden_states(ctx.to(device, non_blocking=True), all_positions=False).cpu().numpy() for ctx, _, _ in loader], axis=0)


def build_probe_specs(config: dict[str, Any], n_probe_windows: int) -> list[dict[str, Any]]:
    split_cfg = config["data"]["split"]
    e_cf = get_or_create_dataset("e_cf", config=config)
    e_cf_sp = split_sequence_chronologically(e_cf.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
    champ = get_or_create_dataset("champernowne", config=config)
    champ_sp = split_sequence_chronologically(champ.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
    rat = get_or_create_dataset("rational_1_97", config=config, length=1_000_050)
    rat_sp = split_sequence_chronologically(rat.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))

    i_ecf = np.arange(e_cf_sp.test_range[0] + 63, e_cf_sp.test_range[0] + 63 + n_probe_windows)
    i_ch = np.arange(champ_sp.test_range[0] + 63, champ_sp.test_range[0] + 63 + n_probe_windows)
    i_rat = np.arange(rat_sp.test_range[0] + 127, rat_sp.test_range[0] + 127 + n_probe_windows)
    ch_ann = _build_champernowne_annotations(champ.digits)
    return [
        {"task": "e_cf_phase", "dataset": "e_cf", "radix": 11, "window_size": 64, "split": e_cf_sp, "labels": _build_e_cf_annotations(e_cf.digits)["triple_phase"][i_ecf].astype(np.int64), "n_classes": 3, "chance": 1.0 / 3.0},
        {"task": "champ_pos", "dataset": "champernowne", "radix": 10, "window_size": 64, "split": champ_sp, "labels": ch_ann["pos_in_num"][i_ch].astype(np.int64), "n_classes": 6, "chance": 1.0 / 6.0},
        {"task": "champ_carry", "dataset": "champernowne", "radix": 10, "window_size": 64, "split": champ_sp, "labels": ch_ann["boundary_9"][i_ch].astype(np.int64), "n_classes": 2, "chance": 0.5},
        {"task": "rat97_phase", "dataset": "rational_1_97", "radix": 10, "window_size": 128, "split": rat_sp, "labels": (i_rat % 96).astype(np.int64), "n_classes": 96, "chance": 1.0 / 96.0},
    ]


def _run_linear_probing(config: dict[str, Any], checkpoints: dict[str, dict[str, Any]], gru_checkpoints: dict[str, dict[str, Any]], device: torch.device, n_probe_windows: int = 10_000) -> pd.DataFrame:
    probe_specs = build_probe_specs(config, n_probe_windows)
    features_cache, rows = {}, []
    for arch_key, arch_name, ckpt_map in (("gru", "DigitGRU", gru_checkpoints), ("transformer", "DigitTransformer", checkpoints)):
        for spec in probe_specs:
            ds_name, win_size, y = spec["dataset"], spec["window_size"], spec["labels"]
            tr_end = int(0.8 * n_probe_windows)
            te_start = tr_end + win_size
            # Зазор в win_size позиций гарантирует, что ни один токен последнего train-окна не попадёт в test-окно
            max_tr_token_pos = (tr_end - 1) + (win_size - 1)
            min_te_token_pos = te_start
            assert max_tr_token_pos < min_te_token_pos, f"Утечка окон probing: {max_tr_token_pos} >= {min_te_token_pos}"

            y_tr, y_te = y[:tr_end], y[te_start:]
            maj_counts = np.bincount(y_te)
            majority_baseline = float(maj_counts.max() / max(len(y_te), 1))
            probe_preds = {}
            for state_name in ("Trained", "Untrained"):
                ck_key = (arch_key, ds_name, state_name)
                if ck_key not in features_cache:
                    if state_name == "Trained":
                        m = _load_checkpoint_model(arch_key, ckpt_map[ds_name]["path"], spec["radix"], win_size, "medium", device)
                    else:
                        seed_everything(int(config.get("seed", 42)) + 777)
                        m = build_model(arch=arch_key, radix=spec["radix"], window_size=win_size, size_tier="medium").to(device).eval()
                    features_cache[ck_key] = _extract_hidden_slice(m, spec["split"].test_digits, win_size, n_probe_windows, device)
                    del m
                    free_gpu()
                xf = features_cache[ck_key]
                clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=42)
                pipe = make_pipeline(StandardScaler(), clf).fit(xf[:tr_end], y_tr)
                if int(np.max(clf.n_iter_)) >= int(clf.max_iter):
                    warnings.warn(
                        f"Linear probing {arch_name}/{spec['task']} ({state_name}) достиг max_iter={clf.max_iter}",
                        stacklevel=2,
                    )
                y_pred = pipe.predict(xf[te_start:])
                probe_preds[state_name] = y_pred
                ci_lo, ci_hi = wilson_ci(int(np.sum(y_pred == y_te)), len(y_te))
                f1_pos: float | None = None
                pr_auc: float | None = None
                if spec["n_classes"] == 2:
                    f1_pos = round(float(f1_score(y_te, y_pred, pos_label=1, zero_division=0)), 6)
                    pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
                    if pos_idx >= 0:
                        pr_auc = round(float(average_precision_score(y_te, pipe.predict_proba(xf[te_start:])[:, pos_idx])), 6)

                rows.append({
                    "model": arch_name, "task": spec["task"], "dataset": ds_name, "n_classes": spec["n_classes"],
                    "state": state_name, "n_train": len(y_tr), "n_test": len(y_te),
                    "probe_acc": round(float(np.mean(y_pred == y_te)), 6), "wilson_ci_low": ci_lo, "wilson_ci_high": ci_hi,
                    "balanced_acc": round(float(balanced_accuracy_score(y_te, y_pred)), 6),
                    "f1_macro": round(float(f1_score(y_te, y_pred, average="macro", zero_division=0)), 6),
                    "f1_pos": f1_pos,
                    "pr_auc": pr_auc,
                    "majority_baseline": round(majority_baseline, 6),
                    "chance_baseline": round(float(spec["chance"]), 6),
                    "mcnemar_vs_untrained_p": "",
                })
            mc = mcnemar_test(probe_preds["Trained"] == y_te, probe_preds["Untrained"] == y_te)
            rows[-2]["mcnemar_vs_untrained_p"] = mc["p_value_str"]
    return pd.DataFrame(rows)


def run_module1_mech_interp(config: dict[str, Any], summary_dir: Path, device: torch.device) -> dict[str, Any]:
    runs_root = _resolve_runs_root(config)
    exp2_dir = _find_latest_dir(runs_root, "exp2_constants_final_")
    exp4_dir = _find_latest_dir(runs_root, "exp4_scaling_")
    exp4_gru = _find_latest_dir(runs_root, "exp4_block_b_phasetransition_h1_gru_rational_1_97_L128_medium_")
    if exp2_dir is None or exp4_dir is None or exp4_gru is None:
        raise FileNotFoundError(
            f"Для Модуля 1 не найдены директории Exp2/Exp4 в {runs_root}. Запустите Exp2 и Exp4 или запустите `--modules 2,3`."
        )
    tr_ckpts = {
        "true_pi": {"path": exp2_dir / "checkpoints" / "transformer_true_pi.pt"},
        "e_cf": {"path": exp2_dir / "checkpoints" / "transformer_e_cf.pt"},
        "champernowne": {"path": exp2_dir / "checkpoints" / "transformer_champernowne.pt"},
        "rational_1_999983": {"path": exp2_dir / "checkpoints" / "transformer_rational_1_999983.pt"},
        "rational_1_97": {"path": exp4_dir / "checkpoints" / "transformer_rational_1_97_L128.pt"},
        "memorization_10k_heavy": {"path": exp4_dir / "checkpoints" / "transformer_memorization_10k_heavy.pt"},
    }
    gru_ckpts = {"e_cf": {"path": exp2_dir / "checkpoints" / "gru_e_cf.pt"}, "champernowne": {"path": exp2_dir / "checkpoints" / "gru_champernowne.pt"}, "rational_1_97": {"path": exp4_gru / "best_model.pt"}}
    missing = [str(v["path"]) for v in (*tr_ckpts.values(), *gru_ckpts.values()) if not Path(v["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Отсутствуют чекпойнты для Модуля 1: {missing}")

    emb_geo = _analyze_embeddings(tr_ckpts, seed=int(config.get("seed", 42)))
    (summary_dir / "embedding_geometry.json").write_text(json.dumps(emb_geo, ensure_ascii=False, indent=2), encoding="utf-8")
    ablation_df = _run_causal_head_ablation(config=config, checkpoints=tr_ckpts, device=device)
    ablation_df.to_csv(summary_dir / "head_ablation.csv", index=False)
    probing_df = _run_linear_probing(config=config, checkpoints=tr_ckpts, gru_checkpoints=gru_ckpts, device=device)
    probing_df.to_csv(summary_dir / "linear_probing.csv", index=False)
    return {"embedding_geometry": emb_geo, "head_ablation": ablation_df, "linear_probing": probing_df}


def _summarize_attn_lag_profile(trig_attn_sum: np.ndarray, nontrig_attn_sum: np.ndarray, trig_count: int, nontrig_count: int, window_size: int) -> dict[str, Any]:
    trig_mean, nontrig_mean = trig_attn_sum / max(trig_count, 1), nontrig_attn_sum / max(nontrig_count, 1)
    lag24_idx, lag7_idx = window_size - 24, window_size - 7
    n_layers, n_heads, _ = trig_mean.shape
    best_score, best_l, best_h, per_head = -1.0, 0, 0, []
    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            m24, m7, m1 = float(trig_mean[l_idx, h_idx, lag24_idx]), float(trig_mean[l_idx, h_idx, lag7_idx]), float(trig_mean[l_idx, h_idx, window_size - 1])
            score = m24 + m7
            per_head.append({"layer": l_idx, "head": h_idx, "mass_t_minus_24": round(m24, 5), "mass_t_minus_7": round(m7, 5), "mass_t_minus_1": round(m1, 5), "sum_24_7": round(score, 5)})
            if score > best_score:
                best_score, best_l, best_h = score, l_idx, h_idx
    return {
        "lags": list(range(-window_size, 0)), "focus_layer": int(best_l), "focus_head": int(best_h), "focus_head_label": f"L{best_l}-H{best_h}",
        "trigger_samples_averaged": int(trig_count), "nontrigger_samples_averaged": int(nontrig_count),
        "trigger_mean_attn": np.round(trig_mean[best_l, best_h], 5).tolist(), "nontrigger_mean_attn": np.round(nontrig_mean[best_l, best_h], 5).tolist(),
        "trigger_lag24_attn": round(float(trig_mean[best_l, best_h, lag24_idx]), 5), "trigger_lag7_attn": round(float(trig_mean[best_l, best_h, lag7_idx]), 5),
        "per_head_lag_mass": per_head,
    }


@torch.no_grad()
def _collect_eval_preds_and_attn(model: torch.nn.Module, test_digits: np.ndarray, test_mask: np.ndarray, window_size: int, device: torch.device, batch_size: int = 512, extract_attn: bool = False) -> tuple[np.ndarray, dict[str, Any] | None]:
    loader = DataLoader(SequenceWindowDataset(test_digits, window_size=window_size, mask=test_mask), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_fast_batch_collate)
    model.eval()
    corr_chunks, trig_sum, nontrig_sum, trig_n, nontrig_n, max_collect = [], None, None, 0, 0, 2_000
    for contexts, targets, mask_flags in loader:
        contexts, targets = contexts.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(contexts)
        corr_chunks.append((torch.argmax(logits.float(), dim=-1) == targets).cpu().numpy())
        if extract_attn and (trig_n < max_collect or nontrig_n < max_collect):
            m_np = mask_flags.numpy() == 1
            if (np.any(m_np) and trig_n < max_collect) or (np.any(~m_np) and nontrig_n < max_collect):
                _, attn = model(contexts, return_attn=True)
                last_q = attn[:, :, :, -1, :].float().cpu().numpy()
                if np.any(m_np) and trig_n < max_collect:
                    take = last_q[m_np][: max_collect - trig_n]
                    trig_sum = take.sum(axis=0) if trig_sum is None else trig_sum + take.sum(axis=0)
                    trig_n += len(take)
                if np.any(~m_np) and nontrig_n < max_collect:
                    take = last_q[~m_np][: max_collect - nontrig_n]
                    nontrig_sum = take.sum(axis=0) if nontrig_sum is None else nontrig_sum + take.sum(axis=0)
                    nontrig_n += len(take)

    corr_vec = np.concatenate(corr_chunks)
    if not extract_attn or trig_sum is None or nontrig_sum is None:
        return corr_vec, None
    return corr_vec, _summarize_attn_lag_profile(trig_sum, nontrig_sum, trig_n, nontrig_n, window_size)


@torch.no_grad()
def _eval_canonical_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str = "float16",
    extract_attn: bool = False,
    window_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Единый канонический прогон test_loader: метрики и McNemar считаются строго по одному массиву."""
    model.eval()
    amp_enabled = device.type == "cuda"
    use_bf16 = amp_dtype == "bfloat16" and amp_enabled and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    corr_chunks, nll_chunks = [], []
    trig_sum, nontrig_sum, trig_n, nontrig_n, max_collect = None, None, 0, 0, 2_000

    for contexts, targets, mask_flags in loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=amp_enabled):
            logits = model(contexts)
        logits_f32 = logits.float()
        corr_chunks.append((torch.argmax(logits_f32, dim=-1) == targets).cpu().numpy())
        nll_chunks.append(F.nll_loss(F.log_softmax(logits_f32, dim=-1), targets, reduction="none").cpu().numpy())

        if extract_attn and (trig_n < max_collect or nontrig_n < max_collect):
            m_np = mask_flags.numpy() == 1
            if (np.any(m_np) and trig_n < max_collect) or (np.any(~m_np) and nontrig_n < max_collect):
                _, attn = model(contexts, return_attn=True)
                last_q = attn[:, :, :, -1, :].float().cpu().numpy()
                if np.any(m_np) and trig_n < max_collect:
                    take = last_q[m_np][: max_collect - trig_n]
                    trig_sum = take.sum(axis=0) if trig_sum is None else trig_sum + take.sum(axis=0)
                    trig_n += len(take)
                if np.any(~m_np) and nontrig_n < max_collect:
                    take = last_q[~m_np][: max_collect - nontrig_n]
                    nontrig_sum = take.sum(axis=0) if nontrig_sum is None else nontrig_sum + take.sum(axis=0)
                    nontrig_n += len(take)

    corr_vec = np.concatenate(corr_chunks)
    nll_vec = np.concatenate(nll_chunks)
    if not extract_attn or trig_sum is None or nontrig_sum is None:
        return corr_vec, nll_vec, None
    return corr_vec, nll_vec, _summarize_attn_lag_profile(trig_sum, nontrig_sum, trig_n, nontrig_n, window_size)


def _ablate_backdoor_heads(
    model: DigitTransformer,
    loader: DataLoader,
    mask_bool: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    """По-головая нулевая абляция на csprng_backdoor (all и targeted mask==1) с поправкой Холма."""
    base_corr, base_nll = _eval_preds_and_nll(model, loader, device=device)
    subsets = {"all": np.ones(len(mask_bool), dtype=bool), "targeted": mask_bool}
    base_stats = {s: (float(np.mean(base_corr[m])), float(np.mean(base_nll[m]) / math.log(2.0))) for s, m in subsets.items()}
    rows: list[dict[str, Any]] = []
    for l_idx in range(model.n_layers):
        for h_idx in range(model.n_heads):
            abl_corr, abl_nll = _eval_preds_and_nll(model, loader, device=device, ablate_heads=[(l_idx, h_idx)])
            for sub_name, sub_mask in subsets.items():
                b_acc, b_bits = base_stats[sub_name]
                a_acc, a_bits = float(np.mean(abl_corr[sub_mask])), float(np.mean(abl_nll[sub_mask]) / math.log(2.0))
                mc = mcnemar_test(base_corr[sub_mask], abl_corr[sub_mask])
                rows.append({
                    "dataset": "csprng_backdoor", "subset": sub_name, "n_subset": int(np.sum(sub_mask)),
                    "layer": int(l_idx), "head": int(h_idx),
                    "base_acc": round(b_acc, 6), "ablated_acc": round(a_acc, 6),
                    "delta_acc_pp": round((a_acc - b_acc) * 100.0, 4),
                    "delta_bits": round(a_bits - b_bits, 6),
                    "mcnemar_p": mc["p_value_str"], "mcnemar_p_raw": float(mc["p_value"]),
                    "mcnemar_log10_p": float(mc["log10_p_value"]), "mcnemar_sig_0_05": bool(mc["significant_0_05"]),
                })
    p_adj, sig_flags = holm_bonferroni_correction([r["mcnemar_p_raw"] for r in rows], alpha=0.05)
    for row, pa, sig in zip(rows, p_adj, sig_flags):
        row["mcnemar_p_holm"] = round(pa, 8) if pa >= 1e-8 else pa
        row["mcnemar_sig_holm"] = bool(sig)
    return rows


def _load_or_train_submodel(arch: str, prefix: str, ds_name: str, radix: int, window_size: int, loaders: dict[str, DataLoader], config: dict[str, Any], device: torch.device, reuse_checkpoints: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    runs_root = _resolve_runs_root(config)
    existing = _find_latest_dir(runs_root, f"{prefix}_{arch}_{ds_name}_") if reuse_checkpoints else None
    if existing is not None and (existing / "best_model.pt").exists():
        best_m = _load_checkpoint_model(arch, existing / "best_model.pt", radix, window_size, "medium", device)
        return best_m, {"summary": {"param_count": count_parameters(best_m), "checkpoint_path": str(existing / "best_model.pt"), "run_dir": str(existing)}}

    seed_everything(int(config.get("seed", 42)))
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier="medium")
    res = train_model(model, loaders, config, f"{prefix}_{arch}_{ds_name}", radix, epochs=10, learning_rate=1e-3, scheduler_type="cosine", early_stopping_patience=3, run_dir=create_run_dir(runs_root, f"{prefix}_{arch}_{ds_name}"), show_progress=True)
    best_m = _load_checkpoint_model(arch, Path(res["summary"]["checkpoint_path"]), radix, window_size, "medium", device)
    del model
    return best_m, res


def _format_backdoor_row(
    m_name: str, params_count: int | None, acc: float, ce: float, n_eval: int, n_mask: int,
    tgt_acc: float | None, tgt_ce: float | None, clean_acc: Any, source: str,
    overall_correct: int | None = None, targeted_correct: int | None = None,
) -> dict[str, Any]:
    ov_k = int(overall_correct) if overall_correct is not None else int(round(acc * n_eval))
    t_val = float(tgt_acc or 0.0)
    t_k = int(targeted_correct) if targeted_correct is not None else int(round(t_val * n_mask))
    ci_lo, ci_hi = wilson_ci(ov_k, n_eval)
    t_lo, t_hi = wilson_ci(t_k, n_mask)
    br = compute_bitrate_metrics(ce, radix=10, correct=ov_k, total=n_eval)
    tgt_z = compute_binomial_z_score(t_val, radix=10, n_samples=n_mask, ci_low=t_lo, ci_high=t_hi)
    return {
        "module": "Module2_CryptoBackdoor", "dataset": "csprng_backdoor", "radix": 10, "model": m_name, "params_count": params_count,
        "overall_acc": round(acc, 6), "overall_correct": ov_k, "wilson_ci_low": ci_lo, "wilson_ci_high": ci_hi,
        "targeted_acc": round(t_val, 6), "targeted_correct": t_k, "targeted_ci_low": t_lo, "targeted_ci_high": t_hi,
        "cross_entropy_nats": round(ce, 6), "targeted_ce_nats": round(float(tgt_ce), 6),
        "bits_per_symbol": round(br["bits_per_symbol"], 6), "shannon_limit_bits": round(br["shannon_limit_bits"], 6),
        "z_score": br["z_score"], "overall_z_score": br["z_score"], "z_p_value": br["z_p_value"],
        "targeted_z_score": tgt_z["z_score"], "targeted_z_p_value": tgt_z["z_p_value"], "targeted_z_log10_p": tgt_z["z_log10_p"],
        "num_eval_samples": n_eval, "num_masked_samples": n_mask,
        "clean_csprng_acc": clean_acc, "source": source,
    }


def run_module2_crypto_auditor(config: dict[str, Any], summary_dir: Path, device: torch.device, reuse_checkpoints: bool = False) -> dict[str, Any]:
    split_cfg = config["data"]["split"]
    win_size = 64
    seed = int(config.get("seed", 42))
    clean_art = get_or_create_dataset("csprng", config=config)
    bd_art = get_or_create_dataset("csprng_backdoor", config=config)
    clean_stats = run_classical_tests("csprng", clean_art.digits, 10)
    bd_stats = run_classical_tests("csprng_backdoor", bd_art.digits, 10)
    bd_split = split_sequence_chronologically(bd_art.digits, bd_art.mask, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))

    uni_m = evaluate_uniform_baseline(bd_split, radix=10, window_size=win_size)
    ppm_m, ppm_probs = fit_and_evaluate_ppm(bd_split, radix=10, window_size=win_size, max_order=8)
    test_targets = bd_split.test_digits[win_size:]
    mask_bool = bd_split.test_mask[win_size:] == 1
    ppm_corr = np.argmax(ppm_probs, axis=1) == test_targets
    uni_corr = test_targets == 0

    loaders = create_dataloaders(split=bd_split, window_size=win_size, batch_size=512, num_workers=int(config["training"].get("num_workers", 4)), pin_memory=(device.type == "cuda"), seed=seed)
    verify_zero_leakage(bd_split, window_size=win_size, train_ds=loaders["train"].dataset, val_ds=loaders["val"].dataset, test_ds=loaders["test"].dataset)

    exp2_dir = _find_latest_dir(_resolve_runs_root(config), "exp2_constants_final_")
    clean_csv = (exp2_dir / "metrics.csv") if exp2_dir and (exp2_dir / "metrics.csv").exists() else (ROOT_DIR / "results" / "04_exp2_constants" / "metrics.csv")
    clean_rows = {str(r["model"]): r.to_dict() for _, r in pd.read_csv(clean_csv).iterrows() if r["dataset"] == "csprng"} if clean_csv.exists() else {}

    m2_rows = [_format_backdoor_row(m_name, None, d["overall_acc"], d["cross_entropy"], int(d["num_eval_samples"]), int(d["num_masked_samples"]), d["targeted_acc"], d["targeted_ce"], clean_rows.get(m_name, {}).get("test_acc"), "exp5_module2") for m_name, d in (("Uniform", uni_m), ("PPM_o8", ppm_m))]
    neural_corr, attn_profile, bd_ablation = {}, None, []
    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for arch, m_name in (("gru", "DigitGRU"), ("transformer", "DigitTransformer")):
        best_model, res = _load_or_train_submodel(arch, "exp5", "csprng_backdoor", 10, win_size, loaders, config, device, reuse_checkpoints)
        corr_vec, nll_vec, attn_res = _eval_canonical_loader(
            best_model, loaders["test"], device=device, amp_dtype="float16", extract_attn=(arch == "transformer"), window_size=win_size
        )
        neural_corr[m_name] = corr_vec
        s = res["summary"]
        m2_rows.append(_format_backdoor_row(
            m_name, s["param_count"], float(corr_vec.mean()), float(nll_vec.mean()),
            len(corr_vec), int(np.sum(mask_bool)), float(corr_vec[mask_bool].mean()), float(nll_vec[mask_bool].mean()),
            clean_rows.get(m_name, {}).get("test_acc"), str(s["run_dir"]),
            overall_correct=int(corr_vec.sum()), targeted_correct=int(corr_vec[mask_bool].sum()),
        ))
        if arch == "transformer":
            attn_profile = attn_res
            shutil.copy2(s["checkpoint_path"], ckpt_dir / "transformer_csprng_backdoor.pt")
            assert isinstance(best_model, DigitTransformer)
            bd_ablation = _ablate_backdoor_heads(best_model, loaders["test"], mask_bool, device)
            pd.DataFrame(bd_ablation).to_csv(summary_dir / "head_ablation_backdoor.csv", index=False)
        del best_model
        free_gpu()

    mc_audit = {
        "Transformer_vs_Uniform_all": mcnemar_test(neural_corr["DigitTransformer"], uni_corr),
        "Transformer_vs_PPM_all": mcnemar_test(neural_corr["DigitTransformer"], ppm_corr),
        "Transformer_vs_GRU_all": mcnemar_test(neural_corr["DigitTransformer"], neural_corr["DigitGRU"]),
        "GRU_vs_PPM_all": mcnemar_test(neural_corr["DigitGRU"], ppm_corr),
        "Transformer_vs_PPM_targeted": mcnemar_test(neural_corr["DigitTransformer"][mask_bool], ppm_corr[mask_bool]),
        "Transformer_vs_GRU_targeted": mcnemar_test(neural_corr["DigitTransformer"][mask_bool], neural_corr["DigitGRU"][mask_bool]),
    }
    audit_plot_data = {
        "classical_stats": {"chi2_p_value": bd_stats["chi2_p_value"], "ks_p_value": bd_stats["ks_p_value"], "acf_max_abs": bd_stats["acf_max_abs"], "acf_max_lag": bd_stats["acf_max_lag"], "lzma_bits_per_digit": bd_stats["lzma_bits_per_digit"], "trigger_count": int(bd_art.mask.sum()), "trigger_fraction": float(bd_art.mask.mean())},
        "clean_classical_stats": {"chi2_p_value": clean_stats["chi2_p_value"], "ks_p_value": clean_stats["ks_p_value"], "acf_max_abs": clean_stats["acf_max_abs"], "lzma_bits_per_digit": clean_stats["lzma_bits_per_digit"]},
        "models": [{"label": r["model"], "overall_acc": r["overall_acc"], "targeted_acc": r["targeted_acc"]} for r in m2_rows],
    }
    return {"rows": m2_rows, "mcnemar": mc_audit, "audit_plot_data": audit_plot_data, "attn_profile": attn_profile, "backdoor_head_ablation": bd_ablation}


def _oracle_row(ds_name: str, radix: int, model_name: str, params_count: int | None, br: dict[str, Any], ce: float, n_test: int, source: str, overall_correct: int | None = None) -> dict[str, Any]:
    ov_k = int(overall_correct) if overall_correct is not None else int(round(br["acc"] * n_test))
    return {
        "module": "Module3_BBP_Oracle", "dataset": ds_name, "radix": radix, "model": model_name, "params_count": params_count,
        "overall_acc": round(br["acc"], 6), "overall_correct": ov_k, "wilson_ci_low": br["wilson_ci_low"], "wilson_ci_high": br["wilson_ci_high"],
        "targeted_acc": None, "targeted_correct": None, "targeted_ci_low": None, "targeted_ci_high": None, "cross_entropy_nats": round(ce, 6),
        "targeted_ce_nats": None, "bits_per_symbol": round(br["bits_per_symbol"], 6), "shannon_limit_bits": round(br["shannon_limit_bits"], 6),
        "z_score": br["z_score"], "overall_z_score": br["z_score"], "z_p_value": br["z_p_value"],
        "targeted_z_score": None, "targeted_z_p_value": None, "targeted_z_log10_p": None,
        "num_eval_samples": n_test, "num_masked_samples": 0,
        "clean_csprng_acc": None, "source": source,
    }


def run_module3_bbp_oracle(config: dict[str, Any], device: torch.device, reuse_checkpoints: bool = False) -> dict[str, Any]:
    split_cfg = config["data"]["split"]
    num_bits = 20
    num_workers = int(config["training"].get("num_workers", 4))
    seed = int(config.get("seed", 42))
    rows, mc_results = [], {}

    for ds_name, radix in (("pi_base16", 16), ("true_pi", 10)):
        art = get_or_create_dataset(ds_name, config=config)
        n_total = len(art.digits)
        tr_end = int(n_total * float(split_cfg["train"]))
        val_end = tr_end + int(n_total * float(split_cfg["val"]))
        idx_all = np.arange(n_total, dtype=np.int64)
        worker_init = seed_worker if num_workers > 0 else None
        loaders = {
            k: DataLoader(
                BinaryCoordinateDataset(idx_all[sl], art.digits[sl], num_bits=num_bits),
                batch_size=1024,
                shuffle=(k == "train"),
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(num_workers > 0),
                worker_init_fn=worker_init,
                generator=make_dataloader_generator(seed) if k == "train" else None,
                collate_fn=_fast_batch_collate,
            )
            for k, sl in (("train", slice(0, tr_end)), ("val", slice(tr_end, val_end)), ("test", slice(val_end, None)))
        }
        uni_corr = art.digits[val_end:] == 0
        n_test = len(uni_corr)
        rows.append(_oracle_row(ds_name, radix, "Uniform", None, compute_bitrate_metrics(math.log(radix), radix, int(np.sum(uni_corr)), n_test), math.log(radix), n_test, "theoretical_uniform", overall_correct=int(np.sum(uni_corr))))

        for arch, m_name in (("mlp", "DigitMLP"), ("transformer", "DigitTransformer")):
            best_m, res = _load_or_train_submodel(arch, "exp5_bbp", ds_name, radix, num_bits, loaders, config, device, reuse_checkpoints)
            model_corr, nll_vec, _ = _eval_canonical_loader(best_m, loaders["test"], device=device, amp_dtype="float16", extract_attn=False)
            mc_results[f"{ds_name}_{m_name}_vs_Uniform"] = mcnemar_test(model_corr, uni_corr)
            s = res["summary"]
            n_ok = int(model_corr.sum())
            ce_val = float(nll_vec.mean())
            rows.append(_oracle_row(ds_name, radix, m_name, s["param_count"], compute_bitrate_metrics(ce_val, radix, n_ok, n_test), ce_val, n_test, str(s["run_dir"]), overall_correct=n_ok))
            del best_m
            free_gpu()
    return {"rows": rows, "mcnemar": mc_results}


def run_experiment_5(modules: list[int] | None = None, reuse_dir: Path | None = None, reuse_checkpoints: bool = False) -> Path:
    t_start = time.perf_counter()
    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config["training"].get("device", "cuda")))
    runs_root = _resolve_runs_root(config)
    summary_dir = reuse_dir if reuse_dir is not None else create_run_dir(runs_root, "exp5_mech_interp")
    summary_dir.mkdir(parents=True, exist_ok=True)
    active = set(modules or [1, 2, 3])

    if 1 in active:
        m1 = run_module1_mech_interp(config=config, summary_dir=summary_dir, device=device)
        emb_geo, ablation_df, probing_df = m1["embedding_geometry"], m1["head_ablation"], m1["linear_probing"]
    else:
        emb_geo = json.loads((summary_dir / "embedding_geometry.json").read_text(encoding="utf-8")) if (summary_dir / "embedding_geometry.json").exists() else {}
        ablation_df = pd.read_csv(summary_dir / "head_ablation.csv") if (summary_dir / "head_ablation.csv").exists() else pd.DataFrame()
        probing_df = pd.read_csv(summary_dir / "linear_probing.csv") if (summary_dir / "linear_probing.csv").exists() else pd.DataFrame()

    m2_out = run_module2_crypto_auditor(config, summary_dir, device, reuse_checkpoints) if 2 in active else {}
    m3_out = run_module3_bbp_oracle(config, device, reuse_checkpoints) if 3 in active else {}
    all_rows = m2_out.get("rows", []) + m3_out.get("rows", [])
    if all_rows:
        pd.DataFrame(all_rows).to_csv(summary_dir / "metrics.csv", index=False)

    stat_audit = {
        "module2_crypto_backdoor": m2_out.get("mcnemar", {}),
        "module2_backdoor_head_ablation": m2_out.get("backdoor_head_ablation", []),
        "module2_classical_vs_clean": {"csprng_backdoor": m2_out.get("audit_plot_data", {}).get("classical_stats", {}), "csprng_clean": m2_out.get("audit_plot_data", {}).get("clean_classical_stats", {})},
        "module3_bbp_oracle": m3_out.get("mcnemar", {}),
    }
    (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (summary_dir / "plot_data.json").write_text(json.dumps({
        "embedding_geometry": emb_geo,
        "head_ablation_summary": ablation_df.to_dict(orient="records") if not ablation_df.empty else [],
        "linear_probing": probing_df.to_dict(orient="records") if not probing_df.empty else [],
        "crypto_audit": m2_out.get("audit_plot_data", {}),
        "crypto_attn_profile": m2_out.get("attn_profile", {}),
        "metrics": all_rows,
        "stat_audit": stat_audit,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    saved_plots = []
    if emb_geo:
        saved_plots.extend(plot_exp5_embedding_clock_pca(emb_geo, summary_dir))
    if not ablation_df.empty:
        saved_plots.extend(plot_exp5_head_ablation_heatmap(ablation_df, summary_dir))
    if not probing_df.empty:
        saved_plots.extend(plot_exp5_linear_probing_bars(probing_df, summary_dir))
    if m2_out.get("audit_plot_data") and m2_out.get("attn_profile"):
        saved_plots.extend(plot_exp5_crypto_backdoor_xray(m2_out["audit_plot_data"], m2_out["attn_profile"], summary_dir))
    print(f"\nЭксперимент 5 завершён за {time.perf_counter() - t_start:.1f}с: {summary_dir} ({len(saved_plots)} графиков)")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 5: Механистическая интерпретируемость, крипто-аудитор и BBP-оракул")
    parser.add_argument("--modules", type=str, default="1,2,3", help="Какие модули запускать: 1,2,3")
    parser.add_argument("--reuse-dir", type=str, default=None, help="Папка runs/exp5_mech_interp_* для дозапуска")
    parser.add_argument("--reuse-checkpoints", action="store_true", help="Переоценить чекпойнты Модулей 2 и 3 без переобучения")
    args = parser.parse_args()
    run_experiment_5(modules=[int(x.strip()) for x in args.modules.split(",") if x.strip()], reuse_dir=Path(args.reuse_dir) if args.reuse_dir else None, reuse_checkpoints=bool(args.reuse_checkpoints))


if __name__ == "__main__":
    main()
