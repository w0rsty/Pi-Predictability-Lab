"""Фаза 4 — Эксперимент 5: Механистическая интерпретируемость, крипто-аудитор ГПСЧ и координатный BBP-оракул."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baselines import evaluate_uniform_baseline, fit_and_evaluate_ppm
from src.data_generator import get_or_create_dataset, load_config, seed_everything
from src.dataset import (
    BinaryCoordinateDataset,
    SequenceWindowDataset,
    _fast_batch_collate,
    create_dataloaders,
    split_sequence_chronologically,
    verify_zero_leakage,
)
from src.evaluator import (
    compute_binomial_z_score,
    compute_bitrate_metrics,
    evaluate_model,
    mcnemar_test,
    wilson_ci,
)
from src.models import DigitGRU, DigitTransformer, build_model, count_parameters
from src.stats_tests import run_classical_tests
from src.trainer import create_run_dir, resolve_device, train_model
from src.visualizer import (
    plot_exp5_crypto_backdoor_xray,
    plot_exp5_embedding_clock_pca,
    plot_exp5_head_ablation_heatmap,
    plot_exp5_linear_probing_bars,
)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _find_latest_dir(runs_root: Path, prefix: str) -> Path | None:
    cands = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
        reverse=True,
    )
    return cands[0] if cands else None


def _load_checkpoint_model(
    arch: str,
    ckpt_path: Path,
    radix: int,
    window_size: int,
    size_tier: str,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ck
    remapped = {
        k.replace(".attn.in_proj_weight", ".qkv_proj.weight")
        .replace(".attn.in_proj_bias", ".qkv_proj.bias")
        .replace(".attn.out_proj.", ".out_proj."): v
        for k, v in sd.items()
    }
    model.load_state_dict(remapped)
    model = model.to(device)
    model.eval()
    return model


def _build_champernowne_annotations(digits: np.ndarray) -> dict[str, np.ndarray]:
    """Восстанавливает точный номер n, позицию внутри числа и флаги переноса для champernowne."""
    n_len = len(digits)
    pos_in_num = np.empty(n_len, dtype=np.int16)
    num_len_arr = np.empty(n_len, dtype=np.int16)
    prev_ends_9 = np.zeros(n_len, dtype=np.uint8)
    boundary_9 = np.zeros(n_len, dtype=np.uint8)
    carry_digit = np.zeros(n_len, dtype=np.uint8)
    inc_digit = np.zeros(n_len, dtype=np.uint8)

    pos = 0
    num = 1
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
            is_carry_num = (num - 1) % 10 == 9
            for j in range(count):
                if s[j] != sp[j]:
                    if is_carry_num:
                        carry_digit[pos + j] = 1
                    else:
                        inc_digit[pos + j] = 1
        pos = end_pos
        num += 1

    return {
        "pos_in_num": pos_in_num,
        "num_len": num_len_arr,
        "prev_ends_9": prev_ends_9,
        "boundary_9": boundary_9,
        "carry_digit": carry_digit,
        "inc_digit": inc_digit,
    }


def _build_e_cf_annotations(digits: np.ndarray) -> dict[str, np.ndarray]:
    """Восстанавливает фазу внутри тройки (1, 2k, 1) и флаг чётных цифр 2k для e_cf."""
    n_len = len(digits)
    # phase: 0 = первый '1' (+ ';'), 1 = '2k' (+ ';'), 2 = второй '1' (+ ';')
    triple_phase = np.zeros(n_len, dtype=np.uint8)
    is_even_2k = np.zeros(n_len, dtype=np.uint8)

    # Начальный заголовок [2, 10]
    pos = min(2, n_len)
    k = 1
    while pos < n_len:
        # 1) первый '1' и ';'
        for _ in (1, 10):
            if pos >= n_len:
                break
            triple_phase[pos] = 0
            pos += 1
        # 2) цифры 2*k и ';'
        for _ in str(2 * k):
            if pos >= n_len:
                break
            triple_phase[pos] = 1
            is_even_2k[pos] = 1
            pos += 1
        if pos < n_len:
            triple_phase[pos] = 1
            pos += 1
        # 3) второй '1' и ';'
        for _ in (1, 10):
            if pos >= n_len:
                break
            triple_phase[pos] = 2
            pos += 1
        k += 1

    return {"triple_phase": triple_phase, "is_even_2k": is_even_2k}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _analyze_embeddings(checkpoints: dict[str, dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    """Модуль 1.1: PCA, косинусная структура соседних цифр, циклический порядок и сдвиг от init."""
    results: dict[str, Any] = {}
    init_cache: dict[tuple[int, str], np.ndarray] = {}

    for name, info in checkpoints.items():
        ckpt = torch.load(info["path"], map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ck
        emb = sd["token_emb.weight"].float().numpy()
        emb_digits = emb[:10]
        vocab_sz = int(emb.shape[0])
        d_model = int(emb.shape[1])
        tier = "heavy" if d_model >= 192 else "medium"
        if (vocab_sz, tier) not in init_cache:
            seed_everything(seed)
            m0 = build_model("transformer", radix=vocab_sz, window_size=64, size_tier=tier)
            init_cache[(vocab_sz, tier)] = m0.token_emb.weight.detach().float().numpy()
        w0 = init_cache[(vocab_sz, tier)]
        dw_digits = emb_digits - w0[:10]
        delta_pca = PCA(n_components=3, random_state=42).fit(dw_digits)
        cos_init = float(np.mean([_cosine(emb_digits[d], w0[d]) for d in range(10)]))
        delta_l2 = float(np.linalg.norm(emb - w0))

        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(emb_digits)
        coords_full = pca.transform(emb)[:, :2]
        var_ratio = [round(float(v), 6) for v in pca.explained_variance_ratio_]

        adj_0_8 = [_cosine(emb_digits[d], emb_digits[d + 1]) for d in range(9)]
        cos_9_0 = _cosine(emb_digits[9], emb_digits[0])
        adj_mod10 = adj_0_8 + [cos_9_0]
        non_adj = [
            _cosine(emb_digits[i], emb_digits[j])
            for i in range(10)
            for j in range(i + 1, 10)
            if (j - i) not in (1, 9)
        ]

        # Оценка циклического порядка («арифметический циферблат») вдоль дуги PC1-PC2
        pc2d = coords_3d[:, :2]
        centered = pc2d - pc2d.mean(axis=0, keepdims=True)
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        step_deltas = np.angle(np.exp(1j * (np.roll(angles, -1) - angles)))
        pos_steps = int(np.sum(step_deltas > 0))
        neg_steps = int(np.sum(step_deltas < 0))
        cyclic_consistency = max(pos_steps, neg_steps) / 10.0

        # Дополнительно: монотонность углов на 0..9 без замыкания (сумма 9 шагов в одном направлении)
        arc_deltas = np.angle(np.exp(1j * np.diff(angles)))
        arc_consistency = max(int(np.sum(arc_deltas > 0)), int(np.sum(arc_deltas < 0))) / 9.0

        results[name] = {
            "d_model": d_model,
            "vocab_size": vocab_sz,
            "explained_variance_ratio_pc1_pc2_pc3": var_ratio,
            "explained_variance_pc1_pc2": round(float(sum(pca.explained_variance_ratio_[:2])), 6),
            "explained_variance_pc1_3": round(float(sum(pca.explained_variance_ratio_[:3])), 6),
            "delta_from_init_l2": round(delta_l2, 4),
            "cos_with_init_mean": round(cos_init, 6),
            "delta_explained_variance_pc1_pc2": round(float(sum(delta_pca.explained_variance_ratio_[:2])), 6),
            "adjacent_cosine_0_to_8": round(float(np.mean(adj_0_8)), 6),
            "cosine_9_to_0": round(float(cos_9_0), 6),
            "adjacent_cosine_mod10_mean": round(float(np.mean(adj_mod10)), 6),
            "nonadjacent_cosine_mean": round(float(np.mean(non_adj)), 6),
            "cyclic_order_consistency_10": round(float(cyclic_consistency), 4),
            "arc_order_consistency_9": round(float(arc_consistency), 4),
            "angles_rad": [round(float(a), 4) for a in angles],
            "coords": np.round(coords_full, 5).tolist(),
        }
    return results


@torch.no_grad()
def _eval_preds_and_nll(
    model: DigitTransformer,
    loader: DataLoader,
    device: torch.device,
    ablate_heads: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (correct_bool_vec, nll_nats_vec) по всему DataLoader для честной разбивки по подмножествам."""
    model.eval()
    amp_enabled = device.type == "cuda"
    corr_chunks: list[np.ndarray] = []
    nll_chunks: list[np.ndarray] = []

    for contexts, targets, _ in loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if ablate_heads:
                logits = model.forward_with_head_ablation(contexts, ablate_heads=ablate_heads, return_attn=False)
            else:
                logits = model(contexts)
        logits_f32 = logits.float()
        log_probs = F.log_softmax(logits_f32, dim=-1)
        preds = torch.argmax(logits_f32, dim=-1)
        nll = F.nll_loss(log_probs, targets, reduction="none")
        corr_chunks.append((preds == targets).cpu().numpy())
        nll_chunks.append(nll.cpu().numpy())

    return np.concatenate(corr_chunks), np.concatenate(nll_chunks)


def _run_causal_head_ablation(
    config: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    """Модуль 1.2: По-головая абляция DigitTransformer на champernowne, e_cf, rational_1_97 и true_pi."""
    split_cfg = config["data"]["split"]
    num_workers = int(config["training"].get("num_workers", 4))
    rows: list[dict[str, Any]] = []

    targets_cfg = [
        ("champernowne", 10, 64, "medium", None),
        ("e_cf", 11, 64, "medium", None),
        ("rational_1_97", 10, 128, "medium", 1_000_050),
        ("true_pi", 10, 64, "medium", None),
    ]

    for ds_name, radix, win_size, size_tier, req_len in targets_cfg:
        art = get_or_create_dataset(ds_name, config=config, length=req_len)
        split = split_sequence_chronologically(
            art.digits, art.mask, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
        )
        te_start, _ = split.test_range
        test_ds = SequenceWindowDataset(split.test_digits, window_size=win_size, mask=split.test_mask)
        bs = 512 if win_size >= 128 else 1024
        loader = DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_fast_batch_collate,
        )

        # Разметка подмножеств для целевых позиций [te_start + win_size : te_start + len(test_digits)]
        target_global_idx = np.arange(te_start + win_size, te_start + len(split.test_digits))
        subsets: dict[str, np.ndarray] = {"all": np.ones(len(target_global_idx), dtype=bool)}
        if ds_name == "champernowne":
            ann = _build_champernowne_annotations(art.digits)
            carry_mask = ann["prev_ends_9"][target_global_idx] == 1
            subsets["carry_over"] = carry_mask
            subsets["non_carry"] = ~carry_mask
            subsets["carry_digit"] = ann["carry_digit"][target_global_idx] == 1
            subsets["inc_digit"] = ann["inc_digit"][target_global_idx] == 1
        elif ds_name == "e_cf":
            ann = _build_e_cf_annotations(art.digits)
            even_mask = ann["is_even_2k"][target_global_idx] == 1
            subsets["even_2k"] = even_mask
            subsets["const_1_sep"] = ~even_mask

        model = _load_checkpoint_model(
            arch="transformer",
            ckpt_path=checkpoints[ds_name]["path"],
            radix=radix,
            window_size=win_size,
            size_tier=size_tier,
            device=device,
        )
        assert isinstance(model, DigitTransformer)

        base_corr, base_nll = _eval_preds_and_nll(model, loader, device=device, ablate_heads=None)
        base_stats: dict[str, tuple[float, float]] = {}
        for sub_name, sub_mask in subsets.items():
            b_acc = float(np.mean(base_corr[sub_mask]))
            b_bits = float(np.mean(base_nll[sub_mask]) / math.log(2.0))
            base_stats[sub_name] = (b_acc, b_bits)

        total_heads = model.n_layers * model.n_heads
        pbar = tqdm(
            [(l, h) for l in range(model.n_layers) for h in range(model.n_heads)],
            total=total_heads,
            desc=f"Head Ablation [{ds_name}]",
            unit="head",
        )
        for l_idx, h_idx in pbar:
            abl_corr, abl_nll = _eval_preds_and_nll(model, loader, device=device, ablate_heads=[(l_idx, h_idx)])
            for sub_name, sub_mask in subsets.items():
                b_acc, b_bits = base_stats[sub_name]
                a_acc = float(np.mean(abl_corr[sub_mask]))
                a_bits = float(np.mean(abl_nll[sub_mask]) / math.log(2.0))
                delta_acc_pp = (a_acc - b_acc) * 100.0
                delta_bits = a_bits - b_bits
                mc = mcnemar_test(base_corr[sub_mask], abl_corr[sub_mask])

                role = "redundant"
                if ds_name != "true_pi" and mc["significant_0_05"]:
                    if delta_acc_pp <= -1.0 or delta_bits >= 0.05:
                        if sub_name in ("carry_over", "carry_digit"):
                            role = "carry_head"
                        elif sub_name == "inc_digit":
                            role = "increment_head"
                        elif sub_name == "even_2k":
                            role = "induction_2k_head"
                        elif ds_name == "rational_1_97":
                            role = "periodic_head"
                        else:
                            role = "critical_head"
                    elif delta_acc_pp <= -0.35 or delta_bits >= 0.015:
                        role = "supporting_head"

                rows.append(
                    {
                        "dataset": ds_name,
                        "window_size": win_size,
                        "subset": sub_name,
                        "n_subset": int(np.sum(sub_mask)),
                        "layer": int(l_idx),
                        "head": int(h_idx),
                        "base_acc": round(b_acc, 6),
                        "ablated_acc": round(a_acc, 6),
                        "delta_acc_pp": round(delta_acc_pp, 4),
                        "base_bits": round(b_bits, 6),
                        "ablated_bits": round(a_bits, 6),
                        "delta_bits": round(delta_bits, 6),
                        "mcnemar_p": mc["p_value_str"],
                        "mcnemar_sig_0_05": mc["significant_0_05"],
                        "head_role": role,
                    }
                )
        del model
        _free_gpu()

    return pd.DataFrame(rows)


@torch.no_grad()
def _extract_hidden_slice(
    model: DigitGRU | DigitTransformer,
    test_digits: np.ndarray,
    window_size: int,
    n_windows: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Извлекает векторы скрытых состояний h_t на первых n_windows тестовых окнах без градиентов."""
    ds = SequenceWindowDataset(test_digits[: n_windows + window_size], window_size=window_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_fast_batch_collate,
    )
    chunks: list[np.ndarray] = []
    model.eval()
    for contexts, _, _ in loader:
        contexts = contexts.to(device, non_blocking=True)
        h_t = model.extract_hidden_states(contexts, all_positions=False)
        chunks.append(h_t.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _run_linear_probing(
    config: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    gru_checkpoints: dict[str, dict[str, Any]],
    device: torch.device,
    n_probe_windows: int = 10_000,
) -> pd.DataFrame:
    """Модуль 1.3: Линейное зондирование скрытых состояний Trained vs Untrained (80/20 сплит)."""
    split_cfg = config["data"]["split"]
    rows: list[dict[str, Any]] = []

    # Загружаем нужные датасеты и разметку
    e_cf_art = get_or_create_dataset("e_cf", config=config)
    e_cf_split = split_sequence_chronologically(
        e_cf_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )
    e_cf_ann = _build_e_cf_annotations(e_cf_art.digits)

    champ_art = get_or_create_dataset("champernowne", config=config)
    champ_split = split_sequence_chronologically(
        champ_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )
    champ_ann = _build_champernowne_annotations(champ_art.digits)

    rat_art = get_or_create_dataset("rational_1_97", config=config, length=1_000_050)
    rat_split = split_sequence_chronologically(
        rat_art.digits, None, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )

    # Позиция последнего входного токена каждого из 10,000 тестовых окон: te_start + window_size - 1 + i
    idx_64_ecf = np.arange(e_cf_split.test_range[0] + 63, e_cf_split.test_range[0] + 63 + n_probe_windows)
    idx_64_ch = np.arange(champ_split.test_range[0] + 63, champ_split.test_range[0] + 63 + n_probe_windows)
    idx_128_rat = np.arange(rat_split.test_range[0] + 127, rat_split.test_range[0] + 127 + n_probe_windows)

    probe_specs = [
        {
            "task": "e_cf_phase",
            "dataset": "e_cf",
            "radix": 11,
            "window_size": 64,
            "split": e_cf_split,
            "labels": e_cf_ann["triple_phase"][idx_64_ecf].astype(np.int64),
            "n_classes": 3,
            "chance": 1.0 / 3.0,
        },
        {
            "task": "champ_pos",
            "dataset": "champernowne",
            "radix": 10,
            "window_size": 64,
            "split": champ_split,
            "labels": champ_ann["pos_in_num"][idx_64_ch].astype(np.int64),
            "n_classes": 6,
            "chance": 1.0 / 6.0,
        },
        {
            "task": "champ_carry",
            "dataset": "champernowne",
            "radix": 10,
            "window_size": 64,
            "split": champ_split,
            "labels": champ_ann["boundary_9"][idx_64_ch].astype(np.int64),
            "n_classes": 2,
            "chance": 0.5,  # Для несбалансированного бинарного признака оцениваем Balanced Acc (random=0.5)
        },
        {
            "task": "rat97_phase",
            "dataset": "rational_1_97",
            "radix": 10,
            "window_size": 128,
            "split": rat_split,
            "labels": (idx_128_rat % 96).astype(np.int64),
            "n_classes": 96,
            "chance": 1.0 / 96.0,
        },
    ]

    # Кешируем извлечённые фичи, чтобы не гонять нейросеть дважды на champernowne
    features_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for arch_key, arch_name, ckpt_map in (
        ("gru", "DigitGRU", gru_checkpoints),
        ("transformer", "DigitTransformer", checkpoints),
    ):
        for spec in probe_specs:
            ds_name = spec["dataset"]
            radix = spec["radix"]
            win_size = spec["window_size"]
            test_digits = spec["split"].test_digits
            y = spec["labels"]

            probe_preds: dict[str, np.ndarray] = {}
            train_end = int(0.8 * n_probe_windows)  # Строго 8,000 Train / 2,000 Test без перемешивания!
            y_tr, y_te = y[:train_end], y[train_end:]

            for state_name in ("Trained", "Untrained"):
                cache_key = (arch_key, ds_name, state_name)
                if cache_key not in features_cache:
                    if state_name == "Trained":
                        model = _load_checkpoint_model(
                            arch=arch_key,
                            ckpt_path=ckpt_map[ds_name]["path"],
                            radix=radix,
                            window_size=win_size,
                            size_tier="medium",
                            device=device,
                        )
                    else:
                        seed_everything(int(config.get("seed", 42)) + 777)
                        model = build_model(arch=arch_key, radix=radix, window_size=win_size, size_tier="medium").to(device)
                        model.eval()
                    assert isinstance(model, (DigitGRU, DigitTransformer))
                    features_cache[cache_key] = _extract_hidden_slice(
                        model=model,
                        test_digits=test_digits,
                        window_size=win_size,
                        n_windows=n_probe_windows,
                        device=device,
                    )
                    del model
                    _free_gpu()

                x_feats = features_cache[cache_key]
                x_tr, x_te = x_feats[:train_end], x_feats[train_end:]

                pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=42),
                )
                pipe.fit(x_tr, y_tr)
                y_pred = pipe.predict(x_te)
                probe_preds[state_name] = y_pred

                acc = float(np.mean(y_pred == y_te))
                bal_acc = float(balanced_accuracy_score(y_te, y_pred))
                f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
                k_corr = int(np.sum(y_pred == y_te))
                ci_lo, ci_hi = wilson_ci(k_corr, len(y_te))

                rows.append(
                    {
                        "model": arch_name,
                        "task": spec["task"],
                        "dataset": ds_name,
                        "n_classes": spec["n_classes"],
                        "state": state_name,
                        "n_train": len(y_tr),
                        "n_test": len(y_te),
                        "probe_acc": round(acc, 6),
                        "wilson_ci_low": ci_lo,
                        "wilson_ci_high": ci_hi,
                        "balanced_acc": round(bal_acc, 6),
                        "f1_macro": round(f1, 6),
                        "chance_baseline": round(float(spec["chance"]), 6),
                        "mcnemar_vs_untrained_p": "",
                    }
                )

            mc = mcnemar_test(probe_preds["Trained"] == y_te, probe_preds["Untrained"] == y_te)
            for row in rows[-2:]:
                if row["state"] == "Trained":
                    row["mcnemar_vs_untrained_p"] = mc["p_value_str"]

    return pd.DataFrame(rows)


def run_module1_mech_interp(config: dict[str, Any], summary_dir: Path, device: torch.device) -> dict[str, Any]:
    """Запускает все 3 подмодуля механистической интерпретируемости."""
    print("\n================ МОДУЛЬ 1: Механистическая интерпретируемость ================")
    runs_root = Path(config["paths"]["runs_dir"])
    exp2_dir = _find_latest_dir(runs_root, "exp2_constants_final_")
    exp4_dir = _find_latest_dir(runs_root, "exp4_scaling_")
    exp4_gru_rat97 = _find_latest_dir(runs_root, "exp4_block_b_phasetransition_h1_gru_rational_1_97_L128_medium_")

    if exp2_dir is None or exp4_dir is None or exp4_gru_rat97 is None:
        raise FileNotFoundError("Не найдены чекпойнты Exp2 или Exp4 для Модуля 1")

    tr_ckpts = {
        "true_pi": {"path": exp2_dir / "checkpoints" / "transformer_true_pi.pt"},
        "e_cf": {"path": exp2_dir / "checkpoints" / "transformer_e_cf.pt"},
        "champernowne": {"path": exp2_dir / "checkpoints" / "transformer_champernowne.pt"},
        "rational_1_999983": {"path": exp2_dir / "checkpoints" / "transformer_rational_1_999983.pt"},
        "rational_1_97": {"path": exp4_dir / "checkpoints" / "transformer_rational_1_97_L128.pt"},
        "memorization_10k_heavy": {"path": exp4_dir / "checkpoints" / "transformer_memorization_10k_heavy.pt"},
    }
    gru_ckpts = {
        "e_cf": {"path": exp2_dir / "checkpoints" / "gru_e_cf.pt"},
        "champernowne": {"path": exp2_dir / "checkpoints" / "gru_champernowne.pt"},
        "rational_1_97": {"path": exp4_gru_rat97 / "best_model.pt"},
    }

    # 1.1 Геометрия эмбеддингов
    print("[1.1] Анализ геометрии эмбеддингов (PCA + Cosine Similarity)...")
    emb_geo = _analyze_embeddings(tr_ckpts, seed=int(config.get("seed", 42)))
    (summary_dir / "embedding_geometry.json").write_text(
        json.dumps(emb_geo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 1.2 Каузальная абляция голов внимания
    print("[1.2] Causal Head Ablation по слоям и головам...")
    ablation_df = _run_causal_head_ablation(config=config, checkpoints=tr_ckpts, device=device)
    ablation_df.to_csv(summary_dir / "head_ablation.csv", index=False)

    # 1.3 Линейное зондирование
    print("[1.3] Linear Probing скрытых состояний (DigitGRU & DigitTransformer)...")
    probing_df = _run_linear_probing(
        config=config, checkpoints=tr_ckpts, gru_checkpoints=gru_ckpts, device=device
    )
    probing_df.to_csv(summary_dir / "linear_probing.csv", index=False)

    return {
        "embedding_geometry": emb_geo,
        "head_ablation": ablation_df,
        "linear_probing": probing_df,
    }


@torch.no_grad()
def _collect_eval_preds_and_attn(
    model: torch.nn.Module,
    test_digits: np.ndarray,
    test_mask: np.ndarray,
    window_size: int,
    device: torch.device,
    batch_size: int = 512,
    extract_attn: bool = False,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Собирает поэлементные правильные ответы на Test и усреднённый профиль внимания на триггерах."""
    ds = SequenceWindowDataset(test_digits, window_size=window_size, mask=test_mask)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_fast_batch_collate,
    )
    model.eval()
    corr_chunks: list[np.ndarray] = []

    trig_attn_sum: np.ndarray | None = None
    nontrig_attn_sum: np.ndarray | None = None
    trig_count = 0
    nontrig_count = 0
    max_collect = 2_000

    for contexts, targets, mask_flags in loader:
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Всегда считаем предсказания единым AMP-путём (совпадает с evaluate_model в metrics.csv)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(contexts)
        preds = torch.argmax(logits.float(), dim=-1)
        corr_chunks.append((preds == targets).cpu().numpy())

        # Извлечение карт внимания на триггерах/фоне делаем отдельно, чтобы не смешивать FP32/FP16 argmax
        if extract_attn and (trig_count < max_collect or nontrig_count < max_collect):
            m_np = mask_flags.numpy() == 1
            need_trig = np.any(m_np) and trig_count < max_collect
            need_nontrig = np.any(~m_np) and nontrig_count < max_collect
            if need_trig or need_nontrig:
                _, attn = model(contexts, return_attn=True)
                last_q_attn = attn[:, :, :, -1, :].float().cpu().numpy()  # (B, n_layers, n_heads, L)
                if need_trig:
                    take = last_q_attn[m_np][: max_collect - trig_count]
                    trig_attn_sum = take.sum(axis=0) if trig_attn_sum is None else trig_attn_sum + take.sum(axis=0)
                    trig_count += len(take)
                if need_nontrig:
                    take = last_q_attn[~m_np][: max_collect - nontrig_count]
                    nontrig_attn_sum = take.sum(axis=0) if nontrig_attn_sum is None else nontrig_attn_sum + take.sum(axis=0)
                    nontrig_count += len(take)

    corr_vec = np.concatenate(corr_chunks)
    if not extract_attn or trig_attn_sum is None or nontrig_attn_sum is None:
        return corr_vec, None

    trig_mean_all = trig_attn_sum / max(trig_count, 1)  # (n_layers, n_heads, L)
    nontrig_mean_all = nontrig_attn_sum / max(nontrig_count, 1)

    # Находим голову с максимальной суммарной концентрацией внимания на скрытых лагах t-24 и t-7
    lag24_idx = window_size - 24
    lag7_idx = window_size - 7
    n_layers, n_heads, _ = trig_mean_all.shape
    best_score = -1.0
    best_l, best_h = 0, 0
    per_head_lag_mass: list[dict[str, Any]] = []
    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            m24 = float(trig_mean_all[l_idx, h_idx, lag24_idx])
            m7 = float(trig_mean_all[l_idx, h_idx, lag7_idx])
            m1 = float(trig_mean_all[l_idx, h_idx, window_size - 1])
            score = m24 + m7
            per_head_lag_mass.append(
                {
                    "layer": l_idx,
                    "head": h_idx,
                    "mass_t_minus_24": round(m24, 5),
                    "mass_t_minus_7": round(m7, 5),
                    "mass_t_minus_1": round(m1, 5),
                    "sum_24_7": round(score, 5),
                }
            )
            if score > best_score:
                best_score = score
                best_l, best_h = l_idx, h_idx

    attn_info = {
        "lags": list(range(-window_size, 0)),
        "focus_layer": int(best_l),
        "focus_head": int(best_h),
        "focus_head_label": f"L{best_l}-H{best_h}",
        "trigger_samples_averaged": int(trig_count),
        "nontrigger_samples_averaged": int(nontrig_count),
        "trigger_mean_attn": np.round(trig_mean_all[best_l, best_h], 5).tolist(),
        "nontrigger_mean_attn": np.round(nontrig_mean_all[best_l, best_h], 5).tolist(),
        "trigger_lag24_attn": round(float(trig_mean_all[best_l, best_h, lag24_idx]), 5),
        "trigger_lag7_attn": round(float(trig_mean_all[best_l, best_h, lag7_idx]), 5),
        "per_head_lag_mass": per_head_lag_mass,
    }
    return corr_vec, attn_info


def run_module2_crypto_auditor(
    config: dict[str, Any],
    summary_dir: Path,
    device: torch.device,
    reuse_checkpoints: bool = False,
) -> dict[str, Any]:
    """Модуль 2: Аудит скрытой стеганографической закладки в csprng_backdoor."""
    print("\n================ МОДУЛЬ 2: Прикладной крипто-аудитор ГПСЧ ================")
    seed = int(config.get("seed", 42))
    split_cfg = config["data"]["split"]
    win_size = 64

    clean_art = get_or_create_dataset("csprng", config=config)
    bd_art = get_or_create_dataset("csprng_backdoor", config=config)
    assert bd_art.mask is not None

    clean_stats = run_classical_tests("csprng", clean_art.digits, radix=10)
    bd_stats = run_classical_tests("csprng_backdoor", bd_art.digits, radix=10)

    bd_split = split_sequence_chronologically(
        bd_art.digits, bd_art.mask, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    )
    verify_zero_leakage(bd_split, window_size=win_size)

    # Бейзлайны на csprng_backdoor: Uniform & PPM_o8
    uni_m = evaluate_uniform_baseline(bd_split, radix=10, window_size=win_size)
    ppm_m, ppm_probs = fit_and_evaluate_ppm(bd_split, radix=10, window_size=win_size, max_order=8)
    test_targets = bd_split.test_digits[win_size:]
    test_mask = bd_split.test_mask[win_size:]
    mask_bool = test_mask == 1

    ppm_corr = np.argmax(ppm_probs, axis=1) == test_targets
    uni_corr = test_targets == 0  # argmax=0 для uniform

    # Обучаем DigitGRU и DigitTransformer (medium ~250K, max_epochs=10, patience=3)
    loaders = create_dataloaders(
        split=bd_split,
        window_size=win_size,
        batch_size=512,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    neural_runs: dict[str, dict[str, Any]] = {}
    neural_corr: dict[str, np.ndarray] = {}
    attn_profile: dict[str, Any] | None = None

    ckpt_dir = summary_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(config["paths"]["runs_dir"])

    for arch, m_name in (("gru", "DigitGRU"), ("transformer", "DigitTransformer")):
        existing_sub = _find_latest_dir(runs_root, f"exp5_{arch}_csprng_backdoor_") if reuse_checkpoints else None
        if existing_sub is not None and (existing_sub / "best_model.pt").exists():
            ckpt_path = existing_sub / "best_model.pt"
            best_model = _load_checkpoint_model(
                arch=arch,
                ckpt_path=ckpt_path,
                radix=10,
                window_size=win_size,
                size_tier="medium",
                device=device,
            )
            tm = evaluate_model(
                model=best_model,
                loader=loaders["test"],
                device=device,
                radix=10,
                use_amp=bool(config["training"].get("use_amp", True)),
                collect_transitions=False,
            )
            res = {
                "test_metrics": tm,
                "summary": {
                    "param_count": count_parameters(best_model),
                    "checkpoint_path": str(ckpt_path),
                    "run_dir": str(existing_sub),
                },
            }
        else:
            seed_everything(seed)
            model = build_model(arch=arch, radix=10, window_size=win_size, size_tier="medium")
            sub_run_dir = create_run_dir(config["paths"]["runs_dir"], f"exp5_{arch}_csprng_backdoor")
            res = train_model(
                model=model,
                loaders=loaders,
                config=config,
                exp_name=f"exp5_{arch}_csprng_backdoor",
                radix=10,
                epochs=10,
                learning_rate=1e-3,
                scheduler_type="cosine",
                early_stopping_patience=3,
                run_dir=sub_run_dir,
                show_progress=True,
            )
            best_model = _load_checkpoint_model(
                arch=arch,
                ckpt_path=Path(res["summary"]["checkpoint_path"]),
                radix=10,
                window_size=win_size,
                size_tier="medium",
                device=device,
            )
            del model

        neural_runs[m_name] = res
        corr_vec, attn_res = _collect_eval_preds_and_attn(
            model=best_model,
            test_digits=bd_split.test_digits,
            test_mask=bd_split.test_mask,
            window_size=win_size,
            device=device,
            extract_attn=(arch == "transformer"),
        )
        neural_corr[m_name] = corr_vec
        if arch == "transformer":
            attn_profile = attn_res
            shutil.copy2(res["summary"]["checkpoint_path"], ckpt_dir / "transformer_csprng_backdoor.pt")
        del best_model
        _free_gpu()

    # Подтягиваем метрики чистого csprng из Exp 2
    runs_root = Path(config["paths"]["runs_dir"])
    exp2_dir = _find_latest_dir(runs_root, "exp2_constants_final_")
    clean_exp2_rows: dict[str, dict[str, Any]] = {}
    if exp2_dir and (exp2_dir / "metrics.csv").exists():
        df_exp2 = pd.read_csv(exp2_dir / "metrics.csv")
        sub_clean = df_exp2[df_exp2["dataset"] == "csprng"]
        for _, r in sub_clean.iterrows():
            clean_exp2_rows[str(r["model"])] = r.to_dict()

    # Сводим метрики Модуля 2
    module2_rows: list[dict[str, Any]] = []
    for m_name, m_dict in (("Uniform", uni_m), ("PPM_o8", ppm_m)):
        n_eval = int(m_dict["num_eval_samples"])
        k_all = int(round(m_dict["overall_acc"] * n_eval))
        ci_lo, ci_hi = wilson_ci(k_all, n_eval)
        n_mask = int(m_dict["num_masked_samples"])
        k_tgt = int(round((m_dict["targeted_acc"] or 0.0) * n_mask))
        t_lo, t_hi = wilson_ci(k_tgt, n_mask)
        br = compute_bitrate_metrics(m_dict["cross_entropy"], radix=10, correct=k_all, total=n_eval)
        module2_rows.append(
            {
                "module": "Module2_CryptoBackdoor",
                "dataset": "csprng_backdoor",
                "radix": 10,
                "model": m_name,
                "params_count": None,
                "overall_acc": round(m_dict["overall_acc"], 6),
                "wilson_ci_low": ci_lo,
                "wilson_ci_high": ci_hi,
                "targeted_acc": round(float(m_dict["targeted_acc"]), 6),
                "targeted_ci_low": t_lo,
                "targeted_ci_high": t_hi,
                "cross_entropy_nats": round(m_dict["cross_entropy"], 6),
                "targeted_ce_nats": round(float(m_dict["targeted_ce"]), 6),
                "bits_per_symbol": round(br["bits_per_symbol"], 6),
                "shannon_limit_bits": round(br["shannon_limit_bits"], 6),
                "z_score": br["z_score"],
                "z_p_value": br["z_p_value"],
                "num_eval_samples": n_eval,
                "num_masked_samples": n_mask,
                "clean_csprng_acc": clean_exp2_rows.get(m_name, {}).get("test_acc"),
                "source": "exp5_module2",
            }
        )

    for m_name, res in neural_runs.items():
        tm = res["test_metrics"]
        s = res["summary"]
        n_eval = int(tm["num_eval_samples"])
        k_all = int(round(tm["overall_acc"] * n_eval))
        ci_lo, ci_hi = wilson_ci(k_all, n_eval)
        n_mask = int(tm["num_masked_samples"])
        k_tgt = int(round(tm["targeted_acc"] * n_mask))
        t_lo, t_hi = wilson_ci(k_tgt, n_mask)
        br = compute_bitrate_metrics(tm["cross_entropy"], radix=10, correct=k_all, total=n_eval)
        module2_rows.append(
            {
                "module": "Module2_CryptoBackdoor",
                "dataset": "csprng_backdoor",
                "radix": 10,
                "model": m_name,
                "params_count": s["param_count"],
                "overall_acc": round(tm["overall_acc"], 6),
                "wilson_ci_low": ci_lo,
                "wilson_ci_high": ci_hi,
                "targeted_acc": round(tm["targeted_acc"], 6),
                "targeted_ci_low": t_lo,
                "targeted_ci_high": t_hi,
                "cross_entropy_nats": round(tm["cross_entropy"], 6),
                "targeted_ce_nats": round(tm["targeted_ce"], 6),
                "bits_per_symbol": round(br["bits_per_symbol"], 6),
                "shannon_limit_bits": round(br["shannon_limit_bits"], 6),
                "z_score": br["z_score"],
                "z_p_value": br["z_p_value"],
                "num_eval_samples": n_eval,
                "num_masked_samples": n_mask,
                "clean_csprng_acc": clean_exp2_rows.get(m_name, {}).get("test_acc"),
                "source": str(s["run_dir"]),
            }
        )

    # Макнемар на всём тесте и строго на позициях закладки (mask==1)
    mc_audit = {
        "Transformer_vs_Uniform_all": mcnemar_test(neural_corr["DigitTransformer"], uni_corr),
        "Transformer_vs_PPM_all": mcnemar_test(neural_corr["DigitTransformer"], ppm_corr),
        "Transformer_vs_GRU_all": mcnemar_test(neural_corr["DigitTransformer"], neural_corr["DigitGRU"]),
        "GRU_vs_PPM_all": mcnemar_test(neural_corr["DigitGRU"], ppm_corr),
        "Transformer_vs_PPM_targeted": mcnemar_test(neural_corr["DigitTransformer"][mask_bool], ppm_corr[mask_bool]),
        "Transformer_vs_GRU_targeted": mcnemar_test(
            neural_corr["DigitTransformer"][mask_bool], neural_corr["DigitGRU"][mask_bool]
        ),
    }

    audit_plot_data = {
        "classical_stats": {
            "chi2_p_value": bd_stats["chi2_p_value"],
            "ks_p_value": bd_stats["ks_p_value"],
            "acf_max_abs": bd_stats["acf_max_abs"],
            "acf_max_lag": bd_stats["acf_max_lag"],
            "lzma_bits_per_digit": bd_stats["lzma_bits_per_digit"],
            "trigger_count": int(bd_art.mask.sum()),
            "trigger_fraction": float(bd_art.mask.mean()),
        },
        "clean_classical_stats": {
            "chi2_p_value": clean_stats["chi2_p_value"],
            "ks_p_value": clean_stats["ks_p_value"],
            "acf_max_abs": clean_stats["acf_max_abs"],
            "lzma_bits_per_digit": clean_stats["lzma_bits_per_digit"],
        },
        "models": [
            {
                "label": r["model"],
                "overall_acc": r["overall_acc"],
                "targeted_acc": r["targeted_acc"],
            }
            for r in module2_rows
        ],
    }

    return {
        "rows": module2_rows,
        "mcnemar": mc_audit,
        "audit_plot_data": audit_plot_data,
        "attn_profile": attn_profile,
    }


def run_module3_bbp_oracle(
    config: dict[str, Any],
    device: torch.device,
    reuse_checkpoints: bool = False,
) -> dict[str, Any]:
    """Модуль 3: Координатный BBP-оракул (20-битный индекс n -> n-я цифра pi в Base-16 vs Base-10)."""
    print("\n================ МОДУЛЬ 3: Координатный BBP-оракул (n -> d_n) ================")
    seed = int(config.get("seed", 42))
    split_cfg = config["data"]["split"]
    num_bits = 20
    num_workers = int(config["training"].get("num_workers", 4))
    runs_root = Path(config["paths"]["runs_dir"])

    rows: list[dict[str, Any]] = []
    mc_results: dict[str, Any] = {}

    for ds_name, radix in (("pi_base16", 16), ("true_pi", 10)):
        art = get_or_create_dataset(ds_name, config=config)
        n_total = len(art.digits)
        train_end = int(n_total * float(split_cfg["train"]))
        val_end = train_end + int(n_total * float(split_cfg["val"]))

        idx_all = np.arange(n_total, dtype=np.int64)
        ds_train = BinaryCoordinateDataset(idx_all[:train_end], art.digits[:train_end], num_bits=num_bits)
        ds_val = BinaryCoordinateDataset(idx_all[train_end:val_end], art.digits[train_end:val_end], num_bits=num_bits)
        ds_test = BinaryCoordinateDataset(idx_all[val_end:], art.digits[val_end:], num_bits=num_bits)

        persistent = num_workers > 0
        loaders = {
            "train": DataLoader(
                ds_train,
                batch_size=1024,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=persistent,
                collate_fn=_fast_batch_collate,
            ),
            "val": DataLoader(
                ds_val,
                batch_size=1024,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=persistent,
                collate_fn=_fast_batch_collate,
            ),
            "test": DataLoader(
                ds_test,
                batch_size=1024,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=persistent,
                collate_fn=_fast_batch_collate,
            ),
        }

        y_test = art.digits[val_end:]
        n_test = len(y_test)
        uni_corr = (y_test == 0)
        k_uni = int(np.sum(uni_corr))
        uni_ce = math.log(radix)
        uni_br = compute_bitrate_metrics(uni_ce, radix=radix, correct=k_uni, total=n_test)
        rows.append(
            {
                "module": "Module3_BBP_Oracle",
                "dataset": ds_name,
                "radix": radix,
                "model": "Uniform",
                "params_count": None,
                "overall_acc": round(uni_br["acc"], 6),
                "wilson_ci_low": uni_br["wilson_ci_low"],
                "wilson_ci_high": uni_br["wilson_ci_high"],
                "targeted_acc": None,
                "targeted_ci_low": None,
                "targeted_ci_high": None,
                "cross_entropy_nats": round(uni_ce, 6),
                "targeted_ce_nats": None,
                "bits_per_symbol": round(uni_br["bits_per_symbol"], 6),
                "shannon_limit_bits": round(uni_br["shannon_limit_bits"], 6),
                "z_score": uni_br["z_score"],
                "z_p_value": uni_br["z_p_value"],
                "num_eval_samples": n_test,
                "num_masked_samples": 0,
                "clean_csprng_acc": None,
                "source": "theoretical_uniform",
            }
        )

        for arch, m_name in (("mlp", "DigitMLP"), ("transformer", "DigitTransformer")):
            existing_sub = _find_latest_dir(runs_root, f"exp5_bbp_{arch}_{ds_name}_") if reuse_checkpoints else None
            if existing_sub is not None and (existing_sub / "best_model.pt").exists():
                ckpt_path = existing_sub / "best_model.pt"
                best_m = _load_checkpoint_model(
                    arch=arch,
                    ckpt_path=ckpt_path,
                    radix=radix,
                    window_size=num_bits,
                    size_tier="medium",
                    device=device,
                )
                tm = evaluate_model(
                    model=best_m,
                    loader=loaders["test"],
                    device=device,
                    radix=radix,
                    use_amp=bool(config["training"].get("use_amp", True)),
                    collect_transitions=False,
                )
                res = {
                    "test_metrics": tm,
                    "summary": {
                        "param_count": count_parameters(best_m),
                        "checkpoint_path": str(ckpt_path),
                        "run_dir": str(existing_sub),
                    },
                }
            else:
                seed_everything(seed)
                model = build_model(arch=arch, radix=radix, window_size=num_bits, size_tier="medium")
                sub_run_dir = create_run_dir(config["paths"]["runs_dir"], f"exp5_bbp_{arch}_{ds_name}")
                res = train_model(
                    model=model,
                    loaders=loaders,
                    config=config,
                    exp_name=f"exp5_bbp_{arch}_{ds_name}",
                    radix=radix,
                    epochs=10,
                    learning_rate=1e-3,
                    scheduler_type="cosine",
                    early_stopping_patience=3,
                    run_dir=sub_run_dir,
                    show_progress=True,
                )
                best_m = _load_checkpoint_model(
                    arch=arch,
                    ckpt_path=Path(res["summary"]["checkpoint_path"]),
                    radix=radix,
                    window_size=num_bits,
                    size_tier="medium",
                    device=device,
                )
                del model
            corr_chunks: list[np.ndarray] = []
            with torch.no_grad():
                for ctx, tgt, _ in loaders["test"]:
                    ctx = ctx.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits = best_m(ctx)
                    corr_chunks.append((torch.argmax(logits.float(), dim=-1) == tgt).cpu().numpy())
            model_corr = np.concatenate(corr_chunks)
            mc = mcnemar_test(model_corr, uni_corr)
            mc_results[f"{ds_name}_{m_name}_vs_Uniform"] = mc

            tm = res["test_metrics"]
            s = res["summary"]
            k_corr = int(np.sum(model_corr))
            br = compute_bitrate_metrics(tm["cross_entropy"], radix=radix, correct=k_corr, total=n_test)
            rows.append(
                {
                    "module": "Module3_BBP_Oracle",
                    "dataset": ds_name,
                    "radix": radix,
                    "model": m_name,
                    "params_count": s["param_count"],
                    "overall_acc": round(br["acc"], 6),
                    "wilson_ci_low": br["wilson_ci_low"],
                    "wilson_ci_high": br["wilson_ci_high"],
                    "targeted_acc": None,
                    "targeted_ci_low": None,
                    "targeted_ci_high": None,
                    "cross_entropy_nats": round(tm["cross_entropy"], 6),
                    "targeted_ce_nats": None,
                    "bits_per_symbol": round(br["bits_per_symbol"], 6),
                    "shannon_limit_bits": round(br["shannon_limit_bits"], 6),
                    "z_score": br["z_score"],
                    "z_p_value": br["z_p_value"],
                    "num_eval_samples": n_test,
                    "num_masked_samples": 0,
                    "clean_csprng_acc": None,
                    "source": str(s["run_dir"]),
                }
            )
            del best_m
            _free_gpu()

    return {"rows": rows, "mcnemar": mc_results}


def run_experiment_5(
    modules: list[int] | None = None,
    reuse_dir: Path | None = None,
    reuse_checkpoints: bool = False,
) -> Path:
    t_start = time.perf_counter()
    config = load_config("configs/config.yaml")
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config["training"].get("device", "cuda")))
    print(f"Устройство: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    summary_dir = reuse_dir if reuse_dir is not None else create_run_dir(config["paths"]["runs_dir"], "exp5_mech_interp")
    summary_dir.mkdir(parents=True, exist_ok=True)
    active = set(modules or [1, 2, 3])

    emb_geo: dict[str, Any] = {}
    ablation_df = pd.DataFrame()
    probing_df = pd.DataFrame()
    m2_out: dict[str, Any] = {}
    m3_out: dict[str, Any] = {}

    if 1 in active:
        m1_out = run_module1_mech_interp(config=config, summary_dir=summary_dir, device=device)
        emb_geo = m1_out["embedding_geometry"]
        ablation_df = m1_out["head_ablation"]
        probing_df = m1_out["linear_probing"]
    else:
        if (summary_dir / "embedding_geometry.json").exists():
            emb_geo = json.loads((summary_dir / "embedding_geometry.json").read_text(encoding="utf-8"))
        if (summary_dir / "head_ablation.csv").exists():
            ablation_df = pd.read_csv(summary_dir / "head_ablation.csv")
        if (summary_dir / "linear_probing.csv").exists():
            probing_df = pd.read_csv(summary_dir / "linear_probing.csv")

    all_metrics_rows: list[dict[str, Any]] = []
    if 2 in active:
        m2_out = run_module2_crypto_auditor(
            config=config, summary_dir=summary_dir, device=device, reuse_checkpoints=reuse_checkpoints
        )
        all_metrics_rows.extend(m2_out["rows"])

    if 3 in active:
        m3_out = run_module3_bbp_oracle(
            config=config, device=device, reuse_checkpoints=reuse_checkpoints
        )
        all_metrics_rows.extend(m3_out["rows"])

    if all_metrics_rows:
        pd.DataFrame(all_metrics_rows).to_csv(summary_dir / "metrics.csv", index=False)

    stat_audit = {
        "module2_crypto_backdoor": m2_out.get("mcnemar", {}),
        "module2_classical_vs_clean": {
            "csprng_backdoor": m2_out.get("audit_plot_data", {}).get("classical_stats", {}),
            "csprng_clean": m2_out.get("audit_plot_data", {}).get("clean_classical_stats", {}),
        },
        "module3_bbp_oracle": m3_out.get("mcnemar", {}),
    }
    (summary_dir / "stat_audit.json").write_text(json.dumps(stat_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_data = {
        "embedding_geometry": emb_geo,
        "head_ablation_summary": ablation_df.to_dict(orient="records") if not ablation_df.empty else [],
        "linear_probing": probing_df.to_dict(orient="records") if not probing_df.empty else [],
        "crypto_audit": m2_out.get("audit_plot_data", {}),
        "crypto_attn_profile": m2_out.get("attn_profile", {}),
        "metrics": all_metrics_rows,
        "stat_audit": stat_audit,
    }
    (summary_dir / "plot_data.json").write_text(json.dumps(plot_data, ensure_ascii=False, indent=2), encoding="utf-8")

    saved_plots: list[Path] = []
    if emb_geo:
        saved_plots.extend(plot_exp5_embedding_clock_pca(emb_geo, summary_dir))
    if not ablation_df.empty:
        saved_plots.extend(plot_exp5_head_ablation_heatmap(ablation_df, summary_dir))
    if not probing_df.empty:
        saved_plots.extend(plot_exp5_linear_probing_bars(probing_df, summary_dir))
    if m2_out.get("audit_plot_data") and m2_out.get("attn_profile"):
        saved_plots.extend(
            plot_exp5_crypto_backdoor_xray(m2_out["audit_plot_data"], m2_out["attn_profile"], summary_dir)
        )

    elapsed = time.perf_counter() - t_start
    print(f"\nЭксперимент 5 полностью завершён за {elapsed:.1f}с! Папка: {summary_dir}")
    for p in saved_plots:
        print(f"- {p}")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент 5: Механистическая интерпретируемость, крипто-аудитор и BBP-оракул")
    parser.add_argument("--modules", type=str, default="1,2,3", help="Какие модули запускать: 1,2,3")
    parser.add_argument("--reuse-dir", type=str, default=None, help="Папка runs/exp5_mech_interp_* для дозапуска")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Переоценить уже обученные чекпойнты Модулей 2 и 3 в новую папку без переобучения",
    )
    args = parser.parse_args()

    modules = [int(x.strip()) for x in args.modules.split(",") if x.strip()]
    reuse_p = Path(args.reuse_dir) if args.reuse_dir else None
    run_experiment_5(modules=modules, reuse_dir=reuse_p, reuse_checkpoints=bool(args.reuse_checkpoints))


if __name__ == "__main__":
    main()
