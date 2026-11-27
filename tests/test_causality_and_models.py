# Тесты каузальности архитектур, абляции голов, ремаппинга весов, обучения и загрузки чекпойнтов
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.dataset import create_dataloaders, split_sequence_chronologically
from src.models import CausalTransformerBlock, DigitCNN, DigitGRU, DigitTransformer, build_model, count_parameters
from src.trainer import _build_adamw_param_groups, remap_legacy_state_dict, train_model


def _load_exp5_module():
    exp5_path = Path(__file__).resolve().parents[1] / "experiments" / "07_exp5_mech_interp_and_crypto.py"
    spec = importlib.util.spec_from_file_location("exp5_module_for_test", exp5_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_causal_transformer_block_enforces_mask_when_none() -> None:
    torch.manual_seed(0)
    block = CausalTransformerBlock(d_model=48, n_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 12, 48, requires_grad=True)
    out, attn = block(x, causal_mask=None, return_attn=True)
    assert attn is not None
    upper = torch.triu(torch.ones((12, 12), dtype=torch.bool), diagonal=1)
    assert torch.all(attn[:, :, upper] == 0.0)

    k = 5
    out[:, k, :].sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad[:, k + 1 :, :] == 0.0)


def test_digit_transformer_and_gru_forward_causality() -> None:
    torch.manual_seed(1)
    x = torch.randint(0, 10, (2, 14), dtype=torch.long)
    x_fut_changed = x.clone()
    k = 7
    x_fut_changed[:, k + 1 :] = (x_fut_changed[:, k + 1 :] + 3) % 10

    tr = DigitTransformer(radix=10, window_size=14, size_tier="light", dropout=0.0).eval()
    # Входной эмбеддинг с retain_grad на forward_with_head_ablation и сверка усечённого префикса с forward()
    emb_box: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        out.retain_grad()
        emb_box.append(out)

    handle = tr.token_emb.register_forward_hook(_hook)
    h_all = tr.extract_hidden_states(x, all_positions=True)
    h_all[:, k, :].sum().backward()
    assert torch.all(emb_box[0].grad[:, k + 1 :, :] == 0.0)
    handle.remove()

    prefix_logits = tr(x[:, : k + 1])
    logits_k_from_full = torch.nn.functional.linear(h_all[:, k, :], tr.head.weight)
    assert torch.allclose(prefix_logits, logits_k_from_full, atol=1e-5)

    logits_abl_a, attn_a = tr.forward_with_head_ablation(x, ablate_heads=[(0, 1)], return_attn=True)
    logits_abl_b, _ = tr.forward_with_head_ablation(x_fut_changed[:, : k + 1], ablate_heads=[(0, 1)], return_attn=True)
    logits_prefix, _ = tr.forward_with_head_ablation(x[:, : k + 1], ablate_heads=[(0, 1)], return_attn=True)
    assert attn_a.shape == (2, tr.n_layers, tr.n_heads, 14, 14)
    assert torch.allclose(logits_abl_b, logits_prefix, atol=1e-6)
    assert logits_abl_a.shape == (2, 10)

    gru = DigitGRU(radix=10, window_size=14, size_tier="light", dropout=0.0).eval()
    emb_gru: list[torch.Tensor] = []
    h_gru = gru.embedding.register_forward_hook(lambda _m, _i, out: (out.retain_grad(), emb_gru.append(out)) and None)
    seq_logits, _ = gru.forward_sequence(x)
    seq_logits[:, k, :].sum().backward()
    assert torch.all(emb_gru[0].grad[:, k + 1 :, :] == 0.0)
    h_gru.remove()

    assert torch.allclose(gru(x[:, : k + 1]), seq_logits[:, k, :], atol=1e-5)


def test_digit_cnn_receptive_field_and_no_future_leak() -> None:
    torch.manual_seed(2)
    win = 64
    cnn = DigitCNN(radix=10, window_size=win, size_tier="light", dropout=0.0).eval()
    x = torch.randint(0, 10, (2, win), dtype=torch.long)

    # Дилатации (1, 2, 8, 16) с kernel=3 дают суммарный левый сдвиг 27 позиций -> индексы < 36 вне RF последнего шага
    emb_box: list[torch.Tensor] = []
    hook = cnn.embedding.register_forward_hook(lambda _m, _i, out: (out.retain_grad(), emb_box.append(out)) and None)
    logits = cnn(x)
    logits.sum().backward()
    hook.remove()

    assert torch.all(emb_box[0].grad[:, : win - 28, :] == 0.0)
    assert torch.any(emb_box[0].grad[:, win - 1 :, :] != 0.0)

    x_perturbed = x.clone()
    x_perturbed[:, : win - 28] = (x_perturbed[:, : win - 28] + 5) % 10
    assert torch.allclose(cnn(x), cnn(x_perturbed), atol=1e-6)


def test_single_head_ablation_index_matches_manual_slice() -> None:
    torch.manual_seed(3)
    tr = DigitTransformer(radix=10, window_size=16, size_tier="medium", dropout=0.0).eval()
    x = torch.randint(0, 10, (3, 16), dtype=torch.long)

    out_h1 = tr.forward_with_head_ablation(x, ablate_heads=[(0, 1)])
    out_h2 = tr.forward_with_head_ablation(x, ablate_heads=[(0, 2)])
    assert not torch.allclose(out_h1, out_h2, atol=1e-5)

    head_dim = tr.d_model // tr.n_heads

    def _zero_head1_slice(_mod, inp):
        inp_mod = inp[0].clone()
        inp_mod[:, :, 1 * head_dim : 2 * head_dim] = 0.0
        return (inp_mod,)

    handle = tr.blocks[0].out_proj.register_forward_pre_hook(_zero_head1_slice)
    out_manual = tr(x)
    handle.remove()
    assert torch.allclose(out_h1, out_manual, atol=1e-6)


def test_real_mha_to_sdpa_weight_remapping() -> None:
    torch.manual_seed(4)
    tr = DigitTransformer(radix=10, window_size=12, size_tier="light", dropout=0.0).eval()
    mha = nn.MultiheadAttention(embed_dim=tr.d_model, num_heads=tr.n_heads, dropout=0.0, batch_first=True).eval()

    legacy_sd = dict(tr.state_dict())
    # Заменяем блок 0 на ключи nn.MultiheadAttention с реальными весами PyTorch
    for suffix in ("qkv_proj.weight", "qkv_proj.bias", "out_proj.weight", "out_proj.bias"):
        legacy_sd.pop(f"blocks.0.{suffix}")
    mha_sd = mha.state_dict()
    legacy_sd["blocks.0.attn.in_proj_weight"] = mha_sd["in_proj_weight"].clone()
    legacy_sd["blocks.0.attn.in_proj_bias"] = mha_sd["in_proj_bias"].clone()
    legacy_sd["blocks.0.attn.out_proj.weight"] = mha_sd["out_proj.weight"].clone()
    legacy_sd["blocks.0.attn.out_proj.bias"] = mha_sd["out_proj.bias"].clone()

    remapped, changed = remap_legacy_state_dict(legacy_sd)
    assert changed is True
    tr.load_state_dict(remapped)

    # Проверяем численное совпадение attention-ветки блока 0 с эталонным nn.MultiheadAttention
    normed = torch.randn(2, 12, tr.d_model)
    mask = torch.triu(torch.ones((12, 12), dtype=torch.bool), diagonal=1)
    ref_out, _ = mha(normed, normed, normed, attn_mask=mask, need_weights=False, is_causal=True)

    qkv = tr.blocks[0].qkv_proj(normed).view(2, 12, 3, tr.n_heads, tr.blocks[0].head_dim)
    q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    sdpa_out = tr.blocks[0].out_proj(sdpa.transpose(1, 2).contiguous().view(2, 12, tr.d_model))
    assert torch.allclose(sdpa_out, ref_out, atol=1e-5)


def test_medium_parameter_counts_exact() -> None:
    expected = {
        "mlp": 247_650,
        "cnn": 249_658,
        "gru": 242_090,
        "transformer": 250_080,
    }
    for arch, count in expected.items():
        m = build_model(arch=arch, radix=10, window_size=64, size_tier="medium")
        assert count_parameters(m) == count


def test_train_model_real_2_epoch_run_param_groups_and_no_restore(tmp_path) -> None:
    torch.manual_seed(42)
    seq = np.tile(np.array([0, 1, 2, 3], dtype=np.uint8), 160)
    split = split_sequence_chronologically(seq, None, 0.8, 0.1, 0.1)
    loaders = create_dataloaders(split, window_size=8, batch_size=64, num_workers=0, pin_memory=False, seed=42)
    model = build_model("mlp", radix=4, window_size=8, size_tier="light", dropout=0.0)

    groups = _build_adamw_param_groups(model, weight_decay=1e-2)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-2 and all(p.ndim >= 2 for p in groups[0]["params"])
    assert groups[1]["weight_decay"] == 0.0 and all(p.ndim < 2 for p in groups[1]["params"])

    cfg = {
        "seed": 42,
        "paths": {"runs_dir": str(tmp_path)},
        "training": {"device": "cpu", "use_amp": False, "learning_rate": 5e-3, "epochs": 2, "early_stopping_patience": 2},
    }
    res = train_model(
        model=model,
        loaders=loaders,
        config=cfg,
        exp_name="unit_test_2ep",
        radix=4,
        epochs=2,
        learning_rate=5e-3,
        enable_early_stopping=False,
        show_progress=False,
    )
    hist = res["history"]
    assert len(hist) == 2
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]

    # С enable_early_stopping=False финальная модель остаётся на последней эпохе без отката к best_model.pt
    res_no_restore = train_model(
        model=build_model("mlp", radix=4, window_size=8, size_tier="light", dropout=0.0),
        loaders=loaders,
        config=cfg,
        exp_name="unit_test_no_restore",
        radix=4,
        epochs=2,
        learning_rate=5e-3,
        enable_early_stopping=False,
        show_progress=False,
    )
    assert res_no_restore["summary"]["early_stopped"] is False


def test_exp5_load_checkpoint_model_raw_state_dict(tmp_path) -> None:
    mod = _load_exp5_module()
    m = build_model("transformer", radix=10, window_size=16, size_tier="light")
    ckpt_path = tmp_path / "raw_sd.pt"
    torch.save(m.state_dict(), ckpt_path)

    loaded = mod._load_checkpoint_model("transformer", ckpt_path, radix=10, window_size=16, size_tier="light", device=torch.device("cpu"))
    x = torch.randint(0, 10, (2, 16), dtype=torch.long)
    m.eval()
    assert torch.allclose(m(x), loaded(x), atol=1e-6)
