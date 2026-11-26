# Тесты хронологического разбиения (zero leakage) и строгой каузальности нейросетей
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import SequenceWindowDataset, split_sequence_chronologically, verify_zero_leakage
from src.models import DigitGRU, DigitTransformer


def test_chronological_split_zero_overlap_and_negative_leak_check():
    digits = np.arange(1000, dtype=np.uint8) % 10
    mask = np.zeros(1000, dtype=np.uint8)
    mask[::10] = 1
    window_size = 32

    split = split_sequence_chronologically(digits, mask=mask, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    report = verify_zero_leakage(split, window_size=window_size)

    assert report["zero_overlap_verified"] is True
    assert split.train_range == (0, 800)
    assert split.val_range == (800, 900)
    assert split.test_range == (900, 1000)

    train_ds = SequenceWindowDataset(split.train_digits, window_size, split.train_mask)
    val_ds = SequenceWindowDataset(split.val_digits, window_size, split.val_mask)
    test_ds = SequenceWindowDataset(split.test_digits, window_size, split.test_mask)

    assert len(train_ds) == 800 - window_size
    assert len(val_ds) == 100 - window_size
    assert len(test_ds) == 100 - window_size

    # Проверяем, что перекрытия сырых индексов при naive-перемешивании окон действительно приводят к утечке
    all_windows = np.lib.stride_tricks.sliding_window_view(np.arange(1000), window_size + 1)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_windows))
    train_idx = set(all_windows[perm[:800], :-1].ravel().tolist())
    test_targets = set(all_windows[perm[900:], -1].tolist())
    leaked_targets = len(train_idx & test_targets)
    assert leaked_targets > 0

    # При хронологическом разбиении пересечение множества индексов строго пустое
    chrono_train_indices = set(range(split.train_range[0], split.train_range[1]))
    chrono_test_indices = set(range(split.test_range[0], split.test_range[1]))
    assert chrono_train_indices.isdisjoint(chrono_test_indices)


def test_transformer_strict_causality_via_autograd():
    torch.manual_seed(42)
    model = DigitTransformer(radix=10, window_size=24, size_tier="light", dropout=0.0)
    model.eval()

    x = torch.randint(0, 10, (2, 24), dtype=torch.long)
    captured = {}

    def capture_emb_grad(_module, _inp, out):
        out.retain_grad()
        captured["emb"] = out

    hook = model.emb_drop.register_forward_hook(capture_emb_grad)
    try:
        h_all = model.extract_hidden_states(x, all_positions=True)  # (2, 24, d_model)
        logits_seq = F.linear(h_all, model.head.weight.float())
        assert logits_seq.shape == (2, 24, 10)

        for query_pos in (0, 5, 11, 18):
            model.zero_grad(set_to_none=True)
            logits_seq[:, query_pos, :].sum().backward(retain_graph=True)

            grad = captured["emb"].grad  # (2, 24, d_model)
            assert grad is not None
            future_grad_max = grad[:, query_pos + 1 :, :].abs().max().item()
            assert future_grad_max == 0.0, f"Утечка из будущего на query_pos={query_pos}"
            assert grad[:, : query_pos + 1, :].abs().max().item() > 0.0
    finally:
        hook.remove()


def test_gru_strict_causality_via_autograd():
    torch.manual_seed(42)
    model = DigitGRU(radix=10, window_size=20, size_tier="light", dropout=0.0)
    model.eval()

    x = torch.randint(0, 10, (2, 20), dtype=torch.long)
    captured = {}

    def capture_emb(_module, _inp, out):
        out.retain_grad()
        captured["emb"] = out

    hook = model.embedding.register_forward_hook(capture_emb)
    try:
        seq_logits, _ = model.forward_sequence(x)  # (2, 20, 10)
        assert seq_logits.shape == (2, 20, 10)

        for query_pos in (0, 7, 14):
            model.zero_grad(set_to_none=True)
            seq_logits[:, query_pos, :].sum().backward(retain_graph=True)
            grad = captured["emb"].grad
            assert grad[:, query_pos + 1 :, :].abs().max().item() == 0.0
            assert grad[:, : query_pos + 1, :].abs().max().item() > 0.0
    finally:
        hook.remove()


def test_transformer_head_ablation_identity_and_effect():
    torch.manual_seed(42)
    model = DigitTransformer(radix=10, window_size=16, size_tier="light", dropout=0.0)
    model.eval()
    x = torch.randint(0, 10, (3, 16), dtype=torch.long)

    with torch.no_grad():
        base_logits = model(x)
        no_ablate_logits = model.forward_with_head_ablation(x, ablate_heads=[])
        assert torch.allclose(base_logits, no_ablate_logits, atol=1e-6)

        n_heads = model.blocks[0].n_heads
        ablated_logits = model.forward_with_head_ablation(x, ablate_heads=[(0, h) for h in range(n_heads)])
        assert not torch.allclose(base_logits, ablated_logits, atol=1e-4)
