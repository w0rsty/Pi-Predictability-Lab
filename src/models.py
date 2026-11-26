# Нейросетевые архитектуры (от простого MLP до GPT-подобного мини-трансформера)
# Все размеры подогнаны к ~50K / ~250K / ~1.5M параметров и кратны 8 под Tensor Cores на RTX 3060 Ti.

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Целевые бюджеты обучаемых параметров для честного сравнения и стресс-теста
CAPACITY_BUDGETS: dict[str, int] = {
    "light": 50_000,
    "medium": 250_000,
    "heavy": 1_500_000,
}


def count_parameters(model: nn.Module) -> int:
    """Считает количество обучаемых параметров нейросети."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _solve_mlp_hidden_dim(
    window_size: int,
    radix: int,
    emb_dim: int,
    target_params: int,
) -> int:
    """Подбирает размер скрытого слоя через квадратное уравнение под целевой бюджет параметров."""
    in_dim = window_size * emb_dim
    emb_params = radix * emb_dim
    # Оценка числа параметров: emb + in_dim*H + 3.5*H + 0.5*H^2 + 0.5*H*radix + radix
    a = 0.5
    b = in_dim + 3.5 + 0.5 * radix
    c = max(target_params - emb_params - radix, 1_000)
    discriminant = b * b + 4.0 * a * c
    h = int((-b + math.sqrt(discriminant)) / (2.0 * a))
    # Округляем до кратности 8 для эффективной работы Tensor Cores на RTX 3060 Ti
    h = max(32, ((h + 4) // 8) * 8)
    return h


class DigitMLP(nn.Module):
    """Многослойный перцептрон: разворот окна в один вектор через Embedding + LayerNorm + GELU."""

    def __init__(
        self,
        radix: int,
        window_size: int,
        size_tier: str = "medium",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if size_tier not in CAPACITY_BUDGETS:
            raise ValueError(f"Неизвестный size_tier='{size_tier}'. Доступны: {list(CAPACITY_BUDGETS)}")

        emb_dims = {"light": 8, "medium": 16, "heavy": 32}
        self.radix = int(radix)
        self.window_size = int(window_size)
        self.size_tier = size_tier
        self.emb_dim = emb_dims[size_tier]

        target_params = CAPACITY_BUDGETS[size_tier]
        hidden1 = _solve_mlp_hidden_dim(self.window_size, self.radix, self.emb_dim, target_params)
        hidden2 = max(32, ((hidden1 // 2 + 4) // 8) * 8)

        # Таблица эмбеддингов переводит каждую цифру 0..radix-1 в вектор из emb_dim чисел
        self.embedding = nn.Embedding(self.radix, self.emb_dim)
        flat_dim = self.window_size * self.emb_dim

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, self.radix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> emb: (B, L, d_emb) -> flat: (B, L * d_emb)
        emb = self.embedding(x)
        flat = emb.reshape(emb.size(0), -1)
        return self.net(flat)


class DilatedConvBlock(nn.Module):
    """Остаточный 1D-свёрточный блок: Conv1d(k=3, padding=dilation) + BatchNorm1d + GELU + skip-связь."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm(self.conv(x))))
        return x + h


def _solve_cnn_dims(radix: int, num_blocks: int, target_params: int) -> tuple[int, int]:
    """Подбирает число каналов C и ширину головы под бюджет параметров (~50K / ~250K / ~1.5M)."""
    # ~60% бюджета отдаём свёрточным слоям, ~40% — линейной голове на последнем шаге
    conv_budget = int(target_params * 0.6)
    a = 3.0 * num_blocks
    b = 3.0 * num_blocks + radix
    channels = int((-b + math.sqrt(b * b + 4.0 * a * conv_budget)) / (2.0 * a))
    channels = max(32, ((channels + 4) // 8) * 8)

    conv_params = radix * channels + num_blocks * (3 * channels * channels + 3 * channels) + 2 * channels
    head_budget = max(target_params - conv_params - radix, 1_000)
    head_hidden = max(64, ((int(head_budget / (channels + radix + 1)) + 4) // 8) * 8)
    return channels, head_hidden


class DigitCNN(nn.Module):
    """Инвариантная к сдвигу дилатированная 1D-CNN без позиционных эмбеддингов."""

    def __init__(
        self,
        radix: int,
        window_size: int,
        size_tier: str = "medium",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if size_tier not in CAPACITY_BUDGETS:
            raise ValueError(f"Неизвестный size_tier='{size_tier}'. Доступны: {list(CAPACITY_BUDGETS)}")

        self.radix = int(radix)
        self.window_size = int(window_size)
        self.size_tier = size_tier

        dilations = (1, 2, 8, 16)
        channels, head_hidden = _solve_cnn_dims(self.radix, len(dilations), CAPACITY_BUDGETS[size_tier])
        self.channels = channels
        self.head_hidden = head_hidden

        self.embedding = nn.Embedding(self.radix, channels)
        self.blocks = nn.ModuleList(
            [DilatedConvBlock(channels=channels, dilation=d, dropout=dropout) for d in dilations]
        )
        self.head_norm = nn.LayerNorm(channels)
        self.head = nn.Sequential(
            nn.Linear(channels, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, self.radix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Нет позиционных эмбеддингов -> сдвиговая инвариантность: на локальном Tier 2 помогает, а на Tier 1 / champernowne мешает
        # x: (B, L) -> h: (B, C, L)
        h = self.embedding(x).transpose(1, 2).contiguous()
        for block in self.blocks:
            h = block(h)
        last_step = self.head_norm(h[:, :, -1])  # (B, C)
        return self.head(last_step)


def _solve_gru_hidden_dim(
    radix: int,
    emb_dim: int,
    num_layers: int,
    target_params: int,
) -> int:
    """Подбирает hidden_size H для многослойного GRU под целевой бюджет параметров."""
    # Считаем параметры: Layer 0 = 3*H*(emb_dim + H + 2), остальные (L-1) слоёв = 3*H*(2*H + 2)
    a = 3.0 * (2 * num_layers - 1)
    b = 3.0 * emb_dim + 6.0 * num_layers + 2.0 + radix
    c = max(target_params - radix * (emb_dim + 1), 1_000)
    discriminant = b * b + 4.0 * a * c
    h = int((-b + math.sqrt(discriminant)) / (2.0 * a))
    # Округляем строго до кратности 8, чтобы cuDNN использовал быстрые ядра Tensor Cores на RTX 3060 Ti
    return max(24, ((h + 4) // 8) * 8)


class DigitGRU(nn.Module):
    """Многослойная однонаправленная GRU с нативным cuDNN-ядром и лёгкой выходной головой."""

    def __init__(
        self,
        radix: int,
        window_size: int,
        size_tier: str = "medium",
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if size_tier not in CAPACITY_BUDGETS:
            raise ValueError(f"Неизвестный size_tier='{size_tier}'. Доступны: {list(CAPACITY_BUDGETS)}")

        self.radix = int(radix)
        self.window_size = int(window_size)
        self.size_tier = size_tier
        self.num_layers = int(num_layers)
        # Компактный эмбеддинг 16 как в MLP, чтобы максимум параметров ушло в рекуррентную память
        self.emb_dim = 16
        self.hidden_size = _solve_gru_hidden_dim(
            radix=self.radix,
            emb_dim=self.emb_dim,
            num_layers=self.num_layers,
            target_params=CAPACITY_BUDGETS[size_tier],
        )

        self.embedding = nn.Embedding(self.radix, self.emb_dim)
        self.gru = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.head_norm = nn.LayerNorm(self.hidden_size)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden_size, self.radix)

    def forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает логиты и скрытые состояния по всем шагам t=0..L-1 для диагностики."""
        if x.is_cuda:
            # Выравниваем веса в памяти GPU для максимальной скорости cuDNN
            self.gru.flatten_parameters()
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        # Приводим выход GRU к FP32 перед LayerNorm и головой, чтобы избежать потерь точности в AMP
        out_f32 = out.float()
        logits_seq = self.head(self.head_drop(self.head_norm(out_f32)))
        return logits_seq, out_f32

    def extract_hidden_states(self, x: torch.Tensor, all_positions: bool = False) -> torch.Tensor:
        """Извлекает скрытые состояния последнего слоя GRU (B, H) или (B, L, H) для зондирования."""
        if x.is_cuda:
            self.gru.flatten_parameters()
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        if all_positions:
            return self.head_norm(out.float())
        return self.head_norm(out[:, -1, :].float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            self.gru.flatten_parameters()
        emb = self.embedding(x)          # (B, L, d_emb)
        out, _ = self.gru(emb)           # (B, L, H), где H кратно 8 под cuDNN Tensor Cores
        last_hidden = out[:, -1, :].float()  # (B, H)
        return self.head(self.head_drop(self.head_norm(last_hidden)))


class CausalTransformerBlock(nn.Module):
    """Pre-LN GPT-блок: LayerNorm -> нативный SDPA -> Residual -> LayerNorm -> FFN(2*d_model)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) должен делиться на n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)

        self.ln1 = nn.LayerNorm(d_model)
        # Объединённая QKV-проекция (в середине проекта переехал с nn.MultiheadAttention на нативный SDPA;
        # старые чекпойнты из Exp 1-3 подхватываются через ремаппинг ключей при загрузке)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        ffn_dim = 2 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        return_attn: bool = False,
        ablate_heads: set[int] | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, _ = x.shape
        normed = self.ln1(x)

        # q, k, v: (B, n_heads, L, head_dim)
        qkv = self.qkv_proj(normed).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights: torch.Tensor | None = None
        if not return_attn:
            # Быстрый SDPA путь (Flash/Memory-Efficient) без аллокации матрицы (B, H, L, L)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Явный расчёт весов внимания (B, n_heads, L, L) для диагностики
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(scores.float(), dim=-1)
            attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_out = attn_probs.to(v.dtype) @ v

        # По-головая нулевая абляция (Causal Head Ablation) без изменения весов модели
        if ablate_heads:
            head_mask = torch.ones((1, self.n_heads, 1, 1), dtype=attn_out.dtype, device=attn_out.device)
            for h_idx in ablate_heads:
                if 0 <= int(h_idx) < self.n_heads:
                    head_mask[0, int(h_idx), 0, 0] = 0.0
            attn_out = attn_out * head_mask

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.drop1(self.out_proj(attn_out))
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


def _solve_transformer_dims(size_tier: str) -> tuple[int, int, int]:
    """Подбирает (d_model, n_heads, n_layers) кратно 8 под бюджеты ~50K / ~250K / ~1.5M."""
    presets = {
        "light": (48, 4, 2),    # ~50.8K параметров
        "medium": (96, 4, 3),   # ~250.1K параметров (head_dim=24, кратно 8)
        "heavy": (192, 6, 5),   # ~1.54M параметров (head_dim=32, кратно 8)
    }
    return presets[size_tier]


class DigitTransformer(nn.Module):
    """Каузальный мини-трансформер (Pre-LN GPT) с Weight Tying и извлечением карт внимания."""

    def __init__(
        self,
        radix: int,
        window_size: int,
        size_tier: str = "medium",
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if size_tier not in CAPACITY_BUDGETS:
            raise ValueError(f"Неизвестный size_tier='{size_tier}'. Доступны: {list(CAPACITY_BUDGETS)}")

        self.radix = int(radix)
        self.window_size = int(window_size)
        self.size_tier = size_tier
        self.max_len = int(max_len)

        d_model, n_heads, n_layers = _solve_transformer_dims(size_tier)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(self.radix, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.radix, bias=False)
        # Weight Tying: входная таблица цифр и выходная проекция делят одни и те же веса
        self.head.weight = self.token_emb.weight

        # Маска нужна только при return_attn=True; при обучении SDPA строит её сам внутри ядра
        bool_mask = torch.triu(torch.ones(self.max_len, self.max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_bool_mask", bool_mask, persistent=False)

    def extract_hidden_states(self, x: torch.Tensor, all_positions: bool = False) -> torch.Tensor:
        """Извлекает скрытые представления после финального блока и ln_f: (B, d_model) или (B, L, d_model)."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Длина окна ({seq_len}) превышает max_len ({self.max_len})")

        positions = torch.arange(seq_len, device=x.device)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions).unsqueeze(0))
        for block in self.blocks:
            h, _ = block(h, causal_mask=None, return_attn=False)

        if all_positions:
            return self.ln_f(h.float())
        return self.ln_f(h[:, -1, :].float())

    def forward_with_head_ablation(
        self,
        x: torch.Tensor,
        ablate_heads: list[tuple[int, int]] | set[tuple[int, int]] | dict[int, list[int]] | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход с обнулением указанных голов внимания (layer_idx, head_idx)."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Длина окна ({seq_len}) превышает max_len ({self.max_len})")

        layer_map: dict[int, list[int]] = {}
        if isinstance(ablate_heads, dict):
            for l_idx, heads in ablate_heads.items():
                layer_map[int(l_idx)] = [int(h) for h in heads]
        elif ablate_heads is not None:
            for l_idx, h_idx in ablate_heads:
                layer_map.setdefault(int(l_idx), []).append(int(h_idx))

        positions = torch.arange(seq_len, device=x.device)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions).unsqueeze(0))
        mask = self.causal_bool_mask[:seq_len, :seq_len] if return_attn else None

        attn_layers: list[torch.Tensor] = []
        for l_idx, block in enumerate(self.blocks):
            h, attn_w = block(
                h,
                causal_mask=mask,
                return_attn=return_attn,
                ablate_heads=layer_map.get(l_idx),
            )
            if return_attn and attn_w is not None:
                attn_layers.append(attn_w)

        last_hidden = self.ln_f(h[:, -1, :].float())
        logits = F.linear(last_hidden, self.head.weight.float())

        if return_attn:
            stacked_attn = torch.stack(attn_layers, dim=1)
            return logits, stacked_attn
        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_with_head_ablation(x, ablate_heads=None, return_attn=return_attn)


def build_model(
    arch: str,
    radix: int,
    window_size: int,
    size_tier: str = "medium",
    **kwargs: Any,
) -> nn.Module:
    """Фабрика моделей: создаёт нейросеть по имени архитектуры и бюджету параметров."""
    arch_norm = arch.lower().strip()
    if arch_norm == "mlp":
        return DigitMLP(
            radix=radix,
            window_size=window_size,
            size_tier=size_tier,
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    if arch_norm == "cnn":
        return DigitCNN(
            radix=radix,
            window_size=window_size,
            size_tier=size_tier,
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    if arch_norm == "gru":
        return DigitGRU(
            radix=radix,
            window_size=window_size,
            size_tier=size_tier,
            num_layers=int(kwargs.get("num_layers", 2)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    if arch_norm == "transformer":
        return DigitTransformer(
            radix=radix,
            window_size=window_size,
            size_tier=size_tier,
            max_len=int(kwargs.get("max_len", 256)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    raise ValueError(f"Архитектура '{arch}' ещё не реализована на текущем шаге")
