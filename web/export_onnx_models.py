"""Экспорт обученных DigitTransformer в компактный ONNX (opset 17) для браузерного инференса."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import build_model
from src.trainer import remap_legacy_state_dict

MODELS_OUT_DIR = ROOT_DIR / "web" / "frontend" / "public" / "models"

EXPORT_SPECS: list[dict[str, Any]] = [
    {
        "key": "true_pi",
        "title": "DigitTransformer (π Base-10)",
        "dataset": "true_pi",
        "radix": 10,
        "window_size": 64,
        "size_tier": "medium",
        "run_prefix": "exp2_constants_final_",
        "ckpt_rel": Path("checkpoints") / "transformer_true_pi.pt",
        "onnx_name": "transformer_true_pi.onnx",
    },
    {
        "key": "csprng_backdoor",
        "title": "DigitTransformer (Backdoor V2: t-24, t-7)",
        "dataset": "csprng_backdoor",
        "radix": 10,
        "window_size": 64,
        "size_tier": "medium",
        "run_prefix": "exp5_mech_interp_",
        "ckpt_rel": Path("checkpoints") / "transformer_csprng_backdoor.pt",
        "onnx_name": "transformer_csprng_backdoor.onnx",
    },
    {
        "key": "champernowne",
        "title": "DigitTransformer (Champernowne C10)",
        "dataset": "champernowne",
        "radix": 10,
        "window_size": 64,
        "size_tier": "medium",
        "run_prefix": "exp2_constants_final_",
        "ckpt_rel": Path("checkpoints") / "transformer_champernowne.pt",
        "onnx_name": "transformer_champernowne.onnx",
    },
    {
        "key": "e_cf",
        "title": "DigitTransformer (e Continued Fraction)",
        "dataset": "e_cf",
        "radix": 11,
        "window_size": 64,
        "size_tier": "medium",
        "run_prefix": "exp2_constants_final_",
        "ckpt_rel": Path("checkpoints") / "transformer_e_cf.pt",
        "onnx_name": "transformer_e_cf.onnx",
    },
]


def _find_latest_run(runs_dir: Path, prefix: str) -> Path:
    cands = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"Не найдена папка эксперимента с префиксом '{prefix}' в {runs_dir}")
    return cands[0]


class OnnxDigitTransformerWrapper(nn.Module):
    """ONNX-совместимая обёртка: возвращает probs (B, V) и attn_last (B, n_layers, n_heads, L)."""

    def __init__(self, base_model: nn.Module, seq_len: int = 64) -> None:
        super().__init__()
        self.model = base_model
        self.seq_len = int(seq_len)
        mask = torch.triu(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.register_buffer("positions", torch.arange(self.seq_len, dtype=torch.long), persistent=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Явный расчёт весов без autocast и SDPA гарантирует совместимость с opset 17 и WebAssembly SIMD
        h = self.model.token_emb(tokens) + self.model.pos_emb(self.positions).unsqueeze(0)
        attn_layers: list[torch.Tensor] = []
        for block in self.model.blocks:
            h, attn_w = block(h, causal_mask=self.causal_mask, return_attn=True, ablate_heads=None)
            attn_layers.append(attn_w)

        last_hidden = self.model.ln_f(h[:, -1, :])
        logits = F.linear(last_hidden, self.model.head.weight)
        probs = F.softmax(logits, dim=-1)
        stacked = torch.stack(attn_layers, dim=1)  # (B, n_layers, n_heads, L, L)
        attn_last = stacked[:, :, :, -1, :]       # (B, n_layers, n_heads, L)
        return probs, attn_last


def _load_transformer_cpu(spec: dict[str, Any], runs_dir: Path) -> tuple[nn.Module, Path]:
    run_dir = _find_latest_run(runs_dir, spec["run_prefix"])
    ckpt_path = run_dir / spec["ckpt_rel"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпойнт не найден: {ckpt_path}")

    model = build_model(
        arch="transformer",
        radix=int(spec["radix"]),
        window_size=int(spec["window_size"]),
        size_tier=str(spec["size_tier"]),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    remapped, _ = remap_legacy_state_dict(sd)
    model.load_state_dict(remapped)
    model = model.float().cpu().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt_path


def _export_one(spec: dict[str, Any], runs_dir: Path, out_dir: Path) -> dict[str, Any]:
    model, ckpt_path = _load_transformer_cpu(spec, runs_dir)
    seq_len = int(spec["window_size"])
    radix = int(spec["radix"])
    wrapper = OnnxDigitTransformerWrapper(model, seq_len=seq_len).eval()

    dummy_tokens = torch.randint(0, radix, (1, seq_len), dtype=torch.long)
    onnx_path = out_dir / spec["onnx_name"]

    # Legacy JIT tracer (dynamo=False) формирует компактный один-файл ONNX граф под ort-wasm
    torch.onnx.export(
        wrapper,
        (dummy_tokens,),
        str(onnx_path),
        input_names=["tokens"],
        output_names=["probs", "attn_last"],
        dynamic_axes={
            "tokens": {0: "batch_size"},
            "probs": {0: "batch_size"},
            "attn_last": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Перепаковываем внешние данные, если экспортёр вдруг создал sidecar .data
    data_sidecar = Path(f"{onnx_path}.data")
    if data_sidecar.exists():
        proto = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(proto, str(onnx_path), save_as_external_data=False)
        data_sidecar.unlink(missing_ok=True)

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Проверка численной эквивалентности с оригинальным DigitTransformer на батче (4, 64)
    rng = np.random.default_rng(12345 + radix)
    test_np = rng.integers(0, radix, size=(4, seq_len), dtype=np.int64)
    test_t = torch.from_numpy(test_np)
    with torch.no_grad():
        ref_logits, ref_attn = model(test_t, return_attn=True)
        ref_probs = F.softmax(ref_logits.float(), dim=-1).numpy()
        ref_attn_last = ref_attn[:, :, :, -1, :].numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_probs, ort_attn_last = session.run(["probs", "attn_last"], {"tokens": test_np})

    diff_probs = float(np.max(np.abs(ref_probs - ort_probs)))
    diff_attn = float(np.max(np.abs(ref_attn_last - ort_attn_last)))
    if diff_probs >= 1e-4 or diff_attn >= 1e-4:
        raise AssertionError(
            f"Расхождение ONNX vs PyTorch для {spec['key']}: probs={diff_probs:.2e}, attn={diff_attn:.2e}"
        )

    size_bytes = onnx_path.stat().st_size
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[OK] {spec['key']}: {onnx_path.name} ({size_bytes / 1024:.1f} KB, "
        f"params={n_params:,}, max_diff_probs={diff_probs:.2e}, max_diff_attn={diff_attn:.2e})"
    )
    return {
        "key": spec["key"],
        "title": spec["title"],
        "dataset": spec["dataset"],
        "file": spec["onnx_name"],
        "radix": radix,
        "window_size": seq_len,
        "n_layers": int(model.n_layers),
        "n_heads": int(model.n_heads),
        "d_model": int(model.d_model),
        "n_params": int(n_params),
        "size_bytes": int(size_bytes),
        "max_abs_diff_probs": diff_probs,
        "max_abs_diff_attn": diff_attn,
        "source_checkpoint": str(ckpt_path.relative_to(ROOT_DIR)),
    }


def main() -> None:
    runs_dir = ROOT_DIR / "runs"
    MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = [_export_one(spec, runs_dir, MODELS_OUT_DIR) for spec in EXPORT_SPECS]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "opset": 17,
        "models": {entry["key"]: entry for entry in entries},
    }
    manifest_path = MODELS_OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Манифест сохранён: {manifest_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
