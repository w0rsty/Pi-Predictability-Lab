# Lifespan-реестр обученных чекпойнтов из runs/ и единый Semaphore для GPU-запросов
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.models import build_model
from src.trainer import remap_legacy_state_dict, resolve_device

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class LoadedModelEntry:
    key: str
    dataset: str
    arch: str
    radix: int
    window_size: int
    size_tier: str
    checkpoint_path: Path
    model: torch.nn.Module


def _find_latest_dir(runs_root: Path, prefix: str) -> Path | None:
    if not runs_root.exists():
        return None
    cands = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
        reverse=True,
    )
    return cands[0] if cands else None


def _load_ckpt(
    arch: str,
    ckpt_path: Path,
    radix: int,
    window_size: int,
    size_tier: str,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(arch=arch, radix=radix, window_size=window_size, size_tier=size_tier)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    remapped, _ = remap_legacy_state_dict(sd)
    model.load_state_dict(remapped)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class ModelRegistry:
    """Держит read-only модели в памяти (lifespan) и сериализует GPU-запросы через Semaphore(1)."""

    def __init__(self, root_dir: Path = ROOT_DIR) -> None:
        self.root_dir = root_dir
        self.runs_dir = root_dir / "runs"
        self.device = resolve_device("cuda")
        self.gpu_semaphore = asyncio.Semaphore(1)
        self.models: dict[str, LoadedModelEntry] = {}
        self.ppm_cache: dict[str, Any] = {}
        self.gpu_name = (
            torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU Local"
        )
        self.vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if self.device.type == "cuda"
            else None
        )

    def load_all(self) -> None:
        exp2_dir = _find_latest_dir(self.runs_dir, "exp2_constants_final_")
        exp4_dir = _find_latest_dir(self.runs_dir, "exp4_scaling_")
        exp5_dir = _find_latest_dir(self.runs_dir, "exp5_mech_interp_")
        exp5_gru_bd_dir = _find_latest_dir(self.runs_dir, "exp5_gru_csprng_backdoor_")
        exp4_gru_97_dir = _find_latest_dir(self.runs_dir, "exp4_block_b_phasetransition_h1_gru_rational_1_97_L64_medium_")

        candidates: list[tuple[str, str, str, int, int, str, Path | None]] = []
        if exp2_dir is not None:
            ck = exp2_dir / "checkpoints"
            candidates.extend([
                ("transformer_true_pi", "true_pi", "transformer", 10, 64, "medium", ck / "transformer_true_pi.pt"),
                ("transformer_e_cf", "e_cf", "transformer", 11, 64, "medium", ck / "transformer_e_cf.pt"),
                ("transformer_champernowne", "champernowne", "transformer", 10, 64, "medium", ck / "transformer_champernowne.pt"),
                ("transformer_rational_1_999983", "rational_1_999983", "transformer", 10, 64, "medium", ck / "transformer_rational_1_999983.pt"),
                ("gru_true_pi", "true_pi", "gru", 10, 64, "medium", ck / "gru_true_pi.pt"),
                ("gru_e_cf", "e_cf", "gru", 11, 64, "medium", ck / "gru_e_cf.pt"),
                ("gru_champernowne", "champernowne", "gru", 10, 64, "medium", ck / "gru_champernowne.pt"),
                ("gru_rational_1_999983", "rational_1_999983", "gru", 10, 64, "medium", ck / "gru_rational_1_999983.pt"),
            ])
        if exp4_dir is not None:
            ck = exp4_dir / "checkpoints"
            candidates.append(
                ("transformer_rational_1_97", "rational_1_97", "transformer", 10, 128, "medium", ck / "transformer_rational_1_97_L128.pt")
            )
        if exp4_gru_97_dir is not None:
            candidates.append(
                ("gru_rational_1_97", "rational_1_97", "gru", 10, 64, "medium", exp4_gru_97_dir / "best_model.pt")
            )
        if exp5_dir is not None:
            ck = exp5_dir / "checkpoints"
            candidates.append(
                ("transformer_csprng_backdoor", "csprng_backdoor", "transformer", 10, 64, "medium", ck / "transformer_csprng_backdoor.pt")
            )
        if exp5_gru_bd_dir is not None:
            candidates.append(
                ("gru_csprng_backdoor", "csprng_backdoor", "gru", 10, 64, "medium", exp5_gru_bd_dir / "best_model.pt")
            )

        for key, dataset, arch, radix, win, tier, path in candidates:
            if path is None or not path.exists():
                continue
            try:
                model = _load_ckpt(arch, path, radix, win, tier, self.device)
                self.models[key] = LoadedModelEntry(
                    key=key,
                    dataset=dataset,
                    arch=arch,
                    radix=radix,
                    window_size=win,
                    size_tier=tier,
                    checkpoint_path=path,
                    model=model,
                )
            except Exception:
                continue

    def clear(self) -> None:
        self.models.clear()
        self.ppm_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def health_payload(self) -> dict[str, Any]:
        ablation_datasets = [
            ds
            for ds in ("csprng_backdoor", "e_cf", "champernowne", "rational_1_97", "true_pi")
            if f"transformer_{ds}" in self.models
        ]
        return {
            "status": "ok",
            "api_version": "1.0.0",
            "device": str(self.device),
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "checkpoints_loaded": len(self.models),
            "available_models": [
                {
                    "key": e.key,
                    "dataset": e.dataset,
                    "arch": e.arch,
                    "radix": e.radix,
                    "window_size": e.window_size,
                    "size_tier": e.size_tier,
                }
                for e in self.models.values()
            ],
            "ablation_datasets": ablation_datasets,
        }
