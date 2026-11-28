# Тесты экспорта ONNX-моделей, целостности dream_seeds.json и эндпойнта POST /api/sandbox/dream
import json
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "web" / "frontend" / "public" / "models"
DREAM_SEEDS_PATH = ROOT_DIR / "web" / "frontend" / "public" / "data" / "dream_seeds.json"


def test_dream_seeds_integrity_and_no_nan():
    assert DREAM_SEEDS_PATH.exists(), "dream_seeds.json должен быть экспортирован"
    bank = json.loads(DREAM_SEEDS_PATH.read_text(encoding="utf-8"))
    seeds = bank["seeds"]
    assert len(seeds) == 12

    by_ds: dict[str, int] = {}
    for s in seeds:
        ds = s["dataset"]
        vocab = int(s["vocab_size"])
        by_ds[ds] = by_ds.get(ds, 0) + 1
        assert len(s["seed_tokens"]) == 64
        assert len(s["ground_truth"]) == 500
        assert all(0 <= x < vocab for x in s["seed_tokens"])
        assert all(0 <= x < vocab for x in s["ground_truth"])
        if ds != "true_pi" or s["id"] != "true_pi_start":
            # Два контрольных среза true_pi и все срезы H1 строго из тестового сплита >= 900_000
            assert s["global_offset"] >= 900_000

    for ds in ("true_pi", "champernowne", "e_cf", "csprng_backdoor"):
        assert by_ds[ds] == 3


def test_onnx_autoregressive_step_equivalence():
    ort = pytest.importorskip("onnxruntime")
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["models"].keys()) == {"true_pi", "csprng_backdoor", "champernowne", "e_cf"}

    for _, meta in manifest["models"].items():
        sess = ort.InferenceSession(str(MODELS_DIR / meta["file"]), providers=["CPUExecutionProvider"])
        ctx = np.arange(64, dtype=np.int64) % int(meta["radix"])
        probs, attn_last = sess.run(["probs", "attn_last"], {"tokens": ctx[None, :]})
        assert probs.shape == (1, meta["radix"])
        assert attn_last.shape == (1, meta["n_layers"], meta["n_heads"], 64)
        assert np.isfinite(probs).all()
        assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_sandbox_dream_endpoint_and_validation(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from web.server import app

    monkeypatch.delenv("LAB_API_TOKEN", raising=False)
    bank = json.loads(DREAM_SEEDS_PATH.read_text(encoding="utf-8"))
    pi_seed = next(s for s in bank["seeds"] if s["id"] == "true_pi_start")
    champ_seed = next(s for s in bank["seeds"] if s["dataset"] == "champernowne")

    with TestClient(app) as client:
        # Ошибка валидации: неверная длина затравки
        r_bad_len = client.post(
            "/api/sandbox/dream",
            json={"dataset": "true_pi", "seed_tokens": [1, 2, 3], "steps": 20},
        )
        assert r_bad_len.status_code == 422

        # Ошибка диапазона токенов (токен 10 в base-10 true_pi)
        r_bad_tok = client.post(
            "/api/sandbox/dream",
            json={"dataset": "true_pi", "seed_tokens": [10] * 64, "steps": 20},
        )
        assert r_bad_tok.status_code in (200, 400)
        if len(client.app.state.registry.models) > 0:
            assert r_bad_tok.status_code == 400

        # Проверка защиты токеном для /api/sandbox/dream (не требует runs/)
        monkeypatch.setenv("LAB_API_TOKEN", "dream-secret")
        r_unauth = client.post(
            "/api/sandbox/dream",
            json={"dataset": "true_pi", "seed_tokens": pi_seed["seed_tokens"], "steps": 20},
        )
        assert r_unauth.status_code == 401
        monkeypatch.delenv("LAB_API_TOKEN", raising=False)

        # Если локальных чекпойнтов runs/ нет (среда CI), пропускаем инференс PyTorch
        if len(client.app.state.registry.models) == 0:
            pytest.skip("Local runs/ checkpoints not present in CI")

        r_pi = client.post(
            "/api/sandbox/dream",
            json={
                "dataset": "true_pi",
                "seed_tokens": pi_seed["seed_tokens"],
                "steps": 40,
                "temperature": 0.0,
                "top_k": 0,
            },
        )
        assert r_pi.status_code == 200
        j = r_pi.json()
        assert len(j["generated_tokens"]) == 40
        assert len(j["step_raw_entropies"]) == 40
        assert len(j["last_probs"]) == 10

        r_ch = client.post(
            "/api/sandbox/dream",
            json={
                "dataset": "champernowne",
                "seed_tokens": champ_seed["seed_tokens"],
                "steps": 60,
                "temperature": 0.0,
                "top_k": 0,
            },
        )
        assert r_ch.status_code == 200
        j_ch = r_ch.json()
        assert len(j_ch["generated_tokens"]) == 60
        truth = champ_seed["ground_truth"][:60]
        acc = sum(1 for a, b in zip(j_ch["generated_tokens"], truth) if a == b) / 60.0
        assert acc > 0.85
