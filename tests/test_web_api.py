# Автотесты локального FastAPI BYO-GPU моста и безопасного AST-парсера формул
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.data_generator import generate_rational_digits
from web.server import app
from web.services.safe_math import SafeMathError, evaluate_formula_digits


def test_safe_math_valid_expressions():
    pi_digits = evaluate_formula_digits("pi", length=120, radix=10)
    assert "".join(map(str, pi_digits[:10].tolist())) == "1415926535"

    sqrt5_digits = evaluate_formula_digits("sqrt(5)", length=120, radix=10)
    # sqrt(5) = 2.2360679774...
    assert "".join(map(str, sqrt5_digits[:8].tolist())) == "23606797"

    phi_hex = evaluate_formula_digits("phi", length=120, radix=16)
    assert len(phi_hex) == 120 and int(phi_hex.max()) < 16

    pi_bin = evaluate_formula_digits("pi", length=120, radix=2)
    assert len(pi_bin) == 120 and set(pi_bin.tolist()).issubset({0, 1})


@pytest.mark.parametrize(
    "bad_expr",
    [
        "__import__('os').system('ls')",
        "9**9**9",
        "open('/etc/passwd')",
        "(1).__class__.__mro__",
        "lambda x: x",
    ],
)
def test_safe_math_rejects_injections_and_dos(bad_expr: str):
    with pytest.raises(SafeMathError):
        evaluate_formula_digits(bad_expr, length=120, radix=10)


def test_safe_math_rejects_bad_limits():
    with pytest.raises(SafeMathError):
        evaluate_formula_digits("pi", length=1_000_000, radix=10)
    with pytest.raises(SafeMathError):
        evaluate_formula_digits("pi", length=120, radix=300)


def test_api_health_and_cors_pna():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert "device" in body
        assert isinstance(body["checkpoints_loaded"], int)

        opt = client.options(
            "/api/sandbox/analyze",
            headers={
                "Origin": "https://w0rsty.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert opt.status_code in (200, 204)
        assert opt.headers.get("access-control-allow-origin") == "https://w0rsty.github.io"
        assert opt.headers.get("access-control-allow-private-network") == "true"

        opt_tuna = client.options(
            "/api/health",
            headers={
                "Origin": "https://w0rsty.github.io",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-lab-token, tuna-skip-browser-warning",
            },
        )
        assert opt_tuna.status_code == 200
        assert opt_tuna.headers.get("access-control-allow-origin") == "https://w0rsty.github.io"
        allowed_hdrs = (opt_tuna.headers.get("access-control-allow-headers") or "").lower()
        assert "x-lab-token" in allowed_hdrs
        assert "tuna-skip-browser-warning" in allowed_hdrs


def test_api_generate_validation_and_analyze_h1_vs_h0():
    with TestClient(app) as client:
        # Плохие параметры и инъекции должны возвращать 400/422
        r_bad_expr = client.post(
            "/api/sandbox/generate",
            json={"mode": "formula", "expression": "__import__('os').system('ls')", "length": 200, "radix": 10},
        )
        assert r_bad_expr.status_code == 400

        r_bad_pow = client.post(
            "/api/sandbox/generate",
            json={"mode": "formula", "expression": "9**9**9", "length": 200, "radix": 10},
        )
        assert r_bad_pow.status_code == 400

        r_bad_limits = client.post(
            "/api/sandbox/generate",
            json={"mode": "formula", "expression": "pi", "length": 1_000_000, "radix": 300},
        )
        assert r_bad_limits.status_code == 422

        # Корректный 1/97 через generate + analyze
        r_gen = client.post(
            "/api/sandbox/generate",
            json={"mode": "rational", "numerator": 1, "denominator": 97, "length": 3000, "radix": 10},
        )
        assert r_gen.status_code == 200
        assert len(r_gen.json()["digits"]) == 3000

        rat_digits = generate_rational_digits(1, 97, length=3000, radix=10).tolist()
        r_h1 = client.post(
            "/api/sandbox/analyze",
            json={"name": "test_1_97", "digits": rat_digits, "radix": 10, "run_ppm": True, "run_models": False},
        )
        assert r_h1.status_code == 200
        j_h1 = r_h1.json()
        assert j_h1["ppm"] is not None
        assert j_h1["ppm"]["bits_per_symbol"] < 0.2
        assert j_h1["stats"]["acf_conf_bound_bonferroni"] > j_h1["stats"]["acf_conf_bound_raw"]

        noise_digits = np.random.default_rng(42).integers(0, 10, size=6000, dtype=np.uint8).tolist()
        r_h0 = client.post(
            "/api/sandbox/analyze",
            json={"name": "test_noise", "digits": noise_digits, "radix": 10, "run_ppm": True, "run_models": False},
        )
        assert r_h0.status_code == 200
        j_h0 = r_h0.json()
        assert j_h0["ppm"] is not None
        assert 2.9 < j_h0["ppm"]["bits_per_symbol"] < 4.1


def test_arena_static_bank_and_live_endpoint():
    import json
    from pathlib import Path

    bank_path = Path(__file__).resolve().parents[1] / "web" / "frontend" / "public" / "data" / "arena_rounds.json"
    assert bank_path.exists(), "arena_rounds.json должен быть сгенерирован"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    for ds in ("true_pi", "csprng_backdoor", "champernowne", "e_cf", "rational_1_97"):
        assert ds in bank["datasets"]
        rounds = bank["datasets"][ds]["rounds"]
        assert len(rounds) == 30
        radix = bank["datasets"][ds]["radix"]
        for r in rounds[:5]:
            assert len(r["context"]) == 64
            assert len(r["transformer"]["probs"]) == radix
            assert abs(sum(r["transformer"]["probs"]) - 1.0) < 0.01
            assert abs(sum(r["ppm"]["probs"]) - 1.0) < 0.01

    with TestClient(app) as client:
        r_bad_ds = client.post("/api/arena/session", json={"dataset": "invalid_ds", "n_rounds": 10})
        assert r_bad_ds.status_code == 422

        r_bad_n = client.post("/api/arena/session", json={"dataset": "true_pi", "n_rounds": 100})
        assert r_bad_n.status_code == 422

        r_ok = client.post(
            "/api/arena/session",
            json={"dataset": "csprng_backdoor", "n_rounds": 8, "seed": 123},
        )
        assert r_ok.status_code in (200, 503)
        if r_ok.status_code == 200:
            payload = r_ok.json()
            assert payload["source"] == "live_gpu"
            assert payload["radix"] == 10
            assert len(payload["rounds"]) == 8
            for r in payload["rounds"]:
                assert abs(sum(r["transformer"]["probs"]) - 1.0) < 0.01
                assert abs(sum(r["ppm"]["probs"]) - 1.0) < 0.01

        r_blind = client.post("/api/arena/session", json={"dataset": "blind_audit", "n_rounds": 10, "seed": 7})
        assert r_blind.status_code in (200, 503)
        if r_blind.status_code == 200:
            assert len(r_blind.json()["rounds"]) == 10


def test_onnx_models_manifest_and_inference():
    import json
    from pathlib import Path
    ort = pytest.importorskip("onnxruntime")

    models_dir = Path(__file__).resolve().parents[1] / "web" / "frontend" / "public" / "models"
    manifest_path = models_dir / "manifest.json"
    assert manifest_path.exists(), "ONNX manifest.json не найден"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in ("true_pi", "csprng_backdoor", "champernowne", "e_cf"):
        assert key in manifest["models"]
        entry = manifest["models"][key]
        onnx_file = models_dir / entry["file"]
        assert onnx_file.exists(), f"Отсутствует файл {onnx_file}"
        assert entry["max_abs_diff_probs"] < 1e-4
        assert entry["max_abs_diff_attn"] < 1e-4

        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        tokens = np.zeros((2, 64), dtype=np.int64)
        probs, attn_last = sess.run(["probs", "attn_last"], {"tokens": tokens})
        assert probs.shape == (2, entry["radix"])
        assert attn_last.shape == (2, entry["n_layers"], entry["n_heads"], 64)
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)


def test_train_backdoor_endpoint():
    with TestClient(app) as client:
        # Одинаковые лаги -> 400
        r_bad = client.post(
            "/api/sandbox/train-backdoor",
            json={"lag_a": 12, "lag_b": 12, "prev_threshold": 4, "coeff_a": 4, "coeff_b": 1, "coeff_const": 5},
        )
        assert r_bad.status_code == 400

        # Быстрый smoke-тест обучения
        r_ok = client.post(
            "/api/sandbox/train-backdoor",
            json={
                "lag_a": 24,
                "lag_b": 7,
                "prev_threshold": 4,
                "coeff_a": 4,
                "coeff_b": 1,
                "coeff_const": 5,
                "length": 2500,
                "epochs": 2,
                "seed": 42,
            },
        )
        assert r_ok.status_code == 200
        j = r_ok.json()
        assert len(j["attention_profile"]) == 64
        assert abs(sum(j["attention_profile"]) - 1.0) < 1e-3
        assert len(j["history"]) == 2
        assert j["bijectivity"]["is_bijective"] is True
        assert 0.0 <= j["overall_acc"] <= 1.0


def test_open_mode_without_lab_api_token(monkeypatch):
    monkeypatch.delenv("LAB_API_TOKEN", raising=False)
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert h["auth_required"] is False
        assert h["auth_valid"] is True

        r = client.post(
            "/api/sandbox/generate",
            json={"mode": "rational", "numerator": 1, "denominator": 7, "length": 150, "radix": 10},
        )
        assert r.status_code == 200


def test_protected_mode_rejects_missing_or_wrong_token(monkeypatch):
    monkeypatch.setenv("LAB_API_TOKEN", "itmo-secret-2026")
    with TestClient(app) as client:
        h_no = client.get("/api/health").json()
        assert h_no["auth_required"] is True
        assert h_no["auth_valid"] is False

        h_bad = client.get("/api/health", headers={"X-Lab-Token": "wrong"}).json()
        assert h_bad["auth_required"] is True
        assert h_bad["auth_valid"] is False

        for endpoint, body in [
            ("/api/sandbox/generate", {"mode": "rational", "numerator": 1, "denominator": 7, "length": 150, "radix": 10}),
            ("/api/sandbox/analyze", {"name": "t", "digits": [1] * 120, "radix": 10, "run_ppm": False, "run_models": False}),
            ("/api/sandbox/train-backdoor", {"lag_a": 24, "lag_b": 7}),
            ("/api/mech-interp/ablate", {"dataset": "e_cf", "ablate_heads": []}),
            ("/api/arena/session", {"dataset": "true_pi", "n_rounds": 6}),
        ]:
            assert client.post(endpoint, json=body).status_code == 401
            assert client.post(endpoint, json=body, headers={"X-Lab-Token": "wrong"}).status_code == 401


def test_protected_mode_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("LAB_API_TOKEN", "itmo-secret-2026")
    with TestClient(app) as client:
        h_ok = client.get("/api/health", headers={"X-Lab-Token": "itmo-secret-2026"}).json()
        assert h_ok["auth_required"] is True
        assert h_ok["auth_valid"] is True

        r_ok = client.post(
            "/api/sandbox/generate",
            headers={"X-Lab-Token": "itmo-secret-2026"},
            json={"mode": "rational", "numerator": 1, "denominator": 7, "length": 150, "radix": 10},
        )
        assert r_ok.status_code == 200
        assert len(r_ok.json()["digits"]) == 150
