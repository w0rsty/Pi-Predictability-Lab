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
