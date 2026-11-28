# FastAPI BYO-GPU Local Bridge (127.0.0.1:8000) + опциональная раздача статической сборки dist/
import asyncio
import inspect
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.services.arena_service import generate_live_arena_session
from web.services.backdoor_trainer import train_custom_backdoor_sync
from web.services.mech_interp_service import run_multi_head_ablation_sync
from web.services.model_registry import ModelRegistry, ROOT_DIR
from web.services.safe_math import SafeMathError
from web.services.sandbox_service import (
    analyze_cpu_part_sync,
    build_verdict,
    eval_loaded_models_sync,
    generate_dream_sync,
    generate_sequence_sync,
)

STATIC_DATA_PATH = ROOT_DIR / "web" / "frontend" / "public" / "data" / "lab_data.json"
DIST_DIR = ROOT_DIR / "web" / "frontend" / "dist"
LAB_API_TOKEN = os.getenv("LAB_API_TOKEN", "").strip()


def _get_lab_token() -> str:
    env_val = os.getenv("LAB_API_TOKEN")
    if env_val is not None:
        return env_val.strip()
    return LAB_API_TOKEN.strip()


def verify_lab_token(x_lab_token: str | None = Header(default=None, alias="X-Lab-Token")) -> None:
    """Проверяет заголовок X-Lab-Token при включённом LAB_API_TOKEN (защита удалённого туннеля)."""
    expected = _get_lab_token()
    if not expected:
        return
    if not x_lab_token or not secrets.compare_digest(x_lab_token.strip(), expected):
        raise HTTPException(
            status_code=401,
            detail="Требуется корректный токен доступа (X-Lab-Token)",
        )


class GenerateRequest(BaseModel):
    mode: Literal["formula", "rational", "backdoor"]
    expression: str | None = Field(default=None, max_length=160)
    numerator: int = Field(default=1, ge=0, le=1_000_000)
    denominator: int = Field(default=97, ge=1, le=500_000)
    length: int = Field(default=2500, ge=100, le=50_000)
    radix: int = Field(default=10, ge=2, le=256)


class AnalyzeRequest(BaseModel):
    name: str = Field(default="custom_sequence", max_length=80)
    digits: list[int] = Field(..., min_length=100, max_length=50_000)
    mask: list[int] | None = Field(default=None, max_length=50_000)
    radix: int = Field(default=10, ge=2, le=256)
    run_ppm: bool = True
    run_models: bool = True


class AblateRequest(BaseModel):
    dataset: Literal["true_pi", "e_cf", "champernowne", "rational_1_97", "csprng_backdoor"]
    ablate_heads: list[tuple[int, int]] = Field(default_factory=list, max_length=12)
    max_windows: int = Field(default=4000, ge=64, le=4000)


class ArenaSessionRequest(BaseModel):
    dataset: Literal["true_pi", "csprng_backdoor", "champernowne", "e_cf", "rational_1_97", "blind_audit"]
    n_rounds: int = Field(default=15, ge=5, le=25)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class TrainBackdoorRequest(BaseModel):
    lag_a: int = Field(default=31, ge=2, le=48)
    lag_b: int = Field(default=11, ge=2, le=48)
    prev_threshold: int = Field(default=4, ge=1, le=9)
    coeff_a: int = Field(default=4, ge=0, le=9)
    coeff_b: int = Field(default=1, ge=0, le=9)
    coeff_const: int = Field(default=5, ge=0, le=9)
    length: int = Field(default=32_000, ge=2000, le=60_000)
    epochs: int | None = Field(default=None, ge=2, le=30)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class DreamRequest(BaseModel):
    dataset: Literal["true_pi", "champernowne", "e_cf", "csprng_backdoor"]
    seed_tokens: list[int] = Field(..., min_length=64, max_length=64)
    steps: int = Field(default=300, ge=10, le=1000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0, le=10)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry(ROOT_DIR)
    await asyncio.to_thread(registry.load_all)
    app.state.registry = registry
    try:
        yield
    finally:
        registry.clear()


app = FastAPI(
    title="Pi-Predictability-Lab BYO-GPU Bridge",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS для локального Vite dev-сервера и публичной витрины GitHub Pages (включая HTTPS-туннели)
_cors_kwargs = {
    "allow_origins": ["https://w0rsty.github.io"],
    "allow_origin_regex": r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": [
        "Content-Type",
        "X-Lab-Token",
        "tuna-skip-browser-warning",
        "ngrok-skip-browser-warning",
        "Accept",
        "Origin",
    ],
}
if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
    _cors_kwargs["allow_private_network"] = True
app.add_middleware(CORSMiddleware, **_cors_kwargs)


@app.middleware("http")
async def add_private_network_access_header(request: Request, call_next):
    response = await call_next(request)
    # Заголовок Chrome Private Network Access для связи https://*.github.io -> http://127.0.0.1:8000
    if request.method == "OPTIONS" or request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.get("/api/health")
async def api_health(
    request: Request,
    x_lab_token: str | None = Header(default=None, alias="X-Lab-Token"),
):
    registry: ModelRegistry = request.app.state.registry
    payload = registry.health_payload()
    expected = _get_lab_token()
    payload["auth_required"] = bool(expected)
    payload["auth_valid"] = (not expected) or bool(
        x_lab_token and secrets.compare_digest(x_lab_token.strip(), expected)
    )
    return payload


@app.get("/api/static-data")
async def api_static_data():
    if not STATIC_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="lab_data.json ещё не собран; запустите web/export_static_data.py")
    data = await asyncio.to_thread(lambda: json.loads(STATIC_DATA_PATH.read_text(encoding="utf-8")))
    return JSONResponse(content=data)


@app.post("/api/sandbox/generate")
async def api_sandbox_generate(req: GenerateRequest, _: None = Depends(verify_lab_token)):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                generate_sequence_sync,
                req.mode,
                req.length,
                req.radix,
                req.expression,
                req.numerator,
                req.denominator,
            ),
            timeout=25.0,
        )
    except (SafeMathError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Превышен таймаут вычисления (25с)") from exc


@app.post("/api/sandbox/analyze")
async def api_sandbox_analyze(req: AnalyzeRequest, request: Request, _: None = Depends(verify_lab_token)):
    t0 = time.perf_counter()
    if req.mask is not None and len(req.mask) != len(req.digits):
        raise HTTPException(status_code=400, detail="Длина mask должна совпадать с длиной digits")
    digits = np.asarray(req.digits, dtype=np.int64)
    if np.any(digits < 0) or np.any(digits >= req.radix):
        raise HTTPException(status_code=400, detail=f"Все цифры должны быть в [0, {req.radix - 1}]")
    digits_u8 = digits.astype(np.uint8)
    mask_u8 = np.asarray(req.mask, dtype=np.uint8) if req.mask is not None else None

    try:
        cpu_out = await asyncio.wait_for(
            asyncio.to_thread(analyze_cpu_part_sync, req.name, digits_u8, req.radix, req.run_ppm),
            timeout=25.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Превышен таймаут статистического анализа (25с)") from exc

    registry: ModelRegistry = request.app.state.registry
    neural_out: list[dict] = []
    if req.run_models:
        async with registry.gpu_semaphore:
            neural_out = await asyncio.wait_for(
                asyncio.to_thread(eval_loaded_models_sync, registry, digits_u8, mask_u8, req.radix),
                timeout=25.0,
            )

    verdict = build_verdict(req.radix, cpu_out["stats"], cpu_out["ppm"], neural_out)
    return {
        "name": req.name,
        "radix": req.radix,
        "length": int(len(digits_u8)),
        "stats": cpu_out["stats"],
        "ppm": cpu_out["ppm"],
        "neural_models": neural_out,
        "verdict": verdict,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


@app.post("/api/mech-interp/ablate")
async def api_mech_interp_ablate(req: AblateRequest, request: Request, _: None = Depends(verify_lab_token)):
    for l_idx, h_idx in req.ablate_heads:
        if not (0 <= l_idx < 3 and 0 <= h_idx < 4):
            raise HTTPException(status_code=400, detail="Индексы голов должны быть в пределах layer=[0..2], head=[0..3]")
    registry: ModelRegistry = request.app.state.registry
    try:
        async with registry.gpu_semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    run_multi_head_ablation_sync,
                    registry,
                    req.dataset,
                    req.ablate_heads,
                    req.max_windows,
                ),
                timeout=25.0,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Превышен таймаут GPU-абляции (25с)") from exc


@app.post("/api/arena/session")
async def api_arena_session(req: ArenaSessionRequest, request: Request, _: None = Depends(verify_lab_token)):
    registry: ModelRegistry = request.app.state.registry
    try:
        async with registry.gpu_semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    generate_live_arena_session,
                    registry,
                    req.dataset,
                    req.n_rounds,
                    req.seed,
                ),
                timeout=30.0,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Превышен таймаут генерации сессии Арены (30с)") from exc


@app.post("/api/sandbox/train-backdoor")
async def api_sandbox_train_backdoor(
    req: TrainBackdoorRequest,
    request: Request,
    _: None = Depends(verify_lab_token),
):
    if req.lag_a == req.lag_b:
        raise HTTPException(status_code=400, detail="Лаги lag_a и lag_b должны различаться")
    registry: ModelRegistry = request.app.state.registry
    try:
        async with registry.gpu_semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    train_custom_backdoor_sync,
                    registry.device,
                    req.lag_a,
                    req.lag_b,
                    req.prev_threshold,
                    req.coeff_a,
                    req.coeff_b,
                    req.coeff_const,
                    req.length,
                    req.epochs,
                    req.seed,
                ),
                timeout=45.0,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Превышен таймаут экспресс-обучения детектора (45с)") from exc


@app.post("/api/sandbox/dream")
async def api_sandbox_dream(
    req: DreamRequest,
    request: Request,
    _: None = Depends(verify_lab_token),
):
    registry: ModelRegistry = request.app.state.registry
    try:
        async with registry.gpu_semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    generate_dream_sync,
                    registry,
                    req.dataset,
                    req.seed_tokens,
                    req.steps,
                    req.temperature,
                    req.top_k,
                    req.random_seed,
                ),
                timeout=30.0,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Превышен таймаут авторегрессионной генерации (30с)") from exc


# Монтируем статическую сборку Vite последней, чтобы не перехватывать /api/*
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
