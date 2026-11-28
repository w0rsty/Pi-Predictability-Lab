import { useCallback, useEffect, useState } from "react";
import type {
  AblationResponse,
  ArenaModeId,
  ArenaSessionResponse,
  BackdoorTrainResponse,
  DreamDatasetId,
  DreamGpuResponse,
  HealthResponse,
  SandboxAnalyzeResponse,
  SandboxGenerateResponse,
} from "../types/lab";

const DEFAULT_GPU_URL = "http://127.0.0.1:8000";
const URL_SESSION_KEY = "pi-lab-gpu-url";
const TOKEN_SESSION_KEY = "pi-lab-gpu-token";

export type GpuConnectionStatus = "offline" | "connecting" | "connected" | "auth_required" | "error";

export function normalizeBridgeUrl(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

function isRemoteHttpBlockedOnHttps(cleanUrl: string): boolean {
  if (typeof window === "undefined" || window.location.protocol !== "https:") {
    return false;
  }
  try {
    const parsed = new URL(cleanUrl);
    if (parsed.protocol !== "http:") return false;
    const h = parsed.hostname.toLowerCase();
    return h !== "localhost" && h !== "127.0.0.1" && h !== "[::1]";
  } catch {
    return false;
  }
}

export function useLocalGpu() {
  const [baseUrl, setBaseUrl] = useState<string>(() => {
    try {
      return normalizeBridgeUrl(sessionStorage.getItem(URL_SESSION_KEY) || "") || DEFAULT_GPU_URL;
    } catch {
      return DEFAULT_GPU_URL;
    }
  });
  const [token, setToken] = useState<string>(() => {
    try {
      return sessionStorage.getItem(TOKEN_SESSION_KEY) || "";
    } catch {
      return "";
    }
  });
  const [status, setStatus] = useState<GpuConnectionStatus>("offline");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const probe = useCallback(
    async (targetUrl: string, silent = false, tokenOverride?: string): Promise<boolean> => {
      const clean = normalizeBridgeUrl(targetUrl);
      const activeToken = (tokenOverride ?? token).trim();
      if (!clean) {
        if (!silent) {
          setStatus("error");
          setError("Укажите адрес FastAPI-моста (например, http://127.0.0.1:8000 или https://...)");
        }
        return false;
      }

      if (isRemoteHttpBlockedOnHttps(clean)) {
        if (!silent) {
          setStatus("error");
          setHealth(null);
          setError(
            "Браузер заблокирует небезопасный HTTP с HTTPS-страницы (Mixed Content). Используйте https://-ссылку туннеля (Tuna / cloudflared / Tailscale)."
          );
        }
        return false;
      }

      if (!silent) {
        setStatus("connecting");
        setError(null);
      }

      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), silent ? 1200 : 4500);
      const t0 = performance.now();
      try {
        const headers: Record<string, string> = {};
        if (activeToken) {
          headers["X-Lab-Token"] = activeToken;
        }
        const res = await fetch(`${clean}/api/health`, {
          method: "GET",
          headers,
          signal: controller.signal,
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = (await res.json()) as HealthResponse;
        const dt = Math.round(performance.now() - t0);
        setBaseUrl(clean);
        setLatencyMs(dt);

        try {
          sessionStorage.setItem(URL_SESSION_KEY, clean);
          if (activeToken) {
            sessionStorage.setItem(TOKEN_SESSION_KEY, activeToken);
          } else {
            sessionStorage.removeItem(TOKEN_SESSION_KEY);
          }
        } catch {
          // игнорируем ограничения sessionStorage
        }

        if (data.auth_required && !data.auth_valid) {
          setHealth(data);
          setStatus(silent ? "offline" : "auth_required");
          if (!silent) {
            setError("Сервер найден, нужен верный токен");
          }
          return false;
        }

        setHealth(data);
        setStatus("connected");
        setError(null);
        return true;
      } catch (err) {
        if (!silent) {
          const msg =
            err instanceof Error && err.name === "AbortError"
              ? "Превышен таймаут ожидания (4.5с). Проверьте, запущен ли uvicorn или туннель."
              : "Не удалось подключиться к мосту. Проверьте адрес и запуск uvicorn web.server:app --port 8000.";
          setError(msg);
          setStatus("error");
          setHealth(null);
        }
        return false;
      } finally {
        window.clearTimeout(timer);
      }
    },
    [token]
  );

  // Тихий авто-probe только если страница открыта локально из uvicorn (8000) или Vite dev (5173)
  useEffect(() => {
    const host = window.location.hostname;
    const port = window.location.port;
    if ((host === "localhost" || host === "127.0.0.1") && (port === "8000" || port === "5173")) {
      const originUrl = `${window.location.protocol}//${window.location.host}`;
      const candidate = port === "8000" ? originUrl : baseUrl || DEFAULT_GPU_URL;
      void probe(candidate, true);
    }
    // Запускаем один раз при монтировании, чтобы не дёргать сервер при каждом нажатии клавиши в токене/URL
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const disconnect = useCallback(() => {
    setStatus("offline");
    setHealth(null);
    setLatencyMs(null);
    setError(null);
  }, []);

  const postJson = useCallback(
    async <T,>(path: string, body: unknown): Promise<T> => {
      if (status !== "connected") {
        throw new Error("GPU-мост не подключён");
      }
      const clean = normalizeBridgeUrl(baseUrl);
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      const activeToken = token.trim();
      if (activeToken) {
        headers["X-Lab-Token"] = activeToken;
      }
      const res = await fetch(`${clean}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const errJson = (await res.json()) as { detail?: string };
          if (errJson.detail) detail = errJson.detail;
        } catch {
          // оставляем статус
        }
        if (res.status === 401) {
          setStatus("auth_required");
          setError("Сервер найден, нужен верный токен");
        }
        throw new Error(detail);
      }
      return (await res.json()) as T;
    },
    [baseUrl, status, token]
  );

  const generateFormula = useCallback(
    (payload: {
      mode: "formula" | "rational" | "backdoor";
      expression?: string;
      numerator?: number;
      denominator?: number;
      length: number;
      radix: number;
    }) => postJson<SandboxGenerateResponse>("/api/sandbox/generate", payload),
    [postJson]
  );

  const analyzeSequence = useCallback(
    (payload: {
      name: string;
      digits: number[];
      mask?: number[] | null;
      radix: number;
      run_ppm?: boolean;
      run_models?: boolean;
    }) => postJson<SandboxAnalyzeResponse>("/api/sandbox/analyze", payload),
    [postJson]
  );

  const ablateHeads = useCallback(
    (payload: {
      dataset: string;
      ablate_heads: [number, number][];
      max_windows?: number;
    }) => postJson<AblationResponse>("/api/mech-interp/ablate", payload),
    [postJson]
  );

  const fetchArenaSession = useCallback(
    (payload: { dataset: ArenaModeId; n_rounds: number; seed?: number }) =>
      postJson<ArenaSessionResponse>("/api/arena/session", payload),
    [postJson]
  );

  const trainBackdoor = useCallback(
    (payload: {
      lag_a: number;
      lag_b: number;
      prev_threshold: number;
      coeff_a: number;
      coeff_b: number;
      coeff_const: number;
      length?: number;
      epochs?: number;
      seed?: number;
    }) => postJson<BackdoorTrainResponse>("/api/sandbox/train-backdoor", payload),
    [postJson]
  );

  const generateDream = useCallback(
    (payload: {
      dataset: DreamDatasetId;
      seed_tokens: number[];
      steps: number;
      temperature: number;
      top_k: number;
      random_seed?: number;
    }) => postJson<DreamGpuResponse>("/api/sandbox/dream", payload),
    [postJson]
  );

  return {
    baseUrl,
    setBaseUrl,
    token,
    setToken,
    status,
    health,
    latencyMs,
    error,
    isConnected: status === "connected",
    connect: (url?: string, tok?: string) => probe(url ?? baseUrl, false, tok),
    disconnect,
    generateFormula,
    analyzeSequence,
    ablateHeads,
    fetchArenaSession,
    trainBackdoor,
    generateDream,
  };
}
