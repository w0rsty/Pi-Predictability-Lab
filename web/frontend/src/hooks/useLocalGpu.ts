import { useCallback, useEffect, useState } from "react";
import type {
  AblationResponse,
  ArenaModeId,
  ArenaSessionResponse,
  HealthResponse,
  SandboxAnalyzeResponse,
  SandboxGenerateResponse,
} from "../types/lab";

const DEFAULT_GPU_URL = "http://127.0.0.1:8000";
const URL_STORAGE_KEY = "pi-lab-gpu-url";

export type GpuConnectionStatus = "offline" | "connecting" | "connected" | "error";

export function useLocalGpu() {
  const [baseUrl, setBaseUrl] = useState<string>(() => {
    try {
      return localStorage.getItem(URL_STORAGE_KEY) || DEFAULT_GPU_URL;
    } catch {
      return DEFAULT_GPU_URL;
    }
  });
  const [status, setStatus] = useState<GpuConnectionStatus>("offline");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const probe = useCallback(
    async (targetUrl: string, silent = false): Promise<boolean> => {
      const clean = targetUrl.trim().replace(/\/+$/, "");
      if (!clean) return false;
      if (!silent) {
        setStatus("connecting");
        setError(null);
      }
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), silent ? 1200 : 3500);
      const t0 = performance.now();
      try {
        const res = await fetch(`${clean}/api/health`, {
          method: "GET",
          signal: controller.signal,
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = (await res.json()) as HealthResponse;
        const dt = Math.round(performance.now() - t0);
        setBaseUrl(clean);
        setHealth(data);
        setLatencyMs(dt);
        setStatus("connected");
        setError(null);
        try {
          localStorage.setItem(URL_STORAGE_KEY, clean);
        } catch {
          // игнорируем
        }
        return true;
      } catch (err) {
        if (!silent) {
          const msg =
            err instanceof Error && err.name === "AbortError"
              ? "Превышен таймаут ожидания (3.5с). Проверьте, запущен ли uvicorn."
              : "Не удалось подключиться к локальному мосту. Запустите uvicorn web.server:app --port 8000.";
          setError(msg);
          setStatus("error");
          setHealth(null);
        }
        return false;
      } finally {
        window.clearTimeout(timer);
      }
    },
    []
  );

  // Тихий авто-probe только если страница открыта локально
  useEffect(() => {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      const originUrl = `${window.location.protocol}//${window.location.host}`;
      const candidate =
        window.location.port === "8000" ? originUrl : baseUrl || DEFAULT_GPU_URL;
      void probe(candidate, true);
    }
  }, [baseUrl, probe]);

  const disconnect = useCallback(() => {
    setStatus("offline");
    setHealth(null);
    setLatencyMs(null);
    setError(null);
  }, []);

  const postJson = useCallback(
    async <T,>(path: string, body: unknown): Promise<T> => {
      if (status !== "connected") {
        throw new Error("Local GPU мост не подключён");
      }
      const res = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
        throw new Error(detail);
      }
      return (await res.json()) as T;
    },
    [baseUrl, status]
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

  return {
    baseUrl,
    setBaseUrl,
    status,
    health,
    latencyMs,
    error,
    isConnected: status === "connected",
    connect: (url?: string) => probe(url ?? baseUrl, false),
    disconnect,
    generateFormula,
    analyzeSequence,
    ablateHeads,
    fetchArenaSession,
  };
}
