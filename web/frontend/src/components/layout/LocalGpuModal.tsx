import { useEffect, useRef, useState } from "react";
import { Check, Copy, X } from "lucide-react";
import type { GpuConnectionStatus } from "../../hooks/useLocalGpu";
import type { HealthResponse } from "../../types/lab";

interface LocalGpuModalProps {
  open: boolean;
  onClose: () => void;
  baseUrl: string;
  onChangeBaseUrl: (url: string) => void;
  token: string;
  onChangeToken: (tok: string) => void;
  status: GpuConnectionStatus;
  health: HealthResponse | null;
  latencyMs: number | null;
  error: string | null;
  onConnect: (url?: string, tok?: string) => Promise<boolean>;
  onDisconnect: () => void;
}

const LAUNCH_CMD = "uvicorn web.server:app --host 127.0.0.1 --port 8000";
const TUNNEL_CMD =
  'LAB_API_TOKEN="my-secret" uvicorn web.server:app --host 127.0.0.1 --port 8000 && tuna http 8000';

export function LocalGpuModal({
  open,
  onClose,
  baseUrl,
  onChangeBaseUrl,
  token,
  onChangeToken,
  status,
  health,
  latencyMs,
  error,
  onConnect,
  onDisconnect,
}: LocalGpuModalProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const [copied, setCopied] = useState(false);
  const [copiedTunnel, setCopiedTunnel] = useState(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    let pressedOutside = false;
    const isOutside = (e: PointerEvent) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    const onDown = (e: PointerEvent) => {
      pressedOutside = isOutside(e);
    };
    const onUp = (e: PointerEvent) => {
      if (pressedOutside && isOutside(e) && dialog.open) {
        onClose();
      }
    };
    const onCancel = (e: Event) => {
      e.preventDefault();
      onClose();
    };
    dialog.addEventListener("pointerdown", onDown);
    dialog.addEventListener("pointerup", onUp);
    dialog.addEventListener("cancel", onCancel);
    return () => {
      dialog.removeEventListener("pointerdown", onDown);
      dialog.removeEventListener("pointerup", onUp);
      dialog.removeEventListener("cancel", onCancel);
    };
  }, [onClose]);

  const handleCopy = async (text = LAUNCH_CMD, isTunnel = false) => {
    try {
      await navigator.clipboard.writeText(text);
      if (isTunnel) {
        setCopiedTunnel(true);
        window.setTimeout(() => setCopiedTunnel(false), 1800);
      } else {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1800);
      }
    } catch {
      // игнорируем
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void onConnect(baseUrl, token);
  };

  return (
    <dialog ref={dialogRef} aria-labelledby="gpu-modal-title" className="text-[var(--ink)]">
      <div className="flex flex-col gap-5 p-6">
        <div className="flex items-start justify-between gap-4 border-b border-[var(--rule)] pb-4">
          <div>
            <div className="font-sans text-xs font-medium text-[var(--ink-muted)]">
              Локальный вычислительный мост
            </div>
            <h1 id="gpu-modal-title" className="mt-1 font-serif text-xl font-medium text-[var(--ink)]">
              Подключение локального GPU-сервера
            </h1>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Закрыть окно"
            className="rounded-[4px] border border-[var(--rule)] p-1.5 text-[var(--ink-2)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="text-xs leading-relaxed text-[var(--ink-2)]">
          Вычисления в лаборатории работают на трёх уровнях: (1) мгновенная математика и статистика
          в браузере, (2) реальный инференс весов{" "}
          <code className="font-mono text-[var(--ink)]">DigitTransformer</code> через{" "}
          <code className="font-mono text-[var(--ink)]">Browser ONNX (WASM SIMD)</code> без сервера,
          (3) локальный мост <code className="font-mono text-[var(--ink)]">127.0.0.1</code> для
          формул <code className="font-mono text-[var(--ink)]">mpmath</code>,{" "}
          <code className="font-mono text-[var(--ink)]">PPM-C</code> и быстрого обучения новых
          детекторов на вашей видеокарте.
        </p>

        <div>
          <div className="mb-1.5 font-sans text-xs font-medium text-[var(--ink-2)]">
            1. Запуск моста из корня репозитория
          </div>
          <div className="flex items-center justify-between gap-2 rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] px-3 py-2 font-mono text-xs">
            <code className="overflow-x-auto whitespace-nowrap text-[var(--ink)]">
              {LAUNCH_CMD}
            </code>
            <button
              type="button"
              onClick={() => void handleCopy(LAUNCH_CMD, false)}
              className="flex shrink-0 items-center gap-1 rounded-[3px] border border-[var(--rule)] bg-[var(--bg)] px-2 py-1 text-[11px] text-[var(--ink-2)] hover:text-[var(--ink)]"
            >
              {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
              {copied ? "Скопировано" : "Копировать"}
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <label className="font-sans text-xs font-medium text-[var(--ink-2)]">
            2. Адрес FastAPI-моста (локальный или HTTPS-туннель)
            <div className="mt-1.5 flex gap-2">
              <input
                type="url"
                data-testid="gpu-url-input"
                value={baseUrl}
                onChange={(e) => onChangeBaseUrl(e.target.value)}
                placeholder="http://127.0.0.1:8000 или https://xxxx.tuna.am"
                className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-2 font-mono text-xs text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
              />
              <button
                type="submit"
                data-testid="gpu-connect-submit"
                disabled={status === "connecting"}
                className="shrink-0 rounded-[4px] bg-[var(--ink)] px-4 py-2 text-xs font-medium text-[var(--bg)] transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {status === "connecting" ? "Проверка…" : "Подключить"}
              </button>
            </div>
          </label>

          <label className="font-sans text-xs font-medium text-[var(--ink-2)]">
            3. Токен доступа (опционально для туннеля)
            <input
              type="password"
              data-testid="gpu-token-input"
              value={token}
              onChange={(e) => onChangeToken(e.target.value)}
              placeholder="Оставьте пустым для локального http://127.0.0.1:8000"
              autoComplete="off"
              className={`mt-1.5 w-full rounded-[4px] border bg-[var(--bg)] px-3 py-2 font-mono text-xs text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none ${
                status === "auth_required"
                  ? "border-[var(--sig-h1)]"
                  : "border-[var(--rule-strong)]"
              }`}
            />
            <span className="mt-1 block text-[11px] font-normal text-[var(--ink-muted)]">
              URL туннеля и токен хранятся только в памяти вкладки (<code className="font-mono">sessionStorage</code>) и стираются при её закрытии.
            </span>
          </label>
        </form>

        <details className="rounded-[4px] border border-[var(--rule)] bg-[var(--bg-alt)]/60 px-3 py-2 text-xs">
          <summary className="cursor-pointer font-sans font-medium text-[var(--ink-2)] hover:text-[var(--ink)]">
            Подключение к домашнему ПК через туннель (Tuna / cloudflared / Tailscale)
          </summary>
          <div className="mt-2.5 space-y-2 text-[11px] leading-relaxed text-[var(--ink-2)]">
            <p>
              На домашнем компьютере с GPU запустите мост с секретным токеном и откройте
              обратный HTTPS-туннель:
            </p>
            <div className="flex items-center justify-between gap-2 rounded-[4px] border border-[var(--rule)] bg-[var(--bg)] px-2.5 py-1.5 font-mono text-[10px]">
              <code className="overflow-x-auto whitespace-nowrap text-[var(--ink)]">
                {TUNNEL_CMD}
              </code>
              <button
                type="button"
                onClick={() => void handleCopy(TUNNEL_CMD, true)}
                className="shrink-0 text-[var(--ink-2)] hover:text-[var(--ink)]"
                title="Скопировать команду запуска с токеном"
              >
                {copiedTunnel ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </div>
            <p className="font-mono text-[10px] text-[var(--ink-muted)]">
              Альтернативы: <code>cloudflared tunnel --url http://127.0.0.1:8000</code> или{" "}
              <code>tailscale funnel 8000</code>
            </p>
          </div>
        </details>

        {error && (
          <div className="rounded-[4px] border border-[var(--sig-h1)]/45 bg-[var(--bg-alt)] px-3.5 py-2.5 text-xs text-[var(--sig-h1)]">
            {error}
          </div>
        )}

        {health && status === "connected" && (
          <div className="rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg-alt)] p-3.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-medium text-[var(--ink)]">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--sig-gpu)]" />
                {health.gpu_name}
              </span>
              {latencyMs !== null && (
                <span className="font-mono tabular-nums text-[var(--ink-muted)]">
                  {latencyMs} ms
                </span>
              )}
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums text-[var(--ink-2)]">
              <div>device: {health.device}</div>
              <div>vram: {health.vram_gb !== null ? `${health.vram_gb} GB` : "—"}</div>
              <div>ckpt: {health.checkpoints_loaded}</div>
            </div>
            <div className="mt-3 flex justify-end border-t border-[var(--rule)] pt-2.5">
              <button
                type="button"
                onClick={onDisconnect}
                className="rounded-[3px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-1 font-sans text-xs text-[var(--ink-2)] hover:text-[var(--ink)]"
              >
                Отключить
              </button>
            </div>
          </div>
        )}
      </div>
    </dialog>
  );
}
