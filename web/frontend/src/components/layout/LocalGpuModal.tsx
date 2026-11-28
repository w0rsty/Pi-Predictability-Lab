import { useEffect, useRef, useState } from "react";
import { Check, Copy, X } from "lucide-react";
import type { GpuConnectionStatus } from "../../hooks/useLocalGpu";
import type { HealthResponse } from "../../types/lab";

interface LocalGpuModalProps {
  open: boolean;
  onClose: () => void;
  baseUrl: string;
  onChangeBaseUrl: (url: string) => void;
  status: GpuConnectionStatus;
  health: HealthResponse | null;
  latencyMs: number | null;
  error: string | null;
  onConnect: (url?: string) => Promise<boolean>;
  onDisconnect: () => void;
}

const LAUNCH_CMD = "uvicorn web.server:app --host 127.0.0.1 --port 8000";

export function LocalGpuModal({
  open,
  onClose,
  baseUrl,
  onChangeBaseUrl,
  status,
  health,
  latencyMs,
  error,
  onConnect,
  onDisconnect,
}: LocalGpuModalProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const [copied, setCopied] = useState(false);

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

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(LAUNCH_CMD);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // игнорируем
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void onConnect(baseUrl);
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
          Статическая витрина работает автономно в браузере. Для вычисления формул{" "}
          <code className="font-mono text-[var(--ink)]">mpmath</code>,{" "}
          <code className="font-mono text-[var(--ink)]">PPM-C</code> и живого инференса PyTorch на
          вашей видеокарте браузер обращается к локальному процессу на{" "}
          <code className="font-mono text-[var(--ink)]">127.0.0.1</code>.
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
              onClick={handleCopy}
              className="flex shrink-0 items-center gap-1 rounded-[3px] border border-[var(--rule)] bg-[var(--bg)] px-2 py-1 text-[11px] text-[var(--ink-2)] hover:text-[var(--ink)]"
            >
              {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
              {copied ? "Скопировано" : "Копировать"}
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-2">
          <label className="font-sans text-xs font-medium text-[var(--ink-2)]">
            2. Адрес FastAPI-моста
            <div className="mt-1.5 flex gap-2">
              <input
                type="url"
                value={baseUrl}
                onChange={(e) => onChangeBaseUrl(e.target.value)}
                placeholder="http://127.0.0.1:8000"
                className="w-full rounded-[4px] border border-[var(--rule-strong)] bg-[var(--bg)] px-3 py-2 font-mono text-xs text-[var(--ink)] focus:border-[var(--sig-h1)] focus:outline-none"
              />
              <button
                type="submit"
                disabled={status === "connecting"}
                className="shrink-0 rounded-[4px] bg-[var(--ink)] px-4 py-2 text-xs font-medium text-[var(--bg)] transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {status === "connecting" ? "Проверка…" : "Подключить"}
              </button>
            </div>
          </label>
        </form>

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
