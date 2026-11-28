import { Moon, Sun } from "lucide-react";
import { motion } from "motion/react";
import type { GpuConnectionStatus } from "../../hooks/useLocalGpu";
import type { HealthResponse, TabId, ThemeMode } from "../../types/lab";

interface NavbarProps {
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  gpuStatus: GpuConnectionStatus;
  gpuHealth: HealthResponse | null;
  onOpenGpuModal: () => void;
}

const TABS: Array<{
  id: TabId;
  label: string;
}> = [
  { id: "overview", label: "Обзор" },
  { id: "atlas", label: "Атлас" },
  { id: "mech-interp", label: "Мехинтерп" },
  { id: "sandbox", label: "Песочница" },
  { id: "arena", label: "Арена" },
];

export function Navbar({
  activeTab,
  onSelectTab,
  theme,
  onToggleTheme,
  gpuStatus,
  gpuHealth,
  onOpenGpuModal,
}: NavbarProps) {
  const isConnected = gpuStatus === "connected" && gpuHealth !== null;
  const gpuLabel = isConnected
    ? gpuHealth.gpu_name.replace("NVIDIA GeForce ", "")
    : "Local GPU";

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--rule)] bg-[var(--bg)]/90 backdrop-blur-sm">
      <div className="mx-auto flex h-[56px] max-w-[1360px] items-center justify-between gap-4 px-5 sm:px-6 lg:px-12">
        {/* SVG-знак π + название Pi-Predictability-Lab */}
        <button
          type="button"
          onClick={() => onSelectTab("overview")}
          className="flex shrink-0 items-center gap-2.5 text-left focus-visible:outline-1 focus-visible:outline-offset-4 focus-visible:outline-[var(--sig-h1)]"
        >
          <svg viewBox="0 0 24 24" className="h-6 w-6 shrink-0" fill="none" aria-hidden="true">
            <path
              d="M2 6V2H6M18 2H22V6M22 18V22H18M6 22H2V18"
              stroke="var(--ink-muted)"
              strokeWidth="1.2"
            />
            <path
              d="M6.5 9H17.5M10 9V16.5M14.5 9V15.5L16.5 16.5"
              stroke="var(--ink)"
              strokeWidth="1.6"
              strokeLinecap="square"
            />
            <circle cx="16.5" cy="16.5" r="1.6" fill="var(--sig-h1)" />
          </svg>
          <span className="hidden font-sans text-[14px] font-medium tracking-tight text-[var(--ink)] sm:inline">
            Pi-Predictability-Lab
          </span>
        </button>

        {/* Навигация на Golos Text */}
        <nav
          aria-label="Разделы лаборатории"
          className="flex h-full min-w-0 items-center gap-3 overflow-x-auto sm:gap-7"
        >
          {TABS.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onSelectTab(tab.id)}
                className={`relative flex h-full shrink-0 items-center font-sans text-[12px] font-medium transition-colors sm:text-[13px] ${
                  active
                    ? "text-[var(--ink)]"
                    : "text-[var(--ink-2)] hover:text-[var(--ink)]"
                }`}
              >
                <span>{tab.label}</span>
                {active && (
                  <motion.span
                    layoutId="nav-underline"
                    transition={{ type: "spring", stiffness: 460, damping: 36 }}
                    className="absolute inset-x-0 bottom-0 h-[2px] bg-[var(--sig-h1)]"
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Правая панель: статус GPU, тема, GitHub */}
        <div className="flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={onOpenGpuModal}
            data-testid="gpu-status-btn"
            className="flex items-center gap-2 rounded-[4px] border border-transparent px-2 py-1 font-mono text-[11px] text-[var(--ink-2)] transition-colors hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
            title="Настройка подключения локального GPU-сервера"
          >
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{
                backgroundColor: isConnected ? "var(--sig-gpu)" : "var(--ink-muted)",
              }}
            />
            <span className="hidden max-w-[160px] truncate sm:inline">{gpuLabel}</span>
            <span className="sm:hidden">GPU</span>
          </button>

          <span className="h-4 w-px bg-[var(--rule)]" aria-hidden="true" />

          <button
            type="button"
            onClick={onToggleTheme}
            data-testid="theme-toggle-btn"
            aria-label={
              theme === "light"
                ? "Переключить на тёмную тему Warm Basalt"
                : "Переключить на светлую тему Archival Desk"
            }
            title={theme === "light" ? "Тёмная тема (Warm Basalt)" : "Светлая тема (Archival Desk)"}
            className="flex h-7 w-7 items-center justify-center text-[var(--ink-2)] transition-colors hover:text-[var(--ink)]"
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          <a
            href="https://github.com/w0rsty/Pi-Predictability-Lab"
            target="_blank"
            rel="noreferrer"
            aria-label="Репозиторий на GitHub"
            className="hidden h-7 w-7 items-center justify-center text-[var(--ink-2)] transition-colors hover:text-[var(--ink)] sm:flex"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
          </a>
        </div>
      </div>
    </header>
  );
}
