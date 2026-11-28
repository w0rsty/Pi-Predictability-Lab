import { useEffect, useState } from "react";
import { MotionConfig } from "motion/react";
import { LocalGpuModal } from "./components/layout/LocalGpuModal";
import { Navbar } from "./components/layout/Navbar";
import type { PianoStreamKey } from "./components/hero/PiPianoSynth";
import { ArenaView } from "./components/arena/ArenaView";
import { AtlasView } from "./components/sections/AtlasView";
import { MechInterpView } from "./components/sections/MechInterpView";
import { OverviewView } from "./components/sections/OverviewView";
import { SandboxView } from "./components/sections/SandboxView";
import { useLocalGpu } from "./hooks/useLocalGpu";
import { useTheme } from "./hooks/useTheme";
import type { LabData, TabId } from "./types/lab";

export function App() {
  const { theme, toggleTheme } = useTheme();
  const gpu = useLocalGpu();
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [gpuModalOpen, setGpuModalOpen] = useState<boolean>(false);
  const [labData, setLabData] = useState<LabData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [externalPianoStream, setExternalPianoStream] = useState<{
    label: string;
    digits: number[];
    mask?: number[] | null;
  } | null>(null);
  const [pianoOverride, setPianoOverride] = useState<PianoStreamKey | undefined>(undefined);

  useEffect(() => {
    const base = import.meta.env.BASE_URL || "/";
    const dataUrl = `${base.endsWith("/") ? base : `${base}/`}data/lab_data.json`;
    fetch(dataUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<LabData>;
      })
      .then((json) => setLabData(json))
      .catch((err) => {
        setLoadError(err instanceof Error ? err.message : "Ошибка загрузки lab_data.json");
      });
  }, []);

  const handleSendToPiano = (stream: { label: string; digits: number[]; mask?: number[] | null }) => {
    setExternalPianoStream(stream);
    setPianoOverride("custom");
    setActiveTab("overview");
    window.scrollTo({ top: 520, behavior: "smooth" });
  };

  return (
    <MotionConfig reducedMotion="user">
      <div className="relative min-h-screen overflow-x-clip bg-[var(--bg)] text-[var(--ink)]">
        {/* Фактура архивной бумаги / сланца */}
        <div className="paper-grain-overlay" aria-hidden="true" />

        <Navbar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          theme={theme}
          onToggleTheme={toggleTheme}
          gpuStatus={gpu.status}
          gpuHealth={gpu.health}
          onOpenGpuModal={() => setGpuModalOpen(true)}
        />

        <main className="relative z-10 mx-auto max-w-[1360px] px-5 pt-8 sm:px-6 lg:px-12 lg:pt-10">
          {loadError ? (
            <div className="rounded-[4px] border border-[var(--sig-h1)]/50 bg-[var(--bg-alt)] p-6 font-mono text-sm text-[var(--sig-h1)]">
              Не удалось загрузить данные лаборатории: {loadError}
            </div>
          ) : !labData ? (
            <div className="flex min-h-[55vh] items-center justify-center font-mono text-xs text-[var(--ink-muted)]">
              Загружаем канонический датасет lab_data.json…
            </div>
          ) : (
            <>
              {activeTab === "overview" && (
                <OverviewView
                  data={labData}
                  theme={theme}
                  onNavigate={(t) => setActiveTab(t)}
                  externalPianoStream={externalPianoStream}
                  onClearExternalPianoStream={() => {
                    setExternalPianoStream(null);
                    setPianoOverride("true_pi");
                  }}
                  pianoOverride={pianoOverride}
                />
              )}
              {activeTab === "atlas" && <AtlasView data={labData} theme={theme} />}
              {activeTab === "mech-interp" && (
                <MechInterpView
                  data={labData}
                  theme={theme}
                  isGpuConnected={gpu.isConnected}
                  onOpenGpuModal={() => setGpuModalOpen(true)}
                  onAblateGpu={gpu.ablateHeads}
                />
              )}
              {activeTab === "sandbox" && (
                <SandboxView
                  data={labData}
                  theme={theme}
                  isGpuConnected={gpu.isConnected}
                  onOpenGpuModal={() => setGpuModalOpen(true)}
                  onSendToPiano={handleSendToPiano}
                  onGenerateGpu={gpu.generateFormula}
                  onAnalyzeGpu={gpu.analyzeSequence}
                  onTrainBackdoorGpu={gpu.trainBackdoor}
                  onGenerateDreamGpu={gpu.generateDream}
                />
              )}
              {activeTab === "arena" && (
                <ArenaView
                  theme={theme}
                  isGpuConnected={gpu.isConnected}
                  onFetchLiveSession={gpu.fetchArenaSession}
                />
              )}
            </>
          )}
        </main>

        <footer className="relative z-10 border-t border-[var(--rule)]">
          <div className="mx-auto flex max-w-[1360px] flex-wrap items-center justify-between gap-4 px-5 py-6 font-sans text-xs text-[var(--ink-muted)] sm:px-6 lg:px-12">
            <div>Pi-Predictability-Lab · 2026 · MIT License</div>
            <div className="flex flex-wrap items-center gap-5">
              <a
                href="https://github.com/w0rsty/Pi-Predictability-Lab"
                target="_blank"
                rel="noreferrer"
                className="transition-colors hover:text-[var(--ink)]"
              >
                GitHub
              </a>
              <a
                href="https://github.com/w0rsty/Pi-Predictability-Lab/blob/main/docs/RESEARCH_LOG.md"
                target="_blank"
                rel="noreferrer"
                className="transition-colors hover:text-[var(--ink)]"
              >
                Документация (Research Log)
              </a>
              <span className="font-mono text-[11px] text-[var(--ink-2)]">
                {gpu.isConnected ? `Compute Bridge: ${gpu.baseUrl}` : "Browser ONNX / Static"}
              </span>
            </div>
          </div>
        </footer>

        <LocalGpuModal
          open={gpuModalOpen}
          onClose={() => setGpuModalOpen(false)}
          baseUrl={gpu.baseUrl}
          onChangeBaseUrl={gpu.setBaseUrl}
          token={gpu.token}
          onChangeToken={gpu.setToken}
          status={gpu.status}
          health={gpu.health}
          latencyMs={gpu.latencyMs}
          error={gpu.error}
          onConnect={gpu.connect}
          onDisconnect={gpu.disconnect}
        />
      </div>
    </MotionConfig>
  );
}
export default App;
