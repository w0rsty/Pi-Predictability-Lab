import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw, Volume2, X } from "lucide-react";
import { SCALE_NOTE_NAMES, type SynthScale, usePiSynth } from "../../hooks/usePiSynth";
import type { LabData, ThemeMode } from "../../types/lab";

export type PianoStreamKey =
  | "true_pi"
  | "rational_1_97"
  | "e_cf"
  | "champernowne"
  | "csprng_backdoor"
  | "custom";

interface PiPianoSynthProps {
  samples: LabData["samples"];
  theme: ThemeMode;
  externalStream?: {
    label: string;
    digits: number[];
    mask?: number[] | null;
  } | null;
  onClearExternalStream?: () => void;
  activeStreamOverride?: PianoStreamKey;
}

const STREAM_OPTIONS: Array<{ key: Exclude<PianoStreamKey, "custom">; label: string }> = [
  { key: "true_pi", label: "π (H₀)" },
  { key: "rational_1_97", label: "1/97 (P=96)" },
  { key: "e_cf", label: "e_cf" },
  { key: "champernowne", label: "Champernowne" },
  { key: "csprng_backdoor", label: "Backdoor (t-24, t-7)" },
];

const TAPE_OFFSETS = Array.from({ length: 41 }, (_, i) => i - 20);

export function PiPianoSynth({
  samples,
  externalStream = null,
  onClearExternalStream,
  activeStreamOverride,
}: PiPianoSynthProps) {
  const synth = usePiSynth();
  const [streamKey, setStreamKey] = useState<PianoStreamKey>("true_pi");
  const scopeRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (activeStreamOverride) {
      setStreamKey(activeStreamOverride);
      synth.setStepIndex(0);
    }
  }, [activeStreamOverride]);

  const currentStream = useMemo(() => {
    if (streamKey === "custom" && externalStream) {
      return externalStream;
    }
    const key = streamKey === "custom" ? "true_pi" : streamKey;
    const s = samples[key];
    return {
      label: STREAM_OPTIONS.find((o) => o.key === key)?.label ?? key,
      digits: s.digits,
      mask: "mask" in s ? s.mask : null,
    };
  }, [externalStream, samples, streamKey]);

  useEffect(() => {
    if (!synth.isPlaying) return;
    const intervalMs = Math.max(90, Math.round(60_000 / synth.bpm));
    const timer = window.setInterval(() => {
      synth.setStepIndex((prev) => {
        const idx = prev % currentStream.digits.length;
        const d = currentStream.digits[idx];
        const trig = currentStream.mask ? currentStream.mask[idx] === 1 : false;
        synth.playNote(d, trig);
        return (idx + 1) % currentStream.digits.length;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [currentStream, synth.bpm, synth.isPlaying, synth.playNote, synth.setStepIndex]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT")
      ) {
        return;
      }
      if (e.key >= "0" && e.key <= "9") {
        synth.playNote(Number(e.key), false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [synth.playNote]);

  // Живой OLED мини-осциллограф внутри тёмного корпуса синтезатора
  useEffect(() => {
    let raf = 0;
    const buf = new Uint8Array(128);
    const renderScope = () => {
      const canvas = scopeRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const w = canvas.width;
          const h = canvas.height;
          ctx.clearRect(0, 0, w, h);
          ctx.strokeStyle = "rgba(239, 233, 222, 0.12)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, h / 2);
          ctx.lineTo(w, h / 2);
          ctx.stroke();

          const analyser = synth.analyserRef.current;
          ctx.strokeStyle = synth.isTriggerPulse ? "#DF6244" : "#EFE9DE";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          if (analyser) {
            analyser.getByteTimeDomainData(buf);
            for (let i = 0; i < buf.length; i++) {
              const x = (i / (buf.length - 1)) * w;
              const y = ((buf[i] - 128) / 128) * (h * 0.42) + h / 2;
              if (i === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
          } else {
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
          }
          ctx.stroke();
        }
      }
      raf = requestAnimationFrame(renderScope);
    };
    raf = requestAnimationFrame(renderScope);
    return () => cancelAnimationFrame(raf);
  }, [synth.analyserRef, synth.isTriggerPulse]);

  const noteNames = SCALE_NOTE_NAMES[synth.scale];
  const len = Math.max(1, currentStream.digits.length);
  const centerIdx = synth.stepIndex === 0 ? 0 : (synth.stepIndex - 1 + len) % len;

  return (
    <section className="hardware-device overflow-hidden">
      {/* Передняя панель синтезатора */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--device-rule)] px-5 py-4">
        <div>
          <div className="font-sans text-xs text-[var(--device-ink-2)]">
            Сонификация последовательностей · Web Audio API
          </div>
          <h2 className="mt-0.5 font-serif text-xl font-medium text-[var(--device-ink)]">
            Пианино π: звук шума против алгоритма
          </h2>
        </div>

        {/* Утопленный OLED мини-осциллограф */}
        <div className="hardware-well hidden items-center gap-3 rounded-[4px] px-3 py-1 sm:flex">
          <canvas ref={scopeRef} width={140} height={30} className="block h-[30px] w-[140px]" />
          <div className="flex items-center gap-1.5 font-sans text-[11px] text-[var(--device-ink-2)]">
            <span
              className="h-1.5 w-1.5 rounded-full transition-shadow"
              style={{
                backgroundColor: synth.isTriggerPulse ? "var(--dev-signal)" : "var(--device-ink-muted)",
                boxShadow: synth.isTriggerPulse ? "0 0 6px var(--dev-signal)" : "none",
              }}
            />
            Колокольчик закладки
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Музыкальный строй"
            value={synth.scale}
            onChange={(e) => synth.setScale(e.target.value as SynthScale)}
            className="device-select"
          >
            <option value="pentatonic">Пентатоника</option>
            <option value="minor">Минор</option>
            <option value="chromatic">Хроматика</option>
          </select>

          <button
            type="button"
            onClick={() => synth.setIsPlaying((v) => !v)}
            data-testid="piano-play-btn"
            className={`flex items-center gap-1.5 rounded-[4px] px-3.5 py-1.5 font-sans text-xs font-medium transition-colors ${
              synth.isPlaying
                ? "bg-[var(--dev-signal)] text-white"
                : "bg-[var(--device-ink)] text-[var(--device-bg)]"
            }`}
          >
            {synth.isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {synth.isPlaying ? "Стоп" : "Слушать поток"}
          </button>

          <button
            type="button"
            onClick={() => {
              synth.setIsPlaying(false);
              synth.setStepIndex(0);
              synth.setActiveDigit(null);
            }}
            aria-label="Сброс позиции синтезатора"
            className="rounded-[4px] border border-[var(--device-rule)] bg-[var(--device-well)] p-1.5 text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Селектор ряда и ручки BPM / громкости */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--device-rule)] bg-[var(--device-elevated)]/45 px-5 py-2.5">
        <div className="flex flex-wrap items-center gap-1.5">
          {STREAM_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              type="button"
              onClick={() => {
                setStreamKey(opt.key);
                synth.setStepIndex(0);
              }}
              className={`rounded-[3px] border px-2.5 py-1 font-mono text-[11px] transition-colors ${
                streamKey === opt.key
                  ? "border-[var(--dev-signal)] bg-[var(--device-well)] font-medium text-[var(--device-ink)]"
                  : "border-transparent text-[var(--device-ink-2)] hover:text-[var(--device-ink)]"
              }`}
            >
              {opt.label}
            </button>
          ))}
          {externalStream && (
            <span className="inline-flex items-center overflow-hidden rounded-[3px] border border-[var(--dev-signal)]">
              <button
                type="button"
                onClick={() => {
                  setStreamKey("custom");
                  synth.setStepIndex(0);
                }}
                className={`px-2.5 py-1 font-mono text-[11px] ${
                  streamKey === "custom"
                    ? "bg-[var(--dev-signal)] text-white"
                    : "text-[var(--dev-signal)]"
                }`}
              >
                Из песочницы: {externalStream.label}
              </button>
              {onClearExternalStream && (
                <button
                  type="button"
                  aria-label="Убрать пользовательский поток"
                  onClick={() => {
                    onClearExternalStream();
                    setStreamKey("true_pi");
                    synth.setStepIndex(0);
                  }}
                  className="bg-[var(--device-well)] px-1.5 py-1 text-[var(--device-ink-muted)] hover:text-[var(--device-ink)]"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-5 font-sans text-xs text-[var(--device-ink-2)]">
          <label className="flex items-center gap-2.5">
            <span className="font-mono text-[11px] tabular-nums">BPM {synth.bpm}</span>
            <input
              type="range"
              min={90}
              max={360}
              step={10}
              value={synth.bpm}
              onChange={(e) => synth.setBpm(Number(e.target.value))}
              className="instrument-slider w-20"
            />
          </label>
          <label className="flex items-center gap-2">
            <Volume2 className="h-3.5 w-3.5 text-[var(--device-ink-muted)]" />
            <input
              type="range"
              min={0}
              max={0.8}
              step={0.05}
              value={synth.volume}
              onChange={(e) => synth.setVolume(Number(e.target.value))}
              className="instrument-slider w-16"
            />
          </label>
        </div>
      </div>

      {/* Утопленный OLED-экран ленты цифр во всю ширину */}
      <div className="hardware-well tape-edge-fade mx-5 mt-4 overflow-hidden rounded-[4px] py-2">
        <div className="flex items-center justify-center gap-1 font-mono text-xs tabular-nums">
          {TAPE_OFFSETS.map((offset) => {
            const idx = (centerIdx + offset + len) % len;
            const d = currentStream.digits[idx] ?? 0;
            const isCurrent = offset === 0;
            const isTrig = currentStream.mask ? currentStream.mask[idx] === 1 : false;
            return (
              <span
                key={offset}
                className={`inline-flex h-6 w-[26px] shrink-0 items-center justify-center rounded-[2px] transition-colors ${
                  isCurrent
                    ? "border border-[var(--dev-signal)] bg-[#DF6244]/15 font-semibold text-[var(--device-ink)]"
                    : isTrig
                      ? "font-semibold text-[var(--dev-signal)]"
                      : Math.abs(offset) <= 4
                        ? "text-[var(--device-ink-2)]"
                        : "text-[var(--device-ink-muted)]"
                }`}
              >
                {d === 10 ? ";" : d}
              </span>
            );
          })}
        </div>
      </div>

      {/* Клавиши слоновой кости с объёмной фаской внутри тёмного лотка */}
      <div className="p-5 pt-4">
        <div className="hardware-well grid grid-cols-5 gap-2 rounded-[6px] p-2.5 sm:grid-cols-10">
          {Array.from({ length: 10 }, (_, digit) => {
            const active = synth.activeDigit === digit;
            return (
              <button
                key={digit}
                type="button"
                data-testid={`piano-key-${digit}`}
                onClick={() => synth.playNote(digit, false)}
                style={{
                  background: active
                    ? "linear-gradient(180deg, #E2DDD2 0%, #D5CFC2 100%)"
                    : "linear-gradient(180deg, #F6F3EC 0%, #EAE5DA 100%)",
                  boxShadow: active
                    ? "inset 0 2px 5px rgba(24, 22, 20, 0.28)"
                    : "inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 1px 2px rgba(0, 0, 0, 0.45)",
                }}
                className={`flex h-[132px] flex-col items-center justify-between rounded-[4px] border border-[#ABA395] py-3 transition-transform duration-75 select-none ${
                  active
                    ? "translate-y-[3px] border-b-[2px] border-b-[#9E9688]"
                    : "border-b-[5px] border-b-[#B8B1A3] hover:brightness-[1.02]"
                }`}
              >
                {/* Круглое отверстие LED-индикатора */}
                <span
                  className="h-[6px] w-[6px] rounded-full border border-[#181614]/20 transition-all"
                  style={{
                    backgroundColor: active ? "#DF6244" : "#8E877C",
                    boxShadow: active ? "0 0 6px rgba(223, 98, 68, 0.85)" : "inset 0 1px 1px rgba(0,0,0,0.4)",
                  }}
                />

                <div className="flex flex-col items-center gap-0.5">
                  <span className="font-mono text-[10px] text-[#6E665C]">{noteNames[digit]}</span>
                  <span
                    className={`font-mono text-lg font-semibold tabular-nums ${
                      active ? "text-[#BF4328]" : "text-[#181614]"
                    }`}
                  >
                    {digit}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
