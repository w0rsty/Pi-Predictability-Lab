import { useCallback, useEffect, useRef, useState } from "react";

export type SynthScale = "pentatonic" | "minor" | "chromatic";

const SCALE_FREQS: Record<SynthScale, number[]> = {
  pentatonic: [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99, 880.0],
  minor: [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25],
  chromatic: [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0],
};

export const SCALE_NOTE_NAMES: Record<SynthScale, string[]> = {
  pentatonic: ["C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5", "A5"],
  minor: ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
  chromatic: ["C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4"],
};

export function usePiSynth() {
  const ctxRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const [scale, setScale] = useState<SynthScale>("pentatonic");
  const [bpm, setBpm] = useState<number>(190);
  const [volume, setVolume] = useState<number>(0.42);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [stepIndex, setStepIndex] = useState<number>(0);
  const [activeDigit, setActiveDigit] = useState<number | null>(null);
  const [isTriggerPulse, setIsTriggerPulse] = useState<boolean>(false);

  const ensureContext = useCallback(() => {
    if (!ctxRef.current) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      const master = ctx.createGain();
      master.gain.value = volume;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      master.connect(analyser);
      analyser.connect(ctx.destination);

      ctxRef.current = ctx;
      masterGainRef.current = master;
      analyserRef.current = analyser;
    }
    if (ctxRef.current.state === "suspended") {
      void ctxRef.current.resume();
    }
    return ctxRef.current;
  }, [volume]);

  useEffect(() => {
    if (masterGainRef.current && ctxRef.current) {
      masterGainRef.current.gain.setTargetAtTime(volume, ctxRef.current.currentTime, 0.02);
    }
  }, [volume]);

  const playNote = useCallback(
    (digit: number, isBackdoorTrigger = false) => {
      const safeDigit = Math.abs(digit) % 10;
      const ctx = ensureContext();
      const master = masterGainRef.current;
      if (!master) return;

      const now = ctx.currentTime;
      const freqs = SCALE_FREQS[scale];
      const baseFreq = freqs[safeDigit] ?? 440;

      const filter = ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(isBackdoorTrigger ? 4200 : 2400, now);

      const env = ctx.createGain();
      env.gain.setValueAtTime(0.0001, now);
      env.gain.linearRampToValueAtTime(0.32, now + 0.012);
      env.gain.exponentialRampToValueAtTime(0.0008, now + 0.28);

      const osc1 = ctx.createOscillator();
      osc1.type = "triangle";
      osc1.frequency.setValueAtTime(baseFreq, now);

      const osc2 = ctx.createOscillator();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(baseFreq * 2, now);
      const harmonicGain = ctx.createGain();
      harmonicGain.gain.value = 0.18;

      osc1.connect(filter);
      osc2.connect(harmonicGain);
      harmonicGain.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.3);
      osc2.stop(now + 0.3);

      if (isBackdoorTrigger) {
        const bellOsc = ctx.createOscillator();
        const bellGain = ctx.createGain();
        bellOsc.type = "sine";
        bellOsc.frequency.setValueAtTime(1046.5, now);
        bellGain.gain.setValueAtTime(0.0001, now);
        bellGain.gain.linearRampToValueAtTime(0.36, now + 0.006);
        bellGain.gain.exponentialRampToValueAtTime(0.0005, now + 0.42);
        bellOsc.connect(bellGain);
        bellGain.connect(master);
        bellOsc.start(now);
        bellOsc.stop(now + 0.44);
      }

      setActiveDigit(safeDigit);
      setIsTriggerPulse(isBackdoorTrigger);
    },
    [ensureContext, scale]
  );

  useEffect(() => {
    return () => {
      if (ctxRef.current) {
        void ctxRef.current.close().catch(() => {});
        ctxRef.current = null;
      }
    };
  }, []);

  return {
    scale,
    setScale,
    bpm,
    setBpm,
    volume,
    setVolume,
    isPlaying,
    setIsPlaying,
    stepIndex,
    setStepIndex,
    activeDigit,
    setActiveDigit,
    isTriggerPulse,
    playNote,
    analyserRef,
  };
}
