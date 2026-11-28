interface CornerReticlesProps {
  active?: boolean;
  variant?: "ink" | "device" | "signal";
  offset?: number;
}

export function CornerReticles({
  active = false,
  variant = "ink",
  offset = 4,
}: CornerReticlesProps) {
  const color =
    variant === "signal"
      ? "var(--sig-h1)"
      : variant === "device"
        ? "var(--device-ink)"
        : "var(--ink)";
  const visibility = active
    ? "opacity-100 scale-100"
    : "opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100";

  return (
    <span
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 transition-all duration-150 ${visibility}`}
    >
      <span
        className="absolute h-[7px] w-[7px] border-t-[1.5px] border-l-[1.5px]"
        style={{ top: offset, left: offset, borderColor: color }}
      />
      <span
        className="absolute h-[7px] w-[7px] border-t-[1.5px] border-r-[1.5px]"
        style={{ top: offset, right: offset, borderColor: color }}
      />
      <span
        className="absolute h-[7px] w-[7px] border-b-[1.5px] border-l-[1.5px]"
        style={{ bottom: offset, left: offset, borderColor: color }}
      />
      <span
        className="absolute h-[7px] w-[7px] border-b-[1.5px] border-r-[1.5px]"
        style={{ bottom: offset, right: offset, borderColor: color }}
      />
    </span>
  );
}
