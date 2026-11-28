import { formatToken } from "../../utils/arenaStats";

interface ArenaKeypadProps {
  radix: number;
  disabled: boolean;
  revealed: boolean;
  target: number;
  humanChoice: number | null;
  aiChoice: number;
  onSelect: (token: number) => void;
}

export function ArenaKeypad({
  radix,
  disabled,
  revealed,
  target,
  humanChoice,
  aiChoice,
  onSelect,
}: ArenaKeypadProps) {
  const keys = Array.from({ length: radix }, (_, i) => i);

  return (
    <div
      className={`hardware-well grid gap-2 rounded-[6px] p-2.5 ${
        radix === 11 ? "grid-cols-4 sm:grid-cols-11" : "grid-cols-5 sm:grid-cols-10"
      }`}
    >
      {keys.map((tok) => {
        const isHuman = humanChoice === tok;
        const isTarget = revealed && target === tok;
        const isHumanWrong = revealed && isHuman && tok !== target;
        const isAiPred = revealed && aiChoice === tok;

        let ledColor = "#8E877C";
        let ledShadow = "inset 0 1px 1px rgba(0,0,0,0.4)";
        if (isTarget) {
          ledColor = "#2F7A4D";
          ledShadow = "0 0 6px rgba(47, 122, 77, 0.85)";
        } else if (isHumanWrong) {
          ledColor = "#BF4328";
          ledShadow = "0 0 6px rgba(191, 67, 40, 0.85)";
        } else if (isHuman) {
          ledColor = "#DF6244";
          ledShadow = "0 0 6px rgba(223, 98, 68, 0.85)";
        }

        const pressed = isHuman && revealed;
        return (
          <button
            key={tok}
            type="button"
            disabled={disabled}
            data-testid={`arena-key-${tok}`}
            onClick={() => onSelect(tok)}
            style={{
              background: pressed
                ? "linear-gradient(180deg, #E2DDD2 0%, #D5CFC2 100%)"
                : "linear-gradient(180deg, #F6F3EC 0%, #EAE5DA 100%)",
              boxShadow: pressed
                ? "inset 0 2px 5px rgba(24, 22, 20, 0.28)"
                : "inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 1px 2px rgba(0, 0, 0, 0.45)",
            }}
            className={`relative flex h-[78px] flex-col items-center justify-between rounded-[4px] border border-[#ABA395] py-2 transition-transform duration-75 select-none ${
              pressed
                ? "translate-y-[3px] border-b-[2px] border-b-[#9E9688]"
                : "border-b-[5px] border-b-[#B8B1A3] hover:brightness-[1.02]"
            } ${disabled && !revealed ? "opacity-60" : "cursor-pointer"}`}
          >
            <span
              className="h-[6px] w-[6px] rounded-full border border-[#181614]/20"
              style={{ backgroundColor: ledColor, boxShadow: ledShadow }}
            />

            <div className="flex flex-col items-center gap-0.5">
              <span
                className={`font-mono text-lg font-semibold tabular-nums ${
                  isTarget
                    ? "text-[#2F7A4D]"
                    : isHumanWrong
                      ? "text-[#BF4328]"
                      : "text-[#181614]"
                }`}
              >
                {formatToken(tok, radix)}
              </span>
              <span className="h-3 font-mono text-[9px] font-medium tracking-tight text-[#56514A]">
                {isAiPred ? "AI" : tok === 10 && radix === 11 ? "sep" : ""}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
