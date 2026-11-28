import type { Config, Layout } from "plotly.js";
import type { ThemeMode } from "../types/lab";

export function getPlotlyColors(theme: ThemeMode) {
  if (theme === "dark") {
    // Warm Basalt
    return {
      paper: "rgba(0,0,0,0)",
      plot: "rgba(0,0,0,0)",
      text: "#EAE4DA",
      muted: "#9C9488",
      grid: "rgba(234, 228, 218, 0.10)",
      zero: "rgba(234, 228, 218, 0.24)",
      h0: "#5E9EA8",
      h1: "#DF6244",
      ochre: "#D4A148",
      noise: "#787066",
      ink: "#EAE4DA",
      // Обратная совместимость ключей
      signal: "#DF6244",
      brass: "#D4A148",
    };
  }
  // Archival Desk (светлая тема по умолчанию)
  return {
    paper: "rgba(0,0,0,0)",
    plot: "rgba(0,0,0,0)",
    text: "#181614",
    muted: "#56514A",
    grid: "rgba(24, 22, 20, 0.11)",
    zero: "rgba(24, 22, 20, 0.26)",
    h0: "#235A63",
    h1: "#BF4328",
    ochre: "#B8862D",
    noise: "#8C857B",
    ink: "#181614",
    signal: "#BF4328",
    brass: "#B8862D",
  };
}

export function buildPlotlyLayout(
  theme: ThemeMode,
  overrides: Partial<Layout> = {}
): Partial<Layout> {
  const c = getPlotlyColors(theme);
  return {
    paper_bgcolor: c.paper,
    plot_bgcolor: c.plot,
    font: {
      family: '"Golos Text", system-ui, sans-serif',
      color: c.text,
      size: 12,
    },
    margin: { l: 56, r: 24, t: 32, b: 56 },
    hoverlabel: {
      bgcolor: theme === "dark" ? "#1C1A17" : "#F4F0E8",
      bordercolor: c.h1,
      font: {
        family: '"JetBrains Mono", monospace',
        color: c.text,
        size: 11,
      },
    },
    xaxis: {
      gridcolor: c.grid,
      zerolinecolor: c.zero,
      tickfont: { family: '"JetBrains Mono", monospace', size: 11, color: c.muted },
      ...overrides.xaxis,
    },
    yaxis: {
      gridcolor: c.grid,
      zerolinecolor: c.zero,
      tickfont: { family: '"JetBrains Mono", monospace', size: 11, color: c.muted },
      ...overrides.yaxis,
    },
    ...overrides,
  };
}

export const PLOTLY_CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};
