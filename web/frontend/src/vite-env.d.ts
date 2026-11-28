/// <reference types="vite/client" />

declare module "plotly.js-cartesian-dist-min" {
  import type Plotly from "plotly.js";
  const mod: typeof Plotly & { default?: typeof Plotly };
  export default mod;
  export * from "plotly.js";
}
