import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  resolve: {
    alias:
      command === "serve"
        ? {
            "@paddleocr/paddleocr-js/viz": resolve(
              __dirname,
              "../../packages/core/src/viz/index.ts"
            ),
            "@paddleocr/paddleocr-js": resolve(__dirname, "../../packages/core/src/index.ts")
          }
        : {}
  },
  worker: {
    format: "es"
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless"
    }
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless"
    }
  }
}));
