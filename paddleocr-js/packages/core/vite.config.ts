import { resolve, dirname, join } from "node:path";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

const require = createRequire(import.meta.url);
const ortPackageDir = dirname(require.resolve("onnxruntime-web"));
const ortVersion = JSON.parse(
  readFileSync(join(ortPackageDir, "..", "package.json"), "utf-8")
).version;

// Post-process library output for npm compatibility.
//
// 1. Rewrite Vite's worker URL references so downstream bundlers handle them
//    correctly (relative paths + split new Worker(new URL()) pattern).
//
// 2. Strip ORT's base64-encoded WASM data URIs from the worker asset.
//    Vite converts ORT's `new URL('./ort-wasm-*.wasm', import.meta.url)`
//    references into `data:application/wasm;base64,...` data URIs, inflating
//    the worker by ~50 MB.  The SDK provides a CDN-based fallback for
//    `ort.env.wasm.wasmPaths`, so the embedded binaries are never loaded.
//    Only `data:application/wasm` URIs are stripped — OpenCV.js embeds its
//    WASM via Emscripten as `data:application/octet-stream` and must be kept
//    since there is no equivalent fallback for it.
//
//    The stripping must run here (main build generateBundle), not in
//    worker.rollupOptions.plugins, because Vite injects the data URIs as a
//    post-processing step after the worker's own Rollup pipeline finishes.
function libraryWorkerPlugin() {
  const ortWasmDataUriPattern = /data:application\/wasm;base64,[A-Za-z0-9+/=]+/g;

  return {
    name: "paddleocr-library-worker",
    apply: "build",
    generateBundle(_, bundle) {
      for (const item of Object.values(bundle)) {
        if (item.type === "asset" && typeof item.source === "string") {
          item.source = item.source.replace(ortWasmDataUriPattern, "data:application/wasm;base64,");
          continue;
        }
        if (item.type !== "chunk") continue;
        if (!item.code.includes("/assets/")) continue;

        item.code = item.code.replace(
          /\/\*\s*@vite-ignore\s*\*\/\s*"\/(assets\/[^"]+)"/g,
          '"./$1"'
        );

        item.code = item.code.replace(
          /new\s+Worker\(\s*new\s+URL\(\s*"(\.\/(assets\/[^"]+))"\s*,\s*(import\.meta\.url)\s*\)\s*,\s*(\{[^}]*\})\s*\)/g,
          (_, fullPath, _assetPath, urlBase, workerOpts) => {
            return `(() => { const _w = new URL("${fullPath}", ${urlBase}); return new Worker(_w, ${workerOpts}); })()`;
          }
        );
      }
    }
  };
}

export default defineConfig({
  plugins: [
    dts({
      rollupTypes: false
    }),
    libraryWorkerPlugin()
  ],
  define: {
    __ORT_WASM_CDN_PREFIX__: JSON.stringify(
      `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`
    )
  },
  worker: {
    format: "es",
    rollupOptions: {
      output: {
        inlineDynamicImports: true
      }
    }
  },
  build: {
    lib: {
      entry: {
        index: resolve(__dirname, "src/index.ts"),
        viz: resolve(__dirname, "src/viz/index.ts")
      },
      name: "paddleocr",
      formats: ["es"],
      fileName: (_format, entryName) => `${entryName}.mjs`
    },
    rollupOptions: {
      external: ["onnxruntime-web", "@techstark/opencv-js", "clipper-lib", "js-yaml"],
      output: {
        globals: {
          "onnxruntime-web": "ort",
          "@techstark/opencv-js": "cv",
          "clipper-lib": "ClipperLib",
          "js-yaml": "jsyaml"
        }
      }
    },
    sourcemap: true,
    minify: false
  }
});
