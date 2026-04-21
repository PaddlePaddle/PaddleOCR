# Development

English | [简体中文](development_cn.md)

## Install

```bash
npm install
```

## Common commands

Commands from the `paddleocr-js/` root:

```bash
npm run build          # build SDK then demo (explicit topological order)
npm run build:sdk      # build only the SDK (packages/core)
npm run build:demo     # build only the demo app (apps/demo)
npm run lint
npm run test
npm run typecheck      # typecheck all workspaces (core + demo)
npm run check          # format:check → lint → build:sdk → typecheck → test → build:demo
npm run clean          # remove all dist/ directories
```

Demo app (Vite dev server):

```bash
npm run dev:demo
```

Single-workspace examples:

```bash
npm run build --workspace packages/core
npm run build --workspace apps/demo
```

## TypeScript

Both the SDK (`packages/core`) and the demo app (`apps/demo`) are written in TypeScript with strict mode enabled. ESLint uses `typescript-eslint` with `strictTypeChecked` for source files under `packages/**/src/` and `apps/**/src/`. Test files under `packages/**/test/` use the lighter `recommendedTypeChecked` preset with relaxed rules (e.g. `no-unsafe-*` and `no-explicit-any` are disabled).

`npm run typecheck` runs `tsc --noEmit` across all workspaces. The demo typechecks directly against the SDK's source using `paths` mapping in its `tsconfig.json`, so it does not strictly require `build:sdk` to run first for typechecking.

## Build

The SDK builds with Vite library mode (`npm run build` in `packages/core`). Output in `dist/`:

- `index.mjs` — ESM entry
- `index.d.ts` — type declarations
- `viz.mjs` — ESM (viz subpath)
- `assets/worker-entry-*.js` — self-contained worker bundle (OpenCV.js + ORT JS runtime)

A custom Vite plugin (`libraryWorkerPlugin`) post-processes the build output for npm compatibility:

1. Rewrites absolute worker asset paths to relative, so the file resolves from the SDK module's location rather than the web origin.
2. Splits the `new Worker(new URL(STRING, import.meta.url))` pattern into a URL variable + Worker construction. This lets downstream bundlers' asset-URL plugins copy the worker file, while preventing their worker-detection plugins from trying to re-bundle it.
3. Strips base64-encoded WASM binaries that Vite inlines into the worker asset. In worker mode, ORT loads WASM at runtime via `ort.env.wasm.wasmPaths` (set by the consumer, or falling back to a CDN URL pinned to the installed ORT version). This significantly reduces the size of the worker file.

The demo app uses a Vite alias during development (`npm run dev`) to build directly from core's TypeScript source, enabling instant HMR. During production builds (`npm run build`), it consumes the SDK's pre-built `dist/` via workspace linking — the downstream-compatible worker URL pattern allows Vite to correctly copy the worker asset into the demo's output.

## Testing strategy

- unit tests for config parsing and registry behavior
- lightweight jsdom checks for browser platform helpers
- no large real-model inference in CI by default
