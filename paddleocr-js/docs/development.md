# Development

## Install

```bash
npm install
```

## Common commands

Commands from the `paddleocr-js/` root:

```bash
npm run build
npm run lint
npm run test
npm run typecheck
npm run check          # runs lint + test + build
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

The SDK is written in TypeScript with strict mode enabled (`packages/core/tsconfig.json`). ESLint uses `typescript-eslint` with `strictTypeChecked`.

`npm run typecheck` runs `tsc --noEmit` to verify types without emitting files.

## Build

The SDK builds with Vite library mode (`npm run build` in `packages/core`). Output in `dist/`:

- `index.mjs` — ESM
- `index.cjs` — CJS
- `index.umd.js` — UMD
- `index.d.ts` — type declarations

## Testing strategy

- unit tests for config parsing, registry behavior, and cache behavior
- lightweight jsdom checks for the demo shell
- no large real-model inference in CI by default
