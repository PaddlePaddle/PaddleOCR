# paddleocr-js TypeScript Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate paddleocr-js from plain JavaScript to TypeScript with a real Vite library build producing CJS/ESM/UMD outputs.

**Architecture:** Bottom-up migration — infrastructure first, then type foundations, then source files in dependency order (leaves → root), then tests, finally verification. Each task produces a commit.

**Tech Stack:** TypeScript 5.8+, Vite 6 (library mode), vite-plugin-dts 4, typescript-eslint

**Design Spec:** `docs/superpowers/specs/2026-03-30-typescript-migration-design.md`

---

## File Structure

### New Files to Create

```
packages/core/
├── tsconfig.json               # Main TS config (strict)
├── tsconfig.test.json          # Test TS config (extends main)
├── vite.config.ts              # Vite library build config
└── src/types/
    ├── clipper-lib.d.ts        # Type declarations for clipper-lib
    ├── opencv.d.ts             # Type declarations for @techstark/opencv-js
    └── index.ts                # Public API types, re-exported from src/index.ts

tsconfig.json                   # Root IDE reference (project references)
```

### Files to Modify

```
package.json                    # Root: add engines, keywords, typecheck script, lint-staged update, new devDeps
packages/core/package.json      # Core: exports, build script, keywords, engines, devDeps
apps/demo/package.json          # Demo: keywords, engines, dependency protocol
eslint.config.js                # TS-aware ESLint config
vitest.config.js                # Coverage pattern update to .ts
```

### Files to Rename (28 source + 29 test)

All `packages/core/src/**/*.js` → `.ts`
All `packages/core/test/**/*.js` → `.ts`

---

## Task 1: Install Dependencies & Create TypeScript Configs

**Files:**
- Create: `packages/core/tsconfig.json`
- Create: `packages/core/tsconfig.test.json`
- Create: `tsconfig.json` (root)
- Modify: `packages/core/package.json` (add devDependencies)
- Modify: `package.json` (add devDependencies)

- [ ] **Step 1: Add devDependencies to packages/core**

```bash
cd paddleocr-js && npm install --save-dev --workspace packages/core typescript@^5.8 vite@^6 vite-plugin-dts@^4 @types/js-yaml@^4
```

- [ ] **Step 2: Add typescript-eslint to root devDependencies**

```bash
cd paddleocr-js && npm install --save-dev typescript-eslint
```

- [ ] **Step 3: Create `packages/core/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable", "WebWorker"],
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "test"]
}
```

- [ ] **Step 4: Create `packages/core/tsconfig.test.json`**

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "rootDir": ".",
    "declaration": false,
    "declarationMap": false
  },
  "include": ["src", "test"]
}
```

- [ ] **Step 5: Create root `tsconfig.json`**

```json
{
  "files": [],
  "references": [
    { "path": "packages/core" }
  ]
}
```

- [ ] **Step 6: Verify TypeScript is installed**

Run: `cd paddleocr-js && npx tsc --version`
Expected: `Version 5.8.x`

- [ ] **Step 7: Commit**

```bash
git add -A paddleocr-js
git commit -m "chore(paddleocr-js): add TypeScript configs and dependencies"
```

---

## Task 2: Vite Library Build Configuration

**Files:**
- Create: `packages/core/vite.config.ts`
- Modify: `packages/core/package.json` (scripts)

- [ ] **Step 1: Create `packages/core/vite.config.ts`**

```typescript
import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'

export default defineConfig({
  plugins: [
    dts({
      rollupTypes: true,
    }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'paddleocr',
      formats: ['es', 'cjs', 'umd'],
      fileName: (format) => {
        if (format === 'es') return 'index.mjs'
        if (format === 'cjs') return 'index.cjs'
        return 'index.umd.js'
      },
    },
    rollupOptions: {
      external: [
        'onnxruntime-web',
        '@techstark/opencv-js',
        'clipper-lib',
        'js-yaml',
      ],
      output: {
        globals: {
          'onnxruntime-web': 'ort',
          '@techstark/opencv-js': 'cv',
          'clipper-lib': 'ClipperLib',
          'js-yaml': 'jsyaml',
        },
      },
    },
    sourcemap: true,
    minify: false,
  },
})
```

- [ ] **Step 2: Update `packages/core/package.json` scripts**

Replace the existing `scripts` section:

```json
{
  "scripts": {
    "build": "vite build",
    "typecheck": "tsc --noEmit"
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add -A paddleocr-js
git commit -m "chore(paddleocr-js): add Vite library build configuration"
```

---

## Task 3: Update All package.json Files

**Files:**
- Modify: `packages/core/package.json`
- Modify: `apps/demo/package.json`
- Modify: `package.json` (root)

- [ ] **Step 1: Update `packages/core/package.json`**

Update/add these fields (keep existing `name`, `version`, `type`, `dependencies`):

```json
{
  "description": "Browser-based OCR SDK powered by PaddleOCR, ONNX Runtime Web and OpenCV.js",
  "main": "./dist/index.cjs",
  "module": "./dist/index.mjs",
  "browser": "./dist/index.umd.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    }
  },
  "files": ["dist", "README.md"],
  "sideEffects": false,
  "engines": {
    "node": ">=18"
  },
  "keywords": [
    "ocr",
    "paddleocr",
    "paddle",
    "text-recognition",
    "text-detection",
    "onnx",
    "onnxruntime",
    "opencv",
    "browser-ocr",
    "wasm",
    "webgpu",
    "deep-learning",
    "machine-learning",
    "computer-vision"
  ]
}
```

Remove the old `"main": "./src/index.js"` and `"module": "./src/index.js"` and `"exports": { ".": "./src/index.js" }` fields.

- [ ] **Step 2: Update `apps/demo/package.json`**

Add `keywords`, `engines`, and change the core dependency:

```json
{
  "keywords": ["paddleocr", "ocr-demo", "browser-ocr", "vite"],
  "engines": { "node": ">=18" }
}
```

Change `"paddleocr-js": "file:../../packages/core"` to `"paddleocr-js": "*"`.

- [ ] **Step 3: Update root `package.json`**

Add `keywords`, `engines`, `typecheck` script, update `lint-staged`:

```json
{
  "keywords": ["paddleocr", "ocr", "monorepo", "browser-ocr", "typescript"],
  "engines": { "node": ">=18" }
}
```

Add to scripts:
```json
{
  "typecheck": "npm run typecheck --workspaces --if-present"
}
```

Update lint-staged:
```json
{
  "lint-staged": {
    "*.{js,ts}": ["eslint --fix", "prettier --write"],
    "*.{json,md,html,css,yaml,yml}": ["prettier --write"]
  }
}
```

- [ ] **Step 4: Run `npm install` to update lockfile**

```bash
cd paddleocr-js && npm install
```

- [ ] **Step 5: Commit**

```bash
git add -A paddleocr-js
git commit -m "chore(paddleocr-js): update package.json exports, keywords, and engines"
```

---

## Task 4: Update ESLint Configuration

**Files:**
- Modify: `eslint.config.js`

- [ ] **Step 1: Rewrite `eslint.config.js`**

Replace the entire file with:

```javascript
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
  {
    ignores: ["**/dist", "**/node_modules", "**/coverage", "**/.cache"],
  },
  eslint.configs.recommended,
  {
    files: ["packages/**/*.ts"],
    extends: [...tseslint.configs.strictTypeChecked],
    languageOptions: {
      globals: { ...globals.browser },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["apps/**/*.js", "*.config.{js,ts}", "packages/**/*.config.*"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
  },
);
```

- [ ] **Step 2: Verify ESLint loads without errors**

Run: `cd paddleocr-js && npx eslint --print-config packages/core/src/index.js`
Expected: Outputs a JSON config object without errors (the file is still .js at this point, so it may not match any rule set — that's fine, just verify no crash).

- [ ] **Step 3: Commit**

```bash
git add -A paddleocr-js
git commit -m "chore(paddleocr-js): update ESLint config for TypeScript support"
```

---

## Task 5: Update Vitest Configuration

**Files:**
- Modify: `vitest.config.js`

- [ ] **Step 1: Update `vitest.config.js`**

Change the coverage include pattern from `.js` to `.ts`:

```javascript
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: ["packages/core/src/**/*.ts"],
    },
  },
});
```

- [ ] **Step 2: Commit**

```bash
git add -A paddleocr-js
git commit -m "chore(paddleocr-js): update Vitest coverage pattern for TypeScript"
```

---

## Task 6: Create Third-Party Type Declarations

**Files:**
- Create: `packages/core/src/types/clipper-lib.d.ts`
- Create: `packages/core/src/types/opencv.d.ts`

- [ ] **Step 1: Create `packages/core/src/types/clipper-lib.d.ts`**

Read the actual usage of `clipper-lib` in `packages/core/src/models/common.js` to understand which APIs are used, then write minimal type declarations covering exactly those usages. The library provides `ClipperLib.Clipper`, `ClipperLib.Paths`, `ClipperLib.ClipType`, `ClipperLib.PolyType`, `ClipperLib.PolyFillType`, `ClipperLib.ClipperOffset`, `ClipperLib.JoinType`, `ClipperLib.EndType` — declare only what's actually imported/used.

- [ ] **Step 2: Create `packages/core/src/types/opencv.d.ts`**

Read the actual usage of `@techstark/opencv-js` across all source files (primarily `src/runtime/opencv.js`, `src/models/det.js`, `src/models/common.js`, `src/platform/browser.js`, `src/platform/worker.js`) to understand which OpenCV APIs are used. Write minimal type declarations covering: `cv.Mat`, `cv.MatVector`, `cv.imread`, `cv.resize`, `cv.cvtColor`, `cv.threshold`, `cv.findContours`, `cv.minAreaRect`, `cv.Size`, `cv.Scalar`, color conversion constants, etc. — only what's actually used.

- [ ] **Step 3: Verify the declaration files are syntactically valid**

Run: `cd paddleocr-js && npx tsc --noEmit --project packages/core/tsconfig.json 2>&1 | head -5`
Expected: May show errors about .js files not being found (since source is still .js) — that's fine at this stage. The key is that the .d.ts files themselves don't have syntax errors.

- [ ] **Step 4: Commit**

```bash
git add -A paddleocr-js
git commit -m "chore(paddleocr-js): add type declarations for clipper-lib and opencv"
```

---

## Task 7: Convert Utility & Foundation Modules

These are leaf modules with no internal dependencies. Convert bottom-up.

**Files:**
- Rename & type: `src/utils/common.js` → `src/utils/common.ts`
- Rename & type: `src/worker/protocol.js` → `src/worker/protocol.ts`
- Rename & type: `src/pipelines/ocr/default-config.js` → `src/pipelines/ocr/default-config.ts`
- Rename & type: `src/pipelines/ocr/runtime-params.js` → `src/pipelines/ocr/runtime-params.ts`

- [ ] **Step 1: Convert `src/utils/common.js` → `src/utils/common.ts`**

Use `git mv` to rename. Then:
- Add type annotations to all function parameters and return types
- Replace `import ... from '...'` removing `.js` extensions
- Use `import type` for type-only imports (enforced by `verbatimModuleSyntax`)

- [ ] **Step 2: Convert `src/worker/protocol.js` → `src/worker/protocol.ts`**

Use `git mv` to rename. This file defines the worker message protocol — convert to TypeScript with explicit interfaces for each message type (request, response, error). These interfaces will be used by `worker/client.ts` and `worker/entry.ts` later.

- [ ] **Step 3: Convert `src/pipelines/ocr/default-config.js` → `src/pipelines/ocr/default-config.ts`**

Use `git mv` to rename. This exports a YAML string constant — straightforward conversion, just add `as const` or explicit `string` type.

- [ ] **Step 4: Convert `src/pipelines/ocr/runtime-params.js` → `src/pipelines/ocr/runtime-params.ts`**

Use `git mv` to rename. Add parameter/return types to the merge function. Define an interface for the runtime params object.

- [ ] **Step 5: Verify no syntax errors in converted files**

Run: `cd paddleocr-js && npx tsc --noEmit --project packages/core/tsconfig.json 2>&1 | grep -c 'error TS'`
Expected: Errors exist (other files still .js) but converted files should not have errors. Check manually for errors in the 4 converted files.

- [ ] **Step 6: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert utility & foundation modules to TypeScript"
```

---

## Task 8: Convert Runtime Modules

**Files:**
- Rename & type: `src/runtime/opencv.js` → `src/runtime/opencv.ts`
- Rename & type: `src/runtime/ort.js` → `src/runtime/ort.ts`
- Rename & type: `src/runtime/index.js` → `src/runtime/index.ts`

- [ ] **Step 1: Convert `src/runtime/opencv.js` → `src/runtime/opencv.ts`**

Use `git mv`. This handles lazy-loading of `@techstark/opencv-js`. Type the module-level state and the init function. Use the OpenCV types from `src/types/opencv.d.ts`.

- [ ] **Step 2: Convert `src/runtime/ort.js` → `src/runtime/ort.ts`**

Use `git mv`. This handles dynamic import of `onnxruntime-web`, WebGPU probing, ONNX session creation, and WASM env config. Use types from `onnxruntime-web` (ships its own `.d.ts`). Type all exported functions.

- [ ] **Step 3: Convert `src/runtime/index.js` → `src/runtime/index.ts`**

Use `git mv`. This is a re-export barrel — straightforward.

- [ ] **Step 4: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert runtime modules to TypeScript"
```

---

## Task 9: Convert Resource Modules

**Files:**
- Rename & type: `src/resources/tar.js` → `src/resources/tar.ts`
- Rename & type: `src/resources/cache.js` → `src/resources/cache.ts`
- Rename & type: `src/resources/registry.js` → `src/resources/registry.ts`
- Rename & type: `src/resources/standard-model.js` → `src/resources/standard-model.ts`
- Rename & type: `src/resources/index.js` → `src/resources/index.ts`

- [ ] **Step 1: Convert `src/resources/tar.js` → `src/resources/tar.ts`**

Use `git mv`. Type the tar parsing functions — input is `ArrayBuffer`/`Uint8Array`, output is parsed file entries. Define a `TarEntry` interface.

- [ ] **Step 2: Convert `src/resources/cache.js` → `src/resources/cache.ts`**

Use `git mv`. Type the cache interface (Cache API + memory fallback). Define interfaces for the cache abstraction.

- [ ] **Step 3: Convert `src/resources/registry.js` → `src/resources/registry.ts`**

Use `git mv`. This contains `DEFAULT_MODEL_ASSETS` — type the asset registry data structure. Define `ModelAssetEntry` and related interfaces.

- [ ] **Step 4: Convert `src/resources/standard-model.js` → `src/resources/standard-model.ts`**

Use `git mv`. Type `loadStandardModelAsset` and related functions.

- [ ] **Step 5: Convert `src/resources/index.js` → `src/resources/index.ts`**

Use `git mv`. Re-export barrel.

- [ ] **Step 6: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert resource modules to TypeScript"
```

---

## Task 10: Convert Model Modules

**Files:**
- Rename & type: `src/models/common.js` → `src/models/common.ts`
- Rename & type: `src/models/det.js` → `src/models/det.ts`
- Rename & type: `src/models/rec.js` → `src/models/rec.ts`
- Rename & type: `src/models/index.js` → `src/models/index.ts`

- [ ] **Step 1: Convert `src/models/common.js` → `src/models/common.ts`**

Use `git mv`. This contains YAML helpers, Clipper polygon operations, geometry utils, BGR→CHW conversion. Type all functions using `clipper-lib` types from `src/types/clipper-lib.d.ts` and OpenCV types from `src/types/opencv.d.ts`. Define geometric interfaces (`Point`, `Polygon`, `BoundingBox`, etc.) if not already defined.

- [ ] **Step 2: Convert `src/models/det.js` → `src/models/det.ts`**

Use `git mv`. Type the detection ONNX session creation, DB postprocessing, and `cropByPoly`. Uses `onnxruntime-web` types (InferenceSession, Tensor) and OpenCV types.

- [ ] **Step 3: Convert `src/models/rec.js` → `src/models/rec.ts`**

Use `git mv`. Type the recognition batching and CTC decode. Uses `onnxruntime-web` types.

- [ ] **Step 4: Convert `src/models/index.js` → `src/models/index.ts`**

Use `git mv`. Re-export barrel.

- [ ] **Step 5: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert model modules to TypeScript"
```

---

## Task 11: Convert Platform Modules

**Files:**
- Rename & type: `src/platform/browser.js` → `src/platform/browser.ts`
- Rename & type: `src/platform/worker.js` → `src/platform/worker.ts`

- [ ] **Step 1: Convert `src/platform/browser.js` → `src/platform/browser.ts`**

Use `git mv`. Type `ensureServedFromHttp`, `sourceToMat` (accepts various image source types → OpenCV Mat), `sourceToWorkerPayload`. Define a union type for accepted image sources (e.g., `HTMLImageElement | HTMLCanvasElement | ImageBitmap | string`).

- [ ] **Step 2: Convert `src/platform/worker.js` → `src/platform/worker.ts`**

Use `git mv`. Type `sourcePayloadToMat` (converts `ImageBitmap` to Mat inside worker context).

- [ ] **Step 3: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert platform modules to TypeScript"
```

---

## Task 12: Convert Worker Modules

**Files:**
- Rename & type: `src/worker/client.js` → `src/worker/client.ts`
- Rename & type: `src/worker/entry.js` → `src/worker/entry.ts`

- [ ] **Step 1: Convert `src/worker/client.js` → `src/worker/client.ts`**

Use `git mv`. Type `WorkerTransportClient` class — uses message types from `worker/protocol.ts` (already converted in Task 7).

- [ ] **Step 2: Convert `src/worker/entry.js` → `src/worker/entry.ts`**

Use `git mv`. Type `attachWorkerMessageHandler` — the worker-scope message handler.

- [ ] **Step 3: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert worker modules to TypeScript"
```

---

## Task 13: Convert Pipeline Modules

**Files:**
- Rename & type: `src/pipelines/ocr/config.js` → `src/pipelines/ocr/config.ts`
- Rename & type: `src/pipelines/ocr/shared.js` → `src/pipelines/ocr/shared.ts`
- Rename & type: `src/pipelines/ocr/core.js` → `src/pipelines/ocr/core.ts`
- Rename & type: `src/pipelines/ocr/worker-backed.js` → `src/pipelines/ocr/worker-backed.ts`
- Rename & type: `src/pipelines/ocr/worker-entry.js` → `src/pipelines/ocr/worker-entry.ts`
- Rename & type: `src/pipelines/ocr/index.js` → `src/pipelines/ocr/index.ts`
- Rename & type: `src/pipelines/index.js` → `src/pipelines/index.ts`

- [ ] **Step 1: Convert `src/pipelines/ocr/config.js` → `src/pipelines/ocr/config.ts`**

Use `git mv`. Type `normalizeOcrPipelineConfig` and `parseOcrPipelineConfigText`. Define `OcrPipelineConfig` interface representing the parsed YAML config structure.

- [ ] **Step 2: Convert `src/pipelines/ocr/shared.js` → `src/pipelines/ocr/shared.ts`**

Use `git mv`. Type model selection logic, `resolvePaddleOCROptions`, worker options. Define `PaddleOCROptions`, `PaddleOCRCreateOptions`, and related interfaces.

- [ ] **Step 3: Convert `src/pipelines/ocr/core.js` → `src/pipelines/ocr/core.ts`**

Use `git mv`. Type `OcrPipelineRunner` class — the main OCR engine. This is the largest file: `init()`, `predict()`, `dispose()`. Define `OcrPredictResult` (or similar) for the prediction output.

- [ ] **Step 4: Convert `src/pipelines/ocr/worker-backed.js` → `src/pipelines/ocr/worker-backed.ts`**

Use `git mv`. Type `WorkerBackedPaddleOCR` class and `createWorkerBackedPaddleOCR` factory.

- [ ] **Step 5: Convert `src/pipelines/ocr/worker-entry.js` → `src/pipelines/ocr/worker-entry.ts`**

Use `git mv`. Type the worker-side message handler that wraps `OcrPipelineRunner`.

- [ ] **Step 6: Convert `src/pipelines/ocr/index.js` → `src/pipelines/ocr/index.ts`**

Use `git mv`. Type the `PaddleOCR` class (public API) — `create()`, `fromPipelineConfig()` static methods.

- [ ] **Step 7: Convert `src/pipelines/index.js` → `src/pipelines/index.ts`**

Use `git mv`. Re-export barrel.

- [ ] **Step 8: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert pipeline modules to TypeScript"
```

---

## Task 14: Convert Entry Point & Create Public Types

**Files:**
- Rename & type: `src/index.js` → `src/index.ts`
- Create: `src/types/index.ts`

- [ ] **Step 1: Create `src/types/index.ts`**

Collect all public-facing types defined across the converted source files into a central types module. Re-export them so consumers can import types:

```typescript
// Re-export all public interfaces and types
export type { OcrPipelineConfig } from '../pipelines/ocr/config'
export type { PaddleOCROptions, PaddleOCRCreateOptions } from '../pipelines/ocr/shared'
export type { OcrPredictResult } from '../pipelines/ocr/core'
// ... add all public-facing types
```

The exact types depend on what was defined during Tasks 7-13. Gather all `export interface` and `export type` that should be part of the public API.

- [ ] **Step 2: Convert `src/index.js` → `src/index.ts`**

Use `git mv`. Update to re-export from typed modules. Also re-export public types from `./types`:

```typescript
export { PaddleOCR, normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from './pipelines'
export type { OcrPipelineConfig, PaddleOCROptions, OcrPredictResult /* ... */ } from './types'
```

- [ ] **Step 3: Run full typecheck**

Run: `cd paddleocr-js && npx tsc --noEmit --project packages/core/tsconfig.json`
Expected: 0 errors. If errors remain, fix them before proceeding.

- [ ] **Step 4: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert entry point and create public types module"
```

---

## Task 15: Convert Test Files

**Files:**
- Rename & type: `test/helpers/mock-ort-tensor.js` → `.ts`
- Rename & type: `test/tar-fixture.js` → `.ts`
- Rename & type: all 27 `test/*.test.js` → `.test.ts`

- [ ] **Step 1: Convert test helpers**

Use `git mv` to rename:
- `test/helpers/mock-ort-tensor.js` → `test/helpers/mock-ort-tensor.ts`
- `test/tar-fixture.js` → `test/tar-fixture.ts`

Add type annotations. These are test utilities — type the mock factories and fixture data.

- [ ] **Step 2: Batch rename all test files**

Use `git mv` for each:
```bash
cd paddleocr-js/packages/core
for f in test/*.test.js; do git mv "$f" "${f%.js}.ts"; done
```

- [ ] **Step 3: Add types to test files**

For each `.test.ts` file:
- Update import paths (remove `.js` extensions)
- Add type annotations where needed (mostly for mock objects and test fixtures)
- Use `import type` for type-only imports
- Vitest's `describe`, `it`, `expect`, `vi` are auto-typed — no changes needed there

Focus on files that have complex mocks or type-heavy assertions. Most test files will just need import path fixes and minor type annotations.

- [ ] **Step 4: Verify tests still pass**

Run: `cd paddleocr-js && npm test`
Expected: All existing tests pass. The test runner (Vitest) handles `.ts` files natively via esbuild.

- [ ] **Step 5: Commit**

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): convert test files to TypeScript"
```

---

## Task 16: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run typecheck**

Run: `cd paddleocr-js && npm run typecheck`
Expected: 0 errors

- [ ] **Step 2: Run tests**

Run: `cd paddleocr-js && npm test`
Expected: All tests pass

- [ ] **Step 3: Run lint**

Run: `cd paddleocr-js && npm run lint`
Expected: 0 errors (or only pre-existing warnings). Fix any new lint errors introduced by the migration.

- [ ] **Step 4: Run build**

Run: `cd paddleocr-js && npm run build:sdk`
Expected: Build succeeds. Verify output files exist:
```bash
ls -la paddleocr-js/packages/core/dist/
```
Should contain: `index.mjs`, `index.cjs`, `index.umd.js`, `index.d.ts`, and their `.map` files.

- [ ] **Step 5: Verify build output sanity**

Check that the built files are reasonable:
```bash
head -5 paddleocr-js/packages/core/dist/index.mjs
head -5 paddleocr-js/packages/core/dist/index.cjs
head -5 paddleocr-js/packages/core/dist/index.d.ts
```

The ESM file should have `export` statements, CJS should have `exports.`, and the `.d.ts` should have type declarations.

- [ ] **Step 6: Fix any issues found and commit**

If any step above failed, fix the issue and re-run. Then:

```bash
git add -A paddleocr-js
git commit -m "refactor(paddleocr-js): fix issues found during verification"
```

(Skip this commit if no fixes were needed.)

---

## Task 17: Update Demo App

**Files:**
- Modify: `apps/demo/src/main.js` (update imports if needed)
- Modify: `apps/demo/vite.config.js` (may need adjustment)

- [ ] **Step 1: Verify demo imports work with new build output**

The demo imports `paddleocr-js` which now resolves to built `dist/` files instead of raw `src/`. Check that `apps/demo/src/main.js` imports are compatible.

Run: `cd paddleocr-js && npm run build:demo`
Expected: Build succeeds.

- [ ] **Step 2: Fix demo if needed**

If the demo build fails, update imports in `apps/demo/src/main.js` to match the new public API exports from `dist/`.

- [ ] **Step 3: Verify demo dev server works**

Run: `cd paddleocr-js && npm run dev:demo`
Expected: Vite dev server starts without errors. (Stop with Ctrl+C after verifying.)

- [ ] **Step 4: Commit if changes were made**

```bash
git add -A paddleocr-js
git commit -m "fix(paddleocr-js): update demo app for TypeScript migration"
```

(Skip this commit if no changes were needed.)

---

## Verification Checklist

After all tasks are complete, confirm:

- [ ] `npm run typecheck` — passes with 0 errors
- [ ] `npm test` — all tests pass
- [ ] `npm run lint` — no errors
- [ ] `npm run build:sdk` — produces `dist/index.{mjs,cjs,umd.js,d.ts}`
- [ ] `npm run build:demo` — demo builds successfully
- [ ] No `.js` files remain in `packages/core/src/` or `packages/core/test/` (except any intentional `.d.ts`)
- [ ] `packages/core/package.json` has correct `main`, `module`, `browser`, `types`, `exports`
- [ ] All package.json files have `keywords` and `engines`
