# paddleocr-js TypeScript 改造设计

> 日期: 2026-03-30
> 状态: 已批准

## 概述

将 `paddleocr-js` 子项目从纯 JavaScript ESM 改造为 TypeScript，引入真正的 library build 流程，完善 package.json 配置。不考虑后向兼容。

## 决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 构建工具 | Vite library mode + vite-plugin-dts | monorepo 已有 Vite；原生支持 CJS/ESM/UMD；配置量适中 |
| TS 改造范围 | 源码 + 测试转 TS，demo 保留 JS | demo 保留 JS 充当外部消费者验证场景 |
| TS 严格程度 | `strict: true` | 源码量可控（~20 文件），一步到位建立类型基线 |
| 改造策略 | 就地改造 | 先搭基础设施，逐步将 .js → .ts，Git rename 追踪好 |
| 后向兼容 | 不考虑 | 破坏性改造，一步到位 |

---

## 1. TypeScript 配置

### 1.1 `packages/core/tsconfig.json`

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

- `target: ES2022` — 匹配 `engines: node >= 18`
- `moduleResolution: bundler` — Vite 打包，不需 Node 解析规则
- `verbatimModuleSyntax: true` — 强制显式 `import type`
- `lib` 包含 `DOM` + `WebWorker` — 浏览器端 SDK

### 1.2 `packages/core/tsconfig.test.json`

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "rootDir": "."
  },
  "include": ["src", "test"]
}
```

### 1.3 根目录 `paddleocr-js/tsconfig.json`

```json
{
  "files": [],
  "references": [
    { "path": "packages/core" }
  ]
}
```

仅作为 IDE 引用入口。

---

## 2. Vite Library Build 配置

### 2.1 `packages/core/vite.config.ts`

```typescript
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
      entry: './src/index.ts',
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

- 依赖全部 external，不打包进产物
- UMD 全局名 `paddleocr`（`window.paddleocr`），小写命名空间避免与 `PaddleOCR` 类名冲突
- `rollupTypes: true` 合并所有类型声明为单个 `dist/index.d.ts`
- 不压缩，交给消费者的 bundler

### 2.2 新增 devDependencies（`packages/core`）

```json
{
  "devDependencies": {
    "@types/js-yaml": "^4",
    "typescript": "^5.8",
    "vite": "^6",
    "vite-plugin-dts": "^4"
  }
}
```

### 2.3 构建脚本

```json
{
  "scripts": {
    "build": "vite build",
    "typecheck": "tsc --noEmit"
  }
}
```

---

## 3. package.json 改造

### 3.1 `packages/core/package.json`

```json
{
  "name": "paddleocr-js",
  "version": "0.1.0",
  "description": "Browser-based OCR SDK powered by PaddleOCR, ONNX Runtime Web and OpenCV.js",
  "type": "module",
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

关键变更：
- `files` 从 `["src"]` 改为 `["dist"]` — 发布构建产物
- 完整三格式导出 + 类型声明
- 14 个关键词
- `engines: node >= 18`

### 3.2 `apps/demo/package.json`

追加：

```json
{
  "keywords": ["paddleocr", "ocr-demo", "browser-ocr", "vite"],
  "engines": { "node": ">=18" }
}
```

对 core 的依赖从 `"paddleocr-js": "file:../../packages/core"` 改为 `"paddleocr-js": "*"`（npm workspaces 会自动解析为本地包）。

### 3.3 根目录 `package.json`

追加：

```json
{
  "keywords": ["paddleocr", "ocr", "monorepo", "browser-ocr", "typescript"],
  "engines": { "node": ">=18" }
}
```

scripts 新增：

```json
{
  "typecheck": "npm run typecheck --workspaces --if-present"
}
```

---

## 4. ESLint + Vitest 配置更新

### 4.1 ESLint

新增根目录 devDependency：`typescript-eslint`

`eslint.config.js` 改造：

```javascript
import eslint from '@eslint/js'
import tseslint from 'typescript-eslint'
import globals from 'globals'

export default tseslint.config(
  {
    ignores: ['**/dist', '**/node_modules', '**/coverage', '**/.cache'],
  },
  eslint.configs.recommended,
  {
    files: ['packages/**/*.ts'],
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
    files: ['apps/**/*.js', '*.config.{js,ts}', 'packages/**/*.config.*'],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
  },
)
```

- `strictTypeChecked` 与 tsconfig `strict: true` 对齐
- `projectService: true` 自动发现 tsconfig
- JS 文件单独配置，不强加 TS 规则

### 4.2 Vitest

`vitest.config.js` 覆盖范围更新：

```javascript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      include: ['packages/core/src/**/*.ts'],
    },
  },
})
```

### 4.3 lint-staged

```json
{
  "lint-staged": {
    "*.{js,ts}": ["eslint --fix", "prettier --write"],
    "*.{json,md,html,css,yaml,yml}": ["prettier --write"]
  }
}
```

---

## 5. 源码 JS → TS 转换

### 5.1 文件重命名

所有 `packages/core/src/**/*.js` → `.ts`，所有 `packages/core/test/**/*.js` → `.test.ts`（含 helpers）。

### 5.2 类型补充策略

| 区域 | 处理方式 |
|------|----------|
| `clipper-lib` | `src/types/clipper-lib.d.ts` 手写声明 |
| `@techstark/opencv-js` | `src/types/opencv.d.ts` 手写声明 |
| `onnxruntime-web` | 自带类型，直接使用 |
| `js-yaml` | `@types/js-yaml` 加到 devDependencies |
| Worker 通信协议 | `worker/protocol.ts` 定义消息类型接口 |
| 公共 API 类型 | 从实现中提取为显式 `interface`/`type` 并导出 |

### 5.3 import 路径

统一移除 `.js` 扩展名（`moduleResolution: bundler` 下 Vite 正确解析）：

```typescript
// 之前
import { OcrPipelineRunner } from './core.js'
// 之后
import { OcrPipelineRunner } from './core'
```

### 5.4 新增 `src/types/` 目录

```
src/types/
├── clipper-lib.d.ts     // clipper-lib 类型声明
├── opencv.d.ts          // @techstark/opencv-js 类型声明
└── index.ts             // 公共类型集中导出
```

---

## 产出物概览

改造完成后 `packages/core` 的构建产物：

```
dist/
├── index.mjs        # ESM
├── index.cjs        # CommonJS
├── index.umd.js     # UMD (CDN)
├── index.d.ts       # 合并的类型声明
├── index.d.ts.map   # 声明映射
├── index.mjs.map    # ESM source map
├── index.cjs.map    # CJS source map
└── index.umd.js.map # UMD source map
```
