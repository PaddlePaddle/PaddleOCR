# 开发指南

[English](development.md) | 简体中文

## 安装依赖

```bash
npm install
```

## 常用命令

在 `paddleocr-js/` 根目录执行：

```bash
npm run build          # 先构建 SDK 再构建 demo（显式拓扑顺序）
npm run build:sdk      # 仅构建 SDK（packages/core）
npm run build:demo     # 仅构建 demo 应用（apps/demo）
npm run lint
npm run test
npm run typecheck      # 对所有 workspace 执行类型检查（core + demo）
npm run check          # format:check → lint → build:sdk → typecheck → test → build:demo
npm run clean          # 删除所有 dist/ 目录
```

Demo 应用（Vite 开发服务器）：

```bash
npm run dev:demo
```

单个 workspace 的示例：

```bash
npm run build --workspace packages/core
npm run build --workspace apps/demo
```

## TypeScript

SDK（`packages/core`）与 demo 应用（`apps/demo`）都使用 TypeScript 严格模式。`packages/**/src/` 与 `apps/**/src/` 下的源码使用 `typescript-eslint` 的 `strictTypeChecked` 规则集；`packages/**/test/` 下的测试使用较轻的 `recommendedTypeChecked`，并放宽了部分规则（例如 `no-unsafe-*` 与 `no-explicit-any`）。

`npm run typecheck` 会在所有 workspace 上执行 `tsc --noEmit`。demo 在 `tsconfig.json` 里通过 `paths` 直接引用 SDK 源码，因此类型检查并不严格依赖先执行 `build:sdk`。

## 构建

SDK 使用 Vite 的库模式构建（在 `packages/core` 中执行 `npm run build`）。`dist/` 下的构建产物包括：

- `index.mjs` — ESM 入口
- `index.d.ts` — 类型声明
- `viz.mjs` — 可视化子路径的 ESM 入口
- `assets/worker-entry-*.js` — 自包含的 Worker bundle（OpenCV.js + ORT JS 运行时）

自定义 Vite 插件 `libraryWorkerPlugin` 会对构建产物做后处理，以兼容 npm 分发场景：

1. 将 Worker 资源的绝对路径改写为相对路径，使文件相对于 SDK 模块路径解析，而不是站点根路径
2. 将 `new Worker(new URL(STRING, import.meta.url))` 拆成 URL 变量与 Worker 构造，便于下游打包工具复制 Worker 文件，并避免被当作二次打包目标
3. 移除 Vite 内联到 Worker 产物中的 base64 WASM 二进制；在 Worker 模式下，ORT 会在运行时通过 `ort.env.wasm.wasmPaths` 加载 WASM（由使用方配置，或回退到与安装 ORT 版本绑定的 CDN），这样可以显著减小 Worker 文件体积

开发阶段，demo 通过 Vite alias 直接引用 core 的 TypeScript 源码，从而获得更好的 HMR 体验；生产构建（`npm run build`）时，则通过 workspace 链接使用 SDK 预构建的 `dist/`，并依靠兼容下游打包器的 Worker URL 形式将 Worker 文件正确复制到 demo 产物中。

## 测试策略

- 配置解析与注册表行为的单元测试
- 面向浏览器平台辅助函数的轻量级 jsdom 测试
- CI 默认不运行大规模真实模型推理
