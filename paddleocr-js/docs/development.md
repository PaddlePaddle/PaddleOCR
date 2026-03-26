# Development

All commands below assume your current working directory is the subproject root **`paddleocr-js/`** inside a clone of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Install

```bash
npm install
```

## Common commands

Root workspace scripts:

```bash
npm run build
npm run lint
npm run test
npm run check
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

## Testing strategy

- unit tests for config parsing, registry behavior, and cache behavior
- lightweight jsdom checks for the demo shell
- no large real-model inference in CI by default
