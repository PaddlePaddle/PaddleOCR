# Development

## Install

```bash
npm install
```

## Common Commands

Repository-level commands:

```bash
npm run build
npm run lint
npm run test
npm run check
```

Current demo app:

```bash
npm run dev:ppocr
```

Single workspace examples:

```bash
npm run build --workspace paddleocr-js
npm run build --workspace ppocr_demo
```

## Testing Strategy

- unit tests for config parsing, registry behavior, and cache behavior
- lightweight jsdom checks for the demo shell
- no large real-model inference in CI by default
