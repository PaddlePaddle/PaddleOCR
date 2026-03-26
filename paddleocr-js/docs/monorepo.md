# Monorepo conventions

## Command execution

When you only want one workspace, use root-level workspace commands with explicit paths:

```bash
npm run build --workspace packages/core
npm run dev --workspace apps/demo
```

(You can also use workspace package names where unambiguous, e.g. `npm run dev --workspace demo`.)

## Workspace roles

- `packages/*`: reusable packages; the SDK lives under `packages/core` but keeps the **npm package name** `paddleocr-js`
- `apps/*`: private applications such as demos (`apps/demo`); not published to npm as products

## Versioning and naming

- **Directory:** `packages/core` — SDK source and publish manifest for the public package
- **npm package name:** `paddleocr-js` — what consumers `npm install` and import in code
- **Directory:** `apps/demo` — private demo, not an npm release target
- future publishable packages should use [Changesets](https://github.com/changesets/changesets)

## Linting and tests

- `packages/**` are linted with browser-oriented globals for SDK source files
- `apps/**` are linted with browser-oriented globals
- test files and config files are allowed to use both Node and browser globals where needed
