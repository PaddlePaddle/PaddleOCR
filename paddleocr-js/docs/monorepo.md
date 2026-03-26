# Monorepo Conventions

## Command Execution

Run `npm install`, `npm run build`, `npm run test`, and `npm run check` from the repository root.

When you only want one workspace, prefer root-level workspace commands such as:

```bash
npm run build --workspace paddleocr-js
npm run dev --workspace ppocr_demo
```

## Workspace Roles

- `packages/*`: reusable packages and future publishable SDK modules
- `apps/*`: private applications such as demos, playgrounds, benchmarks, or docs sites

## Versioning

- `packages/paddleocr-js` is the publishable package
- `apps/ppocr_demo` is a private app and not treated as an npm release target
- Future publishable packages should use Changesets

## Linting and Tests

- `packages/**` are linted with browser-oriented globals for SDK source files
- `apps/**` are linted with browser-oriented globals
- test files and config files are allowed to use both Node and browser globals where needed
