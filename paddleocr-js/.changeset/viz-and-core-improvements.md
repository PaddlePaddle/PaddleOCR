---
"@paddleocr/paddleocr-js": minor
---

### Visualization module improvements

- Refined `draw-boxes` panel rendering with deterministic color assignment
- Improved font loading lifecycle in `font.ts`

### Core SDK refinements

- Improved type exports and internal module organization across models, pipelines, resources, runtime, platform, and worker layers
- Updated test suite for better coverage of cache, models, pipeline core/shared, runtime, worker-backed, and worker-client modules

### Documentation accuracy fixes

- Moved visualization section before API reference in SDK READMEs (EN/CN) for better reading flow
- Added `deterministicColor` usage description to visualization docs
- Added `src/viz`, `src/types`, `src/utils` to package layout in architecture docs and SDK READMEs
- Fixed ESLint rule level description: test files use `recommendedTypeChecked`, not `strictTypeChecked`
- Removed non-existent `index.umd.js` from build output list; added `viz.mjs`/`viz.cjs`
- Corrected worker WASM inflation size from ~78 MB to ~50 MB
