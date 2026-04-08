---
phase: 04-pipeline-orchestration-validation
plan: 02
subsystem: testing
tags: [python, validation, json, paddleocr, paddlex, cli]

# Dependency graph
requires:
  - phase: 04-pipeline-orchestration-validation
    provides: "BoxSorter and PerspectiveCrop from Plan 01 define the pipeline that validation scripts will verify"
provides:
  - "generate_reference.py -- PaddleX OCR reference output generator"
  - "validate.py -- iOS vs reference comparison with exact polygon/text match and 1e-4 confidence tolerance"
  - "Shared JSON schema (polygon, text, confidence, box_count) for cross-platform validation"
  - "README.md documenting the complete validation workflow"
affects: [04-03-PLAN, phase-05, phase-06]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Python validation scripts with CLI interface", "JSON-based cross-platform comparison schema"]

key-files:
  created:
    - deploy/ios_demo/Validation/generate_reference.py
    - deploy/ios_demo/Validation/validate.py
    - deploy/ios_demo/Validation/README.md
    - deploy/ios_demo/Validation/.gitignore
    - deploy/ios_demo/Validation/test_images/.gitkeep
  modified: []

key-decisions:
  - "PaddleOCR import deferred inside generate_reference() so validate.py can be imported without paddleocr installed"
  - "Box extraction handles both PaddleOCR 3.x (rec_texts attribute) and legacy API (dict-based) for forward compatibility"
  - "Confidence tolerance set to 1e-4 (0.0001) per REQUIREMENTS.md to account for ARM vs x86 float differences"
  - "Exit codes 0/1/2 chosen for CI-friendly integration (pass/fail/error)"

patterns-established:
  - "JSON validation schema: {image, box_count, boxes: [{polygon, text, confidence}]}"
  - "Validation script pattern: generate reference -> export iOS output -> compare with tolerance"

requirements-completed: [VALID-01, VALID-02, PIPE-05]

# Metrics
duration: 4min
completed: 2026-04-07
---

# Phase 4 Plan 02: Validation Scripts Summary

**Python validation scripts comparing iOS OCR output against PaddleX reference with exact polygon/text match and 1e-4 confidence tolerance**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-07T09:37:00Z
- **Completed:** 2026-04-07T09:41:37Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- generate_reference.py runs PaddleOCR (PP-OCRv5 mobile models) on test images and exports per-image JSON with polygon coordinates (integer pairs), text strings, and float confidence scores
- validate.py compares iOS-exported JSON against reference JSON reporting per-image PASS/FAIL with detailed diffs for polygons (exact integer), text (exact string), and confidence (1e-4 tolerance)
- README.md documents the complete 4-step validation workflow (add images, generate reference, run iOS, validate)
- CI-friendly exit codes: 0=pass, 1=fail, 2=error

## Task Commits

Each task was committed atomically:

1. **Task 1: generate_reference.py** - `3cdbc29f2` (feat)
2. **Task 2: validate.py + README.md** - `b93ea9fea` (feat)

## Files Created/Modified
- `deploy/ios_demo/Validation/generate_reference.py` - Runs PaddleOCR on test images, exports reference JSON
- `deploy/ios_demo/Validation/validate.py` - Compares iOS output vs reference with tolerance rules
- `deploy/ios_demo/Validation/README.md` - Validation workflow documentation
- `deploy/ios_demo/Validation/.gitignore` - Excludes generated reference/ and ios_output/ directories
- `deploy/ios_demo/Validation/test_images/.gitkeep` - Placeholder for developer-provided test images

## Decisions Made
- PaddleOCR import deferred inside generate_reference() function body so validate.py can be imported standalone without paddleocr installed
- Box extraction handles both PaddleOCR 3.x attribute-based API and legacy dict-based API for forward compatibility
- Confidence tolerance of 1e-4 chosen per REQUIREMENTS.md specification to account for ARM vs x86 floating-point differences
- Exit codes 0/1/2 follow standard Unix conventions for CI integration

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Validation scripts ready for Plan 03 (OCREngine) to wire up ValidationExport JSON serializer in the iOS app
- JSON schema established and documented for cross-platform comparison
- test_images/ directory awaits developer-provided images for actual validation runs

## Self-Check: PASSED

All 5 created files verified on disk. Both task commits (3cdbc29f2, b93ea9fea) found in git log.

---
*Phase: 04-pipeline-orchestration-validation*
*Completed: 2026-04-07*
