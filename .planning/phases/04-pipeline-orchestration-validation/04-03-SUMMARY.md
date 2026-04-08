---
phase: 04-pipeline-orchestration-validation
plan: 03
subsystem: engine
tags: [ocr-pipeline, async, coreml, onnx-runtime, json-export, validation]

# Dependency graph
requires:
  - phase: 04-01
    provides: BoxSorter and PerspectiveCrop components for pipeline composition
  - phase: 02-detection
    provides: DetectionEngine with config-driven DB postprocessing
  - phase: 03-recognition
    provides: RecognitionEngine with config-driven CTC decoding
provides:
  - OCREngine end-to-end pipeline orchestrator (detect -> sort -> crop -> recognize)
  - OCRResult and OCRPipelineResult types with per-stage timing
  - ValidationExport JSON serializer matching Python validation schema
affects: [05-swiftui-integration, 04-04-validation-scripts]

# Tech tracking
tech-stack:
  added: []
  patterns: [pipeline-orchestrator-composition, json-validation-export]

key-files:
  created:
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/OCREngine.swift
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/ValidationExport.swift
  modified: []

key-decisions:
  - "OCREngine is a class (not actor) -- delegates concurrency to ORTSessionManager actor, consistent with DetectionEngine/RecognitionEngine pattern"
  - "Sequential recognition per box (not parallel) -- ORT sessions are not designed for concurrent use; actor ensures serial access"
  - "ValidationExport uses JSONSerialization with sortedKeys for deterministic output diffs against Python reference"

patterns-established:
  - "Pipeline orchestrator pattern: class composes sub-engines via init(sessionManager:), single async run() entry point"
  - "JSON validation export pattern: struct with static toJSON/writeJSON matching cross-platform schema"

requirements-completed: [PIPE-01, PIPE-04, CONF-02, CONF-03, CONF-04, CONF-05]

# Metrics
duration: 2min
completed: 2026-04-08
---

# Phase 4 Plan 3: Pipeline Orchestration Summary

**OCREngine pipeline orchestrator composing detect/sort/crop/recognize with async execution and JSON validation export**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-08T02:22:23Z
- **Completed:** 2026-04-08T02:24:39Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- OCREngine orchestrates the full detect -> sort -> crop -> recognize pipeline as a single async `run(CGImage)` call
- Per-stage timing breakdown (detection time, recognition time, total time) ready for Phase 5 UI display
- ValidationExport serializes OCRPipelineResult to JSON matching the Python validation schema for cross-platform comparison
- Zero hardcoded preprocessing/postprocessing parameters -- entire pipeline is config-driven via inference.yml

## Task Commits

Each task was committed atomically:

1. **Task 1: OCREngine -- end-to-end pipeline orchestrator with timing** - `a97b14a47` (feat)
2. **Task 2: ValidationExport -- JSON serializer for iOS pipeline output** - `1fa343917` (feat)

## Files Created/Modified
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/OCREngine.swift` - End-to-end OCR pipeline orchestrator with OCRResult, OCRPipelineResult types and per-stage timing
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/ValidationExport.swift` - JSON export for validation comparison against Python reference output

## Decisions Made
- OCREngine is a class (not actor) -- delegates concurrency to ORTSessionManager actor, consistent with DetectionEngine and RecognitionEngine pattern
- Sequential recognition per box (not parallel) -- ORT sessions are not designed for concurrent use; the ORTSessionManager actor ensures serial access
- ValidationExport uses JSONSerialization with .sortedKeys for deterministic output diffs against Python reference

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - both files are fully implemented with no placeholder data or TODO markers.

## Next Phase Readiness
- OCREngine provides the single entry point that Phase 5's SwiftUI UI will call via `try await engine.run(image)`
- ValidationExport is ready for the validation scripts plan (04-04) to generate and compare JSON output
- All PIPE-01/04 and CONF-02/03/04/05 requirements satisfied

## Self-Check: PASSED

- FOUND: deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/OCREngine.swift
- FOUND: deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/ValidationExport.swift
- FOUND: commit a97b14a47
- FOUND: commit 1fa343917

---
*Phase: 04-pipeline-orchestration-validation*
*Completed: 2026-04-08*
