---
phase: 02-text-detection
plan: 03
subsystem: engine
tags: [swift, onnx-runtime, detection, inference-pipeline, coreml]

# Dependency graph
requires:
  - phase: 02-text-detection/01
    provides: "InferenceConfig parser, DetPreprocessor with preprocess()"
  - phase: 02-text-detection/02
    provides: "DBPostProcessor with process(), ClipperOffset polygon expansion"
provides:
  - "DetectionEngine class: complete detection pipeline (CGImage -> boxes)"
  - "DetectionResult struct with per-stage timing metrics"
  - "ORTSessionManager.runDetection() for real tensor inference"
  - "PostProcessConfig conformance to DBPostProcessConfigurable"
affects: [03-text-recognition, 04-pipeline-orchestration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Engine pattern: class wrapping preprocessor + session manager + postprocessor"
    - "CFAbsoluteTimeGetCurrent for sub-millisecond per-stage timing"

key-files:
  created:
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/DetectionEngine.swift
  modified:
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/ORTSessionManager.swift
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo.xcodeproj/project.pbxproj

key-decisions:
  - "DetectionEngine is a class (not actor) -- delegates concurrency to ORTSessionManager actor"
  - "PostProcessConfig bridged to DBPostProcessConfigurable via extension conformance"

patterns-established:
  - "Engine integration pattern: init loads config from inference.yml, detect() orchestrates pipeline"
  - "Timing pattern: CFAbsoluteTimeGetCurrent per pipeline stage for performance profiling"

requirements-completed: [CONF-01, PREP-01, PREP-02, PREP-03, PREP-06, POST-01, POST-02]

# Metrics
duration: 3min
completed: 2026-04-07
---

# Phase 2 Plan 3: Detection Engine Integration Summary

**DetectionEngine wiring the full CGImage -> preprocess -> ORT inference -> DBPostProcess -> DetectionBox pipeline with per-stage timing metrics**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-07T04:01:31Z
- **Completed:** 2026-04-07T04:04:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- ORTSessionManager extended with `runDetection(inputData:shape:)` for real preprocessed tensor inference
- DetectionEngine orchestrates full detection pipeline: DetPreprocessor -> ORTSessionManager.runDetection -> DBPostProcessor
- Per-stage timing metrics (preprocess, inference, postprocess) captured via CFAbsoluteTimeGetCurrent
- All parameters loaded from inference.yml at runtime -- zero hardcoded preprocessing/postprocessing values

## Task Commits

Each task was committed atomically:

1. **Task 1: Expose detection inference on ORTSessionManager** - `4abc5d282` (feat)
2. **Task 2: Create DetectionEngine orchestrating the full detection pipeline** - `de0beff24` (feat)

## Files Created/Modified
- `Engine/DetectionEngine.swift` - Complete detection pipeline orchestrator (155 lines): DetectionEngine class, DetectionResult struct, DetectionEngineError enum
- `Engine/ORTSessionManager.swift` - Added runDetection(inputData:shape:) method for real tensor inference
- `PaddleOCRDemo.xcodeproj/project.pbxproj` - Registered DetectionEngine.swift in Xcode project

## Decisions Made
- DetectionEngine is a class (not actor) because it delegates concurrency to ORTSessionManager (which is already an actor) -- avoids unnecessary actor isolation overhead
- PostProcessConfig bridged to DBPostProcessConfigurable via extension conformance (auto-fix: plan assumed direct init compatibility)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added PostProcessConfig conformance to DBPostProcessConfigurable**
- **Found during:** Task 2 (DetectionEngine creation)
- **Issue:** Plan code uses `DBPostProcessor(config: config.postProcess)` but `PostProcessConfig` does not conform to `DBPostProcessConfigurable` protocol required by the initializer
- **Fix:** Added `extension PostProcessConfig: DBPostProcessConfigurable {}` in DetectionEngine.swift -- PostProcessConfig already has all required properties (thresh, boxThresh, maxCandidates, unclipRatio), just lacked formal conformance
- **Files modified:** DetectionEngine.swift
- **Verification:** Type-checked against protocol requirements -- all 4 properties match
- **Committed in:** de0beff24 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for type safety -- the protocol conformance was implicit in the plan but not declared in either existing file. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all data flows are fully wired (config -> preprocessor -> inference -> postprocessor -> result).

## Next Phase Readiness
- DetectionEngine is ready for Phase 4 pipeline orchestration (detect text regions from CGImage)
- Phase 3 (text recognition) can follow the same Engine pattern: RecognitionEngine wrapping RecPreprocessor -> ORT -> RecPostProcessor
- ORTSessionManager pattern established for adding runRecognition() in Phase 3

---
*Phase: 02-text-detection*
*Completed: 2026-04-07*

## Self-Check: PASSED

- All created files verified on disk
- All commit hashes verified in git log
