---
phase: 03-text-recognition
plan: 01
subsystem: inference-engine
tags: [onnx-runtime, recognition, preprocessing, ocr, swift, coregraphics]

# Dependency graph
requires:
  - phase: 01-inference-engine-foundation
    provides: "ORTSessionManager with detSession/recSession loading, ORTEnv, CoreML EP"
  - phase: 02-text-detection
    provides: "InferenceConfig parser with TransformOp.recResizeImg, DetPreprocessor patterns"
provides:
  - "ORTSessionManager.runRecognition() for rec model inference with dynamic-width tensors"
  - "RecPreprocessor implementing OCRResizeNormImg (aspect-ratio resize, pixel/127.5-1.0 normalization, zero-padding)"
  - "Shared runInference() private method eliminating code duplication between det/rec"
affects: [03-text-recognition-plan-02, 04-pipeline-orchestration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared private runInference method for all ORT session inference calls"
    - "RecPreprocessor mirrors DetPreprocessor pattern: config-driven init, CGImage-based preprocess"
    - "Recognition normalization is hardcoded (pixel/127.5-1.0), not config-driven (unlike detection)"

key-files:
  created:
    - "deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/RecPreprocessor.swift"
  modified:
    - "deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/ORTSessionManager.swift"
    - "deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo.xcodeproj/project.pbxproj"

key-decisions:
  - "Recognition normalization is fixed (pixel/127.5-1.0), not parameterized from inference.yml -- matches PaddleX behavior"
  - "Shared runInference private method extracted from runDetection to eliminate code duplication"

patterns-established:
  - "RecPreprocessor pattern: config-driven dimensions from inference.yml, fixed normalization formula, CGContext bilinear resize"
  - "ORT inference methods: thin public wrappers (runDetection, runRecognition) delegating to shared private runInference"

requirements-completed: [PREP-04, PREP-05]

# Metrics
duration: 3min
completed: 2026-04-07
---

# Phase 3 Plan 01: Recognition Preprocessing & Inference Summary

**ORTSessionManager gains runRecognition() with shared inference logic; RecPreprocessor implements OCRResizeNormImg with config-driven aspect-ratio resize and pixel/127.5-1.0 normalization in pure Swift**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-07T06:22:04Z
- **Completed:** 2026-04-07T06:25:19Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- ORTSessionManager refactored with shared `runInference` private method, eliminating tensor creation/output extraction duplication between detection and recognition
- New `runRecognition()` public method for recognition model inference with dynamic-width tensor input [1, 3, 48, W]
- RecPreprocessor implements complete OCRResizeNormImg algorithm: aspect-ratio-aware resize to fixed height, ceil()-based width computation, pixel/127.5-1.0 normalization, HWC-to-CHW transpose, zero-padding to target width

## Task Commits

Each task was committed atomically:

1. **Task 1: Add runRecognition to ORTSessionManager** - `b23f6eaa7` (feat)
2. **Task 2: Create RecPreprocessor implementing OCRResizeNormImg** - `17352e8fe` (feat)

## Files Created/Modified
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/ORTSessionManager.swift` - Added runRecognition(), extracted shared runInference() private method
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/RecPreprocessor.swift` - New file: OCRResizeNormImg preprocessing for recognition (317 lines)
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo.xcodeproj/project.pbxproj` - Registered RecPreprocessor.swift in Xcode project

## Decisions Made
- Recognition normalization uses fixed formula `pixel/127.5 - 1.0` (not the configurable ImageNet mean/std used by detection) -- this matches PaddleX behavior where rec normalization is hardcoded in the processor, not read from config
- Extracted shared `runInference(session:modelName:inputData:shape:)` private method rather than duplicating the tensor creation and output extraction logic -- cleaner architecture and easier to add future model types

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all components are fully implemented with real logic.

## Next Phase Readiness
- RecPreprocessor ready for RecognitionEngine integration (Plan 02)
- runRecognition() ready to be called by RecognitionEngine after preprocessing
- CTCDecoder (Plan 02) will complete the recognition pipeline: preprocess -> infer -> decode

---
*Phase: 03-text-recognition*
*Completed: 2026-04-07*

## Self-Check: PASSED

All files and commits verified:
- RecPreprocessor.swift: FOUND
- ORTSessionManager.swift: FOUND
- 03-01-SUMMARY.md: FOUND
- Commit b23f6eaa7: FOUND
- Commit 17352e8fe: FOUND
