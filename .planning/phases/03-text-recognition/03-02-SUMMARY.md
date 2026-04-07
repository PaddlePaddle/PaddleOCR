---
phase: 03-text-recognition
plan: 02
subsystem: inference-engine
tags: [ctc-decoding, recognition, character-dictionary, swift, ocr, onnx-runtime]

# Dependency graph
requires:
  - phase: 01-inference-engine-foundation
    provides: "ORTSessionManager with recSession loading and runRecognition()"
  - phase: 03-text-recognition
    plan: 01
    provides: "RecPreprocessor implementing OCRResizeNormImg, runRecognition() on ORTSessionManager"
provides:
  - "CTCDecoder implementing CTC label decode with 18,383-character dictionary from inference.yml"
  - "RecognitionEngine orchestrating RecPreprocessor -> ORT inference -> CTCDecoder"
  - "RecognitionEngineResult with text, confidence, and per-stage timing metrics"
affects: [04-pipeline-orchestration, 05-user-interface]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CTCDecoder mirrors Python CTCLabelDecode: blank prepend, argmax, dedup, blank filter, char map, mean confidence"
    - "RecognitionEngine mirrors DetectionEngine pattern: class with config-driven init, compose preprocessor + ORT + postprocessor"

key-files:
  created:
    - "deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/CTCDecoder.swift"
    - "deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/RecognitionEngine.swift"
  modified:
    - "deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo.xcodeproj/project.pbxproj"

key-decisions:
  - "CTCDecoder reads character_dict directly from PostProcess config (already parsed by InferenceConfig), no separate file loading needed"
  - "RecognitionEngine is a class (not actor) mirroring DetectionEngine -- delegates concurrency to ORTSessionManager actor"

patterns-established:
  - "Engine pattern: class wrapping preprocessor + ORT session + postprocessor, initialized from inference.yml config"
  - "CTC decode pattern: argmax -> consecutive dedup -> blank removal -> char mapping -> mean confidence"

requirements-completed: [POST-03, POST-04]

# Metrics
duration: 3min
completed: 2026-04-07
---

# Phase 3 Plan 02: CTC Decoding & RecognitionEngine Integration Summary

**CTCDecoder implements CTC label decode with 18,383-character PP-OCRv5 dictionary; RecognitionEngine orchestrates the full recognition pipeline (preprocess -> infer -> decode) with per-stage timing**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-07T06:31:49Z
- **Completed:** 2026-04-07T06:34:51Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- CTCDecoder ported from Python `CTCLabelDecode` with exact algorithm match: blank prepend at index 0, argmax across class dimension, consecutive duplicate removal, blank token filtering, character dictionary mapping, mean confidence computation
- RecognitionEngine composes RecPreprocessor + ORTSessionManager.runRecognition + CTCDecoder into a single `recognize(CGImage)` call with per-stage timing breakdown
- Complete recognition vertical slice ready: given a cropped text image, produce decoded text with confidence score

## Task Commits

Each task was committed atomically:

1. **Task 1: Create CTCDecoder implementing CTC label decode** - `5a36aa8a2` (feat)
2. **Task 2: Create RecognitionEngine orchestrating full pipeline** - `629764d52` (feat)

## Files Created/Modified
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/CTCDecoder.swift` - CTC label decode: argmax, dedup, blank removal, char mapping, confidence (139 lines)
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/RecognitionEngine.swift` - Recognition pipeline orchestrator: preprocess -> ORT -> CTC decode (114 lines)
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo.xcodeproj/project.pbxproj` - Registered CTCDecoder.swift and RecognitionEngine.swift in Xcode project

## Decisions Made
- CTCDecoder reads `character_dict` array directly from the already-parsed `PostProcessConfig` in `InferenceConfig` -- no separate dictionary file loading needed since `inference.yml` embeds the full 18,383-character dictionary inline
- RecognitionEngine follows the same class (not actor) pattern as DetectionEngine, delegating thread safety to the ORTSessionManager actor for ORT session access

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all components are fully implemented with real logic.

## Next Phase Readiness
- RecognitionEngine ready for pipeline orchestration (Phase 4)
- DetectionEngine + RecognitionEngine both expose per-stage timing for UI display (Phase 5)
- Full recognition vertical slice complete: detect -> crop -> recognize flow can now be wired in Phase 4

---
*Phase: 03-text-recognition*
*Completed: 2026-04-07*

## Self-Check: PASSED

All files and commits verified:
- CTCDecoder.swift: FOUND
- RecognitionEngine.swift: FOUND
- 03-02-SUMMARY.md: FOUND
- Commit 5a36aa8a2: FOUND
- Commit 629764d52: FOUND
