---
phase: 03-text-recognition
plan: 03
subsystem: engine
tags: [ctc, decoder, character-dictionary, postprocessing, swift]

# Dependency graph
requires:
  - phase: 03-02
    provides: CTCDecoder with CTC greedy decode logic and character dictionary from inference.yml
provides:
  - Corrected 18,385-element character dictionary matching PaddleX BaseRecLabelDecode
  - ASCII space character at index 18384 no longer silently dropped
affects: [04-pipeline-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [character-dictionary-construction-matching-paddlex]

key-files:
  created: []
  modified:
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/CTCDecoder.swift

key-decisions:
  - "Append space after dict chars and before storing, matching PaddleX init order (dict -> space -> blank prepend)"

patterns-established:
  - "Character dictionary construction must mirror PaddleX BaseRecLabelDecode.__init__ exactly: blank at 0, dict at 1..N, space at N+1"

requirements-completed: [PREP-04, PREP-05, POST-03, POST-04]

# Metrics
duration: 2min
completed: 2026-04-07
---

# Phase 03 Plan 03: CTC Decoder Space Character Fix Summary

**Fixed CTCDecoder dictionary to include ASCII space at index 18384, achieving 18,385-element parity with PaddleX BaseRecLabelDecode**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-07T07:17:00Z
- **Completed:** 2026-04-07T07:19:10Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Fixed CTCDecoder character dictionary to append ASCII space character matching PaddleX `use_space_char=True` default
- Updated documentation comments to accurately describe the 18,385-element character list layout
- Model output class index 18384 now correctly maps to space instead of being silently dropped

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix CTCDecoder dictionary to include ASCII space at index 18384** - `5556c1a68` (fix)

## Files Created/Modified
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/CTCDecoder.swift` - Added `chars.append(" ")` after dict chars; updated class-level and property doc comments to reflect 18,385-element structure

## Decisions Made
None - followed plan as specified. The fix was a single-line addition (`chars.append(" ")`) plus documentation updates, exactly as the plan prescribed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - no stubs or placeholder code.

## Next Phase Readiness
- CTCDecoder now produces correct output for all 18,385 model output classes
- RecognitionEngine (from plan 03-02) can now correctly decode space characters in recognition results
- Ready for Phase 04 pipeline integration: detection + recognition end-to-end

## Self-Check: PASSED

- [x] CTCDecoder.swift exists at expected path
- [x] 03-03-SUMMARY.md created
- [x] Commit 5556c1a68 found in git log
- [x] `chars.append(" ")` present in CTCDecoder.swift (count: 1)

---
*Phase: 03-text-recognition*
*Completed: 2026-04-07*
