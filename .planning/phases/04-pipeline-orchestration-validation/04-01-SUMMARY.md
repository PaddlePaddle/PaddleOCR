---
phase: 04-pipeline-orchestration-validation
plan: 01
subsystem: engine
tags: [perspective-transform, box-sorting, ocr-pipeline, swift, coregraphics, dlt]

# Dependency graph
requires:
  - phase: 02-text-detection
    provides: "DetectionBox type, DBPostProcessor output"
provides:
  - "BoxSorter: reading-order sort for detection boxes"
  - "PerspectiveCrop: 4-point perspective crop with tall-narrow rotation"
affects: [04-pipeline-orchestration-validation, 05-ui-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure Swift perspective warp via DLT + bilinear backward mapping"
    - "Gaussian elimination with partial pivoting for 8x8 linear system solve"
    - "Analytical 3x3 matrix inversion using cofactor/adjugate formula"

key-files:
  created:
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/BoxSorter.swift
    - deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/PerspectiveCrop.swift
  modified: []

key-decisions:
  - "Pure Swift perspective warp instead of vImage or Accelerate -- avoids API complexity while maintaining fidelity to OpenCV reference"
  - "Float64 for perspective matrix computation matching OpenCV's getPerspectiveTransform precision, Float for pixel sampling loops"
  - "Pixel-center convention (u+0.5, v+0.5) for backward mapping with -0.5 offset before bilinear sampling"

patterns-established:
  - "Static struct pattern for stateless algorithmic components (BoxSorter, PerspectiveCrop)"
  - "Private extensions for grouping related helper functions"

requirements-completed: [PIPE-02, PIPE-03]

# Metrics
duration: 6min
completed: 2026-04-07
---

# Phase 04 Plan 01: Pipeline Bridge Algorithms Summary

**Reading-order box sorting and 4-point perspective crop in pure Swift, ported line-by-line from PaddleX Python reference**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-07T09:36:39Z
- **Completed:** 2026-04-07T09:42:44Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments

- BoxSorter implements SortQuadBoxes with two-phase sort (initial y/x + backward insertion for same-line reordering) with yThreshold=10 matching PaddleX exactly
- PerspectiveCrop implements full get_rotate_crop_image pipeline: DLT homography solve, backward-mapping bilinear warp with BORDER_REPLICATE, and 90-degree CCW rotation for tall-narrow crops (height/width >= 1.5)
- Both components are pure Swift with no external dependencies beyond Foundation/CoreGraphics

## Task Commits

Each task was committed atomically:

1. **Task 1: BoxSorter -- reading-order sort** - `b178d1d78` (feat)
2. **Task 2: PerspectiveCrop -- 4-point perspective transform** - `922568d3d` (feat)

## Files Created/Modified

- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/BoxSorter.swift` - Reading-order box sorting (SortQuadBoxes port)
- `deploy/ios_demo/PaddleOCRDemo/PaddleOCRDemo/Engine/PerspectiveCrop.swift` - Perspective crop with DLT warp and tall-narrow rotation (get_rotate_crop_image port)

## Decisions Made

- **Pure Swift perspective warp:** Chose manual DLT + bilinear interpolation over vImage or Accelerate APIs. This matches the OpenCV getPerspectiveTransform + warpPerspective reference more faithfully and keeps the code understandable for developers reading the demo.
- **Float64 for matrix math, Float for pixel ops:** Matches OpenCV's getPerspectiveTransform (returns float64) while keeping pixel sampling efficient with single precision.
- **Pixel center convention:** Used (u+0.5, v+0.5) as destination coordinate with -0.5 source offset before bilinear sampling, matching OpenCV's standard sub-pixel mapping behavior.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- BoxSorter.sortInReadingOrder and PerspectiveCrop.crop are ready to be composed into the OCR pipeline orchestrator (Plan 04-02/04-03)
- Both consume the DetectionBox type from DBPostProcess.swift
- PerspectiveCrop produces CGImage output compatible with RecPreprocessor input

## Self-Check: PASSED

- BoxSorter.swift: FOUND
- PerspectiveCrop.swift: FOUND
- 04-01-SUMMARY.md: FOUND
- Commit b178d1d78: FOUND
- Commit 922568d3d: FOUND
