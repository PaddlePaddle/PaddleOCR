---
phase: 02-text-detection
verified: 2026-04-07T12:30:00Z
status: human_needed
score: 3/4 must-haves verified
re_verification: false
human_verification:
  - test: "Run detection on a reference test image and compare polygon coordinates against Python/PaddleX output"
    expected: "Polygon coordinates, box count, and scores match the Python reference exactly"
    why_human: "Requires running the app on device/simulator with a test image and comparing numerical output against Python baseline — cannot verify statically"
---

# Phase 2: Text Detection Verification Report

**Phase Goal:** Given an input image, the detection module produces bounding polygons that exactly match the Python reference implementation's detection output
**Verified:** 2026-04-07T12:30:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1   | Detection preprocessing reads transform parameters from inference.yml (not hardcoded) and applies DetResizeForTest + NormalizeImage + HWC-to-CHW in the correct order | VERIFIED | `DetPreprocessor.init(config:)` extracts resizeLong, scale, mean, std, order from `InferenceConfig.preProcess.transformOps` (Preprocessing.swift lines 60-96). Pipeline order: extractRGBPixels -> imagePadding -> DetResizeForTest -> NormalizeImage -> hwcToCHW (lines 104-149). Zero hardcoded preprocessing values — all from config. |
| 2   | DB postprocessing thresholds the probability map, extracts contours, computes minimum area rectangles, and expands polygons using Clipper-equivalent offset -- all in pure Swift with no OpenCV | VERIFIED | `DBPostProcessor.process()` implements: binary threshold (line 121), Suzuki-Abe contour finding (line 125), getMiniBoxes with rotating calipers minAreaRect (line 139), boxScoreFast with scanline fill (line 148), ClipperOffset.offsetPolygon expansion (line 165), coordinate scaling (lines 186-189). No OpenCV imports anywhere. Pure Swift with Foundation/CoreGraphics only. |
| 3   | Given a reference test image, the detection module outputs polygon coordinates that match the Python/PaddleX reference output exactly | UNCERTAIN | Cannot verify statically. DetectionEngine.detect() wires the full pipeline (preprocess -> ORT inference -> postprocess), but numerical exactness against the Python reference requires runtime comparison with a real test image. Deferred to human verification. |
| 4   | The inference.yml parser correctly loads and exposes all preprocessing and postprocessing parameters for the detection model | VERIFIED | `InferenceConfig.load()` parses Global.model_name, PreProcess.transform_ops (DetResizeForTest with resize_long, NormalizeImage with scale/mean/std/order, ToCHWImage), and PostProcess (name, thresh, box_thresh, max_candidates, unclip_ratio). Scale string "1./255." handled by string split (lines 156-178). All fields from det/inference.yml are covered. |

**Score:** 3/4 truths verified (1 needs human verification)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `deploy/ios_demo/PaddleOCRDemo/Podfile` | Yams pod dependency declaration | VERIFIED | Contains `pod 'Yams', '~> 5.0'` inside target block (line 9) |
| `deploy/ios_demo/.../Engine/InferenceConfig.swift` | YAML config parser for inference.yml | VERIFIED | 214 lines (>= 60 min). Contains `InferenceConfig`, `TransformOp`, `PostProcessConfig`. Exports `load(from:)`. Handles scale string parsing. |
| `deploy/ios_demo/.../Engine/Preprocessing.swift` | DetResizeForTest, NormalizeImage, ToCHWImage in pure Swift | VERIFIED | 341 lines (>= 80 min). Contains `DetPreprocessor`, `PreprocessResult`. Imports Accelerate, CoreGraphics — no OpenCV. Config-driven: reads all params from InferenceConfig. |
| `deploy/ios_demo/.../Engine/ClipperOffset.swift` | Pure Swift port of Clipper polygon offset algorithm | VERIFIED | 425 lines (>= 100 min). Contains `class ClipperOffset` with `addPath`, `execute`, `offsetPolygon`. JoinType (.jtRound), EndType (.etClosedPolygon). Arc tolerance via `acos(1 - arcTol / absDelta)` (line 144). |
| `deploy/ios_demo/.../Engine/DBPostProcess.swift` | DB text detection postprocessing | VERIFIED | 763 lines (>= 120 min). Contains `DBPostProcessor`, `DetectionBox`. Full pipeline: threshold, Suzuki-Abe contours, rotating calipers minAreaRect, scanline fill box scoring, Clipper expansion. Params from PostProcessConfig. |
| `deploy/ios_demo/.../Engine/DetectionEngine.swift` | Complete detection pipeline orchestrator | VERIFIED | 155 lines (>= 60 min). Contains `DetectionEngine`, `DetectionResult`. Wires DetPreprocessor -> ORTSessionManager.runDetection -> DBPostProcessor. Loads config from inference.yml. Per-stage timing. |
| `deploy/ios_demo/.../Engine/ORTSessionManager.swift` | Exposed detection session for external callers | VERIFIED | Contains `func runDetection(inputData: [Float], shape: [Int])` (line 95). Existing methods (loadModels, validateDetModel, validateRecModel) preserved. NaN validation included. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| InferenceConfig.swift | Models/det/inference.yml | Yams YAML parsing at runtime | WIRED | `Yams.load(yaml:)` at line 74. `import Yams` at line 2. File reads from `yamlPath` parameter (line 69). |
| Preprocessing.swift | InferenceConfig.swift | Reads transform parameters from InferenceConfig | WIRED | `DetPreprocessor.init(config: InferenceConfig)` at line 60. Iterates `config.preProcess.transformOps` and extracts `.detResizeForTest`, `.normalizeImage` parameters. |
| DBPostProcess.swift | ClipperOffset.swift | Calls ClipperOffset.execute() for polygon expansion | WIRED | `ClipperOffset.offsetPolygon(cgPoints, distance: distance)` at line 165 of DBPostProcess.swift. |
| DBPostProcess.swift | InferenceConfig.swift | Reads thresh/box_thresh/unclip_ratio from PostProcessConfig | WIRED | `DBPostProcessor.init(config: DBPostProcessConfigurable)` at line 84. `PostProcessConfig: DBPostProcessConfigurable` conformance in DetectionEngine.swift line 21. |
| DetectionEngine.swift | Preprocessing.swift | Calls DetPreprocessor.preprocess() | WIRED | `preprocessor.preprocess(image)` at line 108. `DetPreprocessor(config: config)` at line 95. |
| DetectionEngine.swift | ORTSessionManager.swift | Calls ORTSessionManager.runDetection() | WIRED | `sessionManager.runDetection(inputData:shape:)` at line 113. |
| DetectionEngine.swift | DBPostProcess.swift | Calls DBPostProcessor.process() on ORT output | WIRED | `postprocessor.process(outputTensor:tensorHeight:tensorWidth:originalWidth:originalHeight:)` at line 136. |
| DetectionEngine.swift | InferenceConfig.swift | Loads config to initialize preprocessor and postprocessor | WIRED | `InferenceConfig.load(from: modelConfig.configPath)` at line 92. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| DetectionEngine.swift | preprocessed.tensorData | DetPreprocessor.preprocess(CGImage) | Yes -- processes actual pixel data from CGImage | FLOWING |
| DetectionEngine.swift | outputs (ORT result) | ORTSessionManager.runDetection() | Yes -- runs real ORT inference on preprocessed tensor | FLOWING |
| DetectionEngine.swift | boxes (DetectionBox[]) | DBPostProcessor.process() | Yes -- processes actual ORT output tensor | FLOWING |

### Behavioral Spot-Checks

Step 7b: SKIPPED (no runnable entry points -- iOS app requires device/simulator and Xcode build to run)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| CONF-01 | 02-01, 02-03 | App parses inference.yml bundled with each model at runtime using a Swift YAML parser | SATISFIED | InferenceConfig.swift uses Yams to parse inference.yml. `InferenceConfig.load(from:)` reads the YAML file and produces typed Swift values. |
| PREP-01 | 02-01, 02-03 | Detection preprocessing implements DetResizeForTest | SATISFIED | Preprocessing.swift `computeResizeDimensions()` implements resize_long scaling + stride-128 padding, matching Python `resize_image_type2`. `imagePadding()` handles tiny images (h+w < 64). |
| PREP-02 | 02-01, 02-03 | Detection preprocessing implements NormalizeImage with params from inference.yml | SATISFIED | Preprocessing.swift `normalizePixels()` applies `(pixel * scale - mean) / std`. Parameters read from InferenceConfig, not hardcoded. |
| PREP-03 | 02-01, 02-03 | Detection preprocessing handles RGB channel order and HWC->CHW layout conversion | SATISFIED | Preprocessing.swift `hwcToCHW()` converts [H,W,3] to [3,H,W]. RGB extracted from CGImage (line 155-183). Output shape [1,3,H,W]. |
| PREP-06 | 02-01, 02-03 | All preprocessing uses pure Swift (Accelerate/vImage), no OpenCV dependency | SATISFIED | Preprocessing.swift imports Accelerate + CoreGraphics. No OpenCV/opencv2 imports in any Engine file. Resize via CGContext, no external image processing libraries. |
| POST-01 | 02-02, 02-03 | Detection postprocessing implements DBPostProcess | SATISFIED | DBPostProcess.swift (763 lines) implements full pipeline: threshold, Suzuki-Abe contour finding, rotating calipers minAreaRect, scanline polygon fill scoring, Clipper expansion, coordinate scaling. |
| POST-02 | 02-02, 02-03 | DB post-process supports polygon clipping matching pyclipper JT_ROUND, ET_CLOSEDPOLYGON | SATISFIED | ClipperOffset.swift (425 lines) implements JT_ROUND + ET_CLOSEDPOLYGON offset. DBPostProcess.swift calls `ClipperOffset.offsetPolygon()` at line 165. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| DBPostProcess.swift | 115 | `return []` | Info | Defensive guard for malformed output tensor (count < mapSize). Valid error path, not a stub. |

No blockers or warnings found. No TODO/FIXME/placeholder comments. No OpenCV imports. No hardcoded preprocessing values. No console.log-only implementations.

### Human Verification Required

### 1. Numerical Exactness Against Python Reference

**Test:** Build and run the iOS app on a device/simulator. Run detection on a reference test image (same image used for Python validation). Extract the output polygon coordinates, box count, and confidence scores. Compare against Python/PaddleX `DBPostProcess` output for the same image.
**Expected:** Polygon coordinates (4-point bounding quads as Int32 arrays), box count, and confidence scores must match the Python reference output exactly (identical values, not approximate).
**Why human:** Requires building and running the Xcode project on a device or simulator with a real image, then comparing the numerical output against a Python baseline. The static code analysis confirms the algorithms match the Python reference structurally, but floating-point behavior differences between CoreGraphics bilinear resize and cv2.resize, or between Swift float arithmetic and numpy, can only be caught by runtime comparison.

### 2. Pod Install Verification

**Test:** Run `cd deploy/ios_demo/PaddleOCRDemo && pod install` and then `xcodebuild build` on the workspace.
**Expected:** Pod install resolves Yams ~> 5.0 and onnxruntime-objc ~> 1.24 without conflict. Project builds without compilation errors.
**Why human:** Pods/ directory is not committed (gitignored). Build verification requires Xcode toolchain and CocoaPods installed on the developer machine.

### Gaps Summary

No structural gaps found. All 7 artifacts exist, are substantive (well above minimum line counts), properly wired (all key links verified), registered in the Xcode project, and free of stub anti-patterns. All 7 requirement IDs (CONF-01, PREP-01, PREP-02, PREP-03, PREP-06, POST-01, POST-02) are satisfied by the implementation.

The single remaining uncertainty is numerical exactness (Success Criterion 3) which requires runtime validation -- the code structurally matches the Python reference algorithms, but floating-point exactness can only be confirmed by running both implementations on the same input image and comparing outputs.

**Minor documentation note:** The 02-01-SUMMARY.md file is empty (0 bytes). The Plan 01 work was completed (commits 06a5cc7dd and 90ab35135 exist, artifacts verified), but the summary was not written. This has no impact on codebase functionality.

---

_Verified: 2026-04-07T12:30:00Z_
_Verifier: Claude (gsd-verifier)_
