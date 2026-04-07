# Roadmap: PaddleOCR iOS Demo

## Overview

This roadmap delivers a complete PP-OCRv5 iOS demo from bare Xcode project to documented, developer-ready reference application. The build order follows the dependency chain: inference engine foundation first, then detection and recognition as independent vertical slices, then pipeline orchestration that ties them together with config-driven behavior and numerical validation, then the user interface, and finally documentation. The single most important quality bar throughout is **numerical exactness** -- the iOS pipeline must produce identical results to the Python/PaddleX reference implementation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Inference Engine Foundation** - Xcode project with ONNX Runtime integration, both models loading and running raw inference on device
- [ ] **Phase 2: Text Detection** - Complete detection vertical slice: config-driven preprocessing, DB postprocessing with polygon clipping, producing accurate bounding polygons
- [ ] **Phase 3: Text Recognition** - Complete recognition vertical slice: aspect-ratio-aware preprocessing, CTC decoding with character dictionary, producing text with confidence scores
- [ ] **Phase 4: Pipeline Orchestration & Validation** - End-to-end OCR flow (detect -> sort -> crop -> recognize), fully config-driven, numerically validated against Python reference
- [ ] **Phase 5: User Interface** - SwiftUI application with image picker, result visualization, bounding box overlays, timing metrics, and error handling
- [ ] **Phase 6: Documentation** - README with build instructions, architecture guide, and integration guide for developers

## Phase Details

### Phase 1: Inference Engine Foundation
**Goal**: Developer can build and run an iOS app that loads PP-OCRv5 detection and recognition ONNX models, creates ORT sessions with CoreML EP, and runs raw inference (tensor in, tensor out) on device
**Depends on**: Nothing (first phase)
**Requirements**: INFER-01, INFER-02, INFER-03, INFER-04, INFER-05
**Success Criteria** (what must be TRUE):
  1. Xcode project builds and runs on an iOS 16+ device/simulator with ONNX Runtime integrated via CocoaPods
  2. Detection ONNX model loads into an ORTSession with CoreML Execution Provider enabled and produces an output tensor from a dummy input
  3. Recognition ONNX model loads into an ORTSession with CoreML Execution Provider enabled and produces an output tensor from a dummy input
  4. User sees a loading indicator while models are initializing
**Plans:** 2 plans
Plans:
- [x] 01-01-PLAN.md — Xcode project scaffolding, CocoaPods + ORT integration, model bundling
- [ ] 01-02-PLAN.md — ORTSessionManager with CoreML EP, dummy inference validation, MVVM UI with loading states

### Phase 2: Text Detection
**Goal**: Given an input image, the detection module produces bounding polygons that exactly match the Python reference implementation's detection output
**Depends on**: Phase 1
**Requirements**: CONF-01, PREP-01, PREP-02, PREP-03, PREP-06, POST-01, POST-02
**Success Criteria** (what must be TRUE):
  1. Detection preprocessing reads transform parameters from inference.yml (not hardcoded) and applies DetResizeForTest + NormalizeImage + HWC-to-CHW in the correct order
  2. DB postprocessing thresholds the probability map, extracts contours, computes minimum area rectangles, and expands polygons using Clipper-equivalent offset -- all in pure Swift with no OpenCV
  3. Given a reference test image, the detection module outputs polygon coordinates that match the Python/PaddleX reference output exactly
  4. The inference.yml parser correctly loads and exposes all preprocessing and postprocessing parameters for the detection model
**Plans:** 3 plans
Plans:
- [x] 02-01-PLAN.md — InferenceConfig YAML parser (Yams) + pure-Swift preprocessing operators (DetResizeForTest, NormalizeImage, ToCHWImage)
- [x] 02-02-PLAN.md — ClipperOffset pure-Swift port + DBPostProcess (threshold, contours, minAreaRect, scoring, expansion)
- [x] 02-03-PLAN.md — DetectionEngine integration wiring preprocessing + ORT inference + postprocessing

### Phase 3: Text Recognition
**Goal**: Given a cropped text region image, the recognition module produces the correct text string and confidence score, exactly matching the Python reference implementation
**Depends on**: Phase 1
**Requirements**: PREP-04, PREP-05, POST-03, POST-04
**Success Criteria** (what must be TRUE):
  1. Recognition preprocessing implements OCRResizeNormImg with aspect-ratio-aware resize to fixed height, variable width, and correct normalization formula as specified in inference.yml
  2. CTC decoding correctly performs argmax, blank removal, consecutive duplicate removal, and character dictionary mapping using the PP-OCRv5 dictionary (18,385 classes: blank + 18,383 dict + space)
  3. Given a reference cropped text image, the recognition module outputs the correct text string and confidence score matching the Python/PaddleX reference
**Plans:** 3 plans
Plans:
- [ ] 03-01-PLAN.md — ORTSessionManager runRecognition + RecPreprocessor (OCRResizeNormImg preprocessing)
- [ ] 03-02-PLAN.md — CTCDecoder (CTC label decode with 18,385-class dictionary) + RecognitionEngine orchestration
- [ ] 03-03-PLAN.md — Gap closure: fix CTCDecoder missing ASCII space in character dictionary (18,384 -> 18,385) [DONE]

### Phase 4: Pipeline Orchestration & Validation
**Goal**: The complete OCR pipeline (detect -> sort -> crop -> recognize) runs end-to-end, is fully driven by inference.yml configuration with zero hardcoded parameters, and produces numerically exact results validated against the Python reference
**Depends on**: Phase 2, Phase 3
**Requirements**: CONF-02, CONF-03, CONF-04, CONF-05, PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, VALID-01, VALID-02
**Success Criteria** (what must be TRUE):
  1. End-to-end pipeline orchestrates detect -> sort boxes (reading order) -> perspective crop -> recognize, producing an array of (polygon, text, confidence) results
  2. Preprocessing and postprocessing pipelines are built dynamically from inference.yml -- swapping model files + inference.yml changes behavior with zero code changes
  3. A validation script compares iOS pipeline output against Python reference output for test images, and detection polygons, recognized text, and confidence scores match exactly
  4. Pipeline runs on a background thread -- the UI thread remains responsive during inference
  5. Box sorting follows reading order (top-to-bottom, left-to-right with y-threshold) and perspective crop handles tall-narrow boxes with rotation, matching PaddleX behavior
**Plans**: TBD

### Phase 5: User Interface
**Goal**: Users interact with a clean SwiftUI application to select images, run OCR, and see visualized results with bounding boxes, recognized text, confidence scores, and timing metrics
**Depends on**: Phase 4
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07, UI-08
**Success Criteria** (what must be TRUE):
  1. User can select an image from their photo album via PhotosPicker or tap a bundled sample image to run OCR without granting photo library access
  2. Detection results are shown as bounding box polygon overlays drawn on the source image, with recognized text and per-result confidence scores displayed alongside
  3. Per-stage timing breakdown is displayed after inference (detection time, recognition time, total time)
  4. User can copy all recognized text to the clipboard with one tap
  5. Loading indicator appears during model initialization and inference, and meaningful error messages appear when inference fails or input is invalid
**Plans**: TBD
**UI hint**: yes

### Phase 6: Documentation
**Goal**: A developer unfamiliar with the project can clone the repo, build the demo, understand the architecture, and integrate PaddleOCR inference into their own iOS app using only the provided documentation
**Depends on**: Phase 5
**Requirements**: DOC-01, DOC-02, DOC-03
**Success Criteria** (what must be TRUE):
  1. README contains complete build instructions (Xcode version, iOS target, CocoaPods install, model conversion steps, how to run) and a developer can go from clone to running app by following them
  2. Architecture guide explains the code structure, inference pipeline stages, config-driven design, and key design decisions clearly enough that a developer can navigate the codebase independently
  3. Integration guide provides a step-by-step walkthrough showing developers how to extract and use the PaddleOCR inference code in their own iOS app
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6
Note: Phases 2 and 3 can be developed in parallel (both depend only on Phase 1).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Inference Engine Foundation | 0/2 | Planning complete | - |
| 2. Text Detection | 3/3 | Human verification needed | - |
| 3. Text Recognition | 2/3 | Gap closure planned | - |
| 4. Pipeline Orchestration & Validation | 0/0 | Not started | - |
| 5. User Interface | 0/0 | Not started | - |
| 6. Documentation | 0/0 | Not started | - |
