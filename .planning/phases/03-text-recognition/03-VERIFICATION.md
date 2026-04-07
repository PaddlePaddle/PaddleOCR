---
phase: 03-text-recognition
verified: 2026-04-07T09:30:00Z
status: passed
score: 7/7 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 6/7
  gaps_closed:
    - "CTC decoding correctly maps model output tensor indices to characters using the 18,385-element dictionary"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Run recognition on a cropped text image containing spaces and verify output matches Python reference"
    expected: "Decoded text includes space characters at correct positions; confidence score matches Python output"
    why_human: "Requires running on iOS device/simulator with real model and comparing numerical output against Python reference"
  - test: "Run recognition on a cropped text image and verify normalization produces values in [-1, 1] range"
    expected: "Preprocessing tensor values match Python OCRResizeNormImg output within 1e-5 tolerance"
    why_human: "Requires running both iOS and Python implementations on the same input image and comparing floating-point tensor values"
---

# Phase 3: Text Recognition Verification Report

**Phase Goal:** Given a cropped text region image, the recognition module produces the correct text string and confidence score, exactly matching the Python reference implementation
**Verified:** 2026-04-07T09:30:00Z
**Status:** passed
**Re-verification:** Yes -- after gap closure (previous status: gaps_found, score 6/7)

## Gap Closure Summary

The previous verification found one gap: CTCDecoder character dictionary missing the ASCII space character, resulting in 18,384 entries instead of the required 18,385. Plan 03-03 addressed this by adding `chars.append(" ")` at line 72 of CTCDecoder.swift. Commit `5556c1a68` contains the fix.

**Verification of fix:** CTCDecoder.swift lines 69-73 now read:
```swift
var chars = ["blank"]
chars.append(contentsOf: dict)
chars.append(" ")
self.characters = chars
```

This produces `["blank"] + 18,383 dict chars + [" "]` = 18,385 total entries. Index 0 = blank, indices 1..18383 = dict characters, index 18384 = ASCII space. This exactly matches PaddleX `BaseRecLabelDecode.__init__` where `character_list.append(" ")` runs before `["blank"] + character_list`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Recognition model inference can be invoked via ORTSessionManager.runRecognition with dynamic-width input tensors | VERIFIED (regression check) | `func runRecognition(inputData: [Float], shape: [Int])` at line 111 of ORTSessionManager.swift; delegates to shared `runInference(session:modelName:inputData:shape:)` at line 115; guards `recSession != nil` at line 112. Unchanged from previous verification. |
| 2 | A cropped text image is resized to height 48 with aspect-ratio-aware width, normalized to [-1,1], and right-padded to target width | VERIFIED (regression check) | RecPreprocessor.swift: `ceil()` at lines 116/119, `Float(pixels[i]) / 127.5 - 1.0` at line 255, `padToTargetWidth` at line 133. Unchanged from previous verification. |
| 3 | Preprocessing reads image_shape from inference.yml RecResizeImg transform op, not hardcoded values | VERIFIED (regression check) | RecPreprocessor.swift line 70: `case .recResizeImg(let imageShape)`, extracts imgC/imgH/imgW from config at lines 81-83; no hardcoded 48 or 320 in init. Unchanged from previous verification. |
| 4 | CTC decoding correctly maps model output tensor indices to characters using the 18,385-element dictionary | VERIFIED (gap closure) | CTCDecoder.swift lines 69-73: `var chars = ["blank"]` + `chars.append(contentsOf: dict)` + `chars.append(" ")` = 18,385 entries. Blank at index 0, space at index 18384. Matches PaddleX BaseRecLabelDecode exactly. Fix commit: `5556c1a68`. |
| 5 | Consecutive duplicate indices and blank tokens (index 0) are removed before character mapping | VERIFIED (regression check) | CTCDecoder.swift lines 124-135: consecutive duplicate removal at line 125 (`predIndices[t] == predIndices[t - 1]`), blank removal at line 132 (`predIndices[t] == blankIndex`). Unchanged from previous verification. |
| 6 | Confidence score is computed as the mean of max probabilities at selected timesteps | VERIFIED (regression check) | CTCDecoder.swift lines 156-161: `confList.reduce(0, +) / Float(confList.count)`, empty case handled with `confidence = 0`. Unchanged from previous verification. |
| 7 | RecognitionEngine orchestrates preprocess -> ORT inference -> CTC decode into a single recognize(CGImage) call | VERIFIED (regression check) | RecognitionEngine.swift lines 85-127: `preprocessor.preprocess(image)` -> `sessionManager.runRecognition(inputData:shape:)` -> `decoder.decode(outputData:outputShape:)`. Unchanged from previous verification. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Engine/ORTSessionManager.swift` | runRecognition method for rec model inference | VERIFIED | 247 lines; `func runRecognition` at line 111; shared `runInference` at line 123; runDetection at line 95 unchanged |
| `Engine/RecPreprocessor.swift` | OCRResizeNormImg preprocessing for recognition | VERIFIED | 317 lines (min 120); struct RecPreprocessor; struct RecPreprocessResult; reads config from RecResizeImg; pure CoreGraphics, no OpenCV |
| `Engine/CTCDecoder.swift` | CTC greedy decoder with character dictionary | VERIFIED | 165 lines (min 80); struct CTCDecoder; dictionary construction now correct: `["blank"] + dict + [" "]` = 18,385 entries; argmax/dedup/blank removal/confidence all correct |
| `Engine/RecognitionEngine.swift` | Recognition pipeline orchestrator | VERIFIED | 128 lines (min 80); class RecognitionEngine; composes RecPreprocessor + ORTSessionManager + CTCDecoder; per-stage timing via CFAbsoluteTimeGetCurrent (6 occurrences) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| RecPreprocessor.swift | InferenceConfig.swift | `case .recResizeImg` reads imageShape from config.preProcess.transformOps | WIRED | Line 70: `case .recResizeImg(let imageShape)` |
| ORTSessionManager.swift | recSession | runRecognition uses recSession (loaded in loadModels) | WIRED | Line 112: `guard let session = recSession`; line 115: `runInference(session: session, ...)` |
| CTCDecoder.swift | InferenceConfig.swift | reads character_dict from config.postProcess.characterDict | WIRED | Line 61: `guard let dict = config.postProcess.characterDict` |
| RecognitionEngine.swift | RecPreprocessor.swift | calls preprocessor.preprocess(image) | WIRED | Line 88: `let preprocessed = try preprocessor.preprocess(image)` |
| RecognitionEngine.swift | ORTSessionManager.swift | calls sessionManager.runRecognition(inputData:shape:) | WIRED | Line 93: `try await sessionManager.runRecognition(inputData: ..., shape: ...)` |
| RecognitionEngine.swift | CTCDecoder.swift | calls decoder.decode(outputData:outputShape:) | WIRED | Line 114: `try decoder.decode(outputData: outputData, outputShape: outputShape)` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| RecPreprocessor.swift | tensorData | CGImage pixel extraction + normalize + HWC-to-CHW + pad | Real image data flows through pipeline | FLOWING |
| CTCDecoder.swift | characters | config.postProcess.characterDict from inference.yml + space append | 18,383 dict chars + space = 18,384 non-blank entries + blank = 18,385 total | FLOWING |
| RecognitionEngine.swift | outputs | sessionManager.runRecognition() | Real ORT inference output tensor | FLOWING |
| RecognitionEngine.swift | decoded | decoder.decode(outputData:outputShape:) | Decodes real model output to text using complete 18,385-element dictionary | FLOWING |

### Behavioral Spot-Checks

Step 7b: SKIPPED (requires iOS simulator/device to run ORT inference; no runnable entry points in CLI context)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PREP-04 | 03-01 | Recognition preprocessing implements OCRResizeNormImg -- aspect-ratio-aware resize, params from inference.yml | SATISFIED | RecPreprocessor reads image_shape from RecResizeImg config; computes resized_w with ceil(); resizes via CGContext bilinear |
| PREP-05 | 03-01 | Recognition preprocessing implements normalization (pixel/255, then (x-0.5)/0.5) | SATISFIED | RecPreprocessor.swift line 255: `Float(pixels[i]) / 127.5 - 1.0` -- equivalent formula, maps [0,255] to [-1,1] |
| POST-03 | 03-02, 03-03 | Recognition postprocessing implements CTCLabelDecode with character dict from inference.yml | SATISFIED | CTCDecoder implements argmax, dedup, blank removal, char mapping. Dictionary now correct: 18,385 entries matching model output dimension. Space at index 18384 correctly included per use_space_char=True default. |
| POST-04 | 03-02 | Recognition confidence computed as mean of selected token probabilities | SATISFIED | CTCDecoder.swift line 160: `confList.reduce(0, +) / Float(confList.count)`; empty case returns 0 |

No orphaned requirements found. REQUIREMENTS.md maps PREP-04, PREP-05, POST-03, POST-04 to Phase 3, all of which are claimed by plans 03-01, 03-02, and 03-03.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| RecPreprocessor.swift | 225 | `return [UInt8](repeating: 0, count: dstH * dstW * 3)` fallback in resize | Info | Silently returns zero-filled buffer if CGContext creation fails; would produce garbage output without clear error. CGContext failure is extremely rare in practice. |
| CTCDecoder.swift | 144 | `if idx < characters.count` silently drops out-of-bounds index | Info | With the 18,385-element dictionary now matching the model's 18,385 output classes, this is a standard safety guard. No valid model output will trigger it. |

### Human Verification Required

### 1. End-to-End Recognition Accuracy

**Test:** Run RecognitionEngine.recognize() on a cropped text image (e.g., cropped from the PaddleOCR test image "doc/imgs/11.jpg") and compare text output + confidence against Python PaddleX CTCLabelDecode output for the same cropped region.
**Expected:** Decoded text string matches exactly; confidence score matches within 1e-6 tolerance.
**Why human:** Requires running on iOS device/simulator with real ONNX model and comparing against Python reference output.

### 2. Normalization Numerical Exactness

**Test:** Extract the preprocessed tensor from RecPreprocessor.preprocess() for a known input image and compare float values against Python OCRResizeNormImg output.
**Expected:** All tensor values match within 1e-5 tolerance (accounting for CGContext vs cv2.resize bilinear interpolation differences).
**Why human:** Requires running both iOS and Python implementations on the same input and comparing floating-point arrays.

### 3. Space Character in Recognized Text

**Test:** Run recognition on text containing ASCII spaces (e.g., "Hello World") and verify spaces appear correctly in output.
**Expected:** Spaces are present in decoded text at correct positions; dictionary size is 18,385.
**Why human:** Requires running on device with real model output that produces space character predictions.

---

_Verified: 2026-04-07T09:30:00Z_
_Verifier: Claude (gsd-verifier)_
