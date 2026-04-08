# Requirements: PaddleOCR iOS Demo

**Defined:** 2026-04-03
**Core Value:** Developers can see PP-OCRv5 text detection and recognition running on an iOS device with clear, understandable code they can adapt for their own apps.

## Reference Implementation

The iOS demo must produce **identical results** to the Python reference. The following files are the authoritative source of truth for every processing step. Agents **must** read these files during implementation — do not implement from memory or description alone.

### PaddleX Reference (remote — `Bobholamovic/PaddleX` branch `feat/transformers`)

Fetch via: `gh api repos/Bobholamovic/PaddleX/contents/{path}?ref=feat/transformers -q .content | base64 -d`

| Processing Step | File Path | What to Extract |
|----------------|-----------|-----------------|
| Det preprocessing | `paddlex/inference/models/text_detection/processors.py` | `DetResizeForTest`, `NormalizeImage` — exact resize logic, normalization formula, parameter handling |
| Det postprocessing | `paddlex/inference/models/text_detection/processors.py` | `DBPostProcess` — thresh, box_thresh, unclip_ratio, contour finding, polygon scoring, Clipper offset |
| Det predictor flow | `paddlex/inference/models/text_detection/predictor.py` | `TextDetRunnerPredictor` — how inference.yml is loaded, how transforms are chained |
| Det model config | `paddlex/inference/models/text_detection/modeling/_config_pp_ocrv5_mobile.py` | Default parameters for PP-OCRv5 mobile det |
| Rec preprocessing | `paddlex/inference/models/text_recognition/processors.py` | `OCRReisizeNormImg` — aspect-ratio-aware resize, normalization (`x/127.5 - 1`), right-padding |
| Rec postprocessing | `paddlex/inference/models/text_recognition/processors.py` | `CTCLabelDecode` — argmax, duplicate removal, blank removal, character mapping, confidence calc |
| Rec predictor flow | `paddlex/inference/models/text_recognition/predictor.py` | `TextRecRunnerPredictor` — config loading, transform chain |
| Rec model config | `paddlex/inference/models/text_recognition/modeling/_config_pp_ocrv5_mobile_rec.py` | Default parameters for PP-OCRv5 mobile rec |
| Pipeline orchestration | `paddlex/inference/pipelines/ocr/pipeline.py` | `_OCRPipeline.predict()` — det->sort->crop->rec flow, box sorting, perspective crop, score filtering |
| Visualization | `paddlex/inference/pipelines/ocr/result.py` | `OCRResult._to_img()` — side-by-side rendering, polygon overlay, text drawing |
| Config loading | `paddlex/inference/models/utils/model_config.py` | `load_model_config()` — how inference.yml is parsed and passed to predictors |

### PaddleOCR Local Reference (this repo)

| Processing Step | File Path | What to Extract |
|----------------|-----------|-----------------|
| Det preprocessing ops | `ppocr/data/imaug/operators.py` | `DetResizeForTest`, `NormalizeImage`, `ToCHWImage` — operator implementations |
| DB postprocessing | `ppocr/postprocess/db_postprocess.py` | `DBPostProcess` — full algorithm with Clipper polygon expansion |
| CTC decoding | `ppocr/postprocess/rec_postprocess.py` | `CTCLabelDecode` — character dict loading, decode logic |
| Det inference flow | `tools/infer/predict_det.py` | End-to-end detection inference with pre/post-processing |
| Rec inference flow | `tools/infer/predict_rec.py` | End-to-end recognition inference with pre/post-processing |
| System pipeline | `tools/infer/predict_system.py` | Full OCR pipeline: det->crop->rec orchestration |
| inference.yml generation | `ppocr/utils/export_model.py` | `dump_infer_config()` — structure of inference.yml |
| inference.yml loading | `tools/infer/utility.py` | `load_config()` — how inference.yml is parsed at runtime |
| PP-OCRv5 det config | `configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml` | Training config (source for inference.yml det params) |
| PP-OCRv5 rec config | `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` | Training config (source for inference.yml rec params) |
| Character dictionary | `ppocr/utils/dict/ppocrv5_dict.txt` | 18,384 characters used by PP-OCRv5 recognition |

### Numerical Exactness Requirement

**CRITICAL: The iOS inference pipeline must produce numerically EXACTLY THE SAME results as the Python/PaddleX reference implementation.** This means identical detected polygon coordinates, identical recognized text strings, and identical confidence scores for the same input image. Any numerical divergence — even minor floating-point drift from different preprocessing implementations — is a bug that must be investigated and fixed. This is the single most important quality bar for the iOS demo.

### How to Use These References

1. **Before implementing any processing step**: Read the corresponding Python file(s) first
2. **During implementation**: Keep the Python file open for line-by-line comparison
3. **After implementation**: Run validation (VALID-01/02) comparing iOS output vs Python output
4. **When in doubt**: The PaddleX `feat/transformers` branch is the primary authority; PaddleOCR local files are secondary reference

## v1 Requirements

### Inference Engine

- [x] **INFER-01**: App integrates ONNX Runtime for iOS as the inference backend
- [x] **INFER-02**: App enables CoreML Execution Provider for Neural Engine acceleration
- [x] **INFER-03**: Detection model (PP-OCRv5 mobile det ONNX) loads and runs inference on device
- [x] **INFER-04**: Recognition model (PP-OCRv5 mobile rec ONNX) loads and runs inference on device
- [x] **INFER-05**: Model loading shows a progress/loading indicator to the user

### Config-Driven Pipeline

- [x] **CONF-01**: App parses `inference.yml` bundled with each model at runtime using a Swift YAML parser
- [x] **CONF-02**: Preprocessing pipeline is built dynamically from `PreProcess.transform_ops` in inference.yml
- [x] **CONF-03**: Postprocessing is configured from `PostProcess` params in inference.yml (thresholds, character dict, etc.)
- [x] **CONF-04**: Zero hardcoded preprocessing/postprocessing parameters — all behavior driven by model config
- [x] **CONF-05**: Switching to a different model (e.g., server det/rec) requires only replacing model files + inference.yml, no code changes

### Preprocessing

- [x] **PREP-01**: Detection preprocessing implements `DetResizeForTest` — ported from `ppocr/data/imaug/operators.py` and verified against PaddleX `processors.py`
- [x] **PREP-02**: Detection preprocessing implements `NormalizeImage` — ported from `ppocr/data/imaug/operators.py`, params read from inference.yml
- [x] **PREP-03**: Detection preprocessing handles RGB channel order and HWC->CHW layout conversion
- [x] **PREP-04**: Recognition preprocessing implements `OCRReisizeNormImg` — ported from PaddleX `text_recognition/processors.py`, params read from inference.yml
- [x] **PREP-05**: Recognition preprocessing implements its normalization (`pixel/255`, then `(x-0.5)/0.5`) as specified in inference.yml
- [x] **PREP-06**: All preprocessing uses pure Swift (Accelerate/vImage), no OpenCV dependency

### Postprocessing

- [x] **POST-01**: Detection postprocessing implements `DBPostProcess` — ported from `ppocr/postprocess/db_postprocess.py` and verified against PaddleX `processors.py`
- [x] **POST-02**: DB post-process supports polygon clipping (Clipper algorithm equivalent in Swift) — matching pyclipper `JT_ROUND, ET_CLOSEDPOLYGON` behavior
- [x] **POST-03**: Recognition postprocessing implements `CTCLabelDecode` — ported from `ppocr/postprocess/rec_postprocess.py`, character dict loaded from inference.yml
- [x] **POST-04**: Recognition confidence is computed as mean of selected token probabilities

### OCR Pipeline

- [x] **PIPE-01**: End-to-end pipeline orchestrates: detect -> sort boxes -> crop -> recognize — ported from PaddleX `pipelines/ocr/pipeline.py`
- [x] **PIPE-02**: Box sorting follows reading order — matching `SortQuadBoxes` logic in PaddleX pipeline (top-to-bottom, left-to-right with y-threshold of 10px)
- [x] **PIPE-03**: Crop uses perspective transform — matching `CropByPolys.get_minarea_rect_crop()` in PaddleX (minAreaRect -> sortCorners -> warpPerspective -> rot90 for tall-narrow boxes)
- [x] **PIPE-04**: Pipeline runs on background thread, UI remains responsive
- [x] **PIPE-05**: Pipeline results match the Python/PaddleX reference implementation — verified by VALID-01/02

### Validation

- [x] **VALID-01**: Developer-facing validation script compares iOS inference output against Python reference output for a set of test images — results must be numerically exact (not approximate)
- [x] **VALID-02**: Validation covers both detection (polygon coordinates must match exactly) and recognition (text strings and confidence scores must match exactly)

### UI

- [x] **UI-01**: Photo picker allows selecting images from the device photo album
- [x] **UI-02**: 3-5 bundled sample images available for quick testing without photo access
- [ ] **UI-03**: Detection results shown as bounding box overlays on the source image
- [ ] **UI-04**: Recognized text displayed with per-result confidence scores
- [ ] **UI-05**: Per-stage timing breakdown shown (detection time, recognition time, total time)
- [ ] **UI-06**: Copy recognized text to clipboard
- [x] **UI-07**: Loading indicator during model initialization and inference
- [x] **UI-08**: Error states displayed for failed inference or invalid input

### Documentation

- [ ] **DOC-01**: README with build instructions (Xcode version, iOS target, dependencies, how to run)
- [ ] **DOC-02**: Architecture guide explaining code structure, inference pipeline, and design decisions
- [ ] **DOC-03**: Integration guide showing developers how to use PaddleOCR inference in their own iOS app

## v2 Requirements

### Camera

- **CAM-01**: Real-time camera OCR with live bounding box overlay
- **CAM-02**: Camera capture + OCR on captured frame

### Extended Models

- **EXT-01**: Support for PP-OCRv5 server models (larger, higher accuracy)
- **EXT-02**: Detection-only and recognition-only standalone modes
- **EXT-03**: Direction classifier integration for rotated text

### Advanced UI

- **ADV-01**: Side-by-side visualization (detection overlay + recognition text, matching PaddleX style)
- **ADV-02**: Per-box tap to highlight corresponding recognized text
- **ADV-03**: Export results as JSON

## Out of Scope

| Feature | Reason |
|---------|--------|
| PaddleLite integration | No longer maintained by Baidu |
| Model training on device | Inference only — training requires server infrastructure |
| Model conversion in app | PaddleX already handles Paddle->ONNX export; demo accepts pre-converted models |
| App Store distribution | Developer demo, not a consumer product |
| Android demo updates | Separate effort, different codebase |
| Document structure analysis (PP-StructureV3) | Focus on core OCR first; can be added in future milestone |
| OpenCV dependency | Pure Swift preprocessing preferred for lightweight demo |
| Model download at runtime | Models bundled with app for simplicity; runtime download deferred to v2 |
| Multi-language UI | English only for v1 |
| Batch processing of multiple images | Single image at a time for demo simplicity |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFER-01 | Phase 1 | Complete |
| INFER-02 | Phase 1 | Complete |
| INFER-03 | Phase 1 | Complete |
| INFER-04 | Phase 1 | Complete |
| INFER-05 | Phase 1 | Complete |
| CONF-01 | Phase 2 | Complete |
| CONF-02 | Phase 4 | Complete |
| CONF-03 | Phase 4 | Complete |
| CONF-04 | Phase 4 | Complete |
| CONF-05 | Phase 4 | Complete |
| PREP-01 | Phase 2 | Complete |
| PREP-02 | Phase 2 | Complete |
| PREP-03 | Phase 2 | Complete |
| PREP-04 | Phase 3 | Complete |
| PREP-05 | Phase 3 | Complete |
| PREP-06 | Phase 2 | Complete |
| POST-01 | Phase 2 | Complete |
| POST-02 | Phase 2 | Complete |
| POST-03 | Phase 3 | Complete |
| POST-04 | Phase 3 | Complete |
| PIPE-01 | Phase 4 | Complete |
| PIPE-02 | Phase 4 | Complete |
| PIPE-03 | Phase 4 | Complete |
| PIPE-04 | Phase 4 | Complete |
| PIPE-05 | Phase 4 | Complete |
| VALID-01 | Phase 4 | Complete |
| VALID-02 | Phase 4 | Complete |
| UI-01 | Phase 5 | Complete |
| UI-02 | Phase 5 | Complete |
| UI-03 | Phase 5 | Pending |
| UI-04 | Phase 5 | Pending |
| UI-05 | Phase 5 | Pending |
| UI-06 | Phase 5 | Pending |
| UI-07 | Phase 5 | Complete |
| UI-08 | Phase 5 | Complete |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |
| DOC-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 38 total
- Mapped to phases: 38
- Unmapped: 0

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03 after roadmap creation*
