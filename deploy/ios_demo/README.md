# iOS Demo

SwiftUI demo that runs OCR on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/).

## Layout

All app sources, bundled resources, and third-party **source** vendored for this demo live under **`PaddleOCRDemo/`**. Unit tests are in **`PaddleOCRDemoTests/`** next to the Xcode project. The project root also contains `Podfile`, **`scripts/`**, `README.md`, and `NOTICE`.

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

From the **project root**:

```bash
pod install
./scripts/fetch_ios_demo_models.sh
```

`scripts/fetch_ios_demo_models.sh` downloads ONNX bundles into **`PaddleOCRDemo/Models/`**. Intermediate `.tar` caches are stored under **`.fetch_ios_demo_models_work/`** at the project root.

Optionally, pass the **model preset** (bundle name such as `PP-OCRv6_small`) as a positional argument:

```bash
./scripts/fetch_ios_demo_models.sh PP-OCRv6_small
```

Currently, the supported model presets are `PP-OCRv6_small` and `PP-OCRv6_tiny`. The default preset is `PP-OCRv6_small`.

## Quantize ONNX models on the host (optional)

To build **INT8** variants using [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), ensure that Python 3.8 or newer is installed in your host environment.

First, install the required Python dependencies:

```bash
python3 -m pip install -r requirements-onnx-quantize.txt
```

Next, run the quantization script:

```bash
python3 scripts/quantize_onnx_model.py \
  --input-model-dir PaddleOCRDemo/Models/det \
  --output-model-dir /path/to/det_quant \
  --mode dynamic
```

* **`--mode dynamic`**
  Uses `quantize_dynamic`, which performs weight-only quantization and does **not** require a calibration dataset.

* **`--mode static`**
  Uses `quantize_static` (QDQ format). This mode **requires** a calibration dataset specified via `--calib-data-dir`.

  * The calibration directory should contain **float32 `.npy` files**, with **one file per sample**.
  * Each tensor must match the shape of the model’s **single input** (note: in this demo, both `det` and `rec` models have one input).
  * You can choose calibration methods such as *MinMax*, *Entropy*, or others supported by your installed `onnxruntime` version via `--calibration-method`.

**Building calibration `.npy` files (optional):** use **`scripts/build_onnx_calib_npy.py`** to turn a folder of images into tensors.

1. **Dependencies** (host Python):

   ```bash
   python3 -m pip install -r requirements-build-calib.txt
   ```

2. **Run the build script**:

   | Flag | Role |
   | --- | --- |
   | `--task` | `det` = detection model, `rec` = recognition model (must match `--model-dir`). |
   | `--model-dir` | Model directory (e.g. `PaddleOCRDemo/Models/det` or `.../rec`). |
   | `--image-dir` | Images to calibrate on (`.png` / `.jpg` / …). |
   | `--output-dir` | Where to write `.npy` files (created if missing). |
   | `--device` | Optional; e.g. `cpu` (default) or `gpu:0`. |
   | `--det-model-dir` | **Recognition only:** detection model directory. If set, each *full page* in `--image-dir` is run through det; every detected text line is rotated-cropped and written as a separate rec tensor (`…_box000.npy`, …), instead of one tensor per file. |
   | `--max-crops-per-image` | With `--det-model-dir`, limit how many boxes per page are used (default `0` = no cap). |

   For **detection** calibration, one `.npy` is produced per input image:

   ```bash
   python3 scripts/build_onnx_calib_npy.py --task det \
     --model-dir PaddleOCRDemo/Models/det \
     --image-dir /path/to/your/images \
     --output-dir /path/to/calib_npy
   ```

   For **recognition** calibration, you can either use **per-line** crops (if you have them) or pass **full pages** and point at the **det** model so line crops are generated automatically:

   ```bash
   python3 scripts/build_onnx_calib_npy.py --task rec \
     --model-dir PaddleOCRDemo/Models/rec \
     --det-model-dir PaddleOCRDemo/Models/det \
     --image-dir /path/to/full_page_images \
     --output-dir /path/to/calib_rec_crops
   ```

3. **Then static quantize** using that directory:

   ```bash
   python3 scripts/quantize_onnx_model.py \
     --input-model-dir PaddleOCRDemo/Models/det \
     --output-model-dir /path/to/det_int8 \
     --mode static \
     --calib-data-dir /path/to/calib_npy
   ```

4. **(Optional) QDQ debug**  
   For the same [ORT debugging story](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md#debugging) as the upstream *run_qdq_debug* flow: compare a **float** ONNX to your **QDQ** output using the same `.npy` inputs as calibration. This uses ORT’s `qdq_loss_debug` helpers (weight + activation SQNR).  
   * `--float-model` should be the same float graph you fed into `quantize_onnx_model.py` (if you use default `--ort-preprocess`, that means the [pre-processed](https://raw.githubusercontent.com/microsoft/onnxruntime-inference-examples/main/quantization/image_classification/cpu/ReadMe.md#pre-processing) `inference.onnx` in float form—save a copy before quant if you do not have it, or re-run pre-process once).  

   ```bash
   python3 scripts/debug_onnx_qdq.py \
     --float-model /path/to/float_for_quant.onnx \
     --qdq-model /path/to/det_int8/inference.onnx \
     --calib-data-dir /path/to/calib_npy \
     --json-report /tmp/qdq_debug.json
   ```

## Convert to ORT model format (optional)

To produce [ORT format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) weights, install the Python dependencies for the converter, then run the converter:

```bash
python3 -m pip install -r requirements-onnx-convert-ort.txt
./scripts/convert_onnx_to_ort.sh
```

By default, conversion writes `inference*.ort` next to each `inference.onnx` under `PaddleOCRDemo/Models/`, so both formats sit in the same tree and the bundle can grow. The demo loads `inference*.ort` when present, so you do not need to ship ONNX in the bundle. To keep only one weight file in the app, you can either manually delete the `inference.onnx` files, or use the `--out-dir` option to generate `inference*.ort` models in a separate directory. These generated models can then replace the originals in `PaddleOCRDemo/Models/`. Use **`--input-dir`** to point at any ONNX tree on the host; it defaults to `PaddleOCRDemo/Models` when omitted.

Place this script’s options first (`--input-dir`, `--out-dir`), then any ORT converter flags. The script inserts `--` before the model path internally so options like `--optimization_style` (which accept multiple values) are not misparsed. Example:

```bash
./scripts/convert_onnx_to_ort.sh --out-dir ./out/ort_bundles --optimization_style Runtime
```

## Open in Xcode

```bash
open PaddleOCRDemo.xcworkspace
```

If you use CocoaPods, run `pod install` in the project root first so the workspace is generated next to the `Podfile`.

Build the **PaddleOCRDemo** scheme. Ensure **`PaddleOCRDemo/Models/`** and **`PaddleOCRDemo/Resources/SampleImages/`** are included in the app target via folder references / **Copy Bundle Resources**, and **`PaddleOCRDemoTests/Fixtures/`** in the test target (as in the checked-in project). The built-in picker sample is **`general_ocr_002.jpg`**.

## Validation

**Validation** means the **automated check pipeline** for this demo: run a reference OCR, run the same image through the iOS tests, compare the text output to tolerances, and write a short report. In one pass you get:

1. **Does the app match the reference?** The pipeline compares iOS recognition to a Python PaddleOCR run on the **same** photo and models, using fixed accuracy thresholds.
2. **How fast is it on device?** The same test bundle also **times** full OCR runs and records rough **memory** figures so you can track regressions—without using the Python step for that part.

So “validation” is an umbrella for both **accuracy gating** and **on-device performance sampling**; the artifact table below lists which output file serves which role.

### Prerequisites

Complete [One-time asset setup](#one-time-asset-setup) first. For validation specifically you also need:

- PaddleOCR (with ONNX Runtime engine) for the reference step: `python3 -m pip install -r requirements-validation.txt`.
- Xcode 16 or later (validation uses `xcresulttool get test-results`, introduced in 16.0).

### Full pipeline

From the project root, **`./scripts/run_validation.sh`** drives: Python reference OCR → Xcode tests on a simulator or device → extract result attachments → compare accuracy → generate a report.

**Configuring the run:** prefer **`run_validation.sh`** flags (below). **`resolve-image`** picks the file: **`--image`**, **`--fixture`** / **`PADDLEOCR_VALIDATION_IMAGE_NAME`**, or—with none of those set—**exactly one** non-`local-*` image under **`PaddleOCRDemoTests/Fixtures/`** (a typical clone is only **`ios_ocr_validation_reference.jpg`**; zero or multiple candidates → **`--fixture`** or **`--image`**). **`PADDLEOCR_VALIDATION_IMAGE_NAME`** is always passed to the test runner as that basename so it matches **`ref.json`**. **Other** knobs (warmup, measured iterations, inference backend, etc.) are forwarded to [Test runner environment variables](#test-runner-environment-variables) **only when** you set them via a flag or **`PADDLEOCR_VALIDATION_*`**; **flags win over env** when both apply.

| Intent | Flags |
| --- | --- |
| Image from an arbitrary path | `--image <path>` (copied into `Fixtures/` as `local-*` for this run) |
| Image already under `PaddleOCRDemoTests/Fixtures/` | `--fixture <name>` (stem or `stem.ext`) |
| Benchmark intensity | `--warmup <n>`, `--measured-iterations <n>` |
| ONNX Runtime EP | `--inference-backend CORE_ML`, `XNNPACK`, or `CPU` |

```bash
./scripts/run_validation.sh                                              # default simulator (iPhone 16)
./scripts/run_validation.sh --simulator 'iPhone 17'
./scripts/run_validation.sh --udid <device-udid>
./scripts/run_validation.sh --udid <udid> --image /path/to/photo.png
./scripts/run_validation.sh --fixture ios_ocr_validation_reference --warmup 2 --measured-iterations 20
PADDLEOCR_VALIDATION_MEASURED_ITERATIONS=30 ./scripts/run_validation.sh --warmup 0
```

After compare **completes**, the script overwrites **`out/compare-summary.json`**, **`out/run-status.json`**, and **`out/validation-report.md`**. If the pipeline **errors** earlier, those files are **not** updated for this run (any existing copies are from a previous attempt).

Outputs under **`out/`**:

| Artifact | Producer | Purpose |
|---|---|---|
| `ref.json` | `ocr_reference_run.py` | Python reference OCR |
| `result.xcresult` | `xcodebuild test` | iOS test run |
| `ios-ocr-export.json` | `extract_xcresult_attachments.py` | iOS **accuracy** payload (polygons + text) from tests |
| `on-device-performance.json` | `extract_xcresult_attachments.py` | iOS **performance** stats from tests |
| `compare-summary.json` | `compare_ocr_json.py` | **Accuracy** metrics vs thresholds (`pass`) |
| `run-status.json` | `run_validation.sh` | Per-step outcomes |
| `validation-report.md` | `generate_validation_report.py` | Human-readable report |

Exit **`0`** on **PASS**; non-zero on **FAIL** (thresholds not met) or **ERROR** (halted before a finished compare). Use the script’s stderr/stdout and **`logs/`** after **ERROR**.

### Test runner environment variables

The validation **tests** read settings through variables named **`PADDLEOCR_VALIDATION_*`**. They apply to the **test runner process**, not your interactive shell unless you forward them (see table).

| How you launch tests | What to configure |
| --- | --- |
| **Xcode** | Scheme **PaddleOCRDemo** → **Test** (not Run) → **Arguments** → **Environment Variables**. Use the names below **as-is** (`PADDLEOCR_VALIDATION_…`). |
| **`xcodebuild test`** | Set **`TEST_RUNNER_` + the same name** (e.g. `TEST_RUNNER_PADDLEOCR_VALIDATION_IMAGE_NAME=…`). `xcodebuild` injects them into the test runner and strips the prefix so the test still reads **`PADDLEOCR_VALIDATION_…`**. |
| **`run_validation.sh`** | Calls `xcodebuild` with these set for you. You may export `PADDLEOCR_VALIDATION_*` in the shell first; **script flags override env** when both apply to the same setting. |

| Variable | If unset | Role |
| --- | --- | --- |
| `PADDLEOCR_VALIDATION_IMAGE_NAME` | Defaults to **`ios_ocr_validation_reference`**. Set explicitly to use another bundled image. | Bundled test image: **stem** or **`stem.ext`** under **`Fixtures/`**. |
| `PADDLEOCR_VALIDATION_WARMUP_ITERATIONS` | **3** | Untimed full OCR runs before timing (warm caches / JIT). Used only by the **performance** test. |
| `PADDLEOCR_VALIDATION_MEASURED_ITERATIONS` | **10** | Timed runs. Used only by the **performance** test. |
| `PADDLEOCR_VALIDATION_INFERENCE_BACKEND` | **CORE_ML** | ONNX Runtime EP for **`OCRValidationTests`**: **`CORE_ML`**, **`XNNPACK`**, or **`CPU`** (plain CPU execution provider; not XNNPACK). |

Non-negative integers for the two iteration variables.

### XCTest only (without the Python pipeline)

**Read this subsection when** you run or debug the **`PaddleOCRDemoTests`** target directly (**Cmd-U** in Xcode, or **`xcodebuild test`**).

**What it is for:** class **`OCRValidationTests`** produces (1) an OCR JSON attachment for **accuracy** checks, and (2) **`on-device-performance.json`**-style timing and memory stats from repeated runs.

**Environment variables:** set **`PADDLEOCR_VALIDATION_*`** as documented in [Test runner environment variables](#test-runner-environment-variables) (Xcode vs `TEST_RUNNER_` vs `run_validation.sh`).

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).
