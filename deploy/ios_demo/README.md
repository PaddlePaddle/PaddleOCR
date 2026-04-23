# iOS Demo

SwiftUI demo that runs OCR on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/).

## Layout

All app sources, bundled resources, and third-party **source** vendored for this demo live under **`PaddleOCRDemo/`**. Unit tests are in **`PaddleOCRDemoTests/`** next to the Xcode project. The project root also contains `Podfile`, **`Scripts/`**, `README.md`, and `NOTICE`.

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

From the **project root**:

```bash
pod install
./Scripts/fetch_ios_demo_models.sh
```

`Scripts/fetch_ios_demo_models.sh` downloads ONNX bundles into **`PaddleOCRDemo/Models/`**. Intermediate `.tar` caches are stored under **`.fetch_ios_demo_models_work/`** at the project root.

Optionally, pass the **model preset** (bundle name such as `PP-OCRv6_small`) as a positional argument:

```bash
./Scripts/fetch_ios_demo_models.sh PP-OCRv6_small
```

Currently, the supported model presets are `PP-OCRv6_small` and `PP-OCRv6_tiny`. The default preset is `PP-OCRv6_small`.

## Quantize ONNX models on the host (optional)

To build **INT8** variants using [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), ensure that Python 3.8 or newer is installed in your host environment.

First, install the required Python dependencies:

```bash
python3 -m pip install -r Scripts/requirements-onnx-quantize.txt
```

Next, run the quantization script:

```bash
python3 Scripts/quantize_onnx_model.py \
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

**Building calibration `.npy` files (optional):** use **`Scripts/build_onnx_calib_npy.py`** to turn a folder of images into tensors with the **same preprocessing** PaddleX uses for ONNX inference (`paddlex.create_predictor`, `engine="onnxruntime"`). This matches what static quantization expects more reliably than hand-rolled numpy.

* Install [PaddleX](https://github.com/PaddlePaddle/PaddleX) in the same environment (e.g. `pip install -e /path/to/PaddleX` or set `PYTHONPATH` to a checkout that contains the `paddlex` package). You also need common deps (PyYAML, OpenCV, NumPy, `onnxruntime`—overlap with the quantize requirements is fine).
* **Detection** (`--task det`): one `.npy` per image; shapes follow the det config (e.g. `resize_long` in `inference.yml`). **Recognition** (`--task rec`): one `.npy` per image after rec resize/normalize; by default that is **one tensor per file** (e.g. full-page or line image). For the tightest rec calibration, prefer **text-line** crops; whole-page images are still usable for many calibration ranges.
* From the iOS demo project root:

  ```bash
  python3 Scripts/build_onnx_calib_npy.py --task det \
    --model-dir PaddleOCRDemo/Models/det \
    --image-dir /path/to/your/images \
    --output-dir /path/to/calib_npy
  # then: python3 Scripts/quantize_onnx_model.py ... --mode static --calib-data-dir /path/to/calib_npy
  ```

For additional options and details, run:

```bash
python3 Scripts/quantize_onnx_model.py --help
python3 Scripts/build_onnx_calib_npy.py --help
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

- PaddleOCR (with ONNX Runtime engine) for the reference step: `python3 -m pip install -r Scripts/requirements-validation.txt`.
- Xcode 16 or later (validation uses `xcresulttool get test-results`, introduced in 16.0).

### Full pipeline

From the project root, **`./Scripts/run_validation.sh`** drives: Python reference OCR → Xcode tests on a simulator or device → extract result attachments → compare accuracy → generate a report.

**Configuring the run:** prefer **`run_validation.sh`** flags (below). **`resolve-image`** picks the file: **`--image`**, **`--fixture`** / **`PADDLEOCR_VALIDATION_IMAGE_NAME`**, or—with none of those set—**exactly one** non-`local-*` image under **`PaddleOCRDemoTests/Fixtures/`** (a typical clone is only **`ios_ocr_validation_reference.jpg`**; zero or multiple candidates → **`--fixture`** or **`--image`**). **`PADDLEOCR_VALIDATION_IMAGE_NAME`** is always passed to the test runner as that basename so it matches **`ref.json`**. **Other** knobs (warmup, measured iterations, inference backend, etc.) are forwarded to [Test runner environment variables](#test-runner-environment-variables) **only when** you set them via a flag or **`PADDLEOCR_VALIDATION_*`**; **flags win over env** when both apply.

| Intent | Flags |
| --- | --- |
| Image from an arbitrary path | `--image <path>` (copied into `Fixtures/` as `local-*` for this run) |
| Image already under `PaddleOCRDemoTests/Fixtures/` | `--fixture <name>` (stem or `stem.ext`) |
| Benchmark intensity | `--warmup <n>`, `--measured-iterations <n>` |
| ONNX Runtime EP | `--inference-backend CORE_ML` or `XNNPACK` |

```bash
./Scripts/run_validation.sh                                              # default simulator (iPhone 16)
./Scripts/run_validation.sh --simulator 'iPhone 17'
./Scripts/run_validation.sh --udid <device-udid>
./Scripts/run_validation.sh --udid <udid> --image /path/to/photo.png
./Scripts/run_validation.sh --fixture ios_ocr_validation_reference --warmup 2 --measured-iterations 20
PADDLEOCR_VALIDATION_MEASURED_ITERATIONS=30 ./Scripts/run_validation.sh --warmup 0
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
| `PADDLEOCR_VALIDATION_INFERENCE_BACKEND` | **CORE_ML** | ONNX Runtime EP for **`OCRValidationTests`**: **`CORE_ML`** or **`XNNPACK`**. |

Non-negative integers for the two iteration variables.

### XCTest only (without the Python pipeline)

**Read this subsection when** you run or debug the **`PaddleOCRDemoTests`** target directly (**Cmd-U** in Xcode, or **`xcodebuild test`**).

**What it is for:** class **`OCRValidationTests`** produces (1) an OCR JSON attachment for **accuracy** checks, and (2) **`on-device-performance.json`**-style timing and memory stats from repeated runs.

**Environment variables:** set **`PADDLEOCR_VALIDATION_*`** as documented in [Test runner environment variables](#test-runner-environment-variables) (Xcode vs `TEST_RUNNER_` vs `run_validation.sh`).

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).
