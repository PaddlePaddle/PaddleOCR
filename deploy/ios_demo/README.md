# iOS Demo

SwiftUI demo that runs OCR on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/).

## Layout

All app sources, bundled resources, and third-party **source** vendored for this demo live under **`PaddleOCRDemo/`**. Unit tests are in **`PaddleOCRDemoTests/`** next to the Xcode project. The project root also contains `Podfile`, **`Scripts/`**, `README.md`, and `NOTICE`.

> **Paths in this README** assume your shell’s working directory is the **project root** (the folder that contains `Podfile` and `Scripts/`). If you use a checkout of the full PaddleOCR repository, that folder is **`deploy/ios_demo/`**.

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

From the **project root**:

```bash
pod install
./Scripts/fetch_ios_demo_assets.sh
```

`Scripts/fetch_ios_demo_assets.sh` downloads ONNX bundles into **`PaddleOCRDemo/Models/`** and fetches demo images into **`PaddleOCRDemo/Resources/SampleImages/`**. Intermediate `.tar` caches are stored under **`.fetch_ios_demo_assets_work/`** at the project root.

Optionally, pass the **model preset** (bundle name such as `PP-OCRv6_small`) as a positional argument after any options:

```bash
./Scripts/fetch_ios_demo_assets.sh PP-OCRv6_small
```

Currently, the supported model presets are `PP-OCRv6_small` and `PP-OCRv6_tiny`. The default preset is `PP-OCRv6_small`.

Flags:

| Flag | Meaning |
|------|---------|
| `--models-only` | ONNX models only |
| `--samples-only` | Sample image only |

## Open in Xcode

```bash
open PaddleOCRDemo.xcworkspace
```

If you use CocoaPods, run `pod install` in the project root first so the workspace is generated next to the `Podfile`.

Build the **PaddleOCRDemo** scheme. Ensure **`PaddleOCRDemo/Models/`** and **`PaddleOCRDemo/Resources/SampleImages/`** are included via folder references / **Copy Bundle Resources** (as in the checked-in project).

## Validation

One command runs the full validation pipeline on a physical iPhone or simulator:

```bash
./Scripts/run_validation.sh                           # default simulator (iPhone 16)
./Scripts/run_validation.sh --simulator 'iPhone 17'   # specific simulator
./Scripts/run_validation.sh --udid <device-udid>      # connected real device
./Scripts/run_validation.sh --udid <udid> --image /path/to/photo.png   # ad-hoc image
```

Prerequisites:

- `./Scripts/fetch_ios_demo_assets.sh` has populated `PaddleOCRDemo/Models/`
  **and** `PaddleOCRDemoTests/Fixtures/` (both are filled by the same script;
  the default validation fixture is `general_ocr_002.png`).
- PaddleOCR (with ONNX Runtime engine) is installed for the reference step:
  `pip install -r Scripts/requirements-validation.txt`.
- Xcode 16 or later (validation uses `xcresulttool get test-results`, introduced in 16.0).

The runner produces the following under `out/`:

| Artifact | Producer | Purpose |
|---|---|---|
| `ref.json` | `ocr_reference_run.py` | Python reference OCR |
| `result.xcresult` | `xcodebuild test` | iOS test run |
| `ios-ocr-export.json`, `on-device-performance.json` | `extract_xcresult_attachments.py` | iOS outputs pulled from `.xcresult` |
| `compare-summary.json` | `compare_ocr_json.py` | Accuracy verdict |
| `run-status.json` | `run_validation.sh` | Per-step outcomes |
| `validation-report.md` | `generate_validation_report.py` | Human-readable report |

The script exits `0` on PASS (all steps OK, compare under thresholds), non-zero on FAIL or ERROR. The report is always written, including on failure; check the `**Overall:**` line at the top.

### Running individual steps manually

The underlying scripts remain independently invokable — see `./Scripts/<script>.py --help`. Tests read fixtures from the test bundle and write outputs via `XCTAttachment`; `run_validation.sh` orchestrates the full flow.

### Running the benchmark tests directly from Xcode

Set `PADDLEOCR_VALIDATION_IMAGE_NAME=<filename>` on the `PaddleOCRDemo` scheme (Test → Arguments → Environment Variables) to a file committed under `PaddleOCRDemoTests/Fixtures/` (e.g. `table.jpg`). Then Cmd-U.
