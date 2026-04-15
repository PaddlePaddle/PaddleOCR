# iOS Demo

SwiftUI demo that runs text detection and recognition on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/) (CoreML + XNNPACK execution providers).

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

```bash
pod install
./fetch_ios_demo_assets.sh
```

`fetch_ios_demo_assets.sh` downloads ONNX bundles into `PaddleOCRDemo/PaddleOCRDemo/Models/` and fetches demo images into `PaddleOCRDemo/PaddleOCRDemo/Resources/SampleImages/`.

Optionally, pass the model variant as a positional argument after any options (common CLI style):

```bash
./fetch_ios_demo_assets.sh PP-OCR6_small
```

Currently, the supported model variants are `PP-OCRv6_mobile` and `PP-OCRv6_tiny`. The default variant is `PP-OCRv6_small`.

Flags:

| Flag | Meaning |
|------|---------|
| `--models-only` | ONNX models only |
| `--samples-only` | Sample image only |

## Open in Xcode

```bash
open PaddleOCRDemo/PaddleOCRDemo.xcworkspace
```

Build the **PaddleOCRDemo** scheme. Ensure **Models/** and **Resources/SampleImages/** are included via folder references / **Copy Bundle Resources** (as in the checked-in project).
