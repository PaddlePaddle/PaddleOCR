# PP-OCRv6 Android Demo

## Introduction

This project is an Android deployment example for PaddleOCR v6, implementing mobile OCR inference using ONNX Runtime. The project adopts a **SDK and Demo separation** architecture, where the SDK module can be independently integrated into third-party applications.

## Features

- End-to-end text detection and recognition pipeline
- Supports PP-OCRv6 series ONNX models
- Detailed performance timing (detection/recognition stage breakdown)
- MVVM + Jetpack Compose Demo application
- AAR integration support

## Project Structure

```
ppocr-android/
├── ppocr-sdk/                    # OCR SDK (Android Library)
│   ├── src/main/
│   │   ├── assets/models/        # Model files directory
│   │   │   ├── det/              # Detection model: inference.onnx
│   │   │   └── rec/              # Recognition model: inference.onnx, inference.yml
│   │   └── java/com/paddle/ocr/
│   │       ├── PaddleOCR.kt      # [Public API] SDK entry point
│   │       ├── PaddleOCRConfig.kt # [Public API] Inference configuration
│   │       └── ...
│   └── build.gradle.kts
├── app/                          # Demo App
│   ├── src/main/java/com/paddle/ocr/demo/
│   │   ├── OCRApplication.kt     # Initialize SDK
│   │   └── ui/                   # Compose UI
│   └── build.gradle.kts
├── run_benchmark.sh              # Performance test script
└── README.md
```

## Requirements

| Dependency | Version |
|------------|---------|
| Android Studio | Ladybug (2024.2+) |
| JDK | 17 |
| Kotlin | 2.1.0 |
| minSdk | 26 (Android 8.0) |
| ONNX Runtime | 1.21.1 |
| OpenCV | 4.5.3 |

## Quick Start

### 1. Clone the Project

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/ppocr-android
```

### 2. Prepare Models

This project supports the following models:

| Model | HuggingFace | BOS |
|------|-------------|-----|
| **PP-OCRv6_small** | [Detection model](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) / [Recognition model](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx) | [Detection model](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_det_onnx_infer.tar) / [Recognition model](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_onnx_infer.tar) |
| **PP-OCRv6_tiny** | [Detection model](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx) / [Recognition model](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx) | [Detection model](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_onnx_infer.tar) / [Recognition model](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_onnx_infer.tar) |
| **PP-OCRv5_mobile** | [Detection model](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx) / [Recognition model](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx) | [Detection model](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_onnx_infer.tar) / [Recognition model](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_onnx_infer.tar) |

After downloading and extracting, place the files in `ppocr-sdk/src/main/assets/models/`:

- Detection model: place `inference.onnx` in `models/det/`
- Recognition model: place `inference.onnx` and `inference.yml` in `models/rec/`

### 3. Build and Run

```bash
# Build Debug APK
./gradlew :app:assembleDebug

# Install to device
./gradlew :app:installDebug
```

Or run directly from Android Studio.

### 4. Try the Demo

1. Open "PP-OCRv6 Demo" application
2. Wait for model loading to complete
3. Tap "Select from Gallery" to choose an image
4. View recognition results and timing statistics

## SDK Integration

### Option 1: Source Code Dependency

1. Copy `ppocr-sdk/` to your project root
2. Add to `settings.gradle.kts`:
   ```kotlin
   include(":ppocr-sdk")
   ```
3. Add to your app module's `build.gradle.kts`:
   ```kotlin
   implementation(project(":ppocr-sdk"))
   ```

### Option 2: AAR Dependency

```bash
# Build AAR
./gradlew :ppocr-sdk:assembleRelease
```

AAR output: `ppocr-sdk/build/outputs/aar/ppocr-sdk-release.aar`

Add to your app module's `build.gradle.kts`:

```kotlin
dependencies {
    implementation(files("libs/ppocr-sdk-release.aar"))
    // AAR doesn't transit dependencies, add manually
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.21.1")
    implementation("com.quickbirdstudios:opencv:4.5.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
```

## API Reference

### Create Instance

```kotlin
// Default configuration
val ocr = PaddleOCR.create(context)

// Custom configuration
val ocr = PaddleOCR.create(
    context = context,
    config = PaddleOCRConfig(
        detThresh = 0.3f,
        detBoxThresh = 0.6f,
        recScoreThresh = 0.0f,
        recBatchSize = 1,
    ),
    engineConfig = EngineConfig(numThreads = 4),
    detModelAssetPath = "models/det/inference.onnx",
    recModelAssetPath = "models/rec/inference.onnx",
    recConfigAssetPath = "models/rec/inference.yml",
)
```

### Perform OCR

```kotlin
// Pass Bitmap
val result = ocr.recognize(bitmap)

// Pass image bytes (recommended, consistent with Python pipeline)
val result = ocr.recognize(imageBytes)

// Read results
result.results.forEach { item ->
    println("Text: ${item.text}, Confidence: ${item.confidence}")
    println("Box: ${item.box.points}")
}
println("Detection: ${result.detectionTimeMs}ms, Recognition: ${result.recognitionTimeMs}ms")
```

### Release Resources

```kotlin
ocr.release()
```

### Configuration Parameters

```kotlin
data class PaddleOCRConfig(
    val detImgMode: String = "BGR",         // Input color mode
    val detLimitSideLen: Int = 64,          // Detection side length limit
    val detLimitType: String = "min",       // Limit strategy
    val detMaxSideLimit: Int = 4000,        // Maximum side length
    val detThresh: Float = 0.3f,            // Binarization threshold
    val detBoxThresh: Float = 0.6f,         // Detection box confidence threshold
    val detUnclipRatio: Float = 1.5f,       // Detection box expansion ratio
    val detMaxCandidates: Int = 3000,       // Maximum candidate boxes
    val detUseDilation: Boolean = false,    // Whether to dilate
    val detScoreMode: String = "fast",      // Scoring mode
    val detBoxType: String = "quad",        // Detection box type
    val recScoreThresh: Float = 0.0f,       // Recognition confidence threshold
    val recBatchSize: Int = 1,              // Recognition batch size
)
```

### Result Models

```kotlin
data class OCRRunResult(
    val results: List<OCRResult>,       // Recognition result list
    val detectionTimeMs: Long,          // Detection time
    val recognitionTimeMs: Long,        // Recognition time
    val totalTimeMs: Long,              // Total time
    val lineCount: Int,                 // Number of lines
    // Detailed timing...
)

data class OCRResult(
    val box: OCRBox,                    // Detection box coordinates
    val text: String,                   // Recognized text
    val confidence: Float,              // Confidence score
)
```

## Performance Testing

The project provides an automated performance testing script:

```bash
# Run benchmark (10 tests, 3 warmup)
./run_benchmark.sh 10 3

# Sample output
╔═════════════════════════════════════════════════════════════════════════╗
║  PP-OCRv6 Speed Benchmark Results                                       ║
╠═════════════════════════════════════════════════════════════════════════╣
║  Device: GM1900  |  OS: Android 9  |  Lines: 5                          ║
║  Cold load: 158ms  |  Warmup: 3  |  Measured: 10                        ║
╠═════════════════════════════════════════════════════════════════════════╣
+-----------------------------+----------+----------+----------+----------+
| Stage                       |  Mean ms |    Stdev |      P90 |    Min ms|
+-----------------------------+----------+----------+----------+----------+
| Total pipeline              |   420.40 |     6.37 |      427 |      413 |
+-----------------------------+----------+----------+----------+----------+
|   Detection (total)         |   348.70 |     4.67 |      356 |      343 |
|     Preprocess              |    33.30 |     2.90 |       36 |       28 |
|     Inference               |   311.00 |     2.93 |      315 |      304 |
|     Postprocess             |     4.40 |     0.49 |        5 |        4 |
|   Recognition (total)       |    66.20 |     3.16 |       68 |       64 |
|     Preprocess              |     3.00 |     0.89 |        4 |        2 |
|     Inference               |    60.60 |     3.14 |       63 |       58 |
|     Postprocess             |     2.60 |     0.92 |        4 |        1 |
|   Pipeline overhead         |     5.50 |     0.50 |        6 |        5 |
+-----------------------------+----------+----------+----------+----------+
╚═════════════════════════════════════════════════════════════════════════╝
```

## Notes

1. **OpenCV Initialization**: Call `OpenCVUtils.init(context)` before `PaddleOCR.create()`
2. **Coroutine Usage**: `create()` and `recognize()` are suspend functions, call them in coroutines
3. **Memory Management**: Call `release()` when no longer needed
4. **ProGuard Rules**: Refer to `ppocr-sdk/proguard-rules.pro`
