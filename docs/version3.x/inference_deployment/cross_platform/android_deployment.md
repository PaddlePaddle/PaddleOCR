# PP-OCRv6 Android Demo

## 简介

本项目是 PaddleOCR v6 在 Android 平台的部署示例，基于 ONNX Runtime 实现移动端 OCR 推理。项目采用 **SDK 与 Demo 分离** 架构，SDK 模块可独立集成到第三方应用。

## 功能特性

- 文本检测 + 文本识别端到端流程
- 支持 PP-OCRv6 系列 ONNX 模型
- 详细的性能计时（检测/识别各阶段耗时）
- MVVM + Jetpack Compose Demo 应用
- 支持 AAR 方式集成

## 项目结构

```
ppocr-android/
├── ppocr-sdk/                    # OCR SDK（Android Library）
│   ├── src/main/
│   │   ├── assets/models/        # 模型文件目录
│   │   │   ├── det/              # 检测模型：inference.onnx
│   │   │   └── rec/              # 识别模型：inference.onnx, inference.yml
│   │   └── java/com/paddle/ocr/
│   │       ├── PaddleOCR.kt      # [公开 API] SDK 入口
│   │       ├── PaddleOCRConfig.kt # [公开 API] 推理参数配置
│   │       └── ...
│   └── build.gradle.kts
├── app/                          # Demo App
│   ├── src/main/java/com/paddle/ocr/demo/
│   │   ├── OCRApplication.kt     # 初始化 SDK
│   │   └── ui/                   # Compose UI
│   └── build.gradle.kts
├── run_benchmark.sh              # 性能测试脚本
└── README.md
```

## 环境要求

| 依赖 | 版本 |
|------|------|
| Android Studio | Ladybug (2024.2+) |
| JDK | 17 |
| Kotlin | 2.1.0 |
| minSdk | 26 (Android 8.0) |
| ONNX Runtime | 1.21.1 |
| OpenCV | 4.5.3 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/ppocr-android
```

### 2. 准备模型

本项目支持以下模型：

| 模型 | HuggingFace | BOS |
|------|-------------|-----|
| **PP-OCRv6_small** | [检测模型](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) / [识别模型](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx) | [检测模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_det_onnx_infer.tar) / [识别模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_onnx_infer.tar) |
| **PP-OCRv6_tiny** | [检测模型](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx) / [识别模型](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx) | [检测模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_onnx_infer.tar) / [识别模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_onnx_infer.tar) |
| **PP-OCRv5_mobile** | [检测模型](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx) / [识别模型](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx) | [检测模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_onnx_infer.tar) / [识别模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_onnx_infer.tar) |

下载并解压后，将文件放入 `ppocr-sdk/src/main/assets/models/` 目录：

- 检测模型：将 `inference.onnx` 放入 `models/det/`
- 识别模型：将 `inference.onnx` 和 `inference.yml` 放入 `models/rec/`

### 3. 编译运行

```bash
# 编译 Debug APK
./gradlew :app:assembleDebug

# 安装到设备
./gradlew :app:installDebug
```

或使用 Android Studio 直接运行。

### 4. 体验 Demo

1. 打开 "PP-OCRv6 Demo" 应用
2. 等待模型加载完成
3. 点击 "Select from Gallery" 选择图片
4. 查看识别结果和耗时统计

## SDK 集成

### 方式一：源码依赖

1. 将 `ppocr-sdk/` 复制到项目根目录
2. 在 `settings.gradle.kts` 添加：
   ```kotlin
   include(":ppocr-sdk")
   ```
3. 在 App 模块 `build.gradle.kts` 添加：
   ```kotlin
   implementation(project(":ppocr-sdk"))
   ```

### 方式二：AAR 依赖

```bash
# 构建 AAR
./gradlew :ppocr-sdk:assembleRelease
```

AAR 输出：`ppocr-sdk/build/outputs/aar/ppocr-sdk-release.aar`

在 App 模块 `build.gradle.kts` 添加：

```kotlin
dependencies {
    implementation(files("libs/ppocr-sdk-release.aar"))
    // AAR 不传递依赖，需手动添加
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.21.1")
    implementation("com.quickbirdstudios:opencv:4.5.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
```

## API 参考

### 创建实例

```kotlin
// 默认配置
val ocr = PaddleOCR.create(context)

// 自定义配置
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

### 执行 OCR

```kotlin
// 传入 Bitmap
val result = ocr.recognize(bitmap)

// 传入图片字节数据（推荐，与 Python 流程一致）
val result = ocr.recognize(imageBytes)

// 读取结果
result.results.forEach { item ->
    println("文本: ${item.text}, 置信度: ${item.confidence}")
    println("坐标: ${item.box.points}")
}
println("检测: ${result.detectionTimeMs}ms, 识别: ${result.recognitionTimeMs}ms")
```

### 释放资源

```kotlin
ocr.release()
```

### 配置参数

```kotlin
data class PaddleOCRConfig(
    val detImgMode: String = "BGR",         // 输入色彩模式
    val detLimitSideLen: Int = 64,          // 检测侧边长限制
    val detLimitType: String = "min",       // 限制策略
    val detMaxSideLimit: Int = 4000,        // 最长边上限
    val detThresh: Float = 0.3f,            // 二值化阈值
    val detBoxThresh: Float = 0.6f,         // 检测框置信度阈值
    val detUnclipRatio: Float = 1.5f,       // 检测框扩展比例
    val detMaxCandidates: Int = 3000,       // 最大候选框数
    val detUseDilation: Boolean = false,    // 是否膨胀
    val detScoreMode: String = "fast",      // 打分模式
    val detBoxType: String = "quad",        // 检测框类型
    val recScoreThresh: Float = 0.0f,       // 识别置信度阈值
    val recBatchSize: Int = 1,              // 识别批大小
)
```

### 结果模型

```kotlin
data class OCRRunResult(
    val results: List<OCRResult>,       // 识别结果列表
    val detectionTimeMs: Long,          // 检测耗时
    val recognitionTimeMs: Long,        // 识别耗时
    val totalTimeMs: Long,              // 总耗时
    val lineCount: Int,                 // 识别行数
    // 详细计时...
)

data class OCRResult(
    val box: OCRBox,                    // 检测框坐标
    val text: String,                   // 识别文本
    val confidence: Float,              // 置信度
)
```

## 性能测试

项目提供自动化性能测试脚本：

```bash
# 运行 benchmark（10次测试，3次预热）
./run_benchmark.sh 10 3

# 输出示例
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

## 注意事项

1. **OpenCV 初始化**：调用 `PaddleOCR.create()` 前需先调用 `OpenCVUtils.init(context)`
2. **协程调用**：`create()` 和 `recognize()` 都是 suspend 函数，需在协程中调用
3. **内存管理**：不再使用时调用 `release()` 释放资源
4. **混淆规则**：参考 `ppocr-sdk/proguard-rules.pro`
