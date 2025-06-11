# PaddleOCR 3.x Upgrade Notes

## 1. Why Upgrade from PaddleOCR 2.x to 3.x?

Since the release of PaddleOCR 2.0 in February 2021, the community has experienced over four years of rapid growth. The number of GitHub stars, community users and contributors, as well as issues and PRs, have all increased exponentially. With emerging needs such as multilingual recognition and layout analysis, PaddleOCR continued to expand its capabilities in the 2.x series. However, the original lightweight-centric architecture has struggled to accommodate the growing complexity and rising maintenance costs brought by the feature boom.

As more module branches and "bridging" layers were added to the codebase, issues such as code duplication and inconsistent interfaces became increasingly prominent. Testing became more difficult, and development efficiency was severely constrained. In addition, legacy dependencies became incompatible with newer versions of PaddlePaddle, limiting access to its latest features and slowing down training and inference. Under such circumstances, continuing to patch the existing architecture would only increase technical debt and system fragility.

Meanwhile, Transformer-based vision-language models are injecting new momentum into advanced scenarios such as document understanding, image-text summarization, and intelligent proofreading. The community is eager to go beyond traditional OCR recognition and fully harness the powerful contextual understanding and reasoning capabilities of these models. At the same time, lightweight OCR models can still work in tandem with large models—both supporting the input needs of large models in document parsing and achieving complementary strengths to further enhance overall system performance.

Moreover, the official release of PaddlePaddle 3.0 in April 2025 brought groundbreaking upgrades in unified training/inference and domestic hardware adaptation. This calls for a significant update to PaddleOCR in both its training and inference components.

Given this background, we’ve decided to implement a major, non-backward-compatible upgrade—transitioning from 2.x to 3.x. The new version introduces a modular and plugin-based architecture. While retaining familiar usage patterns for users as much as possible, it integrates large model capabilities, offers richer features, and leverages the latest advancements of PaddlePaddle 3.0. The result is reduced maintenance cost, improved performance, and a solid foundation for future feature expansion.

## 2. Key Upgrades from PaddleOCR 2.x to 3.x

The 3.x upgrade consists of three major enhancements:

1. **New Model Pipelines**: Introduced several new pipelines such as PP-OCRv5, PP-StructureV3, and PP-ChatOCR v4, covering a wide range of base models. These significantly enhance recognition capabilities for various text types, including handwriting, to meet the growing demand for high-precision parsing in complex documents. All models are ready-to-use out of the box, improving development efficiency.
2. **Refactored Deployment and Unified Inference Interface**: The deployment module in PaddleOCR 3.x is rebuilt using [PaddleX](../version3.x/paddleocr_and_paddlex.en.md)’s underlying capabilities, fixing design flaws from 2.x and unifying both Python APIs and CLI interfaces. The deployment now supports three main scenarios: high-performance inference, service-oriented deployment, and edge deployment.
3. **PaddlePaddle 3.0 Compatibility and Optimized Training**: The new version is fully compatible with PaddlePaddle 3.0, including features like the CINN compiler. It also introduces a standardized model naming system to streamline future updates and maintenance.

Some legacy features from PaddleOCR 2.x remain partially supported in 3.x. For more information, refer to [Legacy Features](../version2.x/legacy/index.en.md).

## 3. Migrating Inference Code from PaddleOCR 2.x to 3.x

For OCR tasks, PaddleOCR 3.x still supports a usage pattern similar to 2.x. Here’s an example using the Python API in 2.x:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")
result = ocr.ocr("img.png")
for res in result:
    for line in res:
        print(line)

# Visualization
from PIL import Image
from paddleocr import draw_ocr
result = result[0]
image = Image.open(img_path).convert("RGB")
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path="simfang.ttf")
im_show = Image.fromarray(im_show)
im_show.save("result.jpg")
```

In PaddleOCR 3.x, this workflow is further simplified:

```python
from paddleocr import PaddleOCR

# Basic initialization parameters remain the same
ocr = PaddleOCR(lang="en")
result = ocr.ocr("img.png")
# Or use the new unified interface
# result = ocr.predict("img.png")
for res in result:
    # Directly print recognition results, no nested loops required
    res.print()

# Visualization and saving results are simpler
res.save_to_img("result")
```

It’s worth noting that the `PPStructure` module in PaddleOCR 2.x has been removed in 3.x. We recommend switching to `PPStructureV3`, which offers richer functionality and better parsing results. Refer to the relevant documentation for usage details.

Also, in 2.x, the `show_log` parameter could be passed when creating a `PaddleOCR` object to control logging. However, this design affected all `PaddleOCR` instances due to the use of a shared logger—clearly not the expected behavior. PaddleOCR 3.x introduces a brand-new logging system to address this issue. For more details, see [Logging](../version3.x/logging.en.md).

## 4. Known Issues in PaddleOCR 3.0

PaddleOCR 3.0 is still under active development. Current known limitations include:

1. Incomplete support for native C++ deployment.
2. High-performance service-oriented deployment is not yet on par with PaddleServing in 2.x.
3. Edge deployment currently supports only a subset of key models, with broader support pending.

If you encounter any issues during use, feel free to submit feedback via GitHub issues. We also warmly welcome more community members to contribute to PaddleOCR's future. Thank you for your continued support and interest!
