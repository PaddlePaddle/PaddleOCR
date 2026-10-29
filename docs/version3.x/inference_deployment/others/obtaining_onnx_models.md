---
comments: true
---

# 获取 ONNX 模型

PaddleOCR 提供了丰富的预训练模型，这些模型均采用飞桨的静态图格式进行存储。若需在部署阶段使用 ONNX 格式的模型，可借助 PaddleX 提供的 Paddle2ONNX 插件进行转换。关于 PaddleX 及其与 PaddleOCR 之间的关系，请参考 [PaddleOCR 与 PaddleX 的区别与联系](../../paddleocr_and_paddlex.md#1-paddleocr-paddlex)。

首先，执行如下命令，通过 PaddleX CLI 安装 PaddleX 的 Paddle2ONNX 插件：

```bash
# Windows 用户需使用以下命令安装 paddlepaddle dev版本
# python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

paddlex --install paddle2onnx
```

然后，执行如下命令完成模型转换：

```bash
paddlex \
    --paddle2onnx \  # 使用paddle2onnx功能
    --paddle_model_dir /your/paddle_model/dir \  # 指定 Paddle 模型所在的目录
    --onnx_model_dir /your/onnx_model/output/dir \  # 指定转换后 ONNX 模型的输出目录
    --opset_version 7  # 指定要使用的 ONNX opset 版本
```

参数说明如下：

<table>
    <thead>
        <tr>
            <th>参数</th>
            <th>类型</th>
            <th>描述</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>paddle_model_dir</td>
            <td>str</td>
            <td>包含 Paddle 模型的目录。</td>
        </tr>
        <tr>
            <td>onnx_model_dir</td>
            <td>str</td>
            <td>ONNX 模型的输出目录，可以与 Paddle 模型目录相同。默认为 <code>onnx</code>。</td>
        </tr>
        <tr>
            <td>opset_version</td>
            <td>int</td>
            <td>使用的 ONNX opset 版本。当使用低版本 opset 无法完成转换时，将自动选择更高版本的 opset 进行转换。默认为 <code>7</code>。</td>
        </tr>
    </tbody>
</table>

## 直接运行 ONNX 模型

推荐通过 PaddleOCR API 并设置 `engine="onnxruntime"` 来运行 ONNX 模型（参考 [推理引擎与配置](../local_inference/inference_engine.md)），前处理和后处理会自动完成。如果直接调用模型（例如使用 `onnxruntime`，或在其他语言中调用），输入张量必须按照 PaddleOCR 的方式准备。每个模型目录中都有 `inference.yml` 文件，其中的 `PreProcess` 和 `PostProcess` 部分列出了该模型的处理步骤和参数。

对于文本检测模型（例如 `PP-OCRv6_tiny_det`），输入为形状为 `[1, 3, H, W]`（NCHW）的 `float32` 张量：

- 图像保持 `cv2.imread` 返回的 BGR 通道顺序，并直接对其执行 `NormalizeImage`：先乘以 `1/255`，再减去 `mean` 并除以 `std`。
- 图像需要从 HWC 转置为 CHW（`ToCHWImage`）。OpenCV、PIL 和 NumPy 读取的图像默认均为 HWC。
- `H` 和 `W` 都必须是 32 的倍数（`DetResizeForTest`）。

输出为形状为 `[1, 1, H, W]` 的文本概率图，尺寸与输入相同。大于 `PostProcess.thresh` 的像素为文本，`DBPostProcess` 再将这些区域转换为文本框。

```python
import cv2
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("inference.onnx")

img = cv2.imread("input.png")  # BGR, HWC, uint8
h, w = img.shape[:2]
# DetResizeForTest: 宽和高都必须是 32 的倍数
img = cv2.resize(img, (max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)))
x = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.transpose(2, 0, 1)[None].astype(np.float32)  # HWC -> NCHW

prob = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]  # (H, W) 文本概率
mask = (prob > 0.2).astype(np.uint8)  # inference.yml 中的 PostProcess.thresh
```

如果检测区域的形状看起来正确，但位置发生了偏移和拉伸，很可能是 HWC 图像未经转置就直接输入了模型：此时三个"通道"各自包含图像三分之一的行，文本在每一部分中出现在约三倍于原坐标的位置。
