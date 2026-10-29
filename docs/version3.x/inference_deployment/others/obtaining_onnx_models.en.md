---
comments: true
---

# Obtaining ONNX Models

PaddleOCR provides a rich collection of pre-trained models, all stored in PaddlePaddle's static graph format. To use these models in ONNX format during deployment, you can convert them using the Paddle2ONNX plugin provided by PaddleX. For more information about PaddleX and its relationship with PaddleOCR, refer to [Differences and Connections Between PaddleOCR and PaddleX](../../paddleocr_and_paddlex.en.md#1-differences-and-connections-between-paddleocr-and-paddlex).

First, install the Paddle2ONNX plugin for PaddleX using the following command via the PaddleX CLI:

```bash
# Windows users need to use the following command to install the dev version of paddlepaddle
# python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

paddlex --install paddle2onnx
```

Then, execute the following command to complete the model conversion:

```bash
paddlex \
    --paddle2onnx \  # Use the paddle2onnx feature
    --paddle_model_dir /your/paddle_model/dir \  # Specify the directory containing the Paddle model
    --onnx_model_dir /your/onnx_model/output/dir \  # Specify the output directory for the converted ONNX model
    --opset_version 7  # Specify the ONNX opset version to use
```

The parameters are described as follows:

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>paddle_model_dir</td>
            <td>str</td>
            <td>The directory containing the Paddle model.</td>
        </tr>
        <tr>
            <td>onnx_model_dir</td>
            <td>str</td>
            <td>The output directory for the ONNX model. It can be the same as the Paddle model directory. Defaults to <code>onnx</code>.</td>
        </tr>
        <tr>
            <td>opset_version</td>
            <td>int</td>
            <td>The ONNX opset version to use. If conversion fails with a lower opset version, a higher version will be automatically selected for conversion. Defaults to <code>7</code>.</td>
        </tr>
    </tbody>
</table>

## Running the ONNX Model Directly

The recommended way to run an ONNX model is through the PaddleOCR API with `engine="onnxruntime"` (see [Inference Engine and Configuration](../local_inference/inference_engine.en.md)), which handles preprocessing and postprocessing for you. If you call the model directly, for example with `onnxruntime` or from another language, the input tensor must be prepared exactly as PaddleOCR does it. Each model directory contains an `inference.yml` file whose `PreProcess` and `PostProcess` sections list the steps and parameters for that model.

For text detection models (for example, `PP-OCRv6_tiny_det`), the input is a `float32` tensor of shape `[1, 3, H, W]` (NCHW):

- The image keeps the BGR channel order returned by `cv2.imread`, and `NormalizeImage` is applied to it as is: scale by `1/255`, then subtract `mean` and divide by `std`.
- The image is transposed from HWC to CHW (`ToCHWImage`). Images from OpenCV, PIL, or NumPy are HWC by default.
- Both `H` and `W` must be multiples of 32 (`DetResizeForTest`).

The output is a text probability map of shape `[1, 1, H, W]` with the same size as the input. Pixels above `PostProcess.thresh` are text; `DBPostProcess` then turns these regions into boxes.

```python
import cv2
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("inference.onnx")

img = cv2.imread("input.png")  # BGR, HWC, uint8
h, w = img.shape[:2]
# DetResizeForTest: both sides must be multiples of 32
img = cv2.resize(img, (max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)))
x = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.transpose(2, 0, 1)[None].astype(np.float32)  # HWC -> NCHW

prob = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]  # (H, W) text probability
mask = (prob > 0.2).astype(np.uint8)  # PostProcess.thresh in inference.yml
```

If the detected regions look correct in shape but shifted and stretched, the HWC image was most likely passed without the transpose: each of the three "channels" then holds one third of the image rows, and text appears at roughly three times its coordinates within each third.
