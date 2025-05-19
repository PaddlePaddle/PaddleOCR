# Obtaining ONNX Models

PaddleOCR provides a rich collection of pre-trained models, all stored in PaddlePaddle's static graph format. To use these models in ONNX format during deployment, you can convert them using the Paddle2ONNX plugin provided by PaddleX. For more information about PaddleX and its relationship with PaddleOCR, refer to [Differences and Connections Between PaddleOCR and PaddleX](../paddleocr_and_paddlex.en.md#1-Differences-and-Connections-Between-PaddleOCR-and-PaddleX).

First, install the Paddle2ONNX plugin for PaddleX using the following command via the PaddleX CLI:

```bash
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
