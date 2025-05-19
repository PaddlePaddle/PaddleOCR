# 获取 ONNX 模型

PaddleOCR 提供了丰富的预训练模型，这些模型均采用飞桨的静态图格式进行存储。若需在部署阶段使用 ONNX 格式的模型，可借助 PaddleX 提供的 Paddle2ONNX 插件进行转换。关于 PaddleX 及其与 PaddleOCR 之间的关系，请参考 [PaddleOCR 与 PaddleX 的区别与联系](../paddleocr_and_paddlex.md#1-paddleocr-与-paddlex-的区别与联系)。

首先，执行如下命令，通过 PaddleX CLI 安装 PaddleX 的 Paddle2ONNX 插件：

```bash
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
