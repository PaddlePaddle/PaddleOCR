---
comments: true
---

# 文本行方向分类模块使用教程

## 一、概述

文本行方向分类模块主要是将文本行的方向区分出来，并使用后处理将其矫正。在诸如文档扫描、证照拍摄等过程中，有时为了拍摄更清晰，会将拍摄设备进行旋转，导致得到的文本行也是不同方向的。此时，标准的OCR流程无法很好地应对这些数据。利用图像分类技术，可以预先判断文本行方向，并将其进行方向调整，从而提高OCR处理的准确性。

## 二、支持模型列表


<table>
<thead>
<tr>
<th>模型</th>
<th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">训练模型</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>基于PP-LCNet_x0_25的文本行分类模型，含有两个类别，即0度，180度</td>
</tr>
</tbody>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
              <li><strong>测试数据集：</strong>PaddleOCR 自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>其他环境：Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>推理模式说明</b></li>
  </ul>


<table border="1">
    <thead>
        <tr>
            <th>模式</th>
            <th>GPU配置</th>
            <th>CPU配置</th>
            <th>加速技术组合</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>常规模式</td>
            <td>FP32精度 / 无TRT加速</td>
            <td>FP32精度 / 8线程</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>高性能模式</td>
            <td>选择先验精度类型和加速策略的最优组合</td>
            <td>FP32精度 / 8线程</td>
            <td>选择先验最优后端（Paddle/OpenVINO/TRT等）</td>
        </tr>
    </tbody>
</table>

## 三、快速开始

> ❗ 在快速开始前，请先安装 PaddleOCR 的 wheel 包，详细请参考 [安装教程](../installation.md)。

使用一行命令即可快速体验：

```bash
paddleocr text_line_orientation_classification -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg
```

您也可以将文本行方向分类模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg)到本地。

```python
from paddleocr import TextLineOrientationClassification
model = TextLineOrientationClassification(model_name="PP-LCNet_x0_25_textline_ori")
output = model.predict("textline_rot180_demo.jpg",  batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': {'input_path': 'textline_rot180_demo.jpg', 'page_index': None, 'class_ids': array([1], dtype=int32), 'scores': array([1.], dtype=float32), 'label_names': ['180_degree']}}
```

运行结果参数含义如下：
- `input_path`：表示输入图片的路径。
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `class_ids`：表示预测结果的类别 id，含有两个类别，即0度和180度。
- `scores`：表示预测结果的置信度。
- `label_names`：表示预测结果的类别名。

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/textline_ori_classification/textline_rot180_demo_res.jpg">

相关方法、参数等说明如下：

* `TextLineOrientationClassification`实例化文本行方向分类模型（此处以`PP-LCNet_x0_25_textline_ori`为例），具体说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>模型名称</td>
<td><code>str</code></td>
<td>无</td>
<td><code>无</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>模型推理设备</td>
<td><code>str</code></td>
<td>支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理插件</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用文本行方向分类模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input` 和 `batch_size`，具体说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
  <li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg">示例</a></li>
  <li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>任意整数</td>
<td>1</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">获取格式为<code>dict</code>的可视化图像</td>
</tr>

</table>


## 四、二次开发

由于 PaddleOCR 并不直接提供文本行方向分类的训练，因此，如果需要训练文档图像方向分类模型，可以参考 [PaddleX 文本行方向分类二次开发](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#_5)部分进行训练。训练后的模型可以无缝集成到 PaddleOCR 的 API 中进行推理。
