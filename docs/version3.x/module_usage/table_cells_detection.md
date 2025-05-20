---
comments: true
---

# 表格单元格检测模块使用教程

## 一、概述

表格单元格检测模块是表格识别任务的关键组成部分，负责在表格图像中定位和标记每个单元格区域，该模块的性能直接影响到整个表格识别过程的准确性和效率。表格单元格检测模块通常会输出各个单元格区域的边界框（Bounding Boxes），这些边界框将作为输入传递给表格识别相关产线进行后续处理。

## 二、支持模型列表

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR 是一个实时的端到端目标检测模型。百度飞桨视觉团队基于 RT-DETR-L 作为基础模型，在自建表格单元格检测数据集上完成预训练，实现了对有线表格、无线表格均有较好性能的表格单元格检测。
</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
              <li><strong>测试数据集：</strong>自建的内部评测集。</li>
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
paddleocr table_cells_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg
```

您也可以将表格单元格检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg)到本地。

```python
from paddleocr import TableCellsDetection
model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
output = model.predict("table_recognition.jpg", threshold=0.3, batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：

```
{'res': {'input_path': 'table_recognition.jpg', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'cell', 'score': 0.9698355197906494, 'coordinate': [2.3011515, 0, 546.29926, 30.530712]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9690820574760437, 'coordinate': [212.37508, 64.62493, 403.58868, 95.61413]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9668057560920715, 'coordinate': [212.46791, 30.311079, 403.7182, 64.62613]}, {'cls_id': 0, 'label': 'cell', 'score': 0.966505229473114, 'coordinate': [403.56082, 64.62544, 546.83215, 95.66117]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9662341475486755, 'coordinate': [109.48873, 64.66485, 212.5177, 95.631294]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9654079079627991, 'coordinate': [212.39197, 95.63037, 403.60852, 126.78792]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9653300642967224, 'coordinate': [2.2320926, 64.62229, 109.600494, 95.59732]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9639787673950195, 'coordinate': [403.5752, 30.562355, 546.98975, 64.61531]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9636150002479553, 'coordinate': [2.1537683, 30.410172, 109.568306, 64.62762]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9631900191307068, 'coordinate': [2.0534437, 95.57448, 109.57601, 126.71458]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9631181359291077, 'coordinate': [403.65976, 95.68139, 546.84766, 126.713394]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9614537358283997, 'coordinate': [109.56504, 30.391184, 212.65425, 64.6444]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9607433080673218, 'coordinate': [109.525795, 95.62622, 212.44917, 126.8258]}]}}
```

参数含义如下：

- `input_path`：输入的待预测图像的路径
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `boxes`：预测的目标框信息，一个字典列表。每个字典代表一个检出的目标，包含以下信息：
  - `cls_id`：类别ID，一个整数
  - `label`：类别标签，一个字符串
  - `score`：目标框置信度，一个浮点数
  - `coordinate`：目标框坐标，一个浮点数列表，格式为<code>[xmin, ymin, xmax, ymax]</code>

可视化图像如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/table_cells_detection/01.jpg">

相关方法、参数等说明如下：

* `TableCellsDetection`实例化表格单元格检测模型（此处以`RT-DETR-L_wired_table_cell_det`为例），具体说明如下：
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
<td>无</td>
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
<tr>
<td><code>img_size</code></td>
<td>输入图像大小</td>
<td><code>int/list</code></td>
<td>
<ul>
  <li><b>int</b>, 如 640 , 表示将输入图像resize到640x640大小</li>
  <li><b>列表</b>, 如 [640, 512] , 表示将输入图像resize到宽为640，高为512大小</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值。在表格单元格检测任务中，适当降低阈值可能有助于获得更准确的结果</td>
<td><code>float/dict</code></td>
<td>
<ul>
  <li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
  <li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
</ul>
</td>
<td>无</td>
</tr>
</table>

* 其中，`model_name` 必须指定，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用表格单元格检测模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input`、`batch_size`和`threshold`，具体说明如下：

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
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">示例</a></li>
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
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值</td>
<td><code>float/dict</code></td>
<td>
<ul>
  <li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
  <li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
</ul>
</td>
<td>无</td>
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
<td rowspan = "1">获取可视化图像</td>
</tr>

</table>

## 四、二次开发

由于 PaddleOCR 并不直接提供表格单元格检测模块的训练，因此，如果需要训练表格单元格检测模型，可以参考 [PaddleX 表格单元格检测模块二次开发](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_cells_detection.html#_4)部分进行训练。训练后的模型可以无缝集成到 PaddleOCR 的 API 中进行推理。

## 五、FAQ
