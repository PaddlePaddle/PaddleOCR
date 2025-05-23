---
comments: true
---

# 印章文本检测模块使用教程

## 一、概述

印章文本检测模块通常会输出文本区域的多点边界框（Bounding Boxes），这些边界框将作为输入传递给弯曲矫正和文本检测模块进行后续处理，识别出印章的文字内容。印章文本的识别是文档处理的一部分，在很多场景都有用途，例如合同比对，出入库审核以及发票报销审核等场景。印章文本检测模块是OCR（光学字符识别）中的子任务，负责在图像中定位和标记出包含印章文本的区域。该模块的性能直接影响到整个印章文本OCR系统的准确性和效率。

## 二、支持模型列表

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td>
<td>98.40</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td>
<td>96.36</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署</td>
</tr>
</tbody>
</table>
<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
              <li><strong>测试数据集：</strong>自建的内部数据集，包含500张圆形印章图像。</li>
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
paddleocr seal_text_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png
```

您也可以将印章文本检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png)到本地。

```python
from paddleocr import SealTextDetection
model = SealTextDetection(model_name="PP-OCRv4_server_seal_det")
output = model.predict("seal_text_det.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': {'input_path': 'seal_text_det.png', 'page_index': None, 'dt_polys': [array([[463, 477],
       ...,
       [428, 505]]), array([[297, 444],
       ...,
       [230, 443]]), array([[457, 346],
       ...,
       [267, 345]]), array([[325,  38],
       ...,
       [322,  37]])], 'dt_scores': [0.9912680344777314, 0.9906849624837963, 0.9847219455533163, 0.9914791724153904]}}
```

运行结果参数含义如下：
- `input_path`：表示输入待预测图像的路径
- `dt_polys`：表示预测的文本检测框，其中每个文本检测框包含一个多边形的多个顶点。其中每个顶点都是一个列表，分别表示该顶点的x坐标和y坐标
- `dt_scores`：表示预测的文本检测框的置信度

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/seal_text_det/seal_text_det_res.png"/>

相关方法、参数等说明如下：

* `SealTextDetection`实例化文本检测模型（此处以`PP-OCRv4_server_seal_det`为例），具体说明如下：
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
<td>所有支持的印章文本检测模型名称</td>
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
<td><code>limit_side_len</code></td>
<td>检测的图像边长限制</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: 大于0的任意整数
<li><b>None</b>: 如果设置为None, 将使用默认值：736</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>检测的图像边长限制,检测的边长限制类型 </td>
<td><code>str/None</code></td>
<td>
<ul>
<li><b>str</b>: 支持min和max. min表示保证图像最短边不小于det_limit_side_len, max: 表示保证图像最长边不大于limit_side_len
<li><b>None</b>: 如果设置为None, 将使用默认值：“min”</li></li></ul></td>


<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点 </td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: 大于0的任意浮点数
<li><b>None</b>: 如果设置为None, 将使用默认值：0.2</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域 </td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: 大于0的任意浮点数
<li><b>None</b>: 如果设置为None, 将使用默认值：0.6</li></li></ul></td>
<td>None</td>
</tr>
<tr>
<td><code>max_candidates</code></td>
<td>输出的最大文本框数量</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: 大于0的任意整数
<li><b>None</b>: 如果设置为None, 将使用默认值：1000</li></li></ul></td>
<td>None</td>
<tr>
<td><code>unclip_ratio</code></td>
<td>Vatti clipping算法的扩张系数，使用该方法对文字区域进行扩张 </td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: 大于0的任意浮点数
<li><b>None</b>: 如果设置为None, 将使用默认值：0.5</li></li></ul></td>
<td>None</td>
</tr>
<tr>
<td><code>use_dilation</code></td>
<td>是否对分割结果进行膨胀 </td>
<td><code>bool/None</code></td>
<td>True/False/None</td>
<td>None</td>
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

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用印章文本检测模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input`、 `batch_size`、 `limit_side_len`、 `limit_type`、 `thresh`、 `box_thresh`、 `max_candidates`、`unclip_ratio`和`use_dilation`，具体说明如下：

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
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
  <li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">示例</a></li>
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
<td>大于0的任意整数</td>
<td>1</td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>检测的图像边长限制</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: 大于0的任意整数
<li><b>None</b>: 如果设置为None, 将默认使用模型初始化的该参数值</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>检测的图像边长限制,检测的边长限制类型 </td>
<td><code>str/None</code></td>
<td>
<ul>
<li><b>str</b>: 支持min和max. min表示保证图像最短边不小于det_limit_side_len, max: 表示保证图像最长边不大于limit_side_len
<li><b>None</b>: 如果设置为None, 将默认使用模型初始化的该参数值</li></li></ul></td>


<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点 </td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: 大于0的任意浮点数
<li><b>None</b>: 如果设置为None, 将默认使用模型初始化的该参数值</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域 </td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: 大于0的任意浮点数
<li><b>None</b>: 如果设置为None, 将默认使用模型初始化的该参数值</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>max_candidates</code></td>
<td>输出的最大文本框数量 </td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: 大于0的任意整数
<li><b>None</b>: 如果设置为None, 将默认使用模型初始化的该参数值</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Vatti clipping算法的扩张系数，使用该方法对文字区域进行扩张 </td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: 大于0的任意浮点数
<li><b>None</b>: 如果设置为None, 将默认使用模型初始化的该参数值</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>use_dilation</code></td>
<td>是否对分割结果进行膨胀 </td>
<td><code>bool/None</code></td>
<td>True/False/None</td>
<td>None</td>
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
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
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
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
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
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">获取格式为<code>dict</code>的可视化图像</td>
</tr>
</table>

## 四、二次开发

如果以上模型在您的场景上效果仍然不理想，您可以尝试以下步骤进行二次开发，此处以训练 `PP-OCRv4_server_seal_det` 举例，其他模型替换对应配置文件即可。首先，您需要准备文本检测的数据集，可以参考[印章文本检测 Demo 数据](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_curve_det_dataset_examples.tar)的格式准备，准备好后，即可按照以下步骤进行模型训练和导出，导出后，可以将模型快速集成到上述 API 中。此处以印章文本检测 Demo 数据示例。在训练模型之前，请确保已经按照[安装文档](../installation.md)安装了 PaddleOCR 所需要的依赖。


### 4.1 数据集、预训练模型准备

#### 4.1.1 准备数据集

```shell
# 下载示例数据集
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_curve_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_curve_det_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 下载预训练模型

```shell
# 下载 PP-OCRv4_server_seal_det 预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams
```

### 4.2 模型训练

PaddleOCR 对代码进行了模块化，训练 `PP-OCRv4_server_seal_det` 模型时需要使用 `PP-OCRv4_server_seal_det` 的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml)。


训练命令如下：

```bash
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml \
   -o Global.pretrained_model=./PP-OCRv4_server_seal_det_pretrained.pdparams

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml \
        -o Global.pretrained_model=./PP-OCRv4_server_seal_det_pretrained.pdparams
```


### 4.3 模型评估

您可以评估已经训练好的权重，如，`output/xxx/xxx.pdparams`，使用如下命令进行评估：

```bash
# 注意将pretrained_model的路径设置为本地路径。若使用自行训练保存的模型，请注意修改路径和文件名为{path/to/weights}/{model_name}。
 # demo 测试集评估
 python3 tools/eval.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml -o \
 Global.pretrained_model=output/xxx/xxx.pdparams
```

### 4.4 模型导出

```bash
 python3 tools/export_model.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml -o \
 Global.pretrained_model=output/xxx/xxx.pdparams \
 save_inference_dir="./PP-OCRv4_server_seal_det_infer/"
```

 导出模型后，静态图模型会存放于当前目录的`./PP-OCRv4_server_seal_det_infer/`中，在该目录下，您将看到如下文件：
 ```
 ./PP-OCRv4_server_seal_det_infer/
 ├── inference.json
 ├── inference.pdiparams
 ├── inference.yml
 ```
至此，二次开发完成，该静态图模型可以直接集成到 PaddleOCR 的 API 中。

## 五、FAQ
