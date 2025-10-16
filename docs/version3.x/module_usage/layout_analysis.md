---
comments: true
---

# 版面分析模块使用教程

## 一、概述

版面分析是文档解析系统中的重要组成环节，目标是对文档的整体布局进行解析，准确检测出其中所包含的各种元素（例如，文本段落、标题、图片、表格、公式等），并恢复这些元素正确的阅读顺序。该模块的性能直接影响到整个文档解析系统的准确性和可用性。


## 二、支持模型列表

该模块目前仅支持 PP-DocLayoutV2 一个模型。模型结构上，PP-DocLayoutV2 是在版面检测模型 [PP-DocLayout_plus-L](./layout_detection.md)（基于 RT-DETR-L 模型） 的基础上级联一个含 6 层 Transformer 层的轻量级指针网络（Pointer network）组成，原先 PP-DocLayout_plus-L 部分继续用于版面检测，识别文档图像中的不同元素（如文字、图表、图像、公式、段落、摘要、参考文献等），将其归类为预定义的类别并确定这些区域在文档中的位置。检测到的边界框和类别标签作为后续的指针网络的输入来对版面元素进行排序从而得到正确的阅读顺序。

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl/methods/PP-DocLayoutV2.png" width="600"/>
</div>

具体如上图所示，PP-DocLayoutV2 对来自 RT-DETR 检出的目标利用绝对二维位置编码和类别标签进行嵌入表示。此外，指针网络的注意力机制融合了 Relation-DETR中的几何偏置机制，以显式地建模元素之间的成对几何关系。成对关系头（pairwise relation head）将元素表示线性投影为查询（query）向量和键（key）向量，然后计算双线性相似度以生成成对的 logits，最终得到一个表示每对元素之间相对顺序的 N×N 矩阵。最后，一种确定性的“胜者累积”（win-accumulation）解码算法会为检测到的版面元素恢复出一个拓扑一致的阅读顺序。

下表仅给出 PP-DocLayoutV2 的版面检测精度。该精度指标的评估数据集是自建的版面区域检测数据集，包含了中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1000 张文档类型图片，包含 25 类常见的版面元素：文档标题、段落标题、文本、竖排文本、页码、摘要、目录、参考文献、脚注、图像脚注、页眉、页脚、页眉图像、页脚图像、算法、行内公式、行间公式、公式编号、图像、表格、图和表标题（图标题、表格标题和图表标题）、印章、图表、侧栏文本和参考文献内容。

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayoutV2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV2_infer.tar">推理模型</a></td>
<td>81.4</td>
<td> - </td>
<td> - </td>
<td>203.8</td>
<td>自研的版面分析模型在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位和版面阅读顺序恢复模型</td>
</tr>
<tr>
</tbody>
</table>


## 三、快速开始

> ❗ 在快速开始前，请先安装 PaddleOCR 的 wheel 包，详细请参考 [安装教程](../installation.md)。

您可以将版面区域检测模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg)到本地。

```python
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': {'input_path': 'layout.jpg', 'page_index': None, 'boxes': [{'cls_id': 7, 'label': 'figure_title', 'score': 0.9764086604118347, 'coordinate': [34.227077, 19.217348, 362.48834, 80.243195]}, {'cls_id': 21, 'label': 'table', 'score': 0.9881672263145447, 'coordinate': [73.759026, 105.72034, 322.67398, 298.82642]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9405199885368347, 'coordinate': [35.089825, 330.67575, 144.06467, 346.7406]}, {'cls_id': 22, 'label': 'text', 'score': 0.9889925122261047, 'coordinate': [33.540825, 349.37204, 363.5848, 615.04504]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9430180788040161, 'coordinate': [35.032074, 627.2185, 188.24695, 643.66693]}, {'cls_id': 22, 'label': 'text', 'score': 0.988863468170166, 'coordinate': [33.72388, 646.5885, 363.05005, 852.60236]}, {'cls_id': 7, 'label': 'figure_title', 'score': 0.9749509692192078, 'coordinate': [385.14465, 19.341825, 715.4594, 78.739395]}, {'cls_id': 21, 'label': 'table', 'score': 0.9876241683959961, 'coordinate': [436.68512, 105.67539, 664.1041, 314.0018]}, {'cls_id': 22, 'label': 'text', 'score': 0.9878363013267517, 'coordinate': [385.06494, 345.74847, 715.173, 463.18677]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.951842725276947, 'coordinate': [386.3095, 476.34277, 703.0732, 493.00302]}, {'cls_id': 22, 'label': 'text', 'score': 0.9884033799171448, 'coordinate': [385.1237, 496.70123, 716.09717, 702.33386]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.940105676651001, 'coordinate': [386.66754, 715.5732, 527.5737, 731.77826]}, {'cls_id': 22, 'label': 'text', 'score': 0.9876392483711243, 'coordinate': [384.9688, 734.457, 715.63086, 853.71]}]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `boxes`：按照阅读顺序排序的版面元素预测信息，一个字典列表。按照阅读顺序，每个字典代表一个检出版面的元素，包含以下信息：
  - `cls_id`：类别ID，一个整数
  - `label`：类别标签，一个字符串
  - `score`：目标框置信度，一个浮点数
  - `coordinate`：目标框坐标，一个浮点数列表，格式为<code>[xmin, ymin, xmax, ymax]</code>


相关方法、参数等说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model_name</code></td>
<td>模型名称。如果设置为<code>None</code>，则使用<code>PP-DocLayout-L</code></td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。<br/>
<b>例如：</b><code>"cpu"</code>、<code>"gpu"</code>、<code>"npu"</code>、<code>"gpu:0"</code>、<code>"gpu:0,1"</code>。<br/>
如指定多个设备，将进行并行推理。<br/>
默认情况下，优先使用 GPU 0；若不可用则使用 CPU。
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>是否启用高性能推理。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>是否启用 Paddle Inference 的 TensorRT 子图引擎。如果模型不支持通过 TensorRT 加速，即使设置了此标志，也不会使用加速。<br/>
对于 CUDA 11.8 版本的飞桨，兼容的 TensorRT 版本为 8.x（x>=6），建议安装 TensorRT 8.6.1.6。<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>当使用 Paddle Inference 的 TensorRT 子图引擎时设置的计算精度。<br/><b>可选项：</b><code>"fp32"</code>、<code>"fp16"</code>。</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。<br/>
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN 缓存容量。
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上推理时使用的线程数量。</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>输入图像大小。
<ul>
<li><b>int</b>：如<code>640</code>，表示将输入图像resize到640x640大小。</li>
<li><b>list</b>：如<code>[640, 512]</code>，表示将输入图像resize到宽为640、高为512。</li>
</ul>
</td>
<td><code>int|list|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值。
<ul>
<li><b>float</b>：如<code>0.2</code>，表示过滤掉所有阈值小于0.2的目标框。</li>
<li><b>dict</b>：字典的键为<code>int</code>类型，代表类别ID；值为<code>float</code>类型阈值。如<code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对ID为0的类别应用阈值0.45、ID为1的类别应用阈值0.48、ID为7的类别应用阈值0.4。</li>
<li><b>None</b>：使用模型默认的配置。</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>是否使用NMS后处理，过滤重叠框。
<ul>
<li><b>bool</b>表示使用/不使用NMS进行检测框的后处理过滤重叠框。</li>
<li><b>None</b>使用模型默认的配置。</li>
</ul>
</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>检测框的边长缩放倍数。</b>
<ul>
<li><b>float</b>：大于0的浮点数，如<code>1.1</code>，表示将模型输出的检测框中心不变，宽和高都扩张1.1倍。</li>
<li><b>list</b>：如<code>[1.2, 1.5]</code>，表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍。</li>
<li><b>dict</b>：字典的键为<code>int</code>类型，代表类别ID；值为<code>tuple</code>类型，如<code>{0: (1.1, 2.0)}</code>，表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍。</li>
<li><b>None</b>：使用模型默认的配置。</li>
</ul>
</td>
<td><code>float|list|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>模型输出的检测框的合并处理模式。
<ul>
<li><b>"large"</b>：设置为<code>"large"</code>，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>"small"</b>：设置为<code>"small"</code>，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>"union"</b>：不进行框的过滤处理，内外框都保留。</li>
<li><b>dict</b>：字典的键为<code>int</code>类型，代表类别ID；值为<code>str</code>类型, 如<code>{0: "large", 2: "small"}</code>, 表示对第0类别检测框使用<code>large</code>模式，对第2类别检测框使用<code>small</code>。</li>
<li><b>None</b>：使用模型默认的配置。</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td>None</td>
</tr>
</tbody>
</table>


* 调用目标检测模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input`、`batch_size`和`threshold`，具体说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填。
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>list</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小，可设置为任意正整数。</td>
<td><code>int</code></td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</b>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。
<td><code>float|list|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。
</td>
<td><code>str|dict|None</code></td>
<td>None</td>
</tr>
</tbody>
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
