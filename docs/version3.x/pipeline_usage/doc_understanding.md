---
comments: true
---

# 文档理解产线使用教程

## 1. 文档理解产线介绍

文档理解产线是基于视觉-语言模型（VLM）打造的先进文档处理技术，旨在突破传统文档处理的局限。传统方法依赖固定模板或预定义规则解析文档，而该产线借助VLM的多模态能力，仅需输入文档图片和用户问题，即可通过融合视觉与语言信息，精准回答用户提问。这种技术无需针对特定文档格式预训练，能够灵活应对多样化文档内容，显著提升文档处理的泛化性与实用性，在智能问答、信息提取等场景中具有广阔应用前景。本产线目前暂不支持对VLM模型的二次开发，后续计划支持。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/doc_understanding/doc_understanding.png">

<b>文档理解产线中包含以下1个模块。每个模块均可独立进行训练和推理，并包含多个模型。有关详细信息，请点击相应模块以查看文档。</b>

- [文档类视觉语言模型模块](../module_usage/doc_vlm.md)

在本产线中，您可以根据下方的基准测试数据选择使用的模型。

<details>
<summary> <b>文档类视觉语言模型模块：</b></summary>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>模型存储大小（GB）</th>
<th>模型总分</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-DocBee-2B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-2B_infer.tar">推理模型</a></td>
<td>4.2</td>
<td>765</td>
<td rowspan="2">PP-DocBee 是飞桨团队自研的一款专注于文档理解的多模态大模型，在中文文档理解任务上具有卓越表现。该模型通过近 500 万条文档理解类多模态数据集进行微调优化，各种数据集包括了通用VQA类、OCR类、图表类、text-rich文档类、数学和复杂推理类、合成数据类、纯文本数据等，并设置了不同训练数据配比。在学术界权威的几个英文文档理解评测榜单上，PP-DocBee基本都达到了同参数量级别模型的SOTA。在内部业务中文场景类的指标上，PP-DocBee也高于目前的热门开源和闭源模型。</td>
</tr>
<tr>
<td>PP-DocBee-7B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-7B_infer.tar">推理模型</a></td>
<td>15.8</td>
<td>-</td>
</tr>
<tr>
<td>PP-DocBee2-3B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee2-3B_infer.tar">推理模型</a></td>
<td>7.6</td>
<td>852</td>
<td>PP-DocBee2 是飞桨团队自研的一款专注于文档理解的多模态大模型，在PP-DocBee的基础上进一步优化了基础模型，并引入了新的数据优化方案，提高了数据质量，使用自研数据合成策略生成的少量的47万数据便使得PP-DocBee2在中文文档理解任务上表现更佳。在内部业务中文场景类的指标上，PP-DocBee2相较于PP-DocBee提升了约11.4%，同时也高于目前的同规模热门开源和闭源模型。</td>
</tr>
</table>

<b>注：以上模型总分为内部评估集模型测试结果，内部评估集所有图像分辨率 (height, width) 为 (1680,1204)，共1196条数据，包括了财报、法律法规、理工科论文、说明书、文科论文、合同、研报等场景，暂时未有计划公开。</b>
</details>

<br />
<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>

## 2. 快速开始

在本地使用文档理解产线前，请确保您已经按照[安装教程](../installation.md)完成了wheel包安装。安装完成后，可以在本地使用命令行体验或 Python 集成。

### 2.1 命令行方式体验

一行命令即可快速体验 doc_understanding 产线效果：

```bash
paddleocr doc_understanding -i "{'image': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png', 'query': '识别这份表格的内容, 以markdown格式输出'}"
```

<details><summary><b>命令行支持更多参数设置，点击展开以查看命令行参数的详细说明</b></summary>
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
<td><code>input</code></td>
<td>预测的输入数据，需要是Python字典
<ul>
<li><b>Dict</b>: PP-DocBee 系列的输入形式为: <code>{"image":/path/to/image, "query": user question}</code>, 代表输入的图像和对应的用户问题。</li>
</ul>
</td>
<td><code>Dict</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径。如果设置为<code>None</code>, 推理结果将不会保存到本地。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_model_name</code></td>
<td>文档理解模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_model_dir</code></td>
<td>文档理解模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_batch_size</code></td>
<td>文档理解模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号。
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>str</code></td>
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
<td>是否使用 TensorRT 进行推理加速。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>最小子图大小，用于优化模型子图的计算。</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速库。如果设置为<code>None</code>, 将默认启用。
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上进行推理时使用的线程数。</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />

运行结果会被打印到终端上，默认配置的 doc_understanding 产线的运行结果如下：

```bash
{'res': {'image': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png', 'query': '识别这份表格的内容, 以markdown格式输出', 'result': '| 名次 | 国家/地区 | 金牌 | 银牌 | 铜牌 | 奖牌总数 |\n| --- | --- | --- | --- | --- | --- |\n| 1 | 中国（CHN） | 48 | 22 | 30 | 100 |\n| 2 | 美国（USA） | 36 | 39 | 37 | 112 |\n| 3 | 俄罗斯（RUS） | 24 | 13 | 23 | 60 |\n| 4 | 英国（GBR） | 19 | 13 | 19 | 51 |\n| 5 | 德国（GER） | 16 | 11 | 14 | 41 |\n| 6 | 澳大利亚（AUS） | 14 | 15 | 17 | 46 |\n| 7 | 韩国（KOR） | 13 | 11 | 8 | 32 |\n| 8 | 日本（JPN） | 9 | 8 | 8 | 25 |\n| 9 | 意大利（ITA） | 8 | 9 | 10 | 27 |\n| 10 | 法国（FRA） | 7 | 16 | 20 | 43 |\n| 11 | 荷兰（NED） | 7 | 5 | 4 | 16 |\n| 12 | 乌克兰（UKR） | 7 | 4 | 11 | 22 |\n| 13 | 肯尼亚（KEN） | 6 | 4 | 6 | 16 |\n| 14 | 西班牙（ESP） | 5 | 11 | 3 | 19 |\n| 15 | 牙买加（JAM） | 5 | 4 | 2 | 11 |\n'}}
```

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddleocr import DocUnderstanding

pipeline = DocUnderstanding()
output = pipeline.predict(
    {
        "image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png",
        "query": "识别这份表格的内容, 以markdown格式输出"
    }
)
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `DocUnderstanding()` 实例化 文档理解产线 产线对象，具体参数说明如下：

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
<td><code>doc_understanding_model_name</code></td>
<td>文档理解模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_model_dir</code></td>
<td>文档理解模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_batch_size</code></td>
<td>文档理解模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号。
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>str</code></td>
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
<td>是否使用 TensorRT 进行推理加速。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>最小子图大小，用于优化模型子图的计算。</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速库。如果设置为<code>None</code>, 将默认启用。
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上进行推理时使用的线程数。</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（2）调用 文档理解产线 产线对象的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。

另外，产线还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。

以下是 `predict()` 方法的参数及其说明：

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
<td>待预测数据，目前仅支持字典类型的输入
<ul>
  <li><b>Python Dict</b>：如PP-DocBee的输入形式为: <code>{"image":/path/to/image, "query": user question}</code> ,分别表示输入的图像和对应的用户问题</li>
</ul>
</td>
<td><code>Python Dict</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>与实例化时的参数相同。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为`json`文件的操作:

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
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `image`: `(str)` 图像的输入路径

    - `query`: `(str)` 针对输入图像的问题

    - `result`: `(str)` 模型的输出结果

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。

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
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。

## 3. 开发集成/部署

如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成) 中的示例代码。

此外，PaddleOCR 也提供了其他两种部署方式，详细说明如下：

🚀 高性能推理：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleOCR 提供高性能推理功能，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[高性能推理](../deployment/high_performance_inference.md)。

☁️ 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。详细的产线服务化部署流程请参考[服务化部署](../deployment/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>

<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>对输入消息进行推理生成响应。</p>
<p><code>POST /document-understanding</code></p>
<p>说明 以上接口别名/chat/completion，openai兼容的接口</p>

<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model</code></td>
<td><code>string</code></td>
<td>要使用的模型名称</td>
<td>是</td>
<td>-</td>
</tr>
<tr>
<td><code>messages</code></td>
<td><code>array</code></td>
<td>对话消息列表</td>
<td>是</td>
<td>-</td>
</tr>
<tr>
<td><code>max_tokens</code></td>
<td><code>integer</code></td>
<td>生成的最大token数</td>
<td>否</td>
<td>1024</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>float</code></td>
<td>采样温度</td>
<td>否</td>
<td>0.1</td>
</tr>
<tr>
<td><code>top_p</code></td>
<td><code>float</code></td>
<td>核心采样概率</td>
<td>否</td>
<td>0.95</td>
</tr>
<tr>
<td><code>stream</code></td>
<td><code>boolean</code></td>
<td>是否流式输出</td>
<td>否</td>
<td>false</td>
</tr>
<tr>
<td><code>max_image_tokens</code></td>
<td><code>int</code></td>
<td>图像的最大输入token数</td>
<td>否</td>
<td>None</td>
</tr>
</tbody>
</table>

<p><code>messages</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>role</code></td>
<td><code>string</code></td>
<td>消息角色（user/assistant/system）</td>
<td>是</td>
</tr>
<tr>
<td><code>content</code></td>
<td><code>string</code>或<code>array</code></td>
<td>消息内容（文本或图文混合）</td>
<td>是</td>
</tr>
</tbody>
</table>

<p>当<code>content</code>为数组时，每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>type</code></td>
<td><code>string</code></td>
<td>内容类型（text/image_url）</td>
<td>是</td>
<td>-</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>文本内容（当type为text时）</td>
<td>条件必填</td>
<td>-</td>
</tr>
<tr>
<td><code>image_url</code></td>
<td><code>string</code>或<code>object</code></td>
<td>图片URL或对象（当type为image_url时）</td>
<td>条件必填</td>
<td>-</td>
</tr>
</tbody>
</table>

<p>当<code>image_url</code>为对象时，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>url</code></td>
<td><code>string</code></td>
<td>图片URL</td>
<td>是</td>
<td>-</td>
</tr>
<tr>
<td><code>detail</code></td>
<td><code>string</code></td>
<td>图片细节处理方式（low/high/auto）</td>
<td>否</td>
<td>auto</td>
</tr>
</tbody>
</table>

<p>请求处理成功时，响应体的<code>result</code>具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>id</code></td>
<td><code>string</code></td>
<td>请求ID</td>
</tr>
<tr>
<td><code>object</code></td>
<td><code>string</code></td>
<td>对象类型（chat.completion）</td>
</tr>
<tr>
<td><code>created</code></td>
<td><code>integer</code></td>
<td>创建时间戳</td>
</tr>
<tr>
<td><code>choices</code></td>
<td><code>array</code></td>
<td>生成结果选项</td>
</tr>
<tr>
<td><code>usage</code></td>
<td><code>object</code></td>
<td>token使用情况</td>
</tr>
</tbody>
</table>

<p><code>choices</code>中的每个元素为一个<code>Choice</code>对象，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>可选值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>finish_reason</code></td>
<td><code>string</code></td>
<td>模型停止生成token的原因</td>
<td><code>stop</code>（自然停止）<br><code>length</code>（达到最大token数）<br><code>tool_calls</code>（调用了工具）<br><code>content_filter</code>（内容过滤）<br><code>function_call</code>（调用了函数，已弃用）</td>
</tr>
<tr>
<td><code>index</code></td>
<td><code>integer</code></td>
<td>选项在列表中的索引</td>
<td>-</td>
</tr>
<tr>
<td><code>logprobs</code></td>
<td><code>object</code> | <code>null</code></td>
<td>选项的log概率信息</td>
<td>-</td>
</tr>
<tr>
<td><code>message</code></td>
<td><code>ChatCompletionMessage</code></td>
<td>模型生成的聊天消息</td>
<td>-</td>
</tr>
</tbody>
</table>

<p><code>message</code>对象具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>content</code></td>
<td><code>string</code> | <code>null</code></td>
<td>消息内容</td>
<td>可能为空</td>
</tr>
<tr>
<td><code>refusal</code></td>
<td><code>string</code> | <code>null</code></td>
<td>模型生成的拒绝消息</td>
<td>当内容被拒绝时提供</td>
</tr>
<tr>
<td><code>role</code></td>
<td><code>string</code></td>
<td>消息作者角色</td>
<td>固定为<code>"assistant"</code></td>
</tr>
<tr>
<td><code>audio</code></td>
<td><code>object</code> | <code>null</code></td>
<td>音频输出数据</td>
<td>当请求音频输出时提供<br><a href="https://platform.openai.com/docs/guides/audio">了解更多</a></td>
</tr>
<tr>
<td><code>function_call</code></td>
<td><code>object</code> | <code>null</code></td>
<td>应调用的函数名称和参数</td>
<td>已弃用，推荐使用<code>tool_calls</code></td>
</tr>
<tr>
<td><code>tool_calls</code></td>
<td><code>array</code> | <code>null</code></td>
<td>模型生成的工具调用</td>
<td>如函数调用等</td>
</tr>
</tbody>
</table>

<p><code>usage</code>对象具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>prompt_tokens</code></td>
<td><code>integer</code></td>
<td>提示token数</td>
</tr>
<tr>
<td><code>completion_tokens</code></td>
<td><code>integer</code></td>
<td>生成token数</td>
</tr>
<tr>
<td><code>total_tokens</code></td>
<td><code>integer</code></td>
<td>总token数</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
    "id": "ed960013-eb19-43fa-b826-3c1b59657e35",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "| 名次 | 国家/地区 | 金牌 | 银牌 | 铜牌 | 奖牌总数 |\n| --- | --- | --- | --- | --- | --- |\n| 1 | 中国（CHN） | 48 | 22 | 30 | 100 |\n| 2 | 美国（USA） | 36 | 39 | 37 | 112 |\n| 3 | 俄罗斯（RUS） | 24 | 13 | 23 | 60 |\n| 4 | 英国（GBR） | 19 | 13 | 19 | 51 |\n| 5 | 德国（GER） | 16 | 11 | 14 | 41 |\n| 6 | 澳大利亚（AUS） | 14 | 15 | 17 | 46 |\n| 7 | 韩国（KOR） | 13 | 11 | 8 | 32 |\n| 8 | 日本（JPN） | 9 | 8 | 8 | 25 |\n| 9 | 意大利（ITA） | 8 | 9 | 10 | 27 |\n| 10 | 法国（FRA） | 7 | 16 | 20 | 43 |\n| 11 | 荷兰（NED） | 7 | 5 | 4 | 16 |\n| 12 | 乌克兰（UKR） | 7 | 4 | 11 | 22 |\n| 13 | 肯尼亚（KEN） | 6 | 4 | 6 | 16 |\n| 14 | 西班牙（ESP） | 5 | 11 | 3 | 19 |\n| 15 | 牙买加（JAM） | 5 | 4 | 2 | 11 |\n",
                "role": "assistant"
            }
        }
    ],
    "created": 1745218041,
    "model": "pp-docbee",
    "object": "chat.completion"
}
</code></pre></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>
openai接口调用示例

<pre><code class="language-python">import base64
from openai import OpenAI

API_BASE_URL = "http://0.0.0.0:8080"

# 初始化OpenAI客户端
client = OpenAI(
    api_key='xxxxxxxxx',
    base_url=f'{API_BASE_URL}'
)

#图片转base64函数
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#输入图片路径
image_path = "medal_table.png"

#原图片转base64
base64_image = encode_image(image_path)

#提交信息至PP-DocBee模型
response = client.chat.completions.create(
    model="pp-docbee",#选择模型
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content":[
                {
                    "type": "text",
                    "text": "识别这份表格的内容,输出html格式的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
            ]
        },
    ],
)
content = response.choices[0].message.content
print('Reply:', content)
</code></pre></details>
</details>
<br/>

## 4. 二次开发

当前产线暂时不支持微调训练，仅支持推理集成。关于该产线的微调训练，计划在未来支持。
