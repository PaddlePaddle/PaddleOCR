---
comments: true
---

# 图表解析模块使用教程

## 一、概述

多模态图表解析是一项OCR领域的前沿技术，专注于将各类可视化图表（如柱状图、折线图、饼图等）自动转化为底层数据表，并进行格式化输出。传统方法依赖于图表关键点检测等模型进行复杂串联编排，先验假设较多，鲁棒性较差，该模块中的模型使用最新的VLM技术，数据驱动，从海量的现实数据中学习鲁棒的特征。其应用场景覆盖金融分析、学术研究、商业报告等场景——例如快速提取财报中的增长趋势数据、科研论文中的实验对比数值，或市场调研中的用户分布统计，助力用户从“看图”转向“用数”。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>模型参数规模（B）</th>
<th>模型存储大小（GB）</th>
<th>模型分数 </th>
<th>介绍</th>
</tr>
<tr>
<td>PP-Chart2Table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.tar">推理模型</a></td>
<td>0.58</td>
<td>1.4</td>
<th>80.60</th>
<td>PP-Chart2Table是飞桨团队自研的一款专注于图表解析的多模态模型，在中英文图表解析任务中展现出卓越性能。团队专为图表解析设计了Shuffled Chart Data Retrieval训练任务，并结合精心设计的令牌掩码策略，显著提升其在图表转数据表任务上的性能。此外，团队通过精心设计的数据合成流程增强了PP-Chart2Table的能力，该流程利用高质量的种子数据，并结合RAG和大语言模型人格设计，以生成更丰富多样化的数据。为了处理大量未标记的分布外 (OOD) 数据，团队采用了两阶段大模型蒸馏训练过程，确保模型在广泛的真实世界数据集中具有出色的适应性和泛化能力。在内部业务的中英文场景测试中，PP-Chart2Table不仅达到同参数量级模型中的SOTA水平，更在关键场景中实现了与7B参数量级VLM模型相当的精度。</td>
</tr>
</table>

<b>注：以上模型分数为内部评估集模型测试结果，共1801条数据，包括了各个场景（财报、法律法规、合同等）下的各种图表类型（柱状图、折线图、饼图等）的测试样本，暂时未有计划公开。</b>

> ❗ <b>注</b>：PP-Chart2Table模型于 2025.6.27 升级，如需使用升级前的模型权重，请点击<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.bak.tar">下载链接</a>

## 三、快速开始

> ❗ 在快速开始前，请先安装 PaddleOCR 的 wheel 包，详细请参考 [安装教程](../installation.md)。

使用一行命令即可快速体验：

```bash
paddleocr chart_parsing -i "{'image': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png'}"
```

<b>注：</b>PaddleOCR 官方模型默认从 HuggingFace 获取，如运行环境访问 HuggingFace 不便，可通过环境变量修改模型源为 BOS：`PADDLE_PDX_MODEL_SOURCE="BOS"`，未来将支持更多主流模型源；

您也可以将开放文档类视觉语言模型模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png)到本地。

```python
from paddleocr import ChartParsing
model = ChartParsing(model_name="PP-Chart2Table")
results = model.predict(
    input={"image": "chart_parsing_02.png"},
    batch_size=1
)
for res in results:
    res.print()
    res.save_to_json(f"./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': {'image': 'chart_parsing_02.png', 'result': '年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)\n2018 | 104.22 | 9.87\n2019 | 99.11 | 7.47\n2020 | 57.87 | -3.87\n2021 | 68.99 | -2.9\n2022 | 56.29 | -9.48\n2023 | 87.99 | 5.96'}}
```

运行结果参数含义如下：

- `image`: 表示输入待预测图像的路径
- `result`: 模型预测的结果信息

预测结果打印可视化如下：

```bash
年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)
2018 | 104.22 | 9.87
2019 | 99.11 | 7.47
2020 | 57.87 | -3.87
2021 | 68.99 | -2.9
2022 | 56.29 | -9.48
2023 | 87.99 | 5.96
```

相关方法、参数等说明如下：

* `ChartParsing`实例化文档类视觉语言模型，具体说明如下：
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
<td>>模型名称。如果设置为<code>None</code>，则使用<code>PP-Chart2Table</code>。</td>
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
<b>例如：</b><code>"cpu"</code>、<code>"gpu"</code>、<code>"npu"</code>、<code>"gpu:0"</code></code>。
默认情况下，优先使用 GPU 0；若不可用则使用 CPU。
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* 调用图表解析模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input` 、 `batch_size`，具体说明如下：

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
<td>待预测数据，必填。由于多模态模型对输入要求不同，请根据具体模型设定输入格式。<br/>
<li>PP-Chart2Table的输入形式为<code>{'image': image_path}</code></li>
</td>
<td><code>dict</code></td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小，可设置为任意正整数。</td>
<td><code>int</code></td>
<td>1</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为`json`文件的操作:

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

* 此外，也支持通过属性获取预测结果，具体如下：

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
</table>

## 四、二次开发

当前模块暂时不支持微调训练，仅支持推理集成。关于该模块的微调训练，计划在未来支持。

## 五、FAQ
