---
comments: true
---

# Chart Parsing Module Tutorial

## 1. Overview

Multimodal chart parsing is a cutting-edge OCR technology that focuses on automatically converting various types of visual charts (such as bar charts, line charts, pie charts, etc.) into structured data tables with formatted output. Traditional methods rely on complex pipeline designs with chart keypoint detection models, which involve many prior assumptions and tend to lack robustness. The models in this module leverage the latest VLM (Vision-Language Model) techniques and are data-driven, learning robust features from vast real-world datasets. Application scenarios include financial analysis, academic research, business reporting, and more—for instance, quickly extracting growth trend data from financial reports, experimental comparison figures from research papers, or user distribution statistics from market surveys—empowering users to transition from “viewing charts” to “using data”.

## 2. Supported Model List

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Model Size (B)</th>
<th>Storage Size (GB)</th>
<th>Score</th>
<th>Description</th>
</tr>
<tr>
<td>PP-Chart2Table</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.tar">Inference Model</a></td>
<td>0.58</td>
<td>1.4</td>
<th>80.60</th>
<td>PP-Chart2Table is a multimodal chart parsing model developed by the PaddlePaddle team. It demonstrates exceptional performance on both Chinese and English chart parsing tasks. The team designed a specialized “Shuffled Chart Data Retrieval” training task and adopted a carefully designed token masking strategy, significantly improving performance on chart-to-table conversion. Additionally, the team enhanced the model with a high-quality data synthesis process using seed data, RAG, and LLM persona-driven generation to diversify training data. To handle large amounts of out-of-distribution (OOD) unlabeled data, a two-stage large model distillation process was used to ensure excellent adaptability and generalization to diverse real-world data. In internal Chinese-English use case evaluations, PP-Chart2Table achieved state-of-the-art performance among models of similar size and reached accuracy comparable to 7B-parameter VLMs in key scenarios.</td>
</tr>
</table>

**Note:** The scores above are based on internal evaluation on a test set of 1801 samples, covering various chart types (bar, line, pie, etc.) across scenarios such as financial reports, regulations, and contracts. There is currently no plan for public release.

> ❗ **Note:** The PP-Chart2Table model was upgraded on June 27, 2025. To use the previous version, please download it [here](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.bak.tar)

## 3. Quick Start

> ❗ Before getting started, please install the PaddleOCR wheel package. Refer to the [Installation Guide](../installation.md) for details.

Run the following command to get started instantly:

```bash
paddleocr chart_parsing -i "{'image': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png'}"
````

**Note:** By default, PaddleOCR retrieves models from HuggingFace. If HuggingFace access is restricted in your environment, you can switch the model source to BOS by setting the environment variable: `PADDLE_PDX_MODEL_SOURCE="BOS"`. Support for more mainstream sources is planned.

You can also integrate the inference of the vision-language model into your own project. Please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png) locally before running the following code:

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

The output result will be:

```bash
{'res': {'image': 'chart_parsing_02.png', 'result': 'Year | Avg Revenue per 5-star Hotel (Million CNY) | Avg Profit per 5-star Hotel (Million CNY)\n2018 | 104.22 | 9.87\n2019 | 99.11 | 7.47\n2020 | 57.87 | -3.87\n2021 | 68.99 | -2.9\n2022 | 56.29 | -9.48\n2023 | 87.99 | 5.96'}}
```

Explanation of output parameters:

* `image`: The path to the input image
* `result`: The model's prediction output

The visualized result is:

```bash
Year | Avg Revenue per 5-star Hotel (Million CNY) | Avg Profit per 5-star Hotel (Million CNY)
2018 | 104.22 | 9.87
2019 | 99.11 | 7.47
2020 | 57.87 | -3.87
2021 | 68.99 | -2.9
2022 | 56.29 | -9.48
2023 | 87.99 | 5.96
```

Detailed explanation of related methods and parameters:

* Instantiate a vision-language model with `ChartParsing`. Parameters:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model_name</code></td>
<td>Model name. If set to <code>None</code>, defaults to <code>PP-Chart2Table</code>.</td>
<td><code>str | None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path.</td>
<td><code>str | None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device.<br/>
<b>Examples:</b> <code>"cpu"</code>, <code>"gpu"</code>, <code>"npu"</code>, <code>"gpu:0"</code><br/>
Defaults to GPU 0 if available; otherwise falls back to CPU.
</td>
<td><code>str | None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* Use the model's `predict()` method for inference. This returns a list of results. The module also offers a `predict_iter()` method, which behaves identically in terms of inputs and outputs but returns a generator—ideal for large datasets or memory-sensitive scenarios. Choose based on your needs.

`predict()` method parameters:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Input data (required). Input formats vary by model.<br/>
For PP-Chart2Table: <code>{'image': image_path}</code>
</td>
<td><code>dict</code></td>
<td>N/A</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size. Any positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
</table>

* Prediction results are returned as `Result` objects for each sample, with support for printing and saving to JSON:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Explanation</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Format output using JSON indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level for pretty-printed JSON. Only works when <code>format_json=True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-ASCII characters to Unicode. If <code>False</code>, keeps characters as-is.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save results to JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path to save. If a directory, file will use input name as filename.</td>
<td>N/A</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Same as in `print()`</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Same as in `print()`</td>
<td><code>False</code></td>
</tr>
</table>

* You can also access the result via properties:

<table>
<thead>
<tr>
<th>Property</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td><code>json</code></td>
<td>Returns the result in JSON format</td>
</tr>
</table>

## 4. Custom Development

Currently, this module supports inference only and does not yet support fine-tuning. Fine-tuning capabilities are planned for future releases.

## 5. FAQ
