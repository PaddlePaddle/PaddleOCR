---

comments: true

---

# Document Understanding Pipeline Usage Tutorial

## 1. Introduction to the Document Understanding Pipeline

The Document Understanding Pipeline is an advanced document processing technology based on Visual-Language Models (VLM), designed to overcome the limitations of traditional document processing. Traditional methods rely on fixed templates or predefined rules to parse documents, whereas this pipeline leverages the multimodal capabilities of VLM to accurately answer user queries by inputting document images and user questions, integrating visual and language information. This technology does not require pre-training for specific document formats, allowing it to flexibly handle diverse document content, significantly enhancing the generalization and practicality of document processing. It has broad application prospects in intelligent Q&A, information extraction, and other scenarios. Currently, the pipeline does not support secondary development of VLM models, but plans to support it in the future.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/doc_understanding/doc_understanding.png">

<b>The document understanding pipeline includes the following module. Each module can be trained and inferred independently and contains multiple models. For more details, click the corresponding module to view the documentation.</b>

- [Document-like Vision Language Model Module](../module_usage/doc_vlm.md)

In this pipeline, you can choose the model to use based on the benchmark data below.

<details>
<summary> <b>Document-like Vision Language Model Module:</b></summary>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Model Storage Size (GB)</th>
<th>Total Score</th>
<th>Description</th>
</tr>
<tr>
<td>PP-DocBee-2B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-2B_infer.tar">Inference Model</a></td>
<td>4.2</td>
<td>765</td>
<td rowspan="2">PP-DocBee is a multimodal large model independently developed by the PaddlePaddle team, focusing on document understanding, with excellent performance in Chinese document understanding tasks. The model is fine-tuned and optimized using nearly 5 million multimodal datasets related to document understanding, including general VQA, OCR, chart, text-rich documents, math and complex reasoning, synthetic data, pure text data, etc., with different training data ratios set. In several authoritative English document understanding evaluation lists in academia, PP-DocBee has generally achieved SOTA for models of the same parameter scale. In internal business Chinese scenarios, PP-DocBee also outperforms current popular open and closed-source models.</td>
</tr>
<tr>
<td>PP-DocBee-7B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-7B_infer.tar">Inference Model</a></td>
<td>15.8</td>
<td>-</td>
</tr>
<tr>
<td>PP-DocBee2-3B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee2-3B_infer.tar">Inference Model</a></td>
<td>7.6</td>
<td>852</td>
<td>PP-DocBee2 is a multimodal large model independently developed by the PaddlePaddle team, focusing on document understanding. It further optimizes the basic model based on PP-DocBee and introduces new data optimization schemes to improve data quality. With only 470,000 data generated using self-developed data synthesis strategy, PP-DocBee2 performs better in Chinese document understanding tasks. In internal business Chinese scenarios, PP-DocBee2 improves by about 11.4% compared to PP-DocBee and also outperforms current popular open and closed-source models of the same scale.</td>
</tr>
</table>

<b>Note: The above total scores are the model test results of the internal evaluation set. All images in the internal evaluation set have a resolution (height, width) of (1680,1204), with a total of 1196 data, including scenarios such as financial reports, laws and regulations, science and engineering papers, manuals, humanities papers, contracts, and research reports, with no plans to make it public for now.</b>
</details>

<br />
<b>If you focus more on model accuracy, choose a model with higher accuracy; if you care more about inference speed, choose a model with faster inference speed; if you are concerned about storage size, choose a model with a smaller storage volume.</b>

## 2. Quick Start

Before using the document understanding pipeline locally, ensure that you have completed the installation of the wheel package according to the [installation tutorial](../installation.en.md). After installation, you can experience it locally using the command line or Python integration.

### 2.1 Command Line Experience

Experience the doc_understanding pipeline with just one command line:

```bash
paddleocr doc_understanding -i "{'image': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png', 'query': '识别这份表格的内容, 以markdown格式输出'}"
```

<details><summary><b>The command line supports more parameter settings, click to expand for a detailed explanation of the command line parameters</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports dictionary type input, required.
<ul>
<li><b>Python Dict</b>: The input format for PP-DocBee is: <code>{"image":/path/to/image, "query": user question}</code>, representing the input image and corresponding user question.</li>
</ul>
</td>
<td><code>Python Dict</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Specify the path for saving the inference result file. If set to <code>None</code>, the inference result will not be saved locally.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_model_name</code></td>
<td>The name of the document understanding model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_model_dir</code></td>
<td>The directory path of the document understanding model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_batch_size</code></td>
<td>The batch size of the document understanding model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying a specific card number.
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the initialized value of this parameter will be used by default, which will preferentially use the local GPU device 0, or the CPU device if none is available.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use TensorRT for inference acceleration.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>The minimum subgraph size used to optimize model subgraph calculations.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Calculation precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>The number of threads used for inference on the CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>Path to PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />

The results will be printed to the terminal, and the default configuration of the doc_understanding pipeline will produce the following output:

```bash
{'res': {'image': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png', 'query': '识别这份表格的内容, 以markdown格式输出', 'result': '| 名次 | 国家/地区 | 金牌 | 银牌 | 铜牌 | 奖牌总数 |\n| --- | --- | --- | --- | --- | --- |\n| 1 | 中国（CHN） | 48 | 22 | 30 | 100 |\n| 2 | 美国（USA） | 36 | 39 | 37 | 112 |\n| 3 | 俄罗斯（RUS） | 24 | 13 | 23 | 60 |\n| 4 | 英国（GBR） | 19 | 13 | 19 | 51 |\n| 5 | 德国（GER） | 16 | 11 | 14 | 41 |\n| 6 | 澳大利亚（AUS） | 14 | 15 | 17 | 46 |\n| 7 | 韩国（KOR） | 13 | 11 | 8 | 32 |\n| 8 | 日本（JPN） | 9 | 8 | 8 | 25 |\n| 9 | 意大利（ITA） | 8 | 9 | 10 | 27 |\n| 10 | 法国（FRA） | 7 | 16 | 20 | 43 |\n| 11 | 荷兰（NED） | 7 | 5 | 4 | 16 |\n| 12 | 乌克兰（UKR） | 7 | 4 | 11 | 22 |\n| 13 | 肯尼亚（KEN） | 6 | 4 | 6 | 16 |\n| 14 | 西班牙（ESP） | 5 | 11 | 3 | 19 |\n| 15 | 牙买加（JAM） | 5 | 4 | 2 | 11 |\n'}}
```

### 2.2 Python Script Integration

The command line method is for quickly experiencing the effect. Generally, in projects, code integration is often required. You can complete quick inference of the pipeline with just a few lines of code. The inference code is as follows:

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
    res.print() ## Print the structured output of the prediction
    res.save_to_json("./output/")
```

In the above Python script, the following steps are performed:

(1) Instantiate a Document Understanding Pipeline object through `DocUnderstanding()`. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>doc_understanding_model_name</code></td>
<td>The name of the document understanding model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_model_dir</code></td>
<td>The directory path of the document understanding model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_understanding_batch_size</code></td>
<td>The batch size of the document understanding model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying a specific card number.
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the initialized value of this parameter will be used by default, which will preferentially use the local GPU device 0, or the CPU device if none is available.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use TensorRT for inference acceleration.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>The minimum subgraph size used to optimize model subgraph calculations.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Calculation precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>The number of threads used for inference on the CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>Path to PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the Document Understanding Pipeline object for inference prediction, which will return a result list.

Additionally, the pipeline also provides a `predict_iter()` method. Both methods are consistent in terms of parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator` that can process and obtain prediction results step by step, suitable for handling large datasets or scenarios where memory saving is desired. You can choose to use either method according to your actual needs.

Below are the parameters and their descriptions for the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, currently only supports dictionary type input
<ul>
  <li><b>Python Dict</b>: The input format for PP-DocBee is: <code>{"image":/path/to/image, "query": user question}</code>, representing the input image and corresponding user question.</li>
</ul>
</td>
<td><code>Python Dict</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</table>

(3) Process the prediction results. The prediction result for each sample is a corresponding Result object, which supports printing and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters into <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When specified as a directory, the saved file is named consistent with the input file type.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters into <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal. The content printed to the terminal is explained as follows:

    - `image`: `(str)` Input path of the image

    - `query`: `(str)` Question regarding the input image

    - `result`: `(str)` Output result of the model

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the path saved will be `save_path/{your_img_basename}_res.json`, and if specified as a file, it will be saved directly to that file.

* Additionally, the result can be obtained through attributes that provide the visualized images with results and the prediction results, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained through the `json` attribute is data of the dict type, consistent with the content saved by calling the `save_to_json()` method.

## 3. Development Integration/Deployment

If the pipeline meets your requirements for pipeline inference speed and accuracy, you can proceed with development integration/deployment directly.

If you need to apply the pipeline directly to your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

In addition, PaddleOCR also provides two other deployment methods, detailed descriptions are as follows:

🚀 High-Performance Inference: In real production environments, many applications have strict standards for the performance indicators of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleOCR provides high-performance inference capabilities, aiming to deeply optimize the performance of model inference and pre-and post-processing, achieving significant acceleration of the end-to-end process. For detailed high-performance inference processes, refer to [High-Performance Inference](../deployment/high_performance_inference.md).

☁️ Service Deployment: Service deployment is a common form of deployment in real production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. For detailed pipeline service deployment processes, refer to [Serving](../deployment/serving.md).

Below is the API reference for basic service deployment and examples of service invocation in multiple languages:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON object).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform inference on the input message to generate a response.</p>
<p><code>POST /document-understanding</code></p>
<p>Note: The above interface is also known as /chat/completion, compatible with OpenAI interfaces.</p>

<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model</code></td>
<td><code>string</code></td>
<td>The name of the model to use</td>
<td>Yes</td>
<td>-</td>
</tr>
<tr>
<td><code>messages</code></td>
<td><code>array</code></td>
<td>List of dialogue messages</td>
<td>Yes</td>
<td>-</td>
</tr>
<tr>
<td><code>max_tokens</code></td>
<td><code>integer</code></td>
<td>Maximum number of tokens to generate</td>
<td>No</td>
<td>1024</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>float</code></td>
<td>Sampling temperature</td>
<td>No</td>
<td>0.1</td>
</tr>
<tr>
<td><code>top_p</code></td>
<td><code>float</code></td>
<td>Core sampling probability</td>
<td>No</td>
<td>0.95</td>
</tr>
<tr>
<td><code>stream</code></td>
<td><code>boolean</code></td>
<td>Whether to output in streaming mode</td>
<td>No</td>
<td>false</td>
</tr>
<tr>
<td><code>max_image_tokens</code></td>
<td><code>int</code></td>
<td>Maximum number of input tokens for images</td>
<td>No</td>
<td>None</td>
</tr>
</tbody>
</table>

<p>Each element in <code>messages</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>role</code></td>
<td><code>string</code></td>
<td>Message role (user/assistant/system)</td>
<td>Yes</td>
</tr>
<tr>
<td><code>content</code></td>
<td><code>string</code> or <code>array</code></td>
<td>Message content (text or mixed media)</td>
<td>Yes</td>
</tr>
</tbody>
</table>

<p>When <code>content</code> is an array, each element is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>type</code></td>
<td><code>string</code></td>
<td>Content type (text/image_url)</td>
<td>Yes</td>
<td>-</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Text content (when type is text)</td>
<td>Conditionally required</td>
<td>-</td>
</tr>
<tr>
<td><code>image_url</code></td>
<td><code>string</code> or <code>object</code></td>
<td>Image URL or object (when type is image_url)</td>
<td>Conditionally required</td>
<td>-</td>
</tr>
</tbody>
</table>

<p>When <code>image_url</code> is an object, it has the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>url</code></td>
<td><code>string</code></td>
<td>Image URL</td>
<td>Yes</td>
<td>-</td>
</tr>
<tr>
<td><code>detail</code></td>
<td><code>string</code></td>
<td>Image detail processing method (low/high/auto)</td>
<td>No</td>
<td>auto</td>
</tr>
</tbody>
</table>

<p>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>id</code></td>
<td><code>string</code></td>
<td>Request ID</td>
</tr>
<tr>
<td><code>object</code></td>
<td><code>string</code></td>
<td>Object type (chat.completion)</td>
</tr>
<tr>
<td><code>created</code></td>
<td><code>integer</code></td>
<td>Creation timestamp</td>
</tr>
<tr>
<td><code>choices</code></td>
<td><code>array</code></td>
<td>Generated result options</td>
</tr>
<tr>
<td><code>usage</code></td>
<td><code>object</code></td>
<td>Token usage</td>
</tr>
</tbody>
</table>

<p>Each element in <code>choices</code> is a <code>Choice</code> object with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Optional Values</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>finish_reason</code></td>
<td><code>string</code></td>
<td>Reason for the model to stop generating tokens</td>
<td><code>stop</code> (natural stop)<br><code>length</code> (reached max token count)<br><code>tool_calls</code> (called a tool)<br><code>content_filter</code> (content filtering)<br><code>function_call</code> (called a function, deprecated)</td>
</tr>
<tr>
<td><code>index</code></td>
<td><code>integer</code></td>
<td>Index of the option in the list</td>
<td>-</td>
</tr>
<tr>
<td><code>logprobs</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Log probability information of the option</td>
<td>-</td>
</tr>
<tr>
<td><code>message</code></td>
<td><code>ChatCompletionMessage</code></td>
<td>Chat message generated by the model</td>
<td>-</td>
</tr>
</tbody>
</table>

<p>The <code>message</code> object has the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Remarks</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>content</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Message content</td>
<td>May be empty</td>
</tr>
<tr>
<td><code>refusal</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refusal message generated by the model</td>
<td>Provided when content is refused</td>
</tr>
<tr>
<td><code>role</code></td>
<td><code>string</code></td>
<td>Role of the message author</td>
<td>Fixed as <code>"assistant"</code></td>
</tr>
<tr>
<td><code>audio</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Audio output data</td>
<td>Provided when audio output is requested<br><a href="https://platform.openai.com/docs/guides/audio">Learn more</a></td>
</tr>
<tr>
<td><code>function_call</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Name and parameters of the function to be called</td>
<td>Deprecated, recommended to use <code>tool_calls</code></td>
</tr>
<tr>
<td><code>tool_calls</code></td>
<td><code>array</code> | <code>null</code></td>
<td>Tool calls generated by the model</td>
<td>Such as function calls</td>
</tr>
</tbody>
</table>

<p>The <code>usage</code> object has the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>prompt_tokens</code></td>
<td><code>integer</code></td>
<td>Number of prompt tokens</td>
</tr>
<tr>
<td><code>completion_tokens</code></td>
<td><code>integer</code></td>
<td>Number of generated tokens</td>
</tr>
<tr>
<td><code>total_tokens</code></td>
<td><code>integer</code></td>
<td>Total number of tokens</td>
</tr>
</tbody>
</table>
<p>An example of a <code>result</code> is as follows:</p>
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

<details><summary>Multi-language Service Invocation Examples</summary>

<details>
<summary>Python</summary>
OpenAI interface invocation example

<pre><code class="language-python">import base64
from openai import OpenAI

API_BASE_URL = "http://0.0.0.0:8080"

# Initialize OpenAI client
client = OpenAI(
    api_key='xxxxxxxxx',
    base_url=f'{API_BASE_URL}'
)

# Function to convert image to base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Input image path
image_path = "medal_table.png"

# Convert original image to base64
base64_image = encode_image(image_path)

# Submit information to PP-DocBee model
response = client.chat.completions.create(
    model="pp-docbee",# Choose Model
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

## 4. Secondary Development

The current pipeline does not support fine-tuning training and only supports inference integration. Concerning fine-tuning training for this pipeline, there are plans to support it in the future.
