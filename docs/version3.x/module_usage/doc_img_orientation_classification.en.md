---
comments: true
---

# Document Image Orientation Classification Module Tutorial

## 1. Overview

The Document Image Orientation Classification Module is primarily designed to distinguish the orientation of document images and correct them through post-processing. During processes such as document scanning or ID photo capturing, the device might be rotated to achieve clearer images, resulting in images with various orientations. Standard OCR pipelines may not handle these images effectively. By leveraging image classification techniques, the orientation of documents or IDs containing text regions can be pre-determined and adjusted, thereby improving the accuracy of OCR processing.

## 2. Supported Models List

> The inference time only includes the model inference time and does not include the time for pre- or post-processing. The "Normal Mode" values correspond to the local <code>paddle_static</code> inference engine.

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Links</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Pretrained Model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, and 270°.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

<ul>
    <li><b>Performance Test Environment</b>
        <ul>
            <li><strong>Test Dataset:</strong> Self-built multi-scenario dataset (1000 images, including ID/document scenarios)</li>
              <li><strong>Hardware Configuration:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>Software Environment:</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle-gpu 3.0.0 / paddleocr 3.0.3</li>
                  </ul>
              </li>
        </ul>
    </li>
    <li><b>Inference Mode Description</b></li>
</ul>

<table border="1">
    <thead>
        <tr>
            <th>Mode</th>
            <th>GPU Configuration</th>
            <th>CPU Configuration</th>
            <th>Acceleration Technology Combination</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Normal Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of precision type and acceleration strategy</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Optimal backend selected (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## 3. Quick Start

> ❗ Before starting, please install the PaddleOCR wheel package. For details, refer to the [Installation Guide](../installation.en.md).

You can quickly experience it with one command:

```bash
paddleocr doc_img_orientation_classification -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg
```

The example above uses the <code>paddle_static</code> inference engine by default. To run it, first install PaddlePaddle by following [PaddlePaddle Framework Installation](../paddlepaddle_installation.en.md).

If you choose `transformers` as the inference engine, make sure the Transformers environment is configured, and then run the following command:

```bash
# Use the transformers engine for inference
paddleocr doc_img_orientation_classification -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg \
    --engine transformers
```

In most scenarios, the default `paddle_static` inference engine delivers better inference performance and is the recommended first choice.

<b>Note: </b>The official models would be download from HuggingFace by default. If can't access to HuggingFace, please set the environment variable `PADDLE_PDX_MODEL_SOURCE="BOS"` to change the model source to BOS. In the future, more model sources will be supported.

You can also integrate the model inference of the Document Image Orientation Classification Module into your project. Before running the following code, please download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg) to your local machine.

```python
from paddleocr import DocImgOrientationClassification

model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
output = model.predict("img_rot180_demo.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```

The example above uses the <code>paddle_static</code> inference engine by default. To run it, first install PaddlePaddle by following [PaddlePaddle Framework Installation](../paddlepaddle_installation.en.md).

If you choose `transformers` as the inference engine, make sure the Transformers environment is configured, and then run the following code:

```python
from paddleocr import DocImgOrientationClassification

model = DocImgOrientationClassification(
    model_name="PP-LCNet_x1_0_doc_ori",
    engine="transformers",
)
output = model.predict("img_rot180_demo.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```

In most scenarios, the default `paddle_static` inference engine delivers better inference performance and is the recommended first choice.

If you want to use the trained model with the `paddle_dynamic` or `transformers` engine, refer to the [Weight Conversion](#52-weight-conversion) section in the [Inference Engine](#5-inference-engine) section below to convert the model from the `pdparams` format to the `safetensors` format using PaddleX.

After running, the result will be:

```bash
{'res': {'input_path': 'img_rot180_demo.jpg', 'page_index': None, 'class_ids': array([2], dtype=int32), 'scores': array([0.88164], dtype=float32), 'label_names': ['180']}}
```

The meaning of the output parameters is as follows:
<ul>
<li><code>input_path</code>：Represents the path of the input image.</li>
<li><code>class_ids</code>：Represents the predicted class ID, with four categories: 0°, 90°, 180°, and 270°.</li>
<li><code>scores</code>：Represents the confidence level of the prediction result.</li>
<li><code>label_names</code>：Represents the category names of the prediction results.</li>
</ul>

Here is the visualization of the image:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/doc_img_ori_classification/img_rot180_demo_res.jpg">

The explanations of relevant methods and parameters are as follows:

* Instantiate the document image orientation classification model with <code>DocImgOrientationClassification</code> (taking <code>PP-LCNet_x1_0_doc_ori</code> as an example here). The specific explanations are as follows:
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
<td><b>Meaning：</b>Model name. <br/>
<b>Description：</b>
If set to <code>None</code>, <code>PP-LCNet_x1_0_doc_ori</code> will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td><b>Meaning：</b>Model storage path.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td><b>Meaning：</b>Device for inference.<br/>
<b>Description：</b>
<b>For example:</b><code>"cpu"</code>, <code>"gpu"</code>, <code>"npu"</code>, <code>"gpu:0"</code>, <code>"gpu:0,1"</code>.<br/>
If multiple devices are specified, parallel inference will be performed.<br/>
By default, GPU 0 is used if available; otherwise, CPU is used.
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>engine</code></td>
<td><b>Meaning:</b> Inference engine.<br/><b>Description:</b> Supports <code>None</code> (the default), <code>paddle</code>, <code>paddle_static</code>, <code>paddle_dynamic</code>, and <code>transformers</code>. When left as <code>None</code>, local inference uses the <code>paddle_static</code> engine by default. For detailed descriptions, supported values, compatibility rules, and examples, see <a href="../inference_engine.en.md">Inference Engine and Configuration</a>.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>engine_config</code></td>
<td><b>Meaning:</b> Inference-engine configuration.<br/><b>Description:</b> Recommended together with <code>engine</code>. For supported fields, compatibility rules, and examples, see <a href="../inference_engine.en.md">Inference Engine and Configuration</a>.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td><b>Meaning：</b>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td><b>Meaning：</b>Whether to use the Paddle Inference TensorRT subgraph engine.<br/>
 <b>Description：</b>
 If the model does not support acceleration through TensorRT, setting this flag will not enable acceleration.<br/>
For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td><b>Meaning：</b>Computation precision when using the TensorRT subgraph engine in Paddle Inference.<br/>
<b>Description：</b>
<b>Options:</b><code>"fp32"</code>, <code>"fp16"</code>.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
<b>Meaning：</b>Whether to enable MKL-DNN acceleration for inference. <br/>
<b>Description：</b>
If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
<b>Meaning：</b>MKL-DNN cache capacity.
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td><b>Meaning：</b>Number of threads to use for inference on CPUs.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
</tbody>
</table>

* Call the  <code>predict()</code>  method of the document image orientation classification model for inference prediction. This method will return a list of results. In addition, this module also provides the <code>predict_iter()</code> method. The two methods are completely consistent in terms of parameter acceptance and result return. The difference is that  <code>predict_iter()</code>  returns a  <code>generator</code>, which can process and obtain prediction results step by step, suitable for scenarios where large datasets need to be processed or memory needs to be saved. You can choose either of these two methods according to your actual needs. The parameters of the <code>predict()</code>  method are <code>input</code> and <code>batch_size</code>, and the specific explanations are as follows:
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
<td><b>Meaning：</b>Input data to be predicted. Required. <br/>
<b>Description：</b>
Supports multiple input types:
<ul>
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>：Local image or PDF file path: <code>/root/data/img.jpg</code>；
<b>URL</b> of image or PDF file: e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">example</a>;
<b>Local directory</b>: directory containing images for prediction, e.g., <code>/root/data/</code> (Note: directories containing PDF files are not supported; PDFs must be specified by exact file path)</li>
<li><b>list</b>: Elements must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td><b>Meaning：</b>Batch size. <br/>
<b>Description：</b>
Positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is the corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a <code>json</code> file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Description</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. It is only valid when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; when set to <code>False</code>, the original characters will be retained. It is only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a file in <code>json</code> format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save. When it is a directory, the saved file name is consistent with the naming of the input file type.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. It is only valid when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; when set to <code>False</code>, the original characters will be retained. It is only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save. When it is a directory, the saved file name is consistent with the naming of the input file type.</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining the visualization image with results and the prediction results through attributes. The specifics are as follows:

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
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">Get the visualization image in <code>dict</code> format</td>
</tr>

</table>

## 4. Secondary Development

Since PaddleOCR does not directly provide training functionality for document image orientation classification, if you need to train a document image orientation classification model, you can refer to the [PaddleX Secondary Development for Document Image Orientation Classification](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development) section for training guidance. The trained model can be seamlessly integrated into PaddleOCR's API for inference purposes.

If you want to use the `paddle_dynamic` or `transformers` engine with the trained model, please refer to the [Weight Conversion](#52-weight-conversion) section in [Inference Engine](#5-inference-engine) later in this document to convert the model from the `pdparams` format to the `safetensors` format using PaddleX.

## 5. Inference Engine

For detailed descriptions, values, compatibility rules, and examples of the inference engine, please refer to <a href="../inference_engine.en.md">Inference Engine and Configuration Description</a>.

### 5.1 Speed Data

<table border="1">
    <thead>
        <tr>
            <th>model</th>
            <th>engine</th>
            <th>Preprocessing (ms)</th>
            <th>Inference (ms)</th>
            <th>PostProcessing (ms)</th>
            <th>End-to-End (ms)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">PP-LCNet_x1_0_doc_ori</td>
            <td>paddle_static</td>
            <td>2.21</td>
            <td>3.36</td>
            <td>0.06</td>
            <td>5.74</td>
        </tr>
        <tr>
            <td>paddle_dynamic</td>
            <td>2.15</td>
            <td>7.54</td>
            <td>0.07</td>
            <td>9.87</td>
        </tr>
        <tr>
            <td>transformers</td>
            <td>4.46</td>
            <td>3.44</td>
            <td>0.14</td>
            <td>8.36</td>
        </tr>
    </tbody>
</table>

<strong>Test Environment Description:</strong>
<ul>
    <li><strong>Test Data:</strong> [Sample Image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg)</li>
    <li><strong>Hardware Configuration:</strong>
        <ul>
            <li>GPU: NVIDIA A100 40G</li>
            <li>CPU: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz</li>
        </ul>
    </li>
    <li><strong>Software Environment:</strong>
        <ul>
            <li>Ubuntu 22.04 / CUDA 12.6 / cuDNN 9.5</li>
            <li>paddlepaddle-gpu 3.2.1 / paddleocr 3.5 / transformers 5.4.0 / torch 2.10</li>
        </ul>
    </li>
</ul>

### 5.2 Weight Conversion

When using the inference engine, the system will automatically download the official pre-trained model. If you need to use a self-trained model with the `paddle_dynamic` or `transformers` engine, please refer to the [PaddleX Text Image Orientation Classification Module Weight Conversion](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#442) section to convert the model from the `pdparams` format to the `safetensors` format using PaddleX. This allows seamless integration into the PaddleOCR API for inference.

## 6. FAQ
