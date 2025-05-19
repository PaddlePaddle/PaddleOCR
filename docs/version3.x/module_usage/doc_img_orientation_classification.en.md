---
comments: true
---

# Tutorial for Document Image Orientation Classification Module

## 1. Overview

The Document Image Orientation Classification module is primarily used to distinguish the orientation of document images and correct them using post-processing. During processes such as document scanning or ID card photography, the device may be rotated to capture clearer images, resulting in images with varying orientations. In such cases, standard OCR pipelines may not handle these data effectively. By leveraging image classification technology, the orientation of documents or ID cards containing text regions can be pre-determined and adjusted, thereby enhancing the accuracy of OCR processing.

## 2. Supported Models List

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Links</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Pretrained Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, and 270°.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

<ul>
    <li><b>Performance Test Environment</b>
        <ul>
            <li><strong>Test Dataset:</strong> Self-built multi-scenario dataset (1000 images, including ID cards/documents, etc.)</li>
            <li><strong>Hardware Configuration:</strong>
                <ul>
                    <li>GPU: NVIDIA Tesla T4</li>
                    <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                    <li>Other Environments: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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
            <td>Regular Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Select the optimal combination of precision type and acceleration strategy</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Select the optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## 3. Quick Start

> ❗ Before getting started, please install the PaddleOCR wheel package. For details, refer to the [Installation Tutorial](../ppocr/installation.md).

You can quickly experience it with a single command:

```bash
paddleocr doc_img_orientation_classification -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg
```

You can also integrate the model inference from the Document Image Orientation Classification module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg) locally.

```python
from paddleocr import DocImgOrientationClassification

model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
output = model.predict("img_rot180_demo.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```

After running, the obtained result is:

```bash
{'res': {'input_path': 'img_rot180_demo.jpg', 'page_index': None, 'class_ids': array([2], dtype=int32), 'scores': array([0.88164], dtype=float32), 'label_names': ['180']}}
```

The meanings of the parameters in the running result are as follows:
- `input_path`: Indicates the path of the input image.
- `class_ids`: Indicates the class ID of the prediction result, with four categories: 0°, 90°, 180°, and 270°.
- `scores`: Represents the confidence level of the prediction result.
- `label_names`: Represents the category name of the prediction result.

The visualized image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/doc_img_ori_classification/img_rot180_demo_res.jpg">

The descriptions of relevant methods and parameters are as follows:

* Instantiate the document image orientation classification model using `DocImgOrientationClassification` (taking `PP-LCNet_x1_0_doc_ori` as an example here). The specific descriptions are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Model name</td>
<td><code>str</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device for the model</td>
<td><code>str</code></td>
<td>Supports specifying a specific GPU card number, such as "gpu:0", or other hardware-specific card numbers, such as "npu:0". For CPU, use "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</table>

* Among them, `model_name` must be specified. After specifying `model_name`, the built-in model parameters of PaddleX are used by default. On this basis, when `model_dir` is specified, the user-defined model is used.

* Call the `predict()` method of the document image orientation classification model for inference prediction. This method will return a list of results. In addition, this module also provides the `predict_iter()` method. Both methods are completely consistent in terms of parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator`, which can process and obtain prediction results step by step, suitable for scenarios of processing large datasets or wishing to save memory. You can choose to use either of these two methods according to your actual needs. The parameters of the `predict()` method include `input` and `batch_size`, with specific descriptions as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">Example</a></li>
  <li><b>Local directory</b>, which should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, whose elements should be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer</td>
<td>1</td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is a corresponding Result object, which supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. It is only effective when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; when set to <code>False</code>, the original characters are retained. It is only effective when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a file in <code>json</code> format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save. When it is a directory, the saved file name is consistent with the input file type name.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. It is only effective when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; when set to <code>False</code>, the original characters are retained. It is only effective when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save. When it is a directory, the saved file name is consistent with the input file type name.</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining the visualized image with results and the prediction results through attributes. The specific descriptions are as follows:

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
<td rowspan = "1">Get the visualized image in <code>dict</code> format</td>
</tr>

</table>

## 4. Secondary Development

Since PaddleOCR does not directly provide training for document image orientation classification, if you need to train a document image orientation classification model, you can refer to the [Secondary Development for Document Image Orientation Classification in PaddleX](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development) section for training instructions. The trained model can be seamlessly integrated into PaddleOCR's API for inference purposes.