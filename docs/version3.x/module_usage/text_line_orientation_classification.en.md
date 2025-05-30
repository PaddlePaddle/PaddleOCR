---
comments: true
---

# Text Line Orientation Classification Module Tutorial

## 1. Overview
The text line orientation classification module primarily distinguishes the orientation of text lines and corrects them using post-processing. In processes such as document scanning and license/certificate photography, to capture clearer images, the capture device may be rotated, resulting in text lines in various orientations. Standard OCR pipelines cannot handle such data well. By utilizing image classification technology, the orientation of text lines can be predetermined and adjusted, thereby enhancing the accuracy of OCR processing.

## 2. Supported Model List

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>98.85</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>Text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees</td>
</tr>
<tr>
<td>PP-LCNet_x1_0_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>99.42</td>
<td>-</td>
<td>-</td>
<td>6.5</td>
<td>Text line classification model based on PP-LCNet_x1_0, with two classes: 0 degrees and 180 degrees</td>
</tr>
</tbody>
</table>

> ❗ **Note**: The text line orientation classification model has been recently upgraded, and `PP-LCNet_x1_0_textline_ori` has been added. If you need to use the pre-upgrade model weights, please click the <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.bak.tar">download link</a>.

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
             <li><strong>Test Dataset：</strong> PaddleX Self-built Dataset, Covering Multiple Scenarios Such as Documents and Certificates, Containing 1000 Images.</li>
              <li><strong>Hardware Configuration：</strong>
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
            <th>GPU Configuration </th>
            <th>CPU Configuration </th>
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
            <td>Optimal combination of pre-selected precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## 3. Quick Integration

> ❗ Before starting, please install the wheel package of PaddleOCR. For detailed instructions, refer to the [Installation Guide](../installation.en.md).

You can quickly experience the functionality with a single command:

```bash
paddleocr text_line_orientation_classification -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg
```

You can also integrate the text line orientation classification model into your project. Run the following code after downloading the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg) to your local machine. 

```bash
from paddleocr import TextLineOrientationClassification
model = TextLineOrientationClassification(model_name="PP-LCNet_x0_25_textline_ori")
output = model.predict("textline_rot180_demo.jpg",  batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': 'textline_rot180_demo.jpg', 'page_index': None, 'class_ids': array([1], dtype=int32), 'scores': array([0.99864], dtype=float32), 'label_names': ['180_degree']}}
```

The meanings of the running results parameters are as follows:

- `input_path`：Indicates the path of the input image.
- `page_index`：If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.
- `class_ids`：Indicates the class ID of the prediction result.
- `scores`：Indicates the confidence score of the prediction result.
- `label_names`：Indicates the class name of the prediction result.
The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/textline_ori_classification/textline_rot180_demo_res.jpg">

The explanations for the methods, parameters, etc., are as follows:

* `TextLineOrientationClassification` instantiates a textline classification model (here, `PP-LCNet_x0_25_textline_ori` is used as an example), and the specific explanations are as follows:

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
<td>Name of the model</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device(s) to use for inference.<br/>
<b>Examples:</b> <code>cpu</code>, <code>gpu</code>, <code>npu</code>, <code>gpu:0</code>, <code>gpu:0,1</code>.<br/>
If multiple devices are specified, inference will be performed in parallel. Note that parallel inference is not always supported.<br/>
By default, GPU 0 will be used if available; otherwise, the CPU will be used.
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to use the high performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use the Paddle Inference TensorRT subgraph engine.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>Minimum subgraph size for TensorRT when using the Paddle Inference TensorRT subgraph engine.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Precision for TensorRT when using the Paddle Inference TensorRT subgraph engine.<br/><b>Options:</b> <code>fp32</code>, <code>fp16</code>, etc.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
Whether to use MKL-DNN acceleration for inference.
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>Number of threads to use for inference on CPUs.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>top_k</code></td>
<td>The top-k value for prediction results. If not specified, the default value in the official PaddleOCR model configuration is used. If the value is 5, the top 5 categories and their corresponding classification probabilities will be returned.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* `model_name` must be specified. Once `model_name` is set, the default built-in model parameters of PaddleOCR will be used. On this basis, if `model_dir` is specified, the user-defined model will be used.

* Use the `predict()` method of the text line direction classification model to perform inference. This method returns a list of results. In addition, this module also provides the `predict_iter()` method. Both methods accept the same parameters and return the same result format. The difference is that `predict_iter()` returns a `generator`, which processes and retrieves prediction results step by step. It is suitable for handling large datasets or memory-efficient scenarios. You can choose either method based on your actual needs. The `predict()` method accepts the parameters `input` and `batch_size`, which are described in detail below:
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
<td>Input data for prediction. Multiple input types are supported. This parameter is required.
<ul>
<li><b>Python Var</b>: such as <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: such as the local path of an image or PDF file: <code>/root/data/img.jpg</code>; <b>or a URL link</b>, such as the online URL of an image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">Example</a>; <b>or a local directory</b> that contains images for prediction, such as <code>/root/data/</code> (currently, directories containing PDF files are not supported; PDF files must be specified as individual file paths)</li>
<li><b>List</b>: list elements must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size,  positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
</table>

* Call the `predict()` method of the text line orientation classification model for inference. This method will return a list of results. In addition, this module also provides a `predict_iter()` method. Both methods accept the same parameters and return the same results, but `predict_iter()` returns a `generator`, which is more suitable for processing large datasets or when you want to save memory. You can choose either method according to your needs. The parameters of the `predict()` method are `input` and `batch_size`, as described below:



<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
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
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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

* The prediction results are processed, and the prediction result for each sample is of type `dict`. It supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

## 4. Custom Development  

Since PaddleOCR does not natively support training for text line orientation classification, refer to [PaddleX's Custom Development Guide](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#iv-custom-development) for training. Trained models can seamlessly integrate into PaddleOCR's API for inference.
