---

comments: true

---

# Table Cell Detection Module Usage Tutorial

## I. Overview

The Table Cell Detection Module is a key component of the table recognition task, responsible for locating and marking each cell region in table images. The performance of this module directly affects the accuracy and efficiency of the entire table recognition process. The Table Cell Detection Module typically outputs bounding boxes for each cell region, which are then passed as input to the table recognition pipeline for further processing.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is a real-time end-to-end object detection model. The Baidu PaddlePaddle Vision team pre-trained on a self-built table cell detection dataset based on the RT-DETR-L as the base model, achieving good performance in detecting both wired and wireless table cells.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
              <li><strong>Test Dataset:</strong> Internal evaluation set built by PaddleX.</li>
              <li><strong>Hardware Configuration:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environment: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>Inference Mode Explanation</b></li>
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
            <td>Optimal combination of prior precision type and acceleration strategy</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Choose the optimal prior backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## III. Quick Start

> ❗ Before starting quickly, please first install the PaddleOCR wheel package. For details, please refer to the [installation tutorial](../installation.en.md).

You can quickly experience it with one command:

```bash
paddleocr table_cells_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg
```

You can also integrate model inference from the table cell detection module into your project. Before running the following code, please download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) locally.

```python
from paddleocr import TableCellsDetection
model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
output = model.predict("table_recognition.jpg", threshold=0.3, batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```
{'res': {'input_path': 'table_recognition.jpg', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'cell', 'score': 0.9698355197906494, 'coordinate': [2.3011515, 0, 546.29926, 30.530712]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9690820574760437, 'coordinate': [212.37508, 64.62493, 403.58868, 95.61413]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9668057560920715, 'coordinate': [212.46791, 30.311079, 403.7182, 64.62613]}, {'cls_id': 0, 'label': 'cell', 'score': 0.966505229473114, 'coordinate': [403.56082, 64.62544, 546.83215, 95.66117]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9662341475486755, 'coordinate': [109.48873, 64.66485, 212.5177, 95.631294]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9654079079627991, 'coordinate': [212.39197, 95.63037, 403.60852, 126.78792]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9653300642967224, 'coordinate': [2.2320926, 64.62229, 109.600494, 95.59732]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9639787673950195, 'coordinate': [403.5752, 30.562355, 546.98975, 64.61531]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9636150002479553, 'coordinate': [2.1537683, 30.410172, 109.568306, 64.62762]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9631900191307068, 'coordinate': [2.0534437, 95.57448, 109.57601, 126.71458]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9631181359291077, 'coordinate': [403.65976, 95.68139, 546.84766, 126.713394]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9614537358283997, 'coordinate': [109.56504, 30.391184, 212.65425, 64.6444]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9607433080673218, 'coordinate': [109.525795, 95.62622, 212.44917, 126.8258]}]}}
```

The parameter meanings are as follows:

- `input_path`: Path of the input image to be predicted
- `page_index`: If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is `None`
- `boxes`: Predicted bounding box information, a list of dictionaries. Each dictionary represents a detected object and contains the following information:
  - `cls_id`: Class ID, an integer
  - `label`: Class label, a string
  - `score`: Confidence of the bounding box, a float
  - `coordinate`: Coordinates of the bounding box, a list of floats in the format <code>[xmin, ymin, xmax, ymax]</code>

The visualized image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/table_cells_detection/01.jpg">

The relevant methods, parameters, etc., are described as follows:

* `TableCellsDetection` instantiates the table cell detection model (taking `RT-DETR-L_wired_table_cell_det` as an example here), with specific explanations as follows:
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
<td>Model name</td>
<td><code>str</code></td>
<td><code>PP-DocLayout-L</code></td>
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
<td><code>img_size</code></td>
<td>Size of the input image;If not specified, the default configuration of the PaddleOCR official model will be used<br/><b>Examples:</b>
<ul>
<li><b>int</b>: e.g. 640, resizes input image to 640x640</li>
<li><b>list</b>: e.g. [640, 512], resizes input image to 640 width and 512 height</li>
</ul>
</td>
<td><code>int/list/None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold to filter out low-confidence predictions; In table cell detection tasks, lowering the threshold appropriately may help to obtain more accurate results. <br/><b>Examples:</b>
<ul>
  <li><b>float</b>, e.g., 0.3, indicates filtering out all bounding boxes with confidence lower than 0.3</li>
  <li><b>dictionary</b>, where the key is of type <b>int</b> representing <code>cls_id</code>, and the value is of type <b>float</b> representing the threshold. For example, <code>{0: 0.3}</code> applies a threshold of 0.3 for category cls_id 0</li>
</ul>
</td>
<td><code>float/dict/None</code></td>
<td>None</td>
</tr>
</table>

* Among them, `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. When `model_dir` is specified, the user-defined model is used.

* Call the `predict()` method of the table cell detection model for inference prediction. This method will return a result list. Additionally, this module also provides a `predict_iter()` method. Both methods are consistent in terms of parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator`, which can process and obtain prediction results step by step, suitable for handling large datasets or scenarios where memory saving is desired. You can choose to use either of these methods according to your actual needs. The `predict()` method has parameters `input`, `batch_size`, and `threshold`, with specific explanations as follows:

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
<td>Input data to be predicted. Required. Supports multiple input types:
<ul>
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: 
  - Local image or PDF file path: <code>/root/data/img.jpg</code>;
  - <b>URL</b> of image or PDF file: e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">example</a>;
  - <b>Local directory</b>: directory containing images for prediction, e.g., <code>/root/data/</code> (Note: directories containing PDF files are not supported; PDFs must be specified by exact file path)</li>
<li><b>List</b>: Elements must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size, positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering out low-confidence prediction results; <br/><b>Examples:</b>
<ul>
<li><b>float</b>: e.g., 0.2, filters out all boxes with scores below 0.2</li>
<li><b>dict</b>: keys are <code>int</code> representing <code>cls_id</code>, and values are <code>float</code> thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> applies thresholds of 0.45 to class 0, 0.48 to class 2, and 0.4 to class 7</li>
<li><b>None</b>: if not specified, the threshold parameter specified in creat_model will be used by default, and if creat_model also does not specify it, the default PaddleOCR official model configuration will be used</li>
</ul>
</td>
<td><code>float/dict/None</code></td>
<td>None</td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is a corresponding Result object, which supports printing, saving as an image, and saving as a `json` file:

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
<td rowspan = "3">Print result to terminal</td>
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
<td rowspan = "3">Save the result as a json format file</td>
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
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When specified as a directory, the saved file is named consistent with the input file type.</td>
<td>None</td>
</tr>
</table>

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
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">Get the visualized image</td>
</tr>

</table>

## IV. Secondary Development

Since PaddleOCR does not directly provide training for the table cell detection module, if you need to train a table cell detection model, you can refer to the [PaddleX Table Cell Detection Module Secondary Development](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_cells_detection.html#iv-secondary-development) section for training. The trained model can be seamlessly integrated into the PaddleOCR API for inference.

## V. FAQ
