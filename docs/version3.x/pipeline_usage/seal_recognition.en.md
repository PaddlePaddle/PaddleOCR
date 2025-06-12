---
comments: true
---

# Seal Text Recognition Pipeline Usage Tutorial

## 1. Overview

Table structure recognition is an important component of table recognition systems, capable of converting non-editable table images into editable table formats (such as HTML). The goal of table structure recognition is to identify the positions of rows, columns, and cells in tables. The performance of this module directly affects the accuracy and efficiency of the entire table recognition system. The table structure recognition module usually outputs HTML code for the table area, which is then passed as input to the tabl recognition pipeline for further processing.

## 2. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td rowspan="1">SLANet is a table structure recognition model independently developed by Baidu PaddlePaddle Vision Team. By adopting a CPU-friendly lightweight backbone network PP-LCNet, high-low level feature fusion module CSP-PAN, and SLA Head, a feature decoding module aligning structure and position information, this model greatly improves the accuracy and inference speed of table structure recognition.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td rowspan="1">SLANet_plus is an enhanced version of the table structure recognition model SLANet independently developed by the Baidu PaddlePaddle Vision Team. Compared to SLANet, SLANet_plus has greatly improved the recognition ability for wireless and complex tables, and reduced the model's sensitivity to table positioning accuracy. Even if the table positioning is offset, it can still be accurately recognized.
</td>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
<td rowspan="2">The SLANeXt series is a new generation of table structure recognition models independently developed by the Baidu PaddlePaddle Vision Team. Compared to SLANet and SLANet_plus, SLANeXt focuses on table structure recognition, and trains dedicated weights for wired and wireless tables separately. The recognition ability for all types of tables has been significantly improved, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training Model</a></td>
</tr>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
              <li><strong>Test Dataset:</strong> High-difficulty Chinese table recognition dataset.</li>
              <li><strong>Hardware Configuration:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environment: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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
            <td>FP32 precision / No TRT acceleration</td>
            <td>FP32 precision / 8 threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High Performance Mode</td>
            <td>Optimal combination of prior precision type and acceleration strategy</td>
            <td>FP32 precision / 8 threads</td>
            <td>Selects the prior optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>


## 3. Quick Start

> ❗ Before getting started, please install the PaddleOCR wheel package. For details, please refer to the [Installation Tutorial](../installation.en.md).

Quickly experience with a single command:

```bash
paddleocr table_structure_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg
```

<details><summary><b>The command line supports more parameter settings. Click to expand for detailed explanations of command line parameters.</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, required.
Local path of image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, e.g., local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with a specific file path).
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>
Specify the path to save the inference results file. If not set, the inference results will not be saved locally.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td> 
The name of the document orientation classification model. If not set, the default model in pipeline will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td> The name of the text image unwarping model. If not set, the default model in pipeline will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td> The directory path of the  text image unwarping model. If not set, the official model will be downloaded.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>
The name of the layout detection model. If not set, the default model in pipeline will be used. </td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td> The directory path of the  layout detection model. If not set, the official model will be downloaded.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>The name of the seal text detection model. If not set, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>The directory path of the seal text detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>  
<td><code>text_recognition_model_name</code></td>  
<td>Name of the text recognition model. If not set, the default pipeline model is used.</td>  
<td><code>str</code></td>  
<td></td>  
</tr>  
<tr>  
<td><code>text_recognition_model_dir</code></td>  
<td>Directory path of the text recognition model. If not set, the official model is downloaded.</td>  
<td><code>str</code></td>  
<td></td>  
</tr>
<tr>  
<td><code>text_recognition_batch_size</code></td>  
<td>Batch size for the text recognition model. If not set, defaults to <code>1</code>.</td>  
<td><code>int</code></td>  
<td></td>  
</tr>
<tr>  
<td><code>use_doc_orientation_classify</code></td>  
<td>Whether to load and use document orientation classification module. If not set, defaults to pipeline initialization value (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td></td>  
</tr>  
<tr>  
<td><code>use_doc_unwarping</code></td>  
<td>Whether to load and use text image correction module. If not set, defaults to pipeline initialization value (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td></td>  
</tr>   
<tr>
<td><code>use_layout_detection</code></td>
<td>
Whether to load and use the layout detection module. If not set, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model. Any value between <code>0-1</code>. If not set, the default value is used, which is <code>0.5</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use Non-Maximum Suppression (NMS) as post-processing for layout detection. If not set, the parameter will default to the value initialized in the pipeline, which is set to <code>True</code> by default.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Unclip ratio for detected boxes in layout detection model. Any float > <code>0</code>. If not set, the default is <code>1.0</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The merging mode for the detection boxes output by the model in layout region detection.
<ul>
<li><b>large</b>: When set to "large", only the largest outer bounding box will be retained for overlapping bounding boxes, and the inner overlapping boxes will be removed;</li>
<li><b>small</b>: When set to "small", only the smallest inner bounding boxes will be retained for overlapping bounding boxes, and the outer overlapping boxes will be removed;</li>
<li><b>union</b>: No filtering of bounding boxes will be performed, and both inner and outer boxes will be retained;</li>
</ul>If not set, the default is <code>large</code>.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
Any integer > <code>0</code>. If not set, the default is <code>736</code>.
</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Limit type for image side in seal text detection.
Supports <code>min</code> and <code>max</code>; <code>min</code> ensures shortest side ≥ <code>det_limit_side_len</code>, <code>max</code> ensures longest side ≤ <code>limit_side_len</code>. If not set, the default is <code>min</code>.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Pixel threshold. Pixels with scores above this value in the probability map are considered text.
any float > <code>0</code>. If not set, the default is <code>0.2</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Box threshold. Boxes with average pixel scores above this value are considered text regions.
any float > <code>0</code>. If not set, the default is <code>0.6</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. Higher value means larger expansion area.
Any float > <code>0</code>. If not set, the default is <code>0.5</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Recognition score threshold. Text results above this value will be kept.
Any float > <code>0</code>. If not set, the default is <code>0.0</code> (no threshold).
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Support for specifying specific card numbers:
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference.</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference.</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference.</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference.</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference.</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference.</li>
</ul>If not set, the pipeline initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use the Paddle Inference TensorRT subgraph engine.</br>
For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.</br>
For Paddle with CUDA version 12.6, the compatible TensorRT version is 10.x (x>=5), and it is recommended to install TensorRT 10.5.0.18.
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>The computational precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN cache capacity.
</td>
<td><code>int</code></td>
<td><code>10</code></td>
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
<td></td>
</tr>
</tbody>
</table>
</details>
<br />



After running, the results will be printed to the terminal, as follows:

```bash
{'res': {'input_path': './seal_text_det.png', 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 16, 'label': 'seal', 'score': 0.975529670715332, 'coordinate': [6.191284, 0.16680908, 634.39325, 628.85345]}]}, 'seal_res_list': [{'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[320,  38],
       ...,
       [315,  38]]), array([[461, 347],
       ...,
       [456, 346]]), array([[439, 445],
       ...,
       [434, 444]]), array([[158, 468],
       ...,
       [154, 466]])], 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.2, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 0.5}, 'text_type': 'seal', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['天津君和缘商贸有限公司', '发票专用章', '吗繁物', '5263647368706'], 'rec_scores': array([0.99340463, ..., 0.9916274 ]), 'rec_polys': [array([[320,  38],
       ...,
       [315,  38]]), array([[461, 347],
       ...,
       [456, 346]]), array([[439, 445],
       ...,
       [434, 444]]), array([[158, 468],
       ...,
       [154, 466]])], 'rec_boxes': array([], dtype=float64)}]}}
```
The visualized results are saved under `save_path`, and the visualized result of seal OCR is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/04.png"/>

### 2.2 Python Script Integration

* The above command line is for quickly experiencing and viewing the effect. Generally, in a project, you often need to integrate through code. You can complete the quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddleocr import TableStructureRecognition
model = TableStructureRecognition(model_name="SLANet")
output = model.predict(input="table_recognition.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```

After running, the result is:

```
{'res': {'input_path': 'table_recognition.jpg', 'page_index': None, 'bbox': [[42, 2, 390, 2, 388, 27, 40, 26], [11, 35, 89, 35, 87, 63, 11, 63], [113, 34, 192, 34, 186, 64, 109, 64], [219, 33, 399, 33, 393, 62, 212, 62], [413, 33, 544, 33, 544, 64, 407, 64], [12, 67, 98, 68, 96, 93, 12, 93], [115, 66, 205, 66, 200, 91, 111, 91], [234, 65, 390, 65, 385, 92, 227, 92], [414, 66, 537, 67, 537, 95, 409, 95], [7, 97, 106, 97, 104, 128, 7, 128], [113, 96, 206, 95, 201, 127, 109, 127], [236, 96, 386, 96, 381, 128, 230, 128], [413, 96, 534, 95, 533, 127, 408, 127]], 'structure': ['<html>', '<body>', '<table>', '<tr>', '<td', ' colspan="4"', '>', '</td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</table>', '</body>', '</html>'], 'structure_score': 0.99948007}}
```
(1) Instantiate a pipeline object for seal text recognition using the SealRecognition() class, with specific parameter descriptions as follows:

Parameter meanings are as follows:

- `input_path`: The path of the input table image to be predicted
- `page_index`: If the input is a PDF file, indicates the page number of the PDF; otherwise, it is `None`
- `boxes`: Predicted table cell information, a list consisting of the coordinates of predicted table cells. Notably, table cell predictions for the SLANeXt series models are invalid
- `structure`: Predicted table structure HTML expressions, a list consisting of predicted HTML keywords in order
- `structure_score`: Confidence of the predicted table structure

Descriptions of related methods and parameters are as follows:

* `TableStructureRecognition` instantiates a table structure recognition model (using `SLANet` as an example). Details are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td>All model names</td>
<td>None</td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the document unwarping model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the document unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>Name of the seal text detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>Directory of the seal text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>Directory path of the text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to enable the document orientation classification module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to enable the document image unwarping module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to load and use the layout detection module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model.
<ul>
<li><b>float</b>: Any float between <code>0-1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> where the key is the class ID and the value is the threshold for that class;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>0.5</code>.</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use Non-Maximum Suppression (NMS) as post-processing for layout detection. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is set to <code>True</code> by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion ratio for the bounding boxes from the layout detection model.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: Expansion ratios in horizontal and vertical directions;</li>
<li><b>dict</b>: A dictionary with <b>int</b> keys representing <code>cls_id</code>, and <b>tuple</b> values, e.g., <code>{0: (1.1, 2.0)}</code> means width is expanded 1.1× and height 2.0× for class 0 boxes;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>1.0</code>.</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Filtering method for overlapping boxes in layout detection.
<ul>
<li><b>str</b>: Options include <code>large</code>, <code>small</code>, and <code>union</code> to retain the larger box, smaller box, or both;</li>
<li><b>dict</b>: A dictionary with <b>int</b> keys representing <code>cls_id</code>, and <b>str</b> values, e.g., <code>{0: "large", 2: "small"}</code> means using different modes for different classes;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value <code>large</code>.</li>
</ul>
</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>736</code>.</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Limit type for seal text detection image side length.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side is no less than <code>det_limit_side_len</code>, while <code>max</code> ensures the longest side is no greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>min</code>.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels with scores greater than this value in the probability map are considered text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.2</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Bounding box threshold. If the average score of all pixels inside a detection box exceeds this threshold, it is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.6</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. The larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.5</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Score threshold for seal text recognition. Text results with scores above this threshold will be retained.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.0</code> (no threshold).</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device used for inference. Supports specifying device ID:
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> means using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> means using GPU 0;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> means using NPU 0;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> means using XPU 0;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> means using MLU 0;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> means using DCU 0;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used.</li>
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
<td>Whether to use the Paddle Inference TensorRT subgraph engine.</br>
For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.</br>
For Paddle with CUDA version 12.6, the compatible TensorRT version is 10.x (x>=5), and it is recommended to install TensorRT 10.5.0.18.
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computation precision, e.g., fp32, fp16.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN cache capacity.
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>Number of threads used for inference on CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>Path to the PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</table>

* Among them, `model_name` must be specified. If `model_dir` is specified, the user's custom model is used.

* Call the `predict()` method of the table structure recognition model for inference prediction, which returns a result list. In addition, this module also provides the `predict_iter()` method. The two are completely consistent in parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator`, which can process and obtain prediction results step by step, suitable for handling large datasets or scenarios where you want to save memory. You can choose to use either method according to your actual needs. The `predict()` method has parameters `input` and `batch_size`, described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Input data to be predicted. Required. Supports multiple types:
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>;</li>
<li><b>str</b>: Local path of an image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., the network URL of an image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">Example</a>; <b>Local directory</b>, containing images to be predicted, e.g., <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with an exact file path);</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code>.</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the text image correction module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>
Whether to use the layout detection module during inference. </td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
</table>
</details>

* For processing prediction results, the prediction result of each sample is the corresponding Result object, and supports printing and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print result to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation.</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters, effective only when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save result as json format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the results. When it is a directory, the saved file name will be consistent with the input file type.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters, effective only when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the results, supports directory or file path.</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
</table>

## 4. Secondary Development

If the above models are still not ideal for your scenario, you can try the following steps for secondary development. Here, training `SLANet_plus` is used as an example, and for other models, just replace the corresponding configuration file. First, you need to prepare a dataset for table structure recognition, which can be prepared with reference to the format of the [table structure recognition demo data](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar). Once ready, you can train and export the model as follows. After exporting, you can quickly integrate the model into the above API. Here, the table structure recognition demo data is used as an example. Before training the model, please make sure you have installed the dependencies required by PaddleOCR according to the [installation documentation](../installation.en.md).


## 4.1 Dataset and Pretrained Model Preparation

### 4.1.1 Prepare Dataset

```shell
# Download sample dataset
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar
tar -xf table_rec_dataset_examples.tar
```

### 4.1.2 Download Pretrained Model

```shell
# Download SLANet_plus pretrained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams
```

### 4.2 Model Training

PaddleOCR is modularized. When training the `SLANet_plus` recognition model, you need to use the [configuration file](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/table/SLANet_plus.yml) of `SLANet_plus`.


The training commands are as follows:

```bash
# Single card training (default training method)
python3 tools/train.py -c configs/table/SLANet_plus.yml \
    -o Global.pretrained_model=./SLANet_plus_pretrained.pdparams
    Train.dataset.data_dir=./table_rec_dataset_examples \
    Train.dataset.label_file_list='[./table_rec_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./table_rec_dataset_examples \
    Eval.dataset.label_file_list='[./table_rec_dataset_examples/val.txt]'

# Multi-card training, specify card numbers via --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py \
    -c configs/table/SLANet_plus.yml \
    -o Global.pretrained_model=./SLANet_plus_pretrained.pdparams
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./table_rec_dataset_examples \
    Train.dataset.label_file_list='[./table_rec_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./table_rec_dataset_examples \
    Eval.dataset.label_file_list='[./table_rec_dataset_examples/val.txt]'
```


### 4.3 Model Evaluation

You can evaluate the trained weights, such as `output/xxx/xxx.pdparams`, using the following command:

```bash
# Note to set the path of pretrained_model to the local path. If you use the model saved by your own training, please modify the path and file name to {path/to/weights}/{model_name}.
 # Demo test set evaluation
 python3 tools/eval.py -c configs/table/SLANet_plus.yml -o \
    Global.pretrained_model=output/xxx/xxx.pdparams
    Eval.dataset.data_dir=./table_rec_dataset_examples \
    Eval.dataset.label_file_list='[./table_rec_dataset_examples/val.txt]'
```

### 4.4 Model Export

```bash
 python3 tools/export_model.py -c configs/table/SLANet_plus.yml -o \
    Global.pretrained_model=output/xxx/xxx.pdparams \
    Global.save_inference_dir="./SLANet_plus_infer/"
```

 After exporting the model, the static graph model will be stored in `./SLANet_plus_infer/` in the current directory. In this directory, you will see the following files:
 ```
 ./SLANet_plus_infer/
 ├── inference.json
 ├── inference.pdiparams
 ├── inference.yml
 ```
At this point, secondary development is complete, and this static graph model can be directly integrated into the PaddleOCR API.

## 5. FAQ
