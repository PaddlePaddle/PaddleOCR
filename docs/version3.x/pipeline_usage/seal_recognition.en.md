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

- The prediction results obtained through the `json` attribute are of dict type, with content consistent with what is saved by calling the `save_to_json()` method.
- The prediction results returned by the `img` attribute are of dict type. The keys are `layout_det_res`, `seal_res_region1`, and `preprocessed_img`, corresponding to three `Image.Image` objects: one for visualizing layout detection, one for visualizing seal text recognition results, and one for visualizing image preprocessing. If the image preprocessing sub-module is not used, `preprocessed_img` will not be included in the dictionary. If the layout region detection module is not used, `layout_det_res` will not be included.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline into your Python project, you can refer to the example code in [2.2 Python Script Method](#22-python脚本方式集成).

In addition, PaddleOCR also provides three other deployment methods, which are detailed as follows:

🚀 High-Performance Inference: In real-world production environments, many applications have stringent performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and a smooth user experience. To address this, PaddleOCR offers high-performance inference capabilities aimed at deeply optimizing the performance of model inference and pre/post-processing, thereby significantly accelerating the end-to-end process. For detailed high-performance inference procedures, please refer to [High-Performance Inference](../deployment/high_performance_inference.md).

☁️ Service Deployment: Service deployment is a common form of deployment in real-world production environments. By encapsulating inference functionality into a service, clients can access these services via network requests to obtain inference results. For detailed production service deployment procedures, please refer to [Serving](../deployment/serving.md).

Below are the API references for basic serving deployment and multi-language service invocation examples:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain the seal text recognition result.</p>
<p><code>POST /seal-recognition</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the file. By default, for PDF files exceeding 10 pages, only the content of the first 10 pages will be processed.<br />
To remove the page limit, please add the following configuration to the pipeline configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre></td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>The type of file. <code>0</code> indicates a PDF file, <code>1</code> indicates an image file. If this attribute is not present in the request body, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_layout_detection</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_side_len</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_type</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_box_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_rec_score_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>sealRecResults</code></td>
<td><code>object</code></td>
<td>The seal text recognition result. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>sealRecResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation generated by the <code>predict</code> method of the production object, where the <code>input_path</code> and the <code>page_index</code> fields are removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute of the result of the pipeline prediction. The images are in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-language Service Invocation Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/seal-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["sealRecResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost", 8080);

    const std::string filePath = "./demo.jpg";

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::string bufferStr(buffer.data(), static_cast<size_t>(size));
    std::string encodedFile = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/seal-recognition", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["sealRecResults"].is_array()) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["sealRecResults"].size(); ++i) {
            auto res = result["sealRecResults"][i];

            if (res.contains("prunedResult")) {
                std::cout << "Recognized seal result: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImages") && res["outputImages"].is_object()) {
                for (auto& [imgName, imgData] : res["outputImages"].items()) {
                    std::string outputPath = imgName + "_" + std::to_string(i) + ".jpg";
                    std::string decodedImage = base64::from_base64(imgData.get<std::string>());

                    std::ofstream outFile(outputPath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(decodedImage.c_str(), decodedImage.size());
                        outFile.close();
                        std::cout << "Saved image: " << outputPath << std::endl;
                    } else {
                        std::cerr << "Failed to write image: " << outputPath << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "Request failed." << std::endl;
        if (response) {
            std::cerr << "HTTP status: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/seal-recognition";
        String imagePath = "./demo.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String base64Image = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("file", base64Image);
        payload.put("fileType", 1);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(JSON, payload.toString());

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode root = objectMapper.readTree(responseBody);
                JsonNode result = root.get("result");

                JsonNode sealRecResults = result.get("sealRecResults");
                for (int i = 0; i < sealRecResults.size(); i++) {
                    JsonNode item = sealRecResults.get(i);
                    int finalI = i;

                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    JsonNode outputImages = item.get("outputImages");
                    if (outputImages != null && outputImages.isObject()) {
                        outputImages.fieldNames().forEachRemaining(imgName -> {
                            try {
                                String imgBase64 = outputImages.get(imgName).asText();
                                byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                                String imgPath = imgName + "_" + finalI + ".jpg";
                                try (FileOutputStream fos = new FileOutputStream(imgPath)) {
                                    fos.write(imgBytes);
                                    System.out.println("Saved image: " + imgPath);
                                }
                            } catch (IOException e) {
                                System.err.println("Failed to save image: " + e.getMessage());
                            }
                        });
                    }
                }
            } else {
                System.err.println("Request failed with HTTP code: " + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

func main() {
	API_URL := "http://localhost:8080/seal-recognition"
	filePath := "./demo.jpg"

	fileBytes, err := ioutil.ReadFile(filePath)
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}
	fileData := base64.StdEncoding.EncodeToString(fileBytes)

	payload := map[string]interface{}{
		"file":     fileData,
		"fileType": 1,
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		fmt.Printf("Error marshaling payload: %v\n", err)
		return
	}

	client := &http.Client{}
	req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		fmt.Printf("Error creating request: %v\n", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("Error sending request: %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("Unexpected status code: %d\n", resp.StatusCode)
		return
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("Error reading response body: %v\n", err)
		return
	}

	type SealResult struct {
		PrunedResult map[string]interface{}   `json:"prunedResult"`
		OutputImages map[string]string        `json:"outputImages"`
		InputImage   *string                  `json:"inputImage"`
	}

	type Response struct {
		Result struct {
			SealRecResults []SealResult  `json:"sealRecResults"`
			DataInfo       interface{}   `json:"dataInfo"`
		} `json:"result"`
	}

	var respData Response
	if err := json.Unmarshal(body, &respData); err != nil {
		fmt.Printf("Error unmarshaling response: %v\n", err)
		return
	}

	for i, res := range respData.Result.SealRecResults {
		fmt.Printf("Pruned Result %d: %+v\n", i, res.PrunedResult)

		for name, imgBase64 := range res.OutputImages {
			imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
			if err != nil {
				fmt.Printf("Error decoding image %s: %v\n", name, err)
				continue
			}

			filename := fmt.Sprintf("%s_%d.jpg", name, i)
			if err := ioutil.WriteFile(filename, imgBytes, 0644); err != nil {
				fmt.Printf("Error saving image %s: %v\n", filename, err)
				continue
			}
			fmt.Printf("Output image saved at %s\n", filename)
		}
	}
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/seal-recognition";
    static readonly string inputFilePath = "./demo.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);

        var payload = new JObject
        {
            { "file", fileData },
            { "fileType", 1 }
        };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        JArray sealRecResults = (JArray)jsonResponse["result"]["sealRecResults"];
        for (int i = 0; i < sealRecResults.Count; i++)
        {
            var res = sealRecResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            JObject outputImages = res["outputImages"] as JObject;
            if (outputImages != null)
            {
                foreach (var img in outputImages)
                {
                    string imgName = img.Key;
                    string base64Img = img.Value?.ToString();
                    if (!string.IsNullOrEmpty(base64Img))
                    {
                        string imgPath = $"{imgName}_{i}.jpg";
                        byte[] imageBytes = Convert.FromBase64String(base64Img);
                        File.WriteAllBytes(imgPath, imageBytes);
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/seal-recognition';
const imagePath = './demo.jpg';

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: 1
};

axios.post(API_URL, payload)
  .then((response) => {
    const result = response.data["result"];
    const sealRecResults = result["sealRecResults"];

    sealRecResults.forEach((res, i) => {
      console.log(`\n[${i}] prunedResult:\n`, res["prunedResult"]);

      const outputImages = res["outputImages"];
      if (outputImages) {
        for (const [imgName, base64Img] of Object.entries(outputImages)) {
          const imgBuffer = Buffer.from(base64Img, 'base64');
          const fileName = `${imgName}_${i}.jpg`;
          fs.writeFileSync(fileName, imgBuffer);
          console.log(`Output image saved at ${fileName}`);
        }
      } else {
        console.log(`[${i}] No outputImages found.`);
      }
    });
  })
  .catch((error) => {
    console.error('Error occurred while calling the API:', error.message);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/seal-recognition";
$image_path = "./demo.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["sealRecResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $base64_img) {
            if (!empty($base64_img)) {
                $output_path = "{$img_name}_{$i}.jpg";
                file_put_contents($output_path, base64_decode($base64_img));
                echo "Output image saved at $output_path\n";
            }
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>


## 4. Custom Development
If the default model weights provided by the seal text recognition pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing models using <b>your own domain-specific or application data</b> to improve the recognition performance of the seal text recognition pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the seal text recognition pipeline consists of several modules, if the pipeline's performance does not meet expectations, the issue may arise from any one of these modules. You can analyze images with poor recognition results to identify which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Fine-Tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate or missing seal position detection</td>
<td>Layout Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missing text detection</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_detection.html#4-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html#41-dataset-and-pre-trained-model-preparation">Link</a></td>
</tr>
<tr>
<td>Inaccurate full-image rotation correction</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text Image Correction Module</td>
<td>Not supported for fine-tuning</td>
</tr>
</tbody>
</table>


### 4.2 Model Application

After you complete the fine-tuning training with a private dataset, you can obtain the local model weight files. You can then use the fine-tuned model weights by specifying the local model save path through parameters or by using a custom pipeline configuration file.

#### 4.2.1 Specify Local Model Path via Parameters

When initializing the pipeline object, specify the local model path through parameters. Taking the usage of fine-tuned weights for a text detection model as an example, the demonstration is as follows:

Command line method:

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
