---
comments: true
---

# Text Detection Module Usage Guide

## 1. Overview
The text detection module is a critical component of OCR (Optical Character Recognition) systems, responsible for locating and marking text-containing regions in images. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The text detection module typically outputs bounding boxes for text regions, which are then passed to the text recognition module for further processing.

## 2. Supported Models List

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Training Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>371.65 / 371.65</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>63.8</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>

<strong>Testing Environment:</strong>

  <ul>
      <li><b>Performance Testing Environment</b>
          <ul>
              <li><strong>Test Dataset:</strong> PaddleOCR3.0 newly constructed multilingual dataset (including Chinese, Traditional Chinese, English, Japanese), covering street scenes, web images, documents, handwriting, blur, rotation, distortion, etc., totaling 2677 images.</li>
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
            <th>Acceleration Techniques</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Standard Mode</td>
            <td>FP32 precision / No TRT acceleration</td>
            <td>FP32 precision / 8 threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of precision types and acceleration strategies</td>
            <td>FP32 precision / 8 threads</td>
            <td>Optimal backend selection (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## 3. Quick Start

> ❗ Before starting, please install the PaddleOCR wheel package. Refer to the [Installation Guide](../installation.en.md) for details.

Use the following command for a quick experience:

```bash
paddleocr text_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png
```

You can also integrate the model inference into your project. Before running the following code, download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png) locally.

```python
from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_server_det")
output = model.predict("general_ocr_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

The output will be:

```bash
{'res': {'input_path': 'general_ocr_001.png', 'page_index': None, 'dt_polys': array([[[ 75, 549],
        ...,
        [ 77, 586]],

       ...,

       [[ 31, 406],
        ...,
        [ 34, 455]]], dtype=int16), 'dt_scores': [0.873949039891189, 0.8948166013613552, 0.8842595305917041, 0.876953790920377]}}
```

Output parameter meanings:
- `input_path`: Path of the input image.
- `page_index`: If the input is a PDF, this indicates the current page number; otherwise, it is `None`.
- `dt_polys`: Predicted text detection boxes, where each box contains four vertices (x, y coordinates).
- `dt_scores`: Confidence scores of the predicted text detection boxes.

Visualization example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/text_det/general_ocr_001_res.png"/>

Method and parameter descriptions:

* Instantiate the text detection model (e.g., `PP-OCRv5_server_det`):
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
<td><code>model_name</code></td>
<td>Model name</td>
<td><code>str</code></td>
<td>All PaddleX-supported text detection model names</td>
<td>Required</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path</td>
<td><code>str</code></td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device</td>
<td><code>str</code></td>
<td>GPU (e.g., "gpu:0"), NPU (e.g., "npu:0"), CPU ("cpu")</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>Image side length limit for detection</td>
<td><code>int/None</code></td>
<td>Positive integer or None (uses default model config)</td>
<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Side length restriction type</td>
<td><code>str/None</code></td>
<td>"min" (shortest side ≥ limit) or "max" (longest side ≤ limit)</td>
<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>Pixel score threshold for text detection</td>
<td><code>float/None</code></td>
<td>Positive float or None (uses default model config)</td>
<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>Average score threshold for text regions</td>
<td><code>float/None</code></td>
<td>Positive float or None (uses default model config)</td>
<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion coefficient for Vatti clipping algorithm</td>
<td><code>float/None</code></td>
<td>Positive float or None (uses default model config)</td>
<td>None</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Enable high-performance inference plugin</td>
<td><code>bool</code></td>
<td>N/A</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>N/A</td>
<td><code>None</code></td>
</tr>
</table>

* The `predict()` method parameters:
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
<td><code>input</code></td>
<td>Input data (image path, URL, directory, or list)</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>Numpy array, file path, URL, directory, or list of these</td>
<td>Required</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Positive integer</td>
<td>1</td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>Image side length limit for detection</td>
<td><code>int/None</code></td>
<td>Positive integer or None (uses model default)</td>
<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Side length restriction type</td>
<td><code>str/None</code></td>
<td>"min" or "max"</td>
<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>Pixel score threshold for text detection</td>
<td><code>float/None</code></td>
<td>Positive float or None (uses model default)</td>
<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>Average score threshold for text regions</td>
<td><code>float/None</code></td>
<td>Positive float or None (uses model default)</td>
<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion coefficient for Vatti clipping algorithm</td>
<td><code>float/None</code></td>
<td>Positive float or None (uses model default)</td>
<td>None</td>
</tr>
</table>

* Result processing methods:
<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameters</th>
<th>Type</th>
<th>Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Format output as JSON</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>JSON indentation level</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Escape non-ASCII characters</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Output file path</td>
<td>Required</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>JSON indentation level</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Escape non-ASCII characters</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as image</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Output file path</td>
<td>Required</td>
</tr>
</table>

* Additional attributes:
<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td><code>json</code></td>
<td>Get prediction results in JSON format</td>
</tr>
<tr>
<td><code>img</code></td>
<td>Get visualization image as a dictionary</td>
</tr>
</table>

## 4. Custom Development

If the above models do not meet your requirements, follow these steps for custom development (using `PP-OCRv5_server_det` as an example). First, prepare a text detection dataset (refer to the [Demo Dataset](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar) format). After preparation, proceed with model training and export. The exported model can be integrated into the API. Ensure PaddleOCR dependencies are installed as per the [Installation Guide](../installation.en.md).

### 4.1 Dataset and Pretrained Model Preparation

#### 4.1.1 Prepare Dataset

```shell
# Download example dataset
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar
tar -xf ocr_det_dataset_examples.tar
```

#### 4.1.2 Download Pretrained Model

```shell
# Download PP-OCRv5_server_det pretrained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams 
```

### 4.2 Model Training

PaddleOCR modularizes the code. To train the `PP-OCRv5_server_det` model, use its [configuration file](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml).

Training command:

```bash
# Single-GPU training (default)
python3 tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml \
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./ocr_det_dataset_examples \
    Train.dataset.label_file_list='[./ocr_det_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./ocr_det_dataset_examples \
    Eval.dataset.label_file_list='[./ocr_det_dataset_examples/val.txt]'

# Multi-GPU training (specify GPUs with --gpus)
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py \
    -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml \
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./ocr_det_dataset_examples \
    Train.dataset.label_file_list='[./ocr_det_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./ocr_det_dataset_examples \
    Eval.dataset.label_file_list='[./ocr_det_dataset_examples/val.txt]'
```

### 4.3 Model Evaluation

You can evaluate trained weights (e.g., `output/PP-OCRv5_server_det/best_accuracy.pdparams`) using the following command:

```bash
# Note: Set pretrained_model to local path. For custom-trained models, modify the path and filename as {path/to/weights}/{model_name}.
# Demo dataset evaluation
python3 tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml \
    -o Global.pretrained_model=output/PP-OCRv5_server_det/best_accuracy.pdparams \
    Eval.dataset.data_dir=./ocr_det_dataset_examples \
    Eval.dataset.label_file_list='[./ocr_det_dataset_examples/val.txt]' 
```

### 4.4 Model Export

```bash
python3 tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml -o \
    Global.pretrained_model=output/PP-OCRv5_server_det/best_accuracy.pdparams \
    Global.save_inference_dir="./PP-OCRv5_server_det_infer/"
```

After export, the static graph model will be saved in `./PP-OCRv5_server_det_infer/` with the following files:
```
./PP-OCRv5_server_det_infer/
├── inference.json
├── inference.pdiparams
├── inference.yml
```
The custom development is now complete. This static graph model can be directly integrated into PaddleOCR's API.

## 5. FAQ

- Use parameters `limit_type` and `limit_side_len` to constrain image dimensions.  
  - `limit_type` options: [`max`, `min`]  
  - `limit_side_len`: Positive integer (typically multiples of 32, e.g., 960).  
  - For lower-resolution images, use `limit_type=min` and `limit_side_len=960` to balance computational efficiency and detection quality.  
  - For higher-resolution images requiring larger detection scales, set `limit_side_len` to desired values (e.g., 1216).
