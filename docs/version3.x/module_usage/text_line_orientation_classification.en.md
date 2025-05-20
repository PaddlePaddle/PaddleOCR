---
comments: true
---

# Text Line Orientation Classification Module Tutorial

## 1. Overview  
The text line orientation classification module identifies the orientation of text lines and corrects them through post-processing. During processes like document scanning or ID photo capture, users may rotate the shooting device for better clarity, resulting in text lines with varying orientations. Standard OCR workflows often struggle with such data. By employing image classification technology, this module pre-determines text line orientation and adjusts it, thereby enhancing OCR accuracy.

## 2. Supported Models  

<table>
<thead>
<tr>
<th>Model</th>
<th>Download Links</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>A text line classification model based on PP-LCNet_x0_25, with two classes: 0° and 180°.</td>
</tr>
</tbody>
</table>

<strong>Testing Environment:</strong>

  <ul>
      <li><b>Performance Testing Environment</b>
          <ul>
              <li><strong>Test Dataset:</strong> PaddleX's proprietary dataset, covering scenarios like IDs and documents, with 1,000 images.</li>
              <li><strong>Hardware:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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
            <td>Optimal combination of precision and acceleration strategies</td>
            <td>FP32 precision / 8 threads</td>
            <td>Optimal backend selection (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## 3. Quick Start  

> ❗ Before starting, ensure you have installed the PaddleOCR wheel package. Refer to the [Installation Guide](../installation.en.md) for details.  

Run the following command for a quick demo:  

```bash
paddleocr text_line_orientation_classification -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg
```  

Alternatively, integrate the module into your project. Download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg) locally before running the code below.  

```python
from paddleocr import TextLineOrientationClassification
model = TextLineOrientationClassification(model_name="PP-LCNet_x0_25_textline_ori")
output = model.predict("textline_rot180_demo.jpg",  batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```  

The output will be:  

```bash
{'res': {'input_path': 'textline_rot180_demo.jpg', 'page_index': None, 'class_ids': array([1], dtype=int32), 'scores': array([1.], dtype=float32), 'label_names': ['180_degree']}}
```  

Key output fields:  
- `input_path`: Path of the input image.  
- `page_index`: For PDF inputs, indicates the page number; otherwise, `None`.  
- `class_ids`: Predicted class IDs (0° or 180°).  
- `scores`: Confidence scores.  
- `label_names`: Predicted class labels.  

Visualization:  

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/textline_ori_classification/textline_rot180_demo_res.jpg">  

### Method and Parameter Details  

* **`TextLineOrientationClassification` Initialization** (using `PP-LCNet_x0_25_textline_ori` as an example):  

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
<td>N/A</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Custom model path</td>
<td><code>str</code></td>
<td>N/A</td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device</td>
<td><code>str</code></td>
<td>E.g., "gpu:0", "npu:0", "cpu"</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Enable high-performance inference</td>
<td><code>bool</code></td>
<td>N/A</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>HPI configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>N/A</td>
<td><code>None</code></td>
</tr>
</table>

* **`predict()` Method**:  
  - `input`: Supports various input types (numpy array, file path, URL, directory, or list).  
  - `batch_size`: Batch size (default: 1).  

* **Result Handling**:  
  Each prediction result is a `Result` object with methods like `print()`, `save_to_img()`, and `save_to_json()`.  

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameters</th>
<th>Type</th>
<th>Details</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>print()</code></td>
<td>Print results</td>
<td><code>format_json</code>, <code>indent</code>, <code>ensure_ascii</code></td>
<td><code>bool</code>, <code>int</code>, <code>bool</code></td>
<td>Control JSON formatting and ASCII escaping</td>
<td><code>True</code>, 4, <code>False</code></td>
</tr>
<tr>
<td><code>save_to_json()</code></td>
<td>Save results as JSON</td>
<td><code>save_path</code>, <code>indent</code>, <code>ensure_ascii</code></td>
<td><code>str</code>, <code>int</code>, <code>bool</code></td>
<td>Same as <code>print()</code></td>
<td>N/A, 4, <code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save visualized results</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Output path</td>
<td>N/A</td>
</tr>
</table>

* **Attributes**:  
  - `json`: Get results in JSON format.  
  - `img`: Get visualized images as a dictionary.  

## 4. Custom Development  

Since PaddleOCR does not natively support training for text line orientation classification, refer to [PaddleX's Custom Development Guide](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#iv-custom-development) for training. Trained models can seamlessly integrate into PaddleOCR's API for inference.
