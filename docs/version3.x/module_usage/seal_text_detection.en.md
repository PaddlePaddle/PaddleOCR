---
comments: true
---

# Seal Text Detection Module Tutorial

## I. Overview
The seal text detection module typically outputs multi-point bounding boxes around text regions, which are then passed as inputs to the distortion correction and text recognition modules for subsequent processing to identify the textual content of the seal. Recognizing seal text is an integral part of document processing and finds applications in various scenarios such as contract comparison, inventory access auditing, and invoice reimbursement verification. The seal text detection module serves as a subtask within OCR (Optical Character Recognition), responsible for locating and marking the regions containing seal text within an image. The performance of this module directly impacts the accuracy and efficiency of the entire seal text OCR system.

## II. Supported Model List


<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Hmean（%）</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109 M</td>
<td>The server-side seal text detection model of PP-OCRv4 boasts higher accuracy and is suitable for deployment on better-equipped servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6 M</td>
<td>The mobile-side seal text detection model of PP-OCRv4, on the other hand, offers greater efficiency and is suitable for deployment on end devices.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
               <li><strong>Test Dataset：</strong> A Self-built Internal Dataset, Containing 500 Images of Circular Stamps.</li>
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


## III. Quick Integration  <a id="quick"> </a>

> ❗ Before quick integration, please install the PaddleOCR wheel package. For detailed instructions, refer to [PaddleOCR Local Installation Tutorial](../installation.en.md)。

Quickly experience with just one command:

```bash
paddleocr seal_text_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png
```


You can also integrate the model inference from the layout area detection module into your project. Before running the following code, please download [Example Image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png) Go to the local area.

```python
from paddleocr import SealTextDetection
model = SealTextDetection(model_name="PP-OCRv4_server_seal_det")
output = model.predict("seal_text_det.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result is:

```bash
{'res': {'input_path': 'seal_text_det.png', 'page_index': None, 'dt_polys': [array([[463, 477],
       ...,
       [428, 505]]), array([[297, 444],
       ...,
       [230, 443]]), array([[457, 346],
       ...,
       [267, 345]]), array([[325,  38],
       ...,
       [322,  37]])], 'dt_scores': [0.9912680344777314, 0.9906849624837963, 0.9847219455533163, 0.9914791724153904]}}
```

The meanings of the parameters are as follows:
- `input_path`: represents the path of the input image to be predicted
- `dt_polys`: represents the predicted text detection boxes, where each text detection box contains multiple vertices of a polygon. Each vertex is a list of two elements, representing the x and y coordinates of the vertex respectively
- `dt_scores`: represents the confidence scores of the predicted text detection boxes

The visualization image is as follows:

<img alt="Visualization Image" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/seal_text_det/seal_text_det_res.png"/>

The explanations of related methods and parameters are as follows:

* `SealTextDetection` instantiates a text detection model (here we take `PP-OCRv4_server_seal_det` as an example), and the specific explanations are as follows:
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
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>All model names supported for seal text detection</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for model inference</td>
<td><code>str</code></td>
<td>It supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>Limit on the side length of the image for detection</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the default value is 736</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Type of side length limit for detection</td>
<td><code>str/None</code></td>
<td>
<ul>
<li><b>str</b>: Supports min and max. min ensures the shortest side of the image is not less than det_limit_side_len, max ensures the longest side is not greater than limit_side_len
<li><b>None</b>: If set to None, the default value is `min`</li></li></ul></td>


<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>In the output probability map, pixels with scores greater than this threshold will be considered as text pixels</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the default value is 0.2</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>If the average score of all pixels within a detection result box is greater than this threshold, the result will be considered as a text region</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the default value is 0.6</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>max_candidates</code></td>
<td>Maximum number of text boxes to output</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the default value is 1000</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion ratio for the Vatti clipping algorithm, used to expand the text region</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the default value is 0.5</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>use_dilation</code></td>
<td>Whether to dilate the segmentation result</td>
<td><code>bool/None</code></td>
<td>True/False/None</td>
<td>None</td>
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

* The `model_name` must be specified. After specifying `model_name`, the built-in model parameters will be used by default. On this basis, if `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the seal text detection model is called for inference prediction. The parameters of the `predict()` method include `input`, `batch_size`, `limit_side_len`, `limit_type`, `thresh`, `box_thresh`, `max_candidates`, `unclip_ratio`, and `use_dilation`. The specific descriptions are as follows:

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
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL Link</b>, such as the web URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
<li><b>Local Directory</b>, the directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer greater than 0</td>
<td>1</td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>Side length limit for detection</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Type of side length limit for detection</td>
<td><code>str/None</code></td>
<td>
<ul>
<li><b>str</b>: Supports min and max. min indicates that the shortest side of the image is not less than det_limit_side_len, max indicates that the longest side of the image is not greater than limit_side_len
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>


<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>In the output probability map, pixels with scores greater than this threshold will be considered as text pixels</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>If the average score of all pixels within the detection result box is greater than this threshold, the result will be considered as a text area</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>max_candidates</code></td>
<td>Maximum number of text boxes to be output</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion coefficient of the Vatti clipping algorithm, used to expand the text area</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>use_dilation</code></td>
<td>Whether to dilate the segmentation result</td>
<td><code>bool/None</code></td>
<td>True/False/None</td>
<td>None</td>
</tr>
</table>

* Process the prediction results. Each sample's prediction result is a corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a file in JSON format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining visual images with results and prediction results through attributes, as follows:

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
<td rowspan="1">Get the visual image in <code>dict</code> format</td>
</tr>
</table>

## IV. Custom Development

If the above model is still not performing well in your scenario, you can try the following steps for secondary development. Here, we'll use training `PP-OCRv4_server_seal_det` as an example; you can replace it with the corresponding configuration files for other models. First, you need to prepare a text detection dataset. You can refer to the format of the [seal text detection demo data](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_curve_det_dataset_examples.tar) for preparation. Once prepared, you can follow the steps below for model training and export. After export, you can quickly integrate the model into the above API. This example uses a seal text detection demo dataset. Before training the model, please ensure that you have installed the dependencies required by PaddleOCR as per the [installation documentation](../installation.en.md).

### 4.1 Dataset and Pre-trained Model Preparation

#### 4.1.1 Preparing the Dataset

```shell
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_curve_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_curve_det_dataset_examples.tar -C ./dataset/
```

#### 4.1.1 Preparing the pre-trained model


```shell
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams
```

### 4.2 Model Training

PaddleOCR has modularized the code, and when training the `PP-OCRv4_server_seal_det` model, you need to use the [configuration file](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml) for `PP-OCRv4_server_seal_det`.

The training commands are as follows:

```bash
# Single GPU training (default training method)
python3 tools/train.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml \
   -o Global.pretrained_model=./PP-OCRv4_server_seal_det_pretrained.pdparams
   
# Multi-GPU training, specify GPU ids using the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml \
        -o Global.pretrained_model=./PP-OCRv4_server_seal_det_pretrained.pdparams
```

### 4.3 Model Evaluation

You can evaluate the trained weights, such as `output/xxx/xxx.pdparams`, using the following command:

```bash
# Make sure to set the pretrained_model path to the local path. If using a model that was trained and saved by yourself, be sure to modify the path and filename to {path/to/weights}/{model_name}.
# Demo test set evaluation
python3 tools/eval.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml -o \
Global.pretrained_model=output/xxx/xxx.pdparams
```

### 4.4 Model Export

```bash
python3 tools/export_model.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml -o \
Global.pretrained_model=output/xxx/xxx.pdparams \
save_inference_dir="./PP-OCRv4_server_seal_det_infer/"
```

After exporting the model, the static graph model will be stored in the `./PP-OCRv4_server_seal_det_infer/` directory. In this directory, you will see the following files:
```
./PP-OCRv4_server_seal_det_infer/
├── inference.json
├── inference.pdiparams
├── inference.yml
```
With this, the secondary development is complete, and the static graph model can be directly integrated into PaddleOCR's API.

## 5. FAQ
