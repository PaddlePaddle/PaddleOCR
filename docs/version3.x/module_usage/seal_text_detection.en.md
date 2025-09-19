---
comments: true
---

# Seal Text Detection Module Tutorial

## I. Overview
The seal text detection module typically outputs multi-point bounding boxes around text regions, which are then passed as inputs to the distortion correction and text recognition modules for subsequent processing to identify the textual content of the seal. Recognizing seal text is an integral part of document processing and finds applications in various scenarios such as contract comparison, inventory access auditing, and invoice reimbursement verification. The seal text detection module serves as a subtask within OCR (Optical Character Recognition), responsible for locating and marking the regions containing seal text within an image. The performance of this module directly impacts the accuracy and efficiency of the entire seal text OCR system.

## II. Supported Model List

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Hmean（%）</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.40</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td>The server-side seal text detection model of PP-OCRv4 boasts higher accuracy and is suitable for deployment on better-equipped servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.6</td>
<td>The mobile-side seal text detection model of PP-OCRv4, on the other hand, offers greater efficiency and is suitable for deployment on end devices.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
               <li><strong>Test Dataset：</strong> A Self-built Internal Dataset, Containing 500 Images of Circular Stamps.</li>
              <li><strong>Hardware Configuration:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>Software Environment:</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle 3.0.0 / paddleocr 3.0.3</li>
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

<b>Note: </b>The official models would be download from HuggingFace by default. If can't access to HuggingFace, please set the environment variable `PADDLE_PDX_MODEL_SOURCE="BOS"` to change the model source to BOS. In the future, more model sources will be supported.

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
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model_name</code></td>
<td>Model name. If set to <code>None</code>, <code>PP-OCRv4_mobile_seal_det</code> will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device for inference.<br/>
<b>For example:</b> <code>"cpu"</code>, <code>"gpu"</code>, <code>"npu"</code>, <code>"gpu:0"</code>, <code>"gpu:0,1"</code>.<br/>
If multiple devices are specified, parallel inference will be performed.<br/>
By default, GPU 0 is used if available; otherwise, CPU is used.
</td>
<td><code>str|None</code></td>
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
<td>Whether to use the Paddle Inference TensorRT subgraph engine. If the model does not support acceleration through TensorRT, setting this flag will not enable acceleration.<br/>
For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computation precision when using the TensorRT subgraph engine in Paddle Inference.<br/><b>Options:</b> <code>"fp32"</code>, <code>"fp16"</code>.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
</td>
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
<td>Number of threads to use for inference on CPUs.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>Limit on the side length of the input image for detection. <code>int</code> specifies the value. If set to <code>None</code>, the model's default configuration will be used.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Type of image side length limitation. <code>"min"</code> ensures the shortest side of the image is no less than <code>det_limit_side_len</code>; <code>"max"</code> ensures the longest side is no greater than <code>limit_side_len</code>. If set to <code>None</code>, the model's default configuration will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>Pixel score threshold. Pixels in the output probability map with scores greater than this threshold are considered text pixels. Accepts any float value greater than 0. If set to <code>None</code>, the model's default configuration will be used.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>If the average score of all pixels inside the bounding box is greater than this threshold, the result is considered a text region. Accepts any float value greater than 0. If set to <code>None</code>, the model's default configuration will be used.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion ratio for the Vatti clipping algorithm, used to expand the text region. Accepts any float value greater than 0. If set to <code>None</code>, the model's default configuration will be used.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>input_shape</code></td>
<td>Input image size for the model in the format <code>(C, H, W)</code>. If set to <code>None</code>, the model's default size will be used.</td>
<td><code>tuple|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* The `predict()` method of the seal text detection model is called for inference prediction. The parameters of the `predict()` method include `input`, `batch_size`, `limit_side_len`, `limit_type`, `thresh`, `box_thresh`, `max_candidates`, `unclip_ratio`. The specific descriptions are as follows:
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
<td>Input data to be predicted. Required. Supports multiple input types:<ul>
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: 
  - Local image or PDF file path: <code>/root/data/img.jpg</code>;
  - <b>URL</b> of image or PDF file: e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">example</a>;
  - <b>Local directory</b>: directory containing images for prediction, e.g., <code>/root/data/</code> (Note: directories containing PDF files are not supported; PDFs must be specified by exact file path)</li>
<li><b>list</b>: Elements must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size, can be set to any positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
<tr>
<td><code>limit_side_len</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
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
   -o Global.pretrained_model=./PP-OCRv4_server_seal_det_pretrained.pdparams \
   Train.dataset.data_dir=./dataset/ocr_curve_det_dataset_examples Train.dataset.label_file_list=./dataset/ocr_curve_det_dataset_examples/train.txt \
   Eval.dataset.data_dir=./dataset/ocr_curve_det_dataset_examples Eval.dataset.label_file_list=./dataset/ocr_curve_det_dataset_examples/val.txt
   
# Multi-GPU training, specify GPU ids using the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/det/PP-OCRv4/PP-OCRv4_server_seal_det.yml \
   -o Global.pretrained_model=./PP-OCRv4_server_seal_det_pretrained.pdparams \
   Train.dataset.data_dir=./dataset/ocr_curve_det_dataset_examples Train.dataset.label_file_list=./dataset/ocr_curve_det_dataset_examples/train.txt \
   Eval.dataset.data_dir=./dataset/ocr_curve_det_dataset_examples Eval.dataset.label_file_list=./dataset/ocr_curve_det_dataset_examples/val.txt
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
    Global.save_inference_dir="./PP-OCRv4_server_seal_det_infer/"
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
