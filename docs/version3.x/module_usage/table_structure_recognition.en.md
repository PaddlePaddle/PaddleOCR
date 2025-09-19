---
comments: true
---

# Table Structure Recognition Module Tutorial

## 1. Overview

Table structure recognition is an important component of table recognition systems, capable of converting non-editable table images into editable table formats (such as HTML). The goal of table structure recognition is to identify the positions of rows, columns, and cells in tables. The performance of this module directly affects the accuracy and efficiency of the entire table recognition system. The table structure recognition module usually outputs HTML code for the table area, which is then passed as input to the table recognition pipeline for further processing.

## 2. Supported Model List

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>23.96 / 21.75</td>
<td>- / 43.12</td>
<td>6.9</td>
<td rowspan="1">SLANet is a table structure recognition model independently developed by Baidu PaddlePaddle Vision Team. By adopting a CPU-friendly lightweight backbone network PP-LCNet, high-low level feature fusion module CSP-PAN, and SLA Head, a feature decoding module aligning structure and position information, this model greatly improves the accuracy and inference speed of table structure recognition.</td>
</tr>
<tr>
<td>SLANet_plus</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>23.43 / 22.16</td>
<td>- / 41.80</td>
<td>6.9</td>
<td rowspan="1">SLANet_plus is an enhanced version of the table structure recognition model SLANet independently developed by the Baidu PaddlePaddle Vision Team. Compared to SLANet, SLANet_plus has greatly improved the recognition ability for wireless and complex tables, and reduced the model's sensitivity to table positioning accuracy. Even if the table positioning is offset, it can still be accurately recognized.
</td>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">85.92 / 85.92</td>
<td rowspan="2">- / 501.66</td>
<td rowspan="2">351</td>
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

<b>Note: </b>The official models would be download from HuggingFace by default. If can't access to HuggingFace, please set the environment variable `PADDLE_PDX_MODEL_SOURCE="BOS"` to change the model source to BOS. In the future, more model sources will be supported.

You can also integrate the model inference of the table structure recognition module into your own project. Before running the code below, please download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine.

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
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model_name</code></td>
<td>Model name. If set to <code>None</code>, <code>PP-LCNet_x1_0_table_cls</code> will be used.</td>
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
By default, GPU 0 is used if available; otherwise, CPU is used.</td>
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
<td>Computation precision when using the Paddle Inference TensorRT subgraph engine.<br/><b>Options:</b> <code>"fp32"</code>, <code>"fp16"</code>.</td>
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
<td>MKL-DNN cache capacity.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>Number of threads to use for inference on CPUs.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
</tbody>
</table>

* Call the `predict()` method of the table structure recognition model for inference prediction, which returns a result list. In addition, this module also provides the `predict_iter()` method. The two are completely consistent in parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator`, which can process and obtain prediction results step by step, suitable for handling large datasets or scenarios where you want to save memory. You can choose to use either method according to your actual needs. The `predict()` method has parameters `input` and `batch_size`, described as follows:

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
<td>
Data to be predicted. Required. Supports multiple input types:
<ul>
  <li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> representing image data</li>
  <li><b>str</b>: Local path to an image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL</b>: Network URL to an image or PDF file, e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">Example</a>; <b>Directory</b>: A local directory containing images for prediction, e.g., <code>/root/data/</code> (Note: PDF files in directories are not supported; to predict a PDF, specify its file path directly)</li>
  <li><b>list</b>: A list of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
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
</table>
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
<td>Whether to use <code>JSON</code> indentation formatting for the output</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify the output <code>JSON</code> data, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> keeps the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save result as json format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it's a directory, the saved file will be named the same as the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify the output <code>JSON</code> data, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> keeps the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
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
