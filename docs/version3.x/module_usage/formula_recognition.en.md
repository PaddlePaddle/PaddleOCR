---
comments: true
---

# Formula Recognition Module Tutorial

## I. Overview

The formula recognition module is a key component of an OCR (Optical Character Recognition) system, responsible for converting mathematical formulas in images into editable text or computer-readable formats. The performance of this module directly affects the accuracy and efficiency of the entire OCR system. The formula recognition module typically outputs LaTeX or MathML code of the mathematical formulas, which will be passed as input to the text understanding module for further processing.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>85.91</td>
<td>43.50</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.53 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. The model is trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, significantly improving the recognition accuracy of real-world formulas.</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.00</td>
<td>45.71</td>
<td>202.25/-</td>
<td>-/-</td>
<td>224 M</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddlePaddle Vision Team. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network. Through parallel masking and model distillation techniques, it significantly improves inference speed while maintaining high recognition accuracy, making it suitable for applications requiring fast inference. The PP-FormulaNet-L version, on the other hand, uses Vary_VIT_B as its backbone network and is trained on a large-scale formula dataset, showing significant improvements in recognizing complex formulas compared to PP-FormulaNet-S.</td>
</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>695 M</td>
<tr>
<td>PP-FormulaNet_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">Training Model</a></td>
<td>88.71</td>
<td>53.32</td>
<td>191.69/-</td>
<td>-/-</td>
<td>248 M</td>
<td rowspan="3">PP-FormulaNet_plus is an enhanced version of the formula recognition model developed by the Baidu PaddlePaddle Vision Team, building upon the original PP-FormulaNet. Compared to the original version, PP-FormulaNet_plus utilizes a more diverse formula dataset during training, including sources such as Chinese dissertations, professional books, textbooks, exam papers, and mathematics journals. This expansion significantly improves the model’s recognition capabilities. Among the models, PP-FormulaNet_plus-M and PP-FormulaNet_plus-L have added support for Chinese formulas and increased the maximum number of predicted tokens for formulas from 1,024 to 2,560, greatly enhancing the recognition performance for complex formulas. Meanwhile, the PP-FormulaNet_plus-S model focuses on improving the recognition of English formulas. With these improvements, the PP-FormulaNet_plus series models perform exceptionally well in handling complex and diverse formula recognition tasks. </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">Training Model</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1301.56/-</td>
<td>-/-</td>
<td>592 M</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">Training Model</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1745.25/-</td>
<td>-/-</td>
<td>698 M</td>
</tr>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1244.61/-</td>
<td>-/-</td>
<td>99 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone network and a transformer as the decoder, significantly improving the accuracy of formula recognition.</td>
</tr>
</table>
<strong>Test Environment Description:</strong>
    <ul>
        <li><b>Performance Test Environment</b>
            <ul>
                <li><strong>Test Dataset:</strong> PaddleOCR internal custom formula recognition test set</li>
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
            <th>Acceleration Technique Combination</th>
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
            <td>Optimal combination of predefined precision type and acceleration strategy</td>
            <td>FP32 precision / 8 threads</td>
            <td>Optimal predefined backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## III. Quick Start

> ❗ Before getting started, please install the PaddleOCR wheel package. For details, refer to the [Installation Guide](../installation.en.md).

You can quickly try it out with a single command:
```bash
paddleocr formula_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png
```
You can also integrate the model inference from the formula recognition module into your own project.Before running the code below, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) locally.

```python
from paddleocr import FormulaRecognition
model = FormulaRecognition(model_name="PP-FormulaNet_plus-M")
output = model.predict(input="general_formula_rec_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```
After running, the output is:
```bash
{'res': {'input_path': '/root/.paddlex/predict_input/general_formula_rec_001.png', 'page_index': None, 'rec_formula': '\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\int_{C_{+}}d z\\frac{2z^{2}}{(z^{2}+\\omega^{2})^{\\nu+1}}\\breve{\\Psi}(\\omega;z)e^{i\\epsilon z}\\quad,'}}
```
Explanation of the result parameters:

- `input_path`： Indicates the path to the input formula image to be predicted
- `page_index`： If the input is a PDF file, this represents the page number; otherwise, it is None
- `rec_formula`：Indicates the predicted LaTeX source code of the formula image
The visualization image is as follows. The left side is the input formula image, and the right side is the rendered formula from the prediction:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/formula_recog/general_formula_rec_001_res_paddleocr3.png">

<b>Note: If you need to visualize the formula recognition module, you must install the LaTeX rendering environment by running the following command. Currently, visualization is only supported on Ubuntu. Other environments are not supported for now. For complex formulas, the LaTeX result may contain advanced representations that may not render successfully in Markdown or similar environments:</b>
```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-xetex latex-cjk-all texlive-latex-extra -y
```

Related methods and parameter descriptions are as follows:

* `FormulaRecognition` instantiates the formula recognition model (here using `PP-FormulaNet_plus-M` as an example), with detailed description as follows:
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
<td>All model names supported by PaddleX</td>
<td>None</td>
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
<td>Device used for model inference</td>
<td><code>str</code></td>
<td>Supports specifying a specific GPU card such as \"gpu:0\", other hardware card such as \"npu:0\", and CPU such as \"cpu\".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference plugin</td>
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

* Among these, `model_name` must be specified. When `model_name` is provided, the built-in model parameters from PaddleX are used by default. If `model_dir` is also specified, it will use the user-defined model instead.

* Call the `predict()` method of the formula recognition model to perform inference, which returns a result list.  
Additionally, this module provides the `predict_iter()` method. Both accept the same parameters and return the same result format.  
The difference is that `predict_iter()` returns a `generator`, which can process and retrieve results step-by-step, suitable for large datasets or memory-efficient scenarios.  
You can choose either method based on your actual needs. The `predict()` method takes parameters `input` and `batch_size`, described as follows:

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
<td>Input data to be predicted; supports multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as a URL to an image file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png">Example</a></li>
  <li><b>Local directory</b>, which should contain files to be predicted, such as <code>/root/data/</code></li>
  <li><b>List</b>, whose elements must be of the types above, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
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

* The prediction results can be processed. Each result corresponds to a `Result` object, which supports printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Details</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the <code>JSON</code> output; only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. If set to <code>True</code>, all non-ASCII characters are escaped; if <code>False</code>, original characters are kept. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a json-formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file name will match the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the <code>JSON</code> output; only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. If set to <code>True</code>, all non-ASCII characters are escaped; if <code>False</code>, original characters are kept. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file name will match the input file type</td>
<td>None</td>
</tr>
</table>

* In addition, you can also access the visualized image and prediction result via attributes, as follows:

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


## IV. Custom Development

If the models above do not perform well in your scenario, you can try the following steps for custom development.  
Here we take training `PP-FormulaNet_plus-M` as an example. For other models, just replace the corresponding config file.  First, you need to prepare a formula recognition dataset. You can follow the format of the [formula recognition demo data](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_latexocr_dataset_example.tar).  Once the data is ready, follow the steps below to train and export the model. After export, the model can be quickly integrated into the API described above.  This example uses the demo dataset. Before training the model, please ensure you have installed all PaddleOCR dependencies as described in the [installation documentation](../installation.en.md).

### 4.1 Environment Setup

To train the formula recognition model, you need to install additional Python and Linux dependencies. Run the following commands:

```shell
sudo apt-get update
sudo apt-get install libmagickwand-dev
pip install tokenizers==0.19.1 imagesize ftfy Wand
```

### 4.2 Dataset and Pretrained Model Preparation

#### 4.2.1 Prepare the Dataset
```shell
# Download the demo dataset
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_latexocr_dataset_example.tar
tar -xf ocr_rec_latexocr_dataset_example.tar
```

#### 4.2.2 Download the Pretrained Model
```shell
# Download the PP-FormulaNet_plus-M pre-trained model
wget https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_plus_m_train.tar 
tar -xf rec_ppformulanet_plus_m_train.tar
```

### 4.3 Model Training
PaddleOCR is modularized. To train the `PP-FormulaNet_plus-M`  model, you need to use its [config file](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml).

Training commands are as follows:
```bash
# Single GPU training (default)
python3 tools/train.py -c configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml \
   -o Global.pretrained_model=./rec_ppformulanet_plus_m_train/best_accuracy.pdparams

# Multi-GPU training, specify GPU IDs with --gpus
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml \
   -o Global.pretrained_model=./rec_ppformulanet_plus_m_train/best_accuracy.pdparams

```
**Note:**

- By default, evaluation is performed every 1 epoch.If you change the batch size or dataset, modify the following accordingly:
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml \
  -o Global.eval_batch_step=[0,{length_of_dataset//batch_size//4}] \
   Global.pretrained_model=./rec_ppformulanet_plus_m_train/best_accuracy.pdparams
```

### 4.4 Model Evaluation
You can evaluate trained weights, e.g., output/xxx/xxx.pdparams, or use the downloaded [model](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_plus_m_train.tar ) with the following command:

```bash
# Make sure pretrained_model is set to the local path.
# For custom-trained models, modify the path and file name as {path/to/weights}/{model_name}
# Demo test set evaluation
python3 tools/eval.py -c configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml -o \
Global.pretrained_model=./rec_ppformulanet_plus_m_train/best_accuracy.pdparams
```

### 4.5 Model Export
```bash
 python3 tools/export_model.py -c configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml -o \
 Global.pretrained_model=./rec_ppformulanet_plus_m_train/best_accuracy.pdparams \
 Global.save_inference_dir="./PP-FormulaNet_plus-M_infer/"
```

After exporting, the static graph model will be saved in `./PP-FormulaNet_plus-M_infer/`, and you will see the following files:
 ```
 ./PP-FormulaNet_plus-M_infer/
 ├── inference.json
 ├── inference.pdiparams
 ├── inference.yml
 ```
At this point, the secondary development is complete. This static graph model can be directly integrated into the PaddleOCR API.

## V. FAQ

**Q1: Which formula recognition model does PaddleOCR recommend?**

A1: It is recommended to use the PP-FormulaNet series.
If your scenario is mainly in English and inference speed is not a concern, use PP-FormulaNet-L or PP-FormulaNet_plus-L.
For mainly Chinese use cases, use PP-FormulaNet_plus-L or PP-FormulaNet_plus-M.
If your device has limited computing power and you are working with English formulas, use PP-FormulaNet-S.

**Q2: Why does the inference report an error?**
A2: The formula recognition model depends heavily on Paddle 3.0 official release.
Please ensure the correct version is installed.

**Q3: Why is there no visualization image after prediction?**
A3: This may be due to LaTeX not being installed.You need to refer to Section III and install the LaTeX rendering tools.
