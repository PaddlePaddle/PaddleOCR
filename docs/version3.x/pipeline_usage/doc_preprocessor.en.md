---
comments: true
---

# Document Image Preprocessing Pipeline Tutorial

## 1. Introduction to Document Image Preprocessing Pipeline

The Document Image Preprocessing Pipeline integrates two key functions: document orientation classification and geometric distortion correction. The document orientation classification module automatically identifies the four possible orientations of a document (0°, 90°, 180°, 270°), ensuring that the document is processed in the correct direction. The text image unwarping model is designed to correct geometric distortions that occur during document photography or scanning, restoring the document's original shape and proportions. This pipeline is suitable for digital document management, preprocessing tasks for OCR, and any scenario requiring improved document image quality. By automating orientation correction and geometric distortion correction, this module significantly enhances the accuracy and efficiency of document processing, providing a more reliable foundation for image analysis. The pipeline also offers flexible service-oriented deployment options, supporting calls from various programming languages on multiple hardware platforms. Additionally, the pipeline supports secondary development, allowing you to fine-tune the models on your own datasets and seamlessly integrate the trained models.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">

<b>The General Document Image Preprocessing Pipeline includes the following two modules. Each module can be trained and inferred independently and contains multiple models. For detailed information, please click on the corresponding module to view the documentation.</b>

- [Document Image Orientation Classification Module](../module_usage/doc_img_orientation_classification.md) (Optional)
- [Text Image Unwarping Module](../module_usage/text_image_unwarping.md) (Optional)

In this pipeline, you can select the models to use based on the benchmark data provided below.

<details>
<summary> <b>Document Image Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Links</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, which includes four categories: 0°, 90°, 180°, and 270°.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>Text Image Unwarping Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Links</th>
<th>CER </th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>30.3 MB</td>
<td>A high-precision text image unwarping model.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>Test Environment Description:</b></summary>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                      <li><strong>Test Datasets:
             </strong>
                <ul>
                  <li>Document Image Orientation Classification Model: A self-built dataset by PaddleX, covering various scenarios including ID cards and documents, containing 1000 images.</li>
                  <li>Text Image Unwarping Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet.</a></li>
                </ul>
             </li>
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
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of precision type and acceleration strategy selected in advance</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Optimal backend (Paddle/OpenVINO/TRT, etc.) selected in advance</td>
        </tr>
    </tbody>
</table>
</details>

## 2. Quick Start

Before using the General Document Image Preprocessing Pipeline locally, ensure that you have completed the wheel package installation according to the [Installation Guide](../installation.en.md). After installation, you can experience it via the command line or integrate it into Python locally.

### 2.1 Command Line Experience

You can quickly experience the `doc_preprocessor` pipeline with a single command:

```bash
paddleocr doc_preprocessor -i https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg

# Specify whether to use the document orientation classification model via --use_doc_orientation_classify
paddleocr doc_preprocessor -i ./doc_test_rotated.jpg --use_doc_orientation_classify True

# Specify whether to use the text image unwarping module via --use_doc_unwarping
paddleocr doc_preprocessor -i ./doc_test_rotated.jpg --use_doc_unwarping True

# Specify the use of GPU for model inference via --device
paddleocr doc_preprocessor -i ./doc_test_rotated.jpg --device gpu
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
<td>The data to be predicted, supporting multiple input types. This parameter is required.
<ul>
<li><b>Python Var</b>: For example, image data represented as <code>numpy.ndarray</code>.</li>
<li><b>str</b>: For example, the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>or a URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">example</a>; <b>or a local directory</b>, which should contain the images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files need to be specified to a specific file path).</li>
<li><b>List</b>: The list elements should be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Specify the path to save the inference result file. If set to <code>None</code>, the inference result will not be saved locally.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>The name of the document orientation classification model. If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>The name of the text image unwarping model. If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>The directory path of the text image unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load the document orientation classification module. If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized as <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load the text image unwarping module. If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized as <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Support for specifying specific card numbers.
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference.</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference.</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference.</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference.</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference.</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference.</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default. During initialization, the local GPU 0 device will be prioritized; if not available, the CPU device will be used.</li>
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
<td>Whether to use TensorRT for inference acceleration.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>The minimum subgraph size, used to optimize the computation of model subgraphs.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>The computational precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
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
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />

The running results will be printed to the terminal. The running results of the `doc_preprocessor` pipeline with default configuration are as follows:
```bash
{'res': {'input_path': '/root/.paddlex/predict_input/doc_test_rotated.jpg', 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
```

The visualization results are saved under the `save_path`. The visualization results are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg"/>


### 2.2 Integration via Python Script

The command-line approach is for quick experience and viewing results. Generally, in projects, integration through code is often required. You can achieve rapid inference in production lines with just a few lines of code. The inference code is as follows:

```python
from paddleocr import DocPreprocessor

pipeline = DocPreprocessor()
# docpp = DocPreprocessor(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model via use_doc_orientation_classify
# docpp = DocPreprocessor(use_doc_unwarping=True) # Specify whether to use the text image unwarping module via use_doc_unwarping
# docpp = DocPreprocessor(device="gpu") # Specify whether to use GPU for model inference via device
output = pipeline.predict("./doc_test_rotated.jpg")
for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps are executed:

(1) Instantiate the `doc_preprocessor` pipeline object via `DocPreprocessor()`. The specific parameter descriptions are as follows:

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
<td><code>doc_orientation_classify_model_name</code></td>
<td>The name of the document orientation classification model. If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>The name of the text image unwarping model. If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>The directory path of the text image unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load the document orientation classification module. If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized as <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load the text image unwarping module. If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized as <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Support for specifying specific card numbers.
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference.</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference.</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference.</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference.</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference.</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference.</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default. During initialization, the local GPU 0 device will be prioritized; if not available, the CPU device will be used.</li>
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
<td>Whether to use TensorRT for inference acceleration.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>The minimum subgraph size, used to optimize the computation of model subgraphs.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>The computational precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
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
<td><code>None</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the `doc_preprocessor` pipeline object for inference prediction. This method will return a list of results.

In addition, the pipeline also provides the `predict_iter()` method. The two methods are completely consistent in terms of parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator`, which can process and obtain prediction results step by step, suitable for scenarios with large datasets or where memory savings are desired. You can choose either of the two methods according to your actual needs.

The following are the parameters and their descriptions of the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>The data to be predicted, supporting multiple input types. This parameter is required.
<ul>
<li><b>Python Var</b>: For example, image data represented as <code>numpy.ndarray</code>.</li>
<li><b>str</b>: For example, the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>or a URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">example</a>; <b>or a local directory</b>, which should contain the images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files need to be specified to a specific file path).</li>
<li><b>List</b>: The list elements should be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Same as the parameter during instantiation.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the text image unwarping module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>

</table>

(3) Process the prediction results. The prediction result for each sample is a corresponding Result object, which supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Description</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file type name.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. Supports directory or file paths.</td>
<td>None</td>
</tr>
</table>Here's the continuation of the translation:

- Calling the `print()` method will output the results to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters configured for the production line

        - `use_doc_orientation_classify`: `(bool)` Controls whether to enable the document orientation classification module
        - `use_doc_unwarping`: `(bool)` Controls whether to enable the text image rectification module

    - `angle`: `(int)` The prediction result of the document orientation classification. When enabled, the value is one of [0, 90, 180, 270]; when disabled, it is -1

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list form.

- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_doc_preprocessor_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (Production lines usually contain many result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten, and only the last image will be retained)

* In addition, it also supports obtaining visualization images and prediction results with results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Obtain the prediction result in JSON format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Obtain visualization images in dictionary format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is data of type dict, and the content is consistent with that saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary-type data. The key is `preprocessed_img`, and the corresponding value is an `Image.Image` object: a visualization image for displaying the doc_preprocessor result.

## 3. Development Integration/Deployment

If the production line meets your requirements for inference speed and accuracy, you can proceed directly to development integration/deployment.

If you need to apply the production line directly to your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python脚本方式集成).

In addition, PaddleOCR also provides two other deployment methods, which are detailed as follows:

🚀 High-performance inference: In actual production environments, many applications have strict performance requirements (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleOCR provides high-performance inference functionality, aiming to deeply optimize model inference and pre/post-processing to achieve significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [High-Performance Inference Guide](../deployment/high_performance_inference.md).

☁️ Service-oriented deployment: Service-oriented deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. For detailed production line service-oriented deployment procedures, please refer to the [Service-Oriented Deployment Guide](../deployment/serving.md).

Below are the API references for basic service-oriented deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>
<p>Main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the properties of the response body are as follows:</li>
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
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed to <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the properties of the response body are as follows:</li>
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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>Main operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain the preprocessing result of the image document image.</p>
<p><code>POST /document-preprocessing</code></p>
<ul>
<li>Properties of the request body:</li>
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
<td>The URL of an image file or PDF file accessible to the server, or the Base64 encoding result of the content of the above types of files. By default, for PDF files with more than 10 pages, only the first 10 pages will be processed.<br /> To remove the page limit, please add the following configuration to the production line configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre>
</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>File type. <code>0</code> indicates a PDF file, and <code>1</code> indicates an image file. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter in the <code>predict</code> method of the production line object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter in the <code>predict</code> method of the production line object.</td>
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
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>docPreprocessingResults</code></td>
<td><code>object</code></td>
<td>Document image preprocessing results. The array length is 1 (for image input) or the actual number of processed document pages (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Input data information.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>docPreprocessingResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>outputImage</code></td>
<td><code>string</code></td>
<td>The preprocessed image. The image is in PNG format and uses Base64 encoding.</td>
</tr>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the <code>predict</code> method of the production line object, with the <code>input_path</code> and <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>docPreprocessingImage</code></td>
<td><code>string</code> ｜ <code>null</code></td>
<td>Visualization result image. The image is in JPEG format and uses Base64 encoding.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> ｜ <code>null</code></td>
<td>Input image. The image is in JPEG format and uses Base64 encoding.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Call Examples</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/document-preprocessing"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["docPreprocessingResults"]):
    print(res["prunedResult"])
    output_img_path = f"out_{i}.png"
    with open(output_img_path, "wb") as f:
        f.write(base64.b64decode(res["outputImage"]))
    print(f"Output image saved at {output_img_path}")
</code></pre></details>
</details>
<br/>

## 4. Secondary Development

If the default model weights provided by the document image preprocessing pipeline do not meet your accuracy or speed requirements in your specific scenario, you can attempt to further **fine-tune** the existing model using **your own domain-specific or application-specific data** to enhance the recognition performance of the document image preprocessing pipeline in your context.

### 4.1 Model Fine-Tuning

Since the document image preprocessing pipeline comprises multiple modules, any module could potentially contribute to suboptimal performance if the overall pipeline does not meet expectations. You can analyze images with poor recognition results to identify which module is causing the issue and then refer to the corresponding fine-tuning tutorial links in the table below to perform model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Module to Fine-Tune</th>
<th>Fine-Tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate rotation correction of the entire image</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate distortion correction of the image</td>
<td>Text Image Rectification Module</td>
<td>Fine-tuning is currently not supported</td>
</tr>
</tbody>
</table>
