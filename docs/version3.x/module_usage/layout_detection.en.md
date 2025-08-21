---
comments: true
---

# Layout Detection Module Tutorial

## I. Overview
The core task of structure analysis is to parse and segment the content of input document images. By identifying different elements in the image (such as text, charts, images, etc.), they are classified into predefined categories (e.g., pure text area, title area, table area, image area, list area, etc.), and the position and size of these regions in the document are determined.

## II. Supported Model List

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

* <b>The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01</td>
<td>A higher-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L</td>
</tr>
<tr>
</tbody>
</table>


* <b>The layout detection model includes 1 category: Block:</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocBlockLayout</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams">Training Model</a></td>
<td>95.9</td>
<td>34.60 / 28.54</td>
<td>506.43 / 256.83</td>
<td>123.92</td>
<td>A layout block localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L</td>
</tr>
<tr>
</tbody>
</table>


* <b>The layout detection model includes 23 common categories: document title, paragraph title, text, page number, abstract, table of contents, references, footnotes, header, footer, algorithm, formula, formula number, image, figure caption, table, table caption, seal, figure title, figure, header image, footer image, and sidebar text</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-S.</td>
</tr>
</tbody>
</table>

> ❗ The above list includes the <b>4 core models</b> that are key supported by the text recognition module. The module actually supports a total of <b>12 full models</b>, including several predefined models with different categories. The complete model list is as follows:

<details><summary> 👉 Details of Model List</summary>

* <b>Table Layout Detection Model</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>9.57 / 6.63</td>
<td>27.66 / 16.75</td>
<td>7.4</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset using PicoDet-1x, capable of detecting table regions.</td>
</tr>
</tbody></table>

* <b>3-Class Layout Detection Model, including Table, Image, and Stamp</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.43 / 3.44</td>
<td>17.60 / 6.51</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody></table>

* <b>5-Class English Document Area Detection Model, including Text, Title, Table, Image, and List</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.62 / 6.75</td>
<td>26.96 / 12.77</td>
<td>7.4</td>
<td>A high-efficiency English document layout area localization model trained on the PubLayNet dataset using PicoDet-1x.</td>
</tr>
</tbody></table>

* <b>17-Class Area Detection Model, including 17 common layout categories: Paragraph Title, Image, Text, Number, Abstract, Content, Figure Caption, Formula, Table, Table Caption, References, Document Title, Footnote, Header, Algorithm, Footer, and Stamp</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>8.80 / 3.62</td>
<td>17.51 / 6.35</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
             <li><strong>Test Dataset：</strong>
                 <ul>
                   <li>20 types of layout detection models: PaddleOCR's self built layout area detection dataset, including Chinese and English papers, magazines, newspapers, research papers PPT、 1300 images of document types such as test papers and textbooks. </li>
                   <li>Type 1 version face region detection model: PaddleOCR's self built version face region detection dataset, including Chinese and English papers, magazines, newspapers, research reports PPT、 1000 document type images such as test papers and textbooks. </li>      
                   <li>23 categories Layout Detection Model: A self-built layout area detection dataset by PaddleOCR, containing 500 common document type images such as Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</li>
                   <li>Table Layout Detection Model: A self-built table area detection dataset by PaddleOCR, including 7,835 Chinese and English paper document type images with tables.</li>
                   <li> 3-Class Layout Detection Model: A self-built layout area detection dataset by PaddleOCR, comprising 1,154 common document type images such as Chinese and English papers, magazines, and research reports.</li>
                   <li>5-Class English Document Area Detection Model: The evaluation dataset of <a href="https://developer.ibm.com/exchanges/data/all/publaynet">PubLayNet</a>, containing 11,245 images of English documents.</li>
                   <li>17-Class Area Detection Model: A self-built layout area detection dataset by PaddleOCR, including 892 common document type images such as Chinese and English papers, magazines, and research reports.</li>
                 </ul>
             </li>
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

</details>


## III. Quick Integration  <a id="quick"> </a>

> ❗ Before quick integration, please install the PaddleOCR wheel package. For detailed instructions, refer to [PaddleOCR Local Installation Tutorial](../installation.en.md)。

Quickly experience with just one command:

```bash
paddleocr layout_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg
```

<b>Note: </b>The official models would be download from HuggingFace by default. If can't access to HuggingFace, please set the environment variable `PADDLE_PDX_MODEL_SOURCE="BOS"` to change the model source to BOS. In the future, more model sources will be supported.

You can also integrate the model inference from the layout area detection module into your project. Before running the following code, please download [Example Image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg) Go to the local area.

```python
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': 'layout.jpg', 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9870226979255676, 'coordinate': [34.101906, 349.85275, 358.59213, 611.0772]}, {'cls_id': 2, 'label': 'text', 'score': 0.9866003394126892, 'coordinate': [34.500324, 647.1585, 358.29367, 848.66797]}, {'cls_id': 2, 'label': 'text', 'score': 0.9846674203872681, 'coordinate': [385.71445, 497.40973, 711.2261, 697.84265]}, {'cls_id': 8, 'label': 'table', 'score': 0.984126091003418, 'coordinate': [73.76879, 105.94899, 321.95303, 298.84888]}, {'cls_id': 8, 'label': 'table', 'score': 0.9834211468696594, 'coordinate': [436.95642, 105.81531, 662.7168, 313.48462]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832247495651245, 'coordinate': [385.62787, 346.2288, 710.10095, 458.77127]}, {'cls_id': 2, 'label': 'text', 'score': 0.9816061854362488, 'coordinate': [385.7802, 735.1931, 710.56134, 849.9764]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9577341079711914, 'coordinate': [34.421448, 20.055151, 358.71283, 76.53663]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9505634307861328, 'coordinate': [385.72278, 20.053688, 711.29333, 74.92744]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9001723527908325, 'coordinate': [386.46344, 477.03488, 699.4023, 490.07474]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8845751285552979, 'coordinate': [35.413048, 627.73596, 185.58383, 640.52264]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8837394118309021, 'coordinate': [387.17603, 716.3423, 524.7841, 729.258]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8508939743041992, 'coordinate': [35.50064, 331.18445, 141.6444, 344.81097]}]}}
```

The meanings of the parameters are as follows:
- `input_path`: The path to the input image for prediction.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is `None`.
- `boxes`: Information about the predicted bounding boxes, a list of dictionaries. Each dictionary represents a detected object and contains the following information:
  - `cls_id`: Class ID, an integer.
  - `label`: Class label, a string.
  - `score`: Confidence score of the bounding box, a float.
  - `coordinate`: Coordinates of the bounding box, a list of floats in the format <code>[xmin, ymin, xmax, ymax]</code>.


The visualized image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/layout_det/layout_res_plus.jpg"/>

Relevant methods, parameters, and explanations are as follows:

* `LayoutDetection` instantiates a target detection model (here, `PP-DocLayout_plus-L` is used as an example). The detailed explanation is as follows:
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
<td>Model name. If set to <code>None</code>, <code>PP-DocLayout-L</code> will be used.</td>
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
<td><code>img_size</code></td>
<td>Input image size.<ul>
<li><b>int</b>: e.g. <code>640</code>, resizes input image to 640x640.</li>
<li><b>list</b>: e.g. <code>[640, 512]</code>, resizes input image to width 640 and height 512.</li>
</ul>
</td>
<td><code>int|list|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence predictions.<ul>
<li><b>float</b>: e.g. <code>0.2</code>, filters out all boxes with confidence below 0.2.</li>
<li><b>dict</b>: The key is <code>int</code> (class id), the value is <code>float</code> (threshold). For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means class 0 uses threshold 0.45, class 2 uses 0.48, class 7 uses 0.4.</li>
<li><b>None</b>: uses the model's default configuration.</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes.
<ul>
<li><b>bool</b>: whether to use NMS for post-processing to filter overlapping boxes.</li>
<li><b>None</b>: uses the model's default configuration.</li>
</ul>
</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box.<ul>
<li><b>float</b>: A float greater than 0, e.g. <code>1.1</code>, expands width and height by 1.1 times.</li>
<li><b>list</b>: e.g. <code>[1.2, 1.5]</code>, expands width by 1.2x and height by 1.5x.</li>
<li><b>dict</b>: The key is <code>int</code> (class id), the value is <code>tuple</code> of two floats (width ratio, height ratio). For example, <code>{0: (1.1, 2.0)}</code> means for class 0, width is expanded by 1.1x and height by 2.0x.</li>
<li><b>None</b>: uses the model's default configuration.</li>
</ul>
</td>
<td><code>float|list|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merge mode for model output bounding boxes.<ul>
<li><b>"large"</b>: Only keep the largest outer box among overlapping boxes, remove inner boxes.</li>
<li><b>"small"</b>: Only keep the smallest inner box among overlapping boxes, remove outer boxes.</li>
<li><b>"union"</b>: Keep all boxes, no filtering.</li>
<li><b>dict</b>: The key is <code>int</code> (class id), the value is <code>str</code> (mode). For example, <code>{0: "large", 2: "small"}</code> means class 0 uses "large" mode, class 2 uses "small" mode.</li>
<li><b>None</b>: Use the model's default configuration.</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* The `predict()` method of the target detection model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `threshold`, which are explained as follows:

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
<td>Batch size, positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|list|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>str|dict|None</code></td>
<td>None</td>
</tr>
</tbody>
</table>


* Process the prediction results, with each sample's prediction result being the corresponding Result object, and supporting operations such as printing, saving as an image, and saving as a 'json' file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameters</th>
<th>Parameter type</th>
<th>Parameter Description</th>
<th>Default value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Do you want to use <code>JSON</code> indentation formatting for the output content</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to enhance the readability of the <code>JSON</code> data output, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non ASCII characters to Unicode characters. When set to <code>True</code>, all non ASCII </code>characters will be escaped; <code>False</code> preserves the original characters and is only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, when it is a directory, the name of the saved file is consistent with the name of the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to enhance the readability of the <code>JSON</code> data output, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non ASCII characters to Unicode characters. When set to <code>True</code>, all non <code>ASCII</code> characters will be escaped; <code>False</code> preserves the original characters and is only valid when<code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, when it is a directory, the name of the saved file is consistent with the name of the input file type</td>
<td>None</td>
</tr>
</table>


* Additionally, it also supports obtaining the visualized image with results and the prediction results via attributes, as follows:

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
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>


## IV. Custom Development

Since PaddleOCR does not directly provide training for the layout detection module, if you need to train the layout area detection model, you can refer to [PaddleX Layout Detection Module Secondary Development](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#iv-custom-development)Partially conduct training. The trained model can be seamlessly integrated into PaddleOCR's API for inference.

## V. FAQ
