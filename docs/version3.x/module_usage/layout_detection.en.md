---
comments: true
---

# Layout Detection Module Tutorial

## I. Overview
The core task of structure analysis is to parse and segment the content of input document images. By identifying different elements in the image (such as text, charts, images, etc.), they are classified into predefined categories (e.g., pure text area, title area, table area, image area, list area, etc.), and the position and size of these regions in the document are determined.

## II. Supported Model List

* <b>The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocBlockLayout</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams">Training Model</a></td>
<td>95.9</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>123.92 M</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
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

</details>


## III. Quick Integration  <a id="quick"> </a>

> ❗ Before quick integration, please install the PaddleOCR wheel package. For detailed instructions, refer to [PaddleOCR Local Installation Tutorial](../installation.en.md)。

Quickly experience with just one command:

```bash
paddleocr layout_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg
```

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
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>None</td>
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
<td><code>img_size</code></td>
<td>Size of the input image; if not specified, the default 800x800 be used</td>
<td><code>int/list/None</code></td>
<td>
<ul>
<li><b>int</b>, e.g., 640, means resizing the input image to 640x640</li>
<li><b>List</b>, e.g., [640, 512], means resizing the input image to a width of 640 and a height of 512</li>
<li><b>None</b>, not specified, will use the default 0.5</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results; if not specified, the default 0.5 will be used</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.2, means filtering out all bounding boxes with a confidence score less than 0.2</li>
<li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means applying a threshold of 0.45 for cls_id 0, 0.48 for cls_id 2, and 0.4 for cls_id 7</li>
<li><b>None</b>, not specified, will use the default 0.5</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes; if not specified, the default False will be used</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>, True/False, indicates whether to use NMS for post-processing to filter overlapping boxes</li>
<li><b>None</b>, not specified, will use the default False</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box; if not specified, the default 1.0 will be used</td>
<td><code>float/list/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, a positive float number, e.g., 1.1, means expanding the width and height of the detection box by 1.1 times while keeping the center unchanged</li>
<li><b>List</b>, e.g., [1.2, 1.5], means expanding the width by 1.2 times and the height by 1.5 times while keeping the center unchanged</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors, e.g., <code>{0: (1.1, 2.0)}</code> means cls_id 0 expanding the width by 1.1 times and the height by 2.0 times while keeping the center unchanged</li>
<li><b>None</b>, not specified, will use the default 1.0</li>
</ul>
</td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the detection boxes output by the model; if not specified, the default union will be used</td>
<td><code>string/dict/None</code></td>
<td>
<ul>
<li><b>large</b>, when set to large, only the largest external box will be retained for overlapping detection boxes, and the internal overlapping boxes will be deleted</li>
<li><b>small</b>, when set to small, only the smallest internal box will be retained for overlapping detection boxes, and the external overlapping boxes will be deleted</li>
<li><b>union</b>, no filtering of boxes will be performed, and both internal and external boxes will be retained</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes, e.g., <code>{0: "large", 2: "small"}</li>
<li><b>None</b>, not specified, will use the default union</li>
</ul>
</td>
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


* The `predict()` method of the target detection model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `threshold`, which are explained as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data for prediction, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg">示例</a></li>
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
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.2, means filtering out all bounding boxes with a confidence score less than 0.2</li>
<li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means applying a threshold of 0.45 for cls_id 0, 0.48 for cls_id 2, and 0.4 for cls_id 7</li>
<li><b>None</b>, not specified, will use the <code>threshold</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default 0.5 will be used</li>
</ul>
</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes; if not specified, the default False will be used</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>, True/False, indicates whether to use NMS for post-processing to filter overlapping boxes</li>
<li><b>None</b>, not specified, will use the <code>layout_nms</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default False will be used</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box; if not specified, the default 1.0 will be used</td>
<td><code>float/list/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, a positive float number, e.g., 1.1, means expanding the width and height of the detection box by 1.1 times while keeping the center unchanged</li>
<li><b>List</b>, e.g., [1.2, 1.5], means expanding the width by 1.2 times and the height by 1.5 times while keeping the center unchanged</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors, e.g., <code>{0: (1.1, 2.0)}</code> means cls_id 0 expanding the width by 1.1 times and the height by 2.0 times while keeping the center unchanged</li>
<li><b>None</b>, not specified, will use the <code>layout_unclip_ratio</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default 1.0 will be used</li>
</ul>
</td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the detection boxes output by the model; if not specified, the default union will be used</td>
<td><code>string/dict/None</code></td>
<td>
<ul>
<li><b>large</b>, when set to large, only the largest external box will be retained for overlapping detection boxes, and the internal overlapping boxes will be deleted</li>
<li><b>small</b>, when set to small, only the smallest internal box will be retained for overlapping detection boxes, and the external overlapping boxes will be deleted</li>
<li><b>union</b>, no filtering of boxes will be performed, and both internal and external boxes will be retained</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes, e.g., <code>{0: "large", 2: "small"}</li>
<li><b>None</b>, not specified, will use the <code>layout_merge_bboxes_mode</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default union will be used</li>
</ul>
</td>
<td>None</td>
</tr>
</tr></table>

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
