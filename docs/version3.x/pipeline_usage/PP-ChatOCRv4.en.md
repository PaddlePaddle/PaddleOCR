---
comments: true
---
# PP-ChatOCRv4-doc Pipeline Usage Tutorial

## 1. Introduction to PP-ChatOCRv4-doc Pipeline
PP-ChatOCRv4-doc is a unique document and image intelligent analysis solution from PaddlePaddle, combining LLM, MLLM, and OCR technologies to address complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal text recognition. Integrated with ERNIE Bot, it fuses massive data and knowledge, achieving high accuracy and wide applicability. This pipeline also provides flexible service deployment options, supporting deployment on various hardware. Furthermore, it offers custom development capabilities, allowing you to train and fine-tune models on your own datasets, with seamless integration of trained models.

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4.png" width="80%"/>
</div>

The PP-ChatOCRv4 pipeline includes the following 9 modules. Each module can be trained and inferred independently and includes multiple models. For more details, please click on the respective module to view the documentation.

- [Document Image Orientation Classification Module](../module_usage/doc_img_orientation_classification.en.md) (Optional)
- [Text Image Unwarping Module](../module_usage/text_image_unwarping.en.md) (Optional)
- [Layout Detection Module](../module_usage/layout_detection.en.md)
- [Table Structure Recognition Module](../module_usage/table_structure_recognition.en.md) (Optional)
- [Text Detection Module](../module_usage/text_detection.en.md)
- [Text Recognition Module](../module_usage/text_recognition.en.md)
- [Text Line Orientation Classification Module](../module_usage/textline_orientation_classification.en.md)(Optional)
- [Formula Recognition Module](../module_usage/formula_recognition.en.md) (Optional)
- [Seal Text Detection Module](../module_usage/seal_text_detection.en.md) (Optional)


In this pipeline, you can choose the model to use based on the benchmark data below.

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<details>
<summary><b>Document Image Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>Document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, and 270°.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Image Unwarp Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>High-precision Text Image Unwarping model.</td>
</tr>
</tbody>
</table>
</details>


<details>
<summary><b>Layout Detection Module Model:</b></summary>
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
<th>Model</th><th>Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Pretrained Model</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Pretrained Model</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Pretrained Model</a></td>
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
</table>

<p><b>Table Classification Module Models:</b></p>
<table>
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
</table>
</details>
</details>

<details>
<summary><b>Table Structure Recognition Module Models (Optional):</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
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
<td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>23.43 / 22.16</td>
<td>- / 41.80</td>
<td>6.9</td>
<td>SLANet_plus is an enhanced version of SLANet, the table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves the recognition ability for wireless and complex tables and reduces the model's sensitivity to the accuracy of table positioning, enabling more accurate recognition even with offset table positioning.</td>
</tr>
</table>
</details>

<details>
<summary><b>Text Detection Module Models</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Training Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>383.15 / 383.15</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>10.67 / 6.36</td>
<td>57.77 / 28.15</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Recognition Module Models</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Links</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec is a next-generation text recognition model. It aims to efficiently and accurately support the recognition of four major languages—Simplified Chinese, Traditional Chinese, English, and Japanese—as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters using a single model. While maintaining recognition performance, it balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>81.29</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>182</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data, building upon PP-OCRv4_server_rec. It enhances the recognition capabilities for some Traditional Chinese characters, Japanese characters, and special symbols, supporting over 15,000 characters. In addition to improving document-related text recognition, it also enhances general text recognition capabilities.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td>A lightweight recognition model of PP-OCRv4 with high inference efficiency, suitable for deployment on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>85.19</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>173</td>
<td>The server-side model of PP-OCRv4, offering high inference accuracy and deployable on various servers.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td>An ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model, supporting English and numeric character recognition.</td>
</tr>
</table>
* <b>PP-OCRv5 Multi-Scenario Models</b>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Chinese Avg Accuracy (%)</th>
<th>English Avg Accuracy (%)</th>
<th>Traditional Chinese Avg Accuracy (%)</th>
<th>Japanese Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td>PP-OCRv5_server_rec is a new-generation text recognition model. It efficiently and accurately supports four major languages: Simplified Chinese, Traditional Chinese, English, and Japanese, as well as handwriting, vertical text, pinyin, and rare characters, offering robust and efficient support for document understanding.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>136</td>
<td>PP-OCRv5_mobile_rec is a new-generation text recognition model. It efficiently and accurately supports four major languages: Simplified Chinese, Traditional Chinese, English, and Japanese, as well as handwriting, vertical text, pinyin, and rare characters, offering robust and efficient support for document understanding.</td>
</tr>
</table>

* <b>Chinese Recognition Models</b>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>182</td>
<td>Based on PP-OCRv4_server_rec, trained on additional Chinese documents and PP-OCR mixed data. It supports over 15,000 characters including Traditional Chinese, Japanese, and special symbols, enhancing both document-specific and general text recognition accuracy.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td>Lightweight model of PP-OCRv4 with high inference efficiency, suitable for deployment on various edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>85.19</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>173</td>
<td>Server-side model of PP-OCRv4 with high recognition accuracy, suitable for deployment on various servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>72.96</td>
<td>3.89 / 1.16</td>
<td>8.72 / 3.56</td>
<td>10.3</td>
<td>Lightweight model of PP-OCRv3 with high inference efficiency, suitable for deployment on various edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>80.5</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team at the Vision and Learning Lab (FVL) of Fudan University. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 6% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the A-list.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>48.8</td>
<td rowspan="1">
The RepSVTR text recognition model is a mobile-oriented text recognition model based on SVTRv2. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the B-list, while maintaining similar inference speed.
</td>
</tr>
</table>
* <b>English Recognition Models</b>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td>Ultra-lightweight English recognition model trained on PP-OCRv4, supporting English and number recognition.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.69</td>
<td>3.56 / 0.78</td>
<td>8.44 / 5.78</td>
<td>17.3</td>
<td>Ultra-lightweight English recognition model trained on PP-OCRv3, supporting English and number recognition.</td>
</tr>
</table>

* <b>Multilingual Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>60.21</td>
<td>3.73 / 0.98</td>
<td>8.76 / 2.91</td>
<td>9.6</td>
<td>An ultra-lightweight Korean text recognition model trained based on PP-OCRv3, supporting Korean and digits recognition</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>45.69</td>
<td>3.86 / 1.01</td>
<td>8.62 / 2.92</td>
<td>9.8</td>
<td>An ultra-lightweight Japanese text recognition model trained based on PP-OCRv3, supporting Japanese and digits recognition</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>82.06</td>
<td>3.90 / 1.16</td>
<td>9.24 / 3.18</td>
<td>10.8</td>
<td>An ultra-lightweight Traditional Chinese text recognition model trained based on PP-OCRv3, supporting Traditional Chinese and digits recognition</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>95.88</td>
<td>3.59 / 0.81</td>
<td>8.28 / 6.21</td>
<td>8.7</td>
<td>An ultra-lightweight Telugu text recognition model trained based on PP-OCRv3, supporting Telugu and digits recognition</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>96.96</td>
<td>3.49 / 0.89</td>
<td>8.63 / 2.77</td>
<td>17.4</td>
<td>An ultra-lightweight Kannada text recognition model trained based on PP-OCRv3, supporting Kannada and digits recognition</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>76.83</td>
<td>3.49 / 0.86</td>
<td>8.35 / 3.41</td>
<td>8.7</td>
<td>An ultra-lightweight Tamil text recognition model trained based on PP-OCRv3, supporting Tamil and digits recognition</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>76.93</td>
<td>3.53 / 0.78</td>
<td>8.50 / 6.83</td>
<td>8.7</td>
<td>An ultra-lightweight Latin text recognition model trained based on PP-OCRv3, supporting Latin and digits recognition</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>73.55</td>
<td>3.60 / 0.83</td>
<td>8.44 / 4.69</td>
<td>17.3</td>
<td>An ultra-lightweight Arabic script recognition model trained based on PP-OCRv3, supporting Arabic script and digits recognition</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>94.28</td>
<td>3.56 / 0.79</td>
<td>8.22 / 2.76</td>
<td>8.7</td>
<td>An ultra-lightweight Cyrillic script recognition model trained based on PP-OCRv3, supporting Cyrillic script and digits recognition</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>96.44</td>
<td>3.60 / 0.78</td>
<td>6.95 / 2.87</td>
<td>8.7</td>
<td>An ultra-lightweight Devanagari script recognition model trained based on PP-OCRv3, supporting Devanagari script and digits recognition</td>
</tr>
</table>
</details>

<details>
<summary><b>Text Line Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>98.85</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.96</td>
<td>Text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Formula Recognition Module Models  (Optional):</b></summary>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>UniMERNet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>85.91</td>
<td>43.50</td>
<td>1311.84 / 1311.84</td>
<td>- / 8288.07</td>
<td>1530</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. The model is trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, significantly improving the recognition accuracy of real-world formulas.</td>
</tr>
<td>PP-FormulaNet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.00</td>
<td>45.71</td>
<td>182.25 / 182.25</td>
<td>- / 254.39</td>
<td>224</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddlePaddle Vision Team. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network. Through parallel masking and model distillation techniques, it significantly improves inference speed while maintaining high recognition accuracy, making it suitable for applications requiring fast inference. The PP-FormulaNet-L version, on the other hand, uses Vary_VIT_B as its backbone network and is trained on a large-scale formula dataset, showing significant improvements in recognizing complex formulas compared to PP-FormulaNet-S.</td>
</tr>
<td>PP-FormulaNet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1482.03 / 1482.03</td>
<td>- / 3131.54</td>
<td>695</td>
</tr>
<td>PP-FormulaNet_plus-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">Training Model</a></td>
<td>88.71</td>
<td>53.32</td>
<td>179.20 / 179.20</td>
<td>- / 260.99</td>
<td>248</td>
<td rowspan="3">PP-FormulaNet_plus is an enhanced version of the formula recognition model developed by the Baidu PaddlePaddle Vision Team, building upon the original PP-FormulaNet. Compared to the original version, PP-FormulaNet_plus utilizes a more diverse formula dataset during training, including sources such as Chinese dissertations, professional books, textbooks, exam papers, and mathematics journals. This expansion significantly improves the model’s recognition capabilities. Among the models, PP-FormulaNet_plus-M and PP-FormulaNet_plus-L have added support for Chinese formulas and increased the maximum number of predicted tokens for formulas from 1,024 to 2,560, greatly enhancing the recognition performance for complex formulas. Meanwhile, the PP-FormulaNet_plus-S model focuses on improving the recognition of English formulas. With these improvements, the PP-FormulaNet_plus series models perform exceptionally well in handling complex and diverse formula recognition tasks. </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">Training Model</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1040.27 / 1040.27</td>
<td>- / 1615.80</td>
<td>592</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">Training Model</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1476.07 / 1476.07</td>
<td>- / 3125.58</td>
<td>698</td>
</tr>
<tr>
<td>LaTeX_OCR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>99</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone network and a transformer as the decoder, significantly improving the accuracy of formula recognition.</td>
</tr>
</table>
</details>

<details>
<summary><b>Seal Text Detection Module Models (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
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
<td>PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td>PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>Test Environment Description:</b></summary>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                    <li><strong>Test Dataset: </strong>
                        <ul>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Region Detection Model: A self-built layout analysis dataset using PaddleOCR, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Table Structure Recognition Model: A self-built English table recognition dataset using PaddleX.</li>
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>English Recognition Model: A self-built English dataset using PaddleX.</li>
                          <li>Multilingual Recognition Model: A self-built multilingual dataset using PaddleX.</li>
                          <li>Text Line Orientation Classification Model: A self-built dataset using PaddleOCR, covering various scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Seal Text Detection Model: A self-built dataset using PaddleOCR, containing 500 images of circular seal textures.</li>
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

<br />
<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>

## 2. Quick Start

Before using the PP-ChatOCRv4-doc pipeline locally, ensure you have completed the installation of the PaddleOCR wheel package according to the [PaddleOCR Local Installation Tutorial](../installation.en.md). If you prefer to install dependencies selectively, please refer to the relevant instructions in the installation documentation. The corresponding dependency group for this pipeline is `ie`.

Please note: If you encounter issues such as the program becoming unresponsive, unexpected program termination, running out of memory resources, or extremely slow inference during execution, please try adjusting the configuration according to the documentation, such as disabling unnecessary features or using lighter-weight models.

Before performing model inference, you first need to prepare the API key for the large language model. PP-ChatOCRv4 supports large model services on the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or the locally deployed standard OpenAI interface. If using the Baidu Cloud Qianfan Platform, refer to [Authentication and Authorization](https://cloud.baidu.com/doc/qianfan-api/s/ym9chdsy5) to obtain the API key. If using a locally deployed large model service, refer to the [PaddleNLP Large Model Deployment Documentation](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm) for deployment of the dialogue interface and vectorization interface for large models, and fill in the corresponding `base_url` and `api_key`. If you need to use a multimodal large model for data fusion, refer to the OpenAI service deployment in the [PaddleMIX Model Documentation](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2) for multimodal large model deployment, and fill in the corresponding `base_url` and `api_key`.

**Note**: If local deployment of a multimodal large model is restricted due to the local environment, you can comment out the lines containing the `mllm` variable in the code and only use the large language model for information extraction.

### 2.1 Command Line Experience

After updating the configuration file, you can complete quick inference using just a few lines of Python code. You can use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) for testing:


```bash
paddleocr pp_chatocrv4_doc -i vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key

# 通过 --invoke_mllm 和 --pp_docbee_base_url 使用多模态大模型
paddleocr pp_chatocrv4_doc -i vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --invoke_mllm True --pp_docbee_base_url http://127.0.0.1:8080/
```

<details><summary><b>The command line supports more parameter configurations. Click to expand for a detailed explanation of the command line parameters.</b></summary>
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
<td>Data to be predicted, required. Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories, PDF files need to be specified to the specific file path).
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>keys</code></td>
<td>Keys for information extraction.</td>
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
<td><code>invoke_mllm</code></td>
<td>Whether to load and use a multimodal large model. If not set, the default is <code>False</code>.</td>
<td><code>bool</code></td>
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
<td> The directory path of the layout detection model. If not set, the official model will be downloaded.
</td>
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
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If not set, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>Directory path of the text detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If not set, the pipeline's default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>Directory path of the text recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>Batch size for the text recognition model. If not set, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_name</code></td>
<td>Name of the table structure recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_dir</code></td>
<td>Directory path of the table structure recognition model. If not set, the official model will be downloaded.</td>
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
<td><code>seal_text_recognition_model_name</code></td>
<td>The name of the seal text recognition model. If not set, the default model of the pipeline will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>The directory path of the seal text recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>The batch size for the seal text recognition model. If not set, the batch size will default to <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load and use the document orientation classification module. If not set, the parameter value initialized by the pipeline will be used, which defaults to <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load and use the text image unwarping module. If not set, the parameter value initialized by the pipeline will be used, which defaults to <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to load and use the text line orientation classification module. If not set, the parameter value initialized by the pipeline will be used, which defaults to <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to load and use the seal text recognition sub-pipeline. If not set, the parameter's value initialized during pipeline setup will be used, defaulting to <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to load and use the table recognition sub-pipeline. If not set, the parameter's value initialized during pipeline setup will be used, defaulting to <code>True</code>.</td>
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
<td>
Whether to use Non-Maximum Suppression (NMS) as post-processing for layout detection. If not set, the parameter will be set to the value initialized in the pipeline, which defaults to <code>True</code> by default.
</td>
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
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limitation for text detection.
Any integer greater than <code>0</code>. If not set, the pipeline's initialized value for this parameter (initialized to <code>960</code>) will be used.
</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of side length limit for text detection.
Supports <code>min</code> and <code>max</code>. <code>min</code> means ensuring the shortest side of the image is not smaller than <code>det_limit_side_len</code>, and <code>max</code> means ensuring the longest side of the image is not larger than <code>limit_side_len</code>. If not set, the pipeline's initialized value for this parameter (initialized to <code>max</code>) will be used.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for text detection. In the output probability map, pixels with scores higher than this threshold will be considered text pixels.
Any floating-point number greater than <code>0</code>
. If not set, the pipeline's initialized value for this parameter (<code>0.3</code>) will be used.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Text detection box threshold. If the average score of all pixels within the detected result boundary is higher than this threshold, the result will be considered a text region.
 Any floating-point number greater than <code>0</code>. If not set, the pipeline's initialized value for this parameter (<code>0.6</code>) will be used.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion coefficient. This method is used to expand the text region—the larger the value, the larger the expanded area.
Any floating-point number greater than <code>0</code>
. If not set, the pipeline's initialized value for this parameter (<code>2.0</code>) will be used.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold. Text results with scores higher than this threshold will be retained.
 Any floating-point number greater than <code>0</code>
. If not set, the pipeline's initialized value for this parameter (<code>0.0</code>, i.e., no threshold) will be used.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
Any integer > <code>0</code>. If not set, the default is <code>736</code>.
</td>
<td><code>int</code></td>don’t 
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Limit type for image side in seal text detection.
supports <code>min</code> and <code>max</code>; <code>min</code> ensures shortest side ≥ <code>det_limit_side_len</code>, <code>max</code> ensures longest side ≤ <code>limit_side_len</code>. If not set, the default is <code>min</code>.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Pixel threshold. Pixels with scores above this value in the probability map are considered text.
Any float > <code>0</code></li>
</ul>If not set, the default is <code>0.2</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Box threshold. Boxes with average pixel scores above this value are considered text regions.Any float > <code>0</code>. If not set, the default is <code>0.6</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. Higher value means larger expansion area.
any float > <code>0</code>. If not set, the default is <code>0.5</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Recognition score threshold. Text results above this value will be kept.
Any float > <code>0</code></li>
</ul>If not set, the default is <code>0.0</code> (no threshold).
</td>
<td><code>float</code></td>
<td></td>
</tr>
<td><code>qianfan_api_key</code></td>
<td>API key for the Qianfan Platform.</td>
<td><code>str</code></td>
<td></td>
</tr>
<td><code>pp_docbee_base_url</code></td>
<td>URL for the multimodal large language model service.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. You can specify a particular card number:
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
</ul>If not set, the pipeline initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable the high-performance inference plugin.</td>
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
<td>Compute precision, such as FP32 or FP16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
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
<td>
The number of threads to use when performing inference on the CPU.</td>
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

This method will print the results to the terminal. The content printed to the terminal is explained as follows:


```
驾驶室准乘人数 2
```


### 2.2 Python Script Experience

The command-line method is for a quick experience and to view results. Generally, in projects, integration via code is often required. You can download the [Test File](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) and use the following example code for inference:

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

mllm_chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "PP-DocBee2",
    "base_url": "http://127.0.0.1:8080/",  # your local mllm service url
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

pipeline = PPChatOCRv4Doc()

visual_predict_res = pipeline.visual_predict(
    input="vehicle_certificate-1.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]

vector_info = pipeline.build_vector(
    visual_info_list, flag_save_bytes_vector=True, retriever_config=retriever_config
)
mllm_predict_res = pipeline.mllm_pred(
    input="vehicle_certificate-1.png",
    key_list=["Cab Seating Capacity"], # Translated: 驾驶室准乘人数
    mllm_chat_bot_config=mllm_chat_bot_config,
)
mllm_predict_info = mllm_predict_res["mllm_res"]
chat_result = pipeline.chat(
    key_list=["Cab Seating Capacity"], # Translated: 驾驶室准乘人数
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)

```

After running, the output is as follows:

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

The prediction process, API description, and output description for PP-ChatOCRv4 are as follows:

<details><summary>(1) Call <code>PPChatOCRv4Doc</code> to instantiate the PP-ChatOCRv4 pipeline object.</summary>

The relevant parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>The name of the model used for layout region detection. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>The directory path of the layout region detection model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>The name of the document orientation classification model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>The name of the document unwarping model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>The directory path of the document unwarping model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>The name of the text detection model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>The directory path of the text detection model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>The name of the text recognition model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>The directory path of the text recognition model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>The batch size for the text recognition model. If set to<code>None</code>, the batch size will default to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_name</code></td>
<td>The name of the table structure recognition model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_dir</code></td>
<td>The directory path of the table structure recognition model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>The name of the seal text detection model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>The directory path of the seal text detection model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>The name of the seal text recognition model. If set to<code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>The directory path of the seal text recognition model. If set to<code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>The batch size for the seal text recognition model. If set to<code>None</code>, the batch size will default to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load and use the document orientation classification module. If set to<code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>True</code>).</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load and use the document unwarping module. If set to<code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>True</code>).</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to load and use the text line orientation classification function. If set to<code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>True</code>).</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to load and use the seal text recognition sub-pipeline. If set to<code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>True</code>).</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to load and use the table recognition sub-pipeline. If set to<code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>True</code>).</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Layout model score threshold.
<ul>
<li><b>float</b>: Any float between <code>0-1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> where the key is the class ID and the value is the threshold for that class;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>0.5</code>.</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use Non-Maximum Suppression (NMS) as post-processing for layout detection. If set to <code>None</code>, the parameter will be set to the value initialized in the pipeline, which is set to <code>True</code> by default.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion factor for the detection boxes of the layout region detection model.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: Expansion ratios in horizontal and vertical directions;</li>
<li><b>dict</b>: A dictionary with <b>int</b> keys representing <code>cls_id</code>, and <b>tuple</b> values, e.g., <code>{0: (1.1, 2.0)}</code> means width is expanded 1.1× and height 2.0× for class 0 boxes;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>1.0</code>.</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Method for filtering overlapping boxes in layout region detection.
<ul>
<li><b>str</b>: <code>large</code>,<code>small</code>, <code>union</code>, representing whether to keep the large box, small box, or both when filtering overlapping boxes;</li>
<li><b>dict</b>, where the key is of <b>int</b> type, representing <code>cls_id</code>, and the value is of <b>str</b> type, e.g.,<code>{0: "large", 2: "small"}</code>, meaning use "large" mode for class 0 detection boxes and "small" mode for class 2 detection boxes;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>large</code>).</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limitation for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>960</code>).</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of side length limit for text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side of the image is not less than <code>det_limit_side_len</code>. <code>max</code> ensures the longest side of the image is not greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>max</code>).</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold. In the output probability map, pixels with scores greater than this threshold are considered text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.3</code>) will be used.</li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold. If the average score of all pixels within a detection result's bounding box is greater than this threshold, the result is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.6</code>) will be used.</li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion factor. This method is used to expand text regions; the larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>2.0</code>) will be used.</li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold. Text results with scores greater than this threshold will be kept.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.0</code>, i.e., no threshold) will be used.</li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>736</code>).</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Type of image side length limit for seal text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side of the image is not less than <code>det_limit_side_len</code>. <code>max</code> ensures the longest side of the image is not greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter will be used (defaults to <code>min</code>).</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Detection pixel threshold. In the output probability map, pixels with scores greater than this threshold are considered text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;
    <li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.2</code>) will be used.</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Detection box threshold. If the average score of all pixels within a detection result's bounding box is greater than this threshold, the result is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;
    <li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.6</code>) will be used.</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Seal text detection expansion factor. This method is used to expand text regions; the larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;
    <li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.5</code>) will be used.</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Seal text recognition threshold. Text results with scores greater than this threshold will be kept.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;
    <li><b>None</b>: If set to <code>None</code>, the value initialized by the pipeline for this parameter (defaults to <code>0.0</code>, i.e., no threshold) will be used.</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model. The configuration content is the following dictionary:
<pre><code>{
"module_name": "retriever",
"model_name": "embedding-v1",
"base_url": "https://qianfan.baidubce.com/v2",
"api_type": "qianfan",
"api_key": "api_key"  # Please set this to your actual API key
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_chat_bot_config</code></td>
<td>Configuration parameters for the multimodal large model. The configuration content is the following dictionary:
<pre><code>{
"module_name": "chat_bot",
"model_name": "PP-DocBee",
"base_url": "http://127.0.0.1:8080/", # Please set this to the actual URL of your multimodal large model service
"api_type": "openai",
"api_key": "api_key"  # Please set this to your actual API key
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>Configuration information for the large language model. The configuration content is the following dictionary:
<pre><code>{
"module_name": "chat_bot",
"model_name": "ernie-3.5-8k",
"base_url": "https://qianfan.baidubce.com/v2",
"api_type": "openai",
"api_key": "api_key"  # Please set this to your actual API key
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device used for inference. Supports specifying a specific card number:
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used.</li>
</ul>
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
<td>Computation precision, e.g., fp32, fp16.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
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
<td>Number of threads used when performing inference on CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX pipeline configuration file path.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(2) Call the <code>visual_predict()</code> method of the PP-ChatOCRv4 pipeline object to obtain visual prediction results. This method returns a list of results. Additionally, the pipeline also provides the <code>visual_predict_iter()</code> method. Both are identical in terms of parameter acceptance and result return, with the difference being that <code>visual_predict_iter()</code> returns a <code>generator</code>, allowing for step-by-step processing and retrieval of prediction results, suitable for handling large datasets or scenarios where memory saving is desired. You can choose either of these two methods based on your actual needs. The following are the parameters and their descriptions for the <code>visual_predict()</code> method:</summary>

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
<td>Data to be predicted, supports multiple input types, required.
<ul>
  <li><b>Python Var</b>: e.g., image data represented by <code>numpy.ndarray</code>;</li>
  <li><b>str</b>: e.g., local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of an image file or PDF file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">Example</a>; <b>Local directory</b>, which must contain images to be predicted, e.g., local path: <code>/root/data/</code> (Currently, prediction from directories containing PDF files is not supported; PDF files need to be specified by their full path);</li>
  <li><b>list</b>: List elements must be of the above types, e.g.,<code>[numpy.ndarray, numpy.ndarray]</code>,<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>,<code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module during inference.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document image unwarping module during inference.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module during inference.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal text recognition sub-pipeline during inference.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-pipeline during inference.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
</table>
</details>

<details><summary>(3) Process the visual prediction results.</summary>

The prediction result for each sample is of `dict` type, containing two fields: `visual_info` and `layout_parsing_result`. Visual information (including `normal_text_dict`, `table_text_list`, `table_html_list`, etc.) is obtained through `visual_info`, and the information for each sample is placed in the `visual_info_list` list. The content of this list will later be fed into the large language model.

Of course, you can also obtain the layout parsing results through `layout_parsing_result`. This result contains content such as tables, text, and images found in the file or image, and supports operations like printing, saving as an image, and saving as a `json` file:

```python
......
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]
    layout_parsing_result.print()
    layout_parsing_result.save_to_img("./output")
    layout_parsing_result.save_to_json("./output")
    layout_parsing_result.save_to_xlsx("./output")
    layout_parsing_result.save_to_html("./output")
......
```

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
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Prints the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation.</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. Set to <code>True</code> to escape all non-<code>ASCII</code> characters; <code>False</code> to preserve original characters, effective only when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Saves the result as a JSON format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Save file path. When it's a directory, the saved file name will be consistent with the input file name.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. Set to <code>True</code> to escape all non-<code>ASCII</code> characters; <code>False</code> to preserve original characters, effective only when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Saves the visualization images of various intermediate modules as PNG format images.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Save file path, supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Saves the tables in the file as HTML format files.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Save file path, supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Saves the tables in the file as XLSX format files.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Save file path, supports directory or file path.</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` Input path of the image to be predicted.
    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.
    - `model_settings`: `(Dict[str, bool])` Model parameters required to configure the pipeline.
        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessor sub-pipeline.
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal text recognition sub-pipeline.
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition sub-pipeline.
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition sub-pipeline.
    - `parsing_res_list`: `(List[Dict])` List of parsing results, where each element is a dictionary. The list order is the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` Bounding box of the layout region.
        - `block_label`: `(str)` Label of the layout region, e.g., `text`, `table`, etc.
        - `block_content`: `(str)` Content within the layout region.
    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` Dictionary of global OCR results.
      -  `input_path`: `(Union[str, None])` Image path accepted by the image OCR sub-pipeline. When the input is `numpy.ndarray`, it is saved as `None`.
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR sub-pipeline.
      - `dt_polys`: `(List[numpy.ndarray])` List of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with array shape (4, 2) and data type int16.
      - `dt_scores`: `(List[float])` List of confidence scores for text detection boxes.
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` Side length limit value for image preprocessing.
        - `limit_type`: `(str)` Processing method for side length limit.
        - `thresh`: `(float)` Confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` Confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` Expansion factor for text detection boxes.
        - `text_type`: `(str)` Type of text detection, currently fixed to "general".
      - `text_type`: `(str)` Type of text detection, currently fixed to "general".
      - `textline_orientation_angles`: `(List[int])` Prediction results of text line orientation classification. When enabled, returns actual angle values (e.g., [0,0,1]).
      - `text_rec_score_thresh`: `(float)` Filtering threshold for text recognition results.
      - `rec_texts`: `(List[str])` List of text recognition results, containing only text with confidence exceeding `text_rec_score_thresh`.
      - `rec_scores`: `(List[float])` List of text recognition confidence scores, filtered by `text_rec_score_thresh`.
      - `rec_polys`: `(List[numpy.ndarray])` List of text detection boxes filtered by confidence, format same as `dt_polys`.
    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of formula recognition results, each element is a dictionary.
        - `rec_formula`: `(str)` Formula recognition result.
        - `rec_polys`: `(numpy.ndarray)` Formula detection box, shape (4, 2), dtype int16.
        - `formula_region_id`: `(int)` Region number where the formula is located.
    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of seal text recognition results, each element is a dictionary.
        - `input_path`: `(str)` Input path of the seal image.
        - `model_settings`: `(Dict)` Model configuration parameters for the seal text recognition sub-pipeline.
        - `dt_polys`: `(List[numpy.ndarray])` List of seal detection boxes, format same as `dt_polys`.
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the seal detection module, specific parameter meanings are the same as above.
        - `text_type`: `(str)` Type of seal detection, currently fixed to "seal".
        - `text_rec_score_thresh`: `(float)` Filtering threshold for seal text recognition results.
        - `rec_texts`: `(List[str])` List of seal text recognition results, containing only text with confidence exceeding `text_rec_score_thresh`.
        - `rec_scores`: `(List[float])` List of seal text recognition confidence scores, filtered by `text_rec_score_thresh`.
        - `rec_polys`: `(List[numpy.ndarray])` List of seal detection boxes filtered by confidence, format same as `dt_polys`.
        - `rec_boxes`: `(numpy.ndarray)` Array of rectangular bounding boxes for detections, shape (n, 4), dtype int16. Each row represents a rectangle.
    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of table recognition results, each element is a dictionary.
        - `cell_box_list`: `(List[numpy.ndarray])` List of bounding boxes for table cells.
        - `pred_html`: `(str)` HTML format string of the table.
        - `table_ocr_pred`: `(dict)` OCR recognition result for the table.
            - `rec_polys`: `(List[numpy.ndarray])` List of detection boxes for cells.
            - `rec_texts`: `(List[str])` Recognition results for cells.
            - `rec_scores`: `(List[float])` Recognition confidence scores for cells.
            - `rec_boxes`: `(numpy.ndarray)` Array of rectangular bounding boxes for detections, shape (n, 4), dtype int16. Each row represents a rectangle.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list form.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, and only the last image will be retained).

Additionally, it supports obtaining visualization images with results and prediction results through properties, as follows:
<table>
<thead>
<tr>
<th>Property</th>
<th>Property Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Gets the prediction results in <code>json</code> format.</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Gets the visualization images in <code>dict</code> format.</td>
</tr>
</table>

- The prediction result obtained by the `json` property is dict-type data, and its content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` property is a dictionary-type data. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, and the corresponding values are `Image.Image` objects: used to display visualization images of layout region detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, the dictionary will only contain `layout_det_res`.
</details>

<details><summary>(4) Call the <code>build_vector()</code> method of the PP-ChatOCRv4 pipeline object to build vectors for the text content.</summary>

The following are the parameters and their descriptions for the `build_vector()` method:

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
<td><code>visual_info</code></td>
<td>Visual information, can be a dictionary containing visual information, or a list of such dictionaries.</td>
<td><code>list|dict</code></td>
<td></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Minimum number of characters. A positive integer greater than 0, can be determined based on the token length supported by the large language model.</td>
<td><code>int</code></td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>block_size</code></td>
<td>Block size when building a vector library for long text. A positive integer greater than 0, can be determined based on the token length supported by the large language model.</td>
<td><code>int</code></td>
<td><code>300</code></td>
</tr>
<tr>
<td><code>flag_save_bytes_vector</code></td>
<td>Whether to save text as a binary file.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model, same as the parameter during instantiation. If set to <code>None</code>, uses instantiation parameters; otherwise, this parameter takes precedence.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</table>
This method returns a dictionary containing visual text information. The content of the dictionary is as follows:

- `flag_save_bytes_vector`: `(bool)` Whether to save the result as a binary file.
- `flag_too_short_text`: `(bool)` Whether the text length is less than the minimum number of characters.
- `vector`: `(str|list)` Binary content of the text or the text content itself, depending on the values of `flag_save_bytes_vector` and `min_characters`. If `flag_save_bytes_vector=True` and the text length is greater than or equal to the minimum number of characters, it returns binary content; otherwise, it returns the original text.
</details>

<details><summary>(5) Call the <code>mllm_pred()</code> method of the PP-ChatOCRv4 pipeline object to get the extraction results from the multimodal large model.</summary>

The following are the parameters and their descriptions for the `mllm_pred()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types, required.
<ul>
  <li><b>Python Var</b>: e.g., image data represented by <code>numpy.ndarray</code>; </li>
  <li><b>str</b>: e.g., local path of an image file or single-page PDF file: <code>/root/data/img.jpg</code>;<b>URL link</b>, e.g., network URL of an image file or single-page PDF file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">Example</a>.</li>
</ul>
</td>
<td><code>Python Var|str</code></td>
<td></td>
</tr>
<tr>
<td><code>key_list</code></td>
<td>A single key or a list of keys used for extracting information.</td>
<td><code>Union[str, List[str]]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_chat_bot_config</code></td>
<td>Configuration parameters for the multimodal large model, same as the parameter during instantiation. If set to <code>None</code>, uses instantiation parameters; otherwise, this parameter takes precedence.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>(6) Call the <code>chat()</code> method of the PP-ChatOCRv4 pipeline object to extract key information.</summary>

The following are the parameters and their descriptions for the `chat()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>key_list</code></td>
<td>A single key or a list of keys used for extracting information.</td>
<td><code>Union[str, List[str]]</code></td>
<td></td>
</tr>
<tr>
<td><code>visual_info</code></td>
<td>Visual information result.</td>
<td><code>List[dict]</code></td>
<td></td>
</tr>
<tr>
<td><code>use_vector_retrieval</code></td>
<td>Whether to use vector retrieval.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>vector_info</code></td>
<td>Vector information used for retrieval.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Required minimum number of characters. A positive integer greater than 0.</td>
<td><code>int</code></td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>text_task_description</code></td>
<td>Description of the text task.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_output_format</code></td>
<td>Output format for text results.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rules_str</code></td>
<td>Rules for generating text results.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_text_content</code></td>
<td>Text content for few-shot demonstration.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_key_value_list</code></td>
<td>Key-value list for few-shot demonstration.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_task_description</code></td>
<td>Description of the table task.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_output_format</code></td>
<td>Output format for table results.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_rules_str</code></td>
<td>Rules for generating table results.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_text_content</code></td>
<td>Text content for table few-shot demonstration.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_key_value_list</code></td>
<td>Key-value list for table few-shot demonstration.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_predict_info</code></td>
<td>Multimodal large model result.</td>
<td><code>dict|None</code></td>
<td>
<code>None</code>
</td>
</tr>
<td><code>mllm_integration_strategy</code></td>
<td>Data fusion strategy for multimodal large model and large language model, supports using one of them separately or fusing the results of both. Options: "integration", "llm_only", and "mllm_only".</td>
<td><code>str</code></td>
<td><code>"integration"</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>Configuration information for the large language model, same as the parameter during instantiation.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model, same as the parameter during instantiation. If set to <code>None</code>, uses instantiation parameters; otherwise, this parameter takes precedence.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

This method will print the result to the terminal. The content printed to the terminal is explained as follows:
  - `chat_res`: `(dict)` The result of information extraction, which is a dictionary containing the keys to be extracted and their corresponding values.

</details>


## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the sample code in [2.2  Python Script Experience](#22-python-script-experience).

Additionally, PaddleX provides two other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed instructions on high-performance inference, please refer to the [High-Performance Inference Guide](../deployment/high_performance_inference.en.md).

☁️ **Serving**: Serving is a common deployment form in actual production environments. By encapsulating the inference functionality as a service, clients can access these services through network requests to obtain inference results. PaddleX supports multiple serving solutions for pipelines. For detailed instructions on serving, please refer to the [Service Deployment Guide](../deployment/serving.en.md).

Below are the API references for basic serving and multi-language service invocation examples:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the response body has the following properties:</li>
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
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed at <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed at <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not successfully processed, the response body has the following properties:</li>
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
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID of the request.</td>
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
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>analyzeImages</code></b></li>
</ul>
<p>Uses computer vision models to analyze images, obtain OCR, table recognition results, etc., and extract key information from the images.</p>
<p><code>POST /chatocr-visual</code></p>
<ul>
<li>Properties of the request body:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>URL of an image file or PDF file accessible to the server, or Base64 encoded result of the content of the above file types. By default, for PDF files exceeding 10 pages, only the content of the first 10 pages will be processed.<br />
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
<td>File type. <code>0</code> represents a PDF file, <code>1</code> represents an image file. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_seal_recognition</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_table_recognition</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_side_len</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_type</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_box_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_unclip_ratio</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_rec_score_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_side_len</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_type</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_box_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_unclip_ratio</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_rec_score_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>
Whether to return the final visualization image and intermediate images during the processing.<br/>
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>If <code>true</code> is provided: return images.</li>
<li>If <code>false</code> is provided: do not return any images.</li>
<li>If this parameter is omitted from the request body, or if <code>null</code> is explicitly passed, the behavior will follow the value of <code>Serving.visualize</code> in the pipeline configuration.</li>
</ul>
<br/>
For example, adding the following setting to the pipeline config file:<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
will disable image return by default. This behavior can be overridden by explicitly setting the <code>visualize</code> parameter in the request.<br/>
If neither the request body nor the configuration file is set (If <code>visualize</code> is set to <code>null</code> in the request and  not defined in the configuration file), the image is returned by default.
</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>Analysis results obtained using computer vision models. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>array</code></td>
<td>Key information in the image, which can be used as input for other operations.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Input data information.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>layoutParsingResults</code> is an <code>object</code> with the following properties:</p>
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
<td>A simplified version of the <code>res</code> field in the JSON representation of the <code>layout_parsing_result</code> generated by the pipeline object's <code>visual_predict</code> method, with the <code>input_path</code> and <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Refer to the description of <code>img</code> attribute of the pipeline's visual prediction result.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Input image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>buildVectorStore</code></b></li>
</ul>
<p>Builds a vector database.</p>
<p><code>POST /chatocr-vector</code></p>
<ul>
<li>Properties of the request body:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>visualInfo</code></td>
<td><code>array</code></td>
<td>Key information in the image. Provided by the <code>analyzeImages</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>minCharacters</code></td>
<td><code>integer</code></td>
<td>Please refer to the description of the <code>min_characters</code> parameter of the pipeline object's <code>build_vector</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>blockSize</code></td>
<td><code>integer</code></td>
<td>Please refer to the description of the <code>block_size</code> parameter of the pipeline object's <code>build_vector</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>retrieverConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>retriever_config</code> parameter of the pipeline object's <code>build_vector</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>vectorInfo</code></td>
<td><code>object</code></td>
<td>Serialized result of the vector database, which can be used as input for other operations.</td>
</tr>
</tbody>
</table>
<li><b><code>invokeMLLM</code></b></li>
</ul>
<p>Invoke the MLLM.</p>
<p><code>POST /chatocr-mllm</code></p>
<ul>
<li>Properties of the request body:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>URL of an image file accessible by the server or the Base64-encoded content of the image file.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>keyList</code></td>
<td><code>array</code></td>
<td>List of keys.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>mllmChatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>mllm_chat_bot_config</code> parameter of the pipeline object's <code>mllm_pred</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following property:</li>
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
<td><code>mllmPredictInfo</code></td>
<td><code>object</code></td>
<td>MLLM invocation result.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>chat</code></b></li>
</ul>
<p>Interacts with large language models to extract key information using them.</p>
<p><code>POST /chatocr-chat</code></p>
<ul>
<li>Properties of the request body:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>keyList</code></td>
<td><code>array</code></td>
<td>List of keys.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>Key information in the image. Provided by the <code>analyzeImages</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>useVectorRetrieval</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_vector_retrieval</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>vectorInfo</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Serialized result of the vector database. Provided by the <code>buildVectorStore</code> operation. Please note that the deserialization process involves performing an unpickle operation. To prevent malicious attacks, be sure to use data from trusted sources.</td>
<td>No</td>
</tr>
<tr>
<td><code>minCharacters</code></td>
<td><code>integer</code></td>
<td>Please refer to the description of the <code>min_characters</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textTaskDescription</code></td>
<td><code>string</code></td>
<td>Please refer to the description of the <code>text_task_description</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textOutputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_output_format</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_rules_str</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textFewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_few_shot_demo_text_content</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textFewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_few_shot_demo_key_value_list</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableTaskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_task_description</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableOutputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_output_format</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableRulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_rules_str</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableFewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_few_shot_demo_text_content</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableFewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_few_shot_demo_key_value_list</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>mllmPredictInfo</code></td>
<td><code>object</code> | <code>null</code></td>
<td>MLLM invocation result. Provided by the <code>invokeMllm</code> operation.</td>
<td>No</td>
</tr>
<tr>
<td><code>mllmIntegrationStrategy</code></td>
<td><code>string</code></td>
<td>Please refer to the description of the <code>mllm_integration_strategy</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>chatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>chat_bot_config</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>retrieverConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>retriever_config</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>chatResult</code></td>
<td><code>object</code></td>
<td>Key information extraction result.</td>
</tr>
</tbody>
</table>
<li><b>Note:</b></li>
Including sensitive parameters such as API key for large model calls in the request body can be a security risk. If not necessary, set these parameters in the configuration file and do not pass them on request.
<br/><br/>
</details>

<details><summary>Multi-language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">
# This script only shows the use case for images. For calling with other file types, please read the API reference and make adjustments.

import base64
import pprint
import sys
import requests


API_BASE_URL = "http://127.0.0.1:8080"

image_path = "./demo.jpg"
keys = ["name"]

with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data,
    "fileType": 1,
}

resp_visual = requests.post(url=f"{API_BASE_URL}/chatocr-visual", json=payload)
if resp_visual.status_code != 200:
    print(
        f"Request to chatocr-visual failed with status code {resp_visual.status_code}."
    )
    pprint.pp(resp_visual.json())
    sys.exit(1)
result_visual = resp_visual.json()["result"]

for i, res in enumerate(result_visual["layoutParsingResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")

payload = {
    "visualInfo": result_visual["visualInfo"],
}
resp_vector = requests.post(url=f"{API_BASE_URL}/chatocr-vector", json=payload)
if resp_vector.status_code != 200:
    print(
        f"Request to chatocr-vector failed with status code {resp_vector.status_code}."
    )
    pprint.pp(resp_vector.json())
    sys.exit(1)
result_vector = resp_vector.json()["result"]

payload = {
    "image": image_data,
    "keyList": keys,
}
resp_mllm = requests.post(url=f"{API_BASE_URL}/chatocr-mllm", json=payload)
if resp_mllm.status_code != 200:
    print(
        f"Request to chatocr-mllm failed with status code {resp_mllm.status_code}."
    )
    pprint.pp(resp_mllm.json())
    sys.exit(1)
result_mllm = resp_mllm.json()["result"]

payload = {
    "keyList": keys,
    "visualInfo": result_visual["visualInfo"],
    "useVectorRetrieval": True,
    "vectorInfo": result_vector["vectorInfo"],
    "mllmPredictInfo": result_mllm["mllmPredictInfo"],
}
resp_chat = requests.post(url=f"{API_BASE_URL}/chatocr-chat", json=payload)
if resp_chat.status_code != 200:
    print(
        f"Request to chatocr-chat failed with status code {resp_chat.status_code}."
    )
    pprint.pp(resp_chat.json())
    sys.exit(1)
result_chat = resp_chat.json()["result"]
print("Final result:")
print(result_chat["chatResult"])
</code></pre>
</details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

using json = nlohmann::json;

std::string encode_image(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("File open error.");
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    return base64::to_base64(std::string(buf.data(), buf.size()));
}

int main() {
    httplib::Client client("localhost", 8080);
    std::string imagePath = "./demo.jpg";
    std::string imageData = encode_image(imagePath);
    json keys = { "Name" };

    json payload_visual = { {"file", imageData}, {"fileType", 1} };
    auto resp1 = client.Post("/chatocr-visual", payload_visual.dump(), "application/json");
    if (!resp1 || resp1->status != 200) {
        std::cerr << "chatocr-visual failed.\n"; return 1;
    }
    json result_visual = json::parse(resp1->body)["result"];

    for (size_t i = 0; i < result_visual["layoutParsingResults"].size(); ++i) {
        auto& res = result_visual["layoutParsingResults"][i];
        std::cout << "prunedResult: " << res["prunedResult"].dump() << "\n";
        if (res.contains("outputImages")) {
            for (auto& [name, b64] : res["outputImages"].items()) {
                std::string outPath = name + "_" + std::to_string(i) + ".jpg";
                std::string decoded = base64::from_base64(b64.get<std::string>());
                std::ofstream out(outPath, std::ios::binary);
                out.write(decoded.data(), decoded.size());
                out.close();
                std::cout << "Saved: " << outPath << "\n";
            }
        }
    }

    json payload_vector = { {"visualInfo", result_visual["visualInfo"]} };
    auto resp2 = client.Post("/chatocr-vector", payload_vector.dump(), "application/json");
    if (!resp2 || resp2->status != 200) {
        std::cerr << "chatocr-vector failed.\n"; return 1;
    }
    json result_vector = json::parse(resp2->body)["result"];

    json payload_mllm = { {"image", imageData}, {"keyList", keys} };
    auto resp3 = client.Post("/chatocr-mllm", payload_mllm.dump(), "application/json");
    if (!resp3 || resp3->status != 200) {
        std::cerr << "chatocr-mllm failed.\n"; return 1;
    }
    json result_mllm = json::parse(resp3->body)["result"];

    json payload_chat = {
        {"keyList", keys},
        {"visualInfo", result_visual["visualInfo"]},
        {"useVectorRetrieval", true},
        {"vectorInfo", result_vector["vectorInfo"]},
        {"mllmPredictInfo", result_mllm["mllmPredictInfo"]}
    };
    auto resp4 = client.Post("/chatocr-chat", payload_chat.dump(), "application/json");
    if (!resp4 || resp4->status != 200) {
        std::cerr << "chatocr-chat failed.\n"; return 1;
    }

    json result_chat = json::parse(resp4->body)["result"];
    std::cout << "Final chat result: " << result_chat["chatResult"] << std::endl;

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Iterator;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_BASE_URL = "http://localhost:8080";
        String imagePath = "./demo.jpg";
        String[] keys = {"Name"};

        OkHttpClient client = new OkHttpClient();
        ObjectMapper objectMapper = new ObjectMapper();
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        byte[] imageBytes = java.nio.file.Files.readAllBytes(new File(imagePath).toPath());
        String base64Image = Base64.getEncoder().encodeToString(imageBytes);

        ObjectNode visualPayload = objectMapper.createObjectNode();
        visualPayload.put("file", base64Image);
        visualPayload.put("fileType", 1);

        Request requestVisual = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-visual")
                .post(RequestBody.create(JSON, visualPayload.toString()))
                .build();

        Response responseVisual = client.newCall(requestVisual).execute();
        if (!responseVisual.isSuccessful()) {
            System.err.println("chatocr-visual failed: " + responseVisual.code());
            return;
        }

        JsonNode resultVisual = objectMapper.readTree(responseVisual.body().string()).get("result");

        JsonNode layoutResults = resultVisual.get("layoutParsingResults");
        for (int i = 0; i < layoutResults.size(); i++) {
            JsonNode res = layoutResults.get(i);
            System.out.println("prunedResult [" + i + "]: " + res.get("prunedResult").toString());

            JsonNode outputImages = res.get("outputImages");
            if (outputImages != null && outputImages.isObject()) {
                Iterator<String> names = outputImages.fieldNames();
                while (names.hasNext()) {
                    String imgName = names.next();
                    String imgBase64 = outputImages.get(imgName).asText();
                    byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                    String imgPath = imgName + "_" + i + ".jpg";
                    try (FileOutputStream fos = new FileOutputStream(imgPath)) {
                        fos.write(imgBytes);
                        System.out.println("Saved image: " + imgPath);
                    }
                }
            }
        }

        ObjectNode vectorPayload = objectMapper.createObjectNode();
        vectorPayload.set("visualInfo", resultVisual.get("visualInfo"));

        Request requestVector = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-vector")
                .post(RequestBody.create(JSON, vectorPayload.toString()))
                .build();

        Response responseVector = client.newCall(requestVector).execute();
        if (!responseVector.isSuccessful()) {
            System.err.println("chatocr-vector failed: " + responseVector.code());
            return;
        }

        JsonNode resultVector = objectMapper.readTree(responseVector.body().string()).get("result");

        ObjectNode mllmPayload = objectMapper.createObjectNode();
        mllmPayload.put("image", base64Image);
        mllmPayload.putArray("keyList").add(keys[0]);

        Request requestMllm = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-mllm")
                .post(RequestBody.create(JSON, mllmPayload.toString()))
                .build();

        Response responseMllm = client.newCall(requestMllm).execute();
        if (!responseMllm.isSuccessful()) {
            System.err.println("chatocr-mllm failed: " + responseMllm.code());
            return;
        }

        JsonNode resultMllm = objectMapper.readTree(responseMllm.body().string()).get("result");

        ObjectNode chatPayload = objectMapper.createObjectNode();
        chatPayload.putArray("keyList").add(keys[0]);
        chatPayload.set("visualInfo", resultVisual.get("visualInfo"));
        chatPayload.put("useVectorRetrieval", true);
        chatPayload.set("vectorInfo", resultVector.get("vectorInfo"));
        chatPayload.set("mllmPredictInfo", resultMllm.get("mllmPredictInfo"));

        Request requestChat = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-chat")
                .post(RequestBody.create(JSON, chatPayload.toString()))
                .build();

        Response responseChat = client.newCall(requestChat).execute();
        if (!responseChat.isSuccessful()) {
            System.err.println("chatocr-chat failed: " + responseChat.code());
            return;
        }

        JsonNode resultChat = objectMapper.readTree(responseChat.body().string()).get("result");
        System.out.println("Final result:");
        System.out.println(resultChat.get("chatResult").toString());
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
    "os"
)

func sendPostRequest(url string, payload map[string]interface{}) (map[string]interface{}, error) {
    bodyBytes, err := json.Marshal(payload)
    if err != nil {
        return nil, fmt.Errorf("error marshaling payload: %v", err)
    }

    req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
    if err != nil {
        return nil, fmt.Errorf("error creating request: %v", err)
    }
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("error sending request: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("status code error: %d", resp.StatusCode)
    }

    respBytes, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("error reading response: %v", err)
    }

    var result map[string]interface{}
    if err := json.Unmarshal(respBytes, &result); err != nil {
        return nil, fmt.Errorf("error unmarshaling response: %v", err)
    }
    return result["result"].(map[string]interface{}), nil
}

func main() {
    apiBase := "http://localhost:8080"
    imagePath := "./demo.jpg"
    keys := []string{"Name"}

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Printf("read image failed : %v\n", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    visualPayload := map[string]interface{}{
        "file":     imageData,
        "fileType": 1,
    }
    visualResult, err := sendPostRequest(apiBase+"/chatocr-visual", visualPayload)
    if err != nil {
        fmt.Printf("chatocr-visual request error: %v\n", err)
        return
    }

    layoutResults := visualResult["layoutParsingResults"].([]interface{})
    for i, res := range layoutResults {
        layout := res.(map[string]interface{})
        fmt.Println("PrunedResult:", layout["prunedResult"])
        outputImages := layout["outputImages"].(map[string]interface{})
        for name, img := range outputImages {
            imgBytes, _ := base64.StdEncoding.DecodeString(img.(string))
            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            if err := os.WriteFile(filename, imgBytes, 0644); err == nil {
                fmt.Printf("save image：%s\n", filename)
            }
        }
    }

    vectorPayload := map[string]interface{}{
        "visualInfo": visualResult["visualInfo"],
    }
    vectorResult, err := sendPostRequest(apiBase+"/chatocr-vector", vectorPayload)
    if err != nil {
        fmt.Printf("chatocr-vector request error: %v\n", err)
        return
    }

    mllmPayload := map[string]interface{}{
        "image":   imageData,
        "keyList": keys,
    }
    mllmResult, err := sendPostRequest(apiBase+"/chatocr-mllm", mllmPayload)
    if err != nil {
        fmt.Printf("chatocr-mllm request error: %v\n", err)
        return
    }

    chatPayload := map[string]interface{}{
        "keyList":           keys,
        "visualInfo":        visualResult["visualInfo"],
        "useVectorRetrieval": true,
        "vectorInfo":        vectorResult["vectorInfo"],
        "mllmPredictInfo":   mllmResult["mllmPredictInfo"],
    }
    chatResult, err := sendPostRequest(apiBase+"/chatocr-chat", chatPayload)
    if err != nil {
        fmt.Printf("chatocr-chat request error: %v\n", err)
        return
    }

    fmt.Println("final result：", chatResult["chatResult"])
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
    static readonly string API_BASE_URL = "http://localhost:8080";
    static readonly string inputFilePath = "./demo.jpg";
    static readonly string[] keys = { "Name" };

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(inputFilePath);
        string imageData = Convert.ToBase64String(imageBytes);

        var payloadVisual = new JObject
        {
            { "file", imageData },
            { "fileType", 1 }
        };

        var respVisual = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-visual",
            new StringContent(payloadVisual.ToString(), Encoding.UTF8, "application/json"));

        if (!respVisual.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-visual failed: {respVisual.StatusCode}");
            Console.Error.WriteLine(await respVisual.Content.ReadAsStringAsync());
            return;
        }

        JObject resultVisual = JObject.Parse(await respVisual.Content.ReadAsStringAsync())["result"] as JObject;

        var layoutParsingResults = (JArray)resultVisual["layoutParsingResults"];
        for (int i = 0; i < layoutParsingResults.Count; i++)
        {
            var res = layoutParsingResults[i];
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
                        File.WriteAllBytes(imgPath, Convert.FromBase64String(base64Img));
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }

        var payloadVector = new JObject
        {
            { "visualInfo", resultVisual["visualInfo"] }
        };

        var respVector = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-vector",
            new StringContent(payloadVector.ToString(), Encoding.UTF8, "application/json"));

        if (!respVector.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-vector failed: {respVector.StatusCode}");
            Console.Error.WriteLine(await respVector.Content.ReadAsStringAsync());
            return;
        }

        JObject resultVector = JObject.Parse(await respVector.Content.ReadAsStringAsync())["result"] as JObject;

        var payloadMllm = new JObject
        {
            { "image", imageData },
            { "keyList", new JArray(keys) }
        };

        var respMllm = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-mllm",
            new StringContent(payloadMllm.ToString(), Encoding.UTF8, "application/json"));

        if (!respMllm.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-mllm failed: {respMllm.StatusCode}");
            Console.Error.WriteLine(await respMllm.Content.ReadAsStringAsync());
            return;
        }

        JObject resultMllm = JObject.Parse(await respMllm.Content.ReadAsStringAsync())["result"] as JObject;

        var payloadChat = new JObject
        {
            { "keyList", new JArray(keys) },
            { "visualInfo", resultVisual["visualInfo"] },
            { "useVectorRetrieval", true },
            { "vectorInfo", resultVector["vectorInfo"] },
            { "mllmPredictInfo", resultMllm["mllmPredictInfo"] }
        };

        var respChat = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-chat",
            new StringContent(payloadChat.ToString(), Encoding.UTF8, "application/json"));

        if (!respChat.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-chat failed: {respChat.StatusCode}");
            Console.Error.WriteLine(await respChat.Content.ReadAsStringAsync());
            return;
        }

        JObject resultChat = JObject.Parse(await respChat.Content.ReadAsStringAsync())["result"] as JObject;
        Console.WriteLine("Final result:");
        Console.WriteLine(resultChat["chatResult"]);
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_BASE_URL = 'http://localhost:8080';
const imagePath = './demo.jpg';
const keys = ['Name'];

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

(async () => {
  try {
    const imageData = encodeImageToBase64(imagePath);

    const respVisual = await axios.post(`${API_BASE_URL}/chatocr-visual`, {
      file: imageData,
      fileType: 1
    });

    const resultVisual = respVisual.data.result;
    resultVisual.layoutParsingResults.forEach((res, i) => {
      console.log(`\n[${i}] prunedResult:\n`, res.prunedResult);
      const outputImages = res.outputImages || {};
      for (const [imgName, base64Img] of Object.entries(outputImages)) {
        const fileName = `${imgName}_${i}.jpg`;
        fs.writeFileSync(fileName, Buffer.from(base64Img, 'base64'));
        console.log(`Output image saved at ${fileName}`);
      }
    });

    const respVector = await axios.post(`${API_BASE_URL}/chatocr-vector`, {
      visualInfo: resultVisual.visualInfo
    });
    const resultVector = respVector.data.result;

    const respMllm = await axios.post(`${API_BASE_URL}/chatocr-mllm`, {
      image: imageData,
      keyList: keys
    });
    const resultMllm = respMllm.data.result;

    const respChat = await axios.post(`${API_BASE_URL}/chatocr-chat`, {
      keyList: keys,
      visualInfo: resultVisual.visualInfo,
      useVectorRetrieval: true,
      vectorInfo: resultVector.vectorInfo,
      mllmPredictInfo: resultMllm.mllmPredictInfo
    });

    const resultChat = respChat.data.result;
    console.log('\nFinal result:\n', resultChat.chatResult);

  } catch (error) {
    if (error.response) {
      console.error(`❌ Request failed: ${error.response.status}`);
      console.error(error.response.data);
    } else {
      console.error('❌ Error occurred:', error.message);
    }
  }
})();
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_BASE_URL = "http://localhost:8080";
$image_path = "./demo.jpg";
$keys = ["Name"];

$image_data = base64_encode(file_get_contents($image_path));

$payload_visual = [
    "file" => $image_data,
    "fileType" => 1
];
$response_visual_raw = send_post_raw("$API_BASE_URL/chatocr-visual", $payload_visual);
$response_visual = json_decode($response_visual_raw, true);
if (!isset($response_visual["result"])) {
    echo "chatocr-visual request error\n";
    print_r($response_visual);
    exit(1);
}
$result_visual_raw = json_decode($response_visual_raw, false)->result;
$result_visual_arr = $response_visual["result"];

foreach ($result_visual_arr["layoutParsingResults"] as $i => $res) {
    echo "[$i] prunedResult:\n";
    print_r($res["prunedResult"]);
    if (!empty($res["outputImages"])) {
        foreach ($res["outputImages"] as $img_name => $base64_img) {
            $img_path = "{$img_name}_{$i}.jpg";
            file_put_contents($img_path, base64_decode($base64_img));
            echo "Output image saved at $img_path\n";
        }
    }
}

$payload_vector = [
    "visualInfo" => $result_visual_raw->visualInfo
];
$response_vector_raw = send_post_raw("$API_BASE_URL/chatocr-vector", $payload_vector);
$response_vector = json_decode($response_vector_raw, true);
if (!isset($response_vector["result"])) {
    echo "chatocr-vector request error\n";
    print_r($response_vector);
    exit(1);
}
$result_vector_raw = json_decode($response_vector_raw, false)->result;

$payload_mllm = [
    "image" => $image_data,
    "keyList" => $keys
];
$response_mllm_raw = send_post_raw("$API_BASE_URL/chatocr-mllm", $payload_mllm);
$response_mllm = json_decode($response_mllm_raw, true);
if (!isset($response_mllm["result"])) {
    echo "chatocr-mllm request error\n";
    print_r($response_mllm);
    exit(1);
}
$result_mllm_raw = json_decode($response_mllm_raw, false)->result;

$payload_chat = [
    "keyList" => $keys,
    "visualInfo" => $result_visual_raw->visualInfo,
    "useVectorRetrieval" => true,
    "vectorInfo" => $result_vector_raw->vectorInfo,
    "mllmPredictInfo" => $result_mllm_raw->mllmPredictInfo
];
$response_chat_raw = send_post_raw("$API_BASE_URL/chatocr-chat", $payload_chat);
$response_chat = json_decode($response_chat_raw, true);
if (!isset($response_chat["result"])) {
    echo "chatocr-chat request error\n";
    print_r($response_chat);
    exit(1);
}

echo "Final result:\n";
echo json_encode($response_chat["result"]["chatResult"], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT) . "\n";


function send_post_raw($url, $data) {
    $json_str = json_encode($data, JSON_UNESCAPED_UNICODE);
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $json_str);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    $response = curl_exec($ch);
    if ($response === false) {
        echo "cURL error: " . curl_error($ch) . "\n";
    }
    curl_close($ch);
    return $response;
}
?&gt;
</code></pre></details>
</details>
<br/>

## 4. Custom Development
If the default model weights provided by the PP-ChatOCRv4 pipeline do not meet your requirements in terms of accuracy or speed, you can try to fine-tune the existing model using your own domain-specific or application-specific data to improve the recognition performance of the PP-ChatOCRv4 pipeline in your scenario.


### 4.1 Model Fine-Tuning
Since the PP-ChatOCRv4 pipeline includes several modules, the unsatisfactory performance of the pipeline may originate from any one of these modules. You can analyze the cases with poor extraction results, identify which module is problematic through visualizing the images, and refer to the corresponding fine-tuning tutorial links in the table below to fine-tune the model.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-tuning Module</th>
<th>Fine-tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate layout region detection, such as missed detection of seals, tables, etc.</td>
<td>Layout Region Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate table structure recognition</td>
<td>Table Structure Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/table_structure_recognition.html#4-secondary-development">Link</a></td>
</tr>
<tr>
<td>Missed detection of seal text</td>
<td>Seal Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/seal_text_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missed detection of text</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_detection.html#4-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html#41-dataset-and-pre-trained-model-preparation">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of vertical or rotated text lines</td>
<td>Text Line Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of whole-image rotation</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of image distortion</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After you complete fine-tuning with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the production configuration file by replacing the local directory of the fine-tuned model weights to the corresponding position in the production configuration file:

1. Exporting Pipeline Configuration Files

You can call the `export_paddlex_config_to_yaml` method of the pipeline object to export the current pipeline configuration to a YAML file. Here is an example:

```Python
from paddleocr import PPChatOCRv4Doc

pipeline = PPChatOCRv4Doc()
pipeline.export_paddlex_config_to_yaml("PP-ChatOCRv4-doc.yaml")
```

2. Editing Pipeline Configuration Files

Replace the local directory of the fine-tuned model weights to the corresponding position in the pipeline configuration file. For example:

```yaml
......
SubModules:
    TextDetection:
    module_name: text_detection
    model_name: PP-OCRv5_server_det
    model_dir: null # Replace with the fine-tuned text detection model weights directory
    limit_side_len: 960
    limit_type: max
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 1.5

    TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_server_rec
    model_dir: null # Replace with the fine-tuned text recognition model weights directory
        batch_size: 1
    batch_size: 1
            score_thresh: 0
......
```

The exported PaddleX pipeline configuration file not only includes parameters supported by PaddleOCR's CLI and Python API but also allows for more advanced settings. Please refer to the corresponding pipeline usage tutorials in [PaddleX Pipeline Usage Overview](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/pipeline_develop_guide.html) for detailed instructions on adjusting various configurations according to your needs.


3. Loading Pipeline Configuration Files in CLI

By specifying the path to the PaddleX pipeline configuration file using the `--paddlex_config` parameter, PaddleOCR will read its contents as the configuration for inference. Here is an example:

```bash
paddleocr pp_chatocrv4_doc --paddlex_config PP-ChatOCRv4-doc.yaml ...
```

4. Loading Pipeline Configuration Files in Python API

When initializing the pipeline object, you can pass the path to the PaddleX pipeline configuration file or a configuration dictionary through the `paddlex_config` parameter, and PaddleOCR will use it as the configuration for inference. Here is an example:

```python
from paddleocr import PPChatOCRv4Doc

pipeline = PPChatOCRv4Doc(paddlex_config="PP-ChatOCRv4-doc.yaml")
```
