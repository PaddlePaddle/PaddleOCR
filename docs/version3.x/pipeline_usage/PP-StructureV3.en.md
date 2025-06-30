---
comments: true
---

# PP-StructureV3 Pipeline Usage Tutorial

## 1. Introduction to PP-StructureV3 Pipeline

Layout analysis is a technique used to extract structured information from document images. It is primarily used to convert complex document layouts into machine-readable data formats. This technology has broad applications in document management, information extraction, and data digitization. Layout analysis combines Optical Character Recognition (OCR), image processing, and machine learning algorithms to identify and extract text blocks, titles, paragraphs, images, tables, and other layout elements from documents. This process generally includes three main steps: layout analysis, element analysis, and data formatting. The final result is structured document data, which enhances the efficiency and accuracy of data processing. <b>PP-StructureV3 improves upon the general layout analysis v1 pipeline by enhancing layout region detection, table recognition, and formula recognition. It also adds capabilities such as multi-column reading order recovery, chart understanding, and result conversion to Markdown files. It performs excellently across various document types and can handle complex document data.</b>  This pipeline also provides flexible service deployment options, supporting invocation using multiple programming languages on various hardware. In addition, it offers secondary development capabilities, allowing you to train and fine-tune models on your own dataset and integrate the trained models seamlessly.

<b>The PP-StructureV3 pipeline consists of the following six modules or sub-pipelines. Each module or sub-pipeline can be trained and inferred independently and contains multiple models. For more details, please click the corresponding links to view the documentation.</b>

- [Layout Detection Module](../module_usage/layout_detection.en.md)
- [General OCR Subline](./OCR.en.md)
- [Document Image Preprocessing Subline](./doc_preprocessor.en.md) （Optional）
- [Table Recognition Subline ](./table_recognition_v2.en.md) （Optional）
- [Seal Text Recognition Subline](./seal_recognition.en.md) （Optional）
- [Formula Recognition Subline](./formula_recognition.en.md) （Optional）

In this pipeline, you can choose the model to use based on the benchmark data below.

<details>
<summary><b>Document Image Orientation Classification Module :</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Pretrained Model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>Document image classification model based on PP-LCNet_x1_0, supporting four categories: 0°, 90°, 180°, 270°</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Image Rectification Module:</b></summary>
<p><b>Text Image Rectification Module (Optional):</b></p>
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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Pretrained Model</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>High-precision text image rectification model</td>
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
<summary><b>Table Structure Recognition Module (Optional):</b></summary>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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

<p><b>Table Classification Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
<td>94.2</td>
<td>2.62 / 0.60</td>
<td>3.17 / 1.14</td>
<td>6.6</td>
</tr>
</table>
<p><b>Table Cell Detection Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">33.47 / 27.02</td>
<td rowspan="2">402.55 / 256.56</td>
<td rowspan="2">124</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The Baidu PaddlePaddle vision team based RT-DETR-L as the base model, completing pre-training on a self-built table cell detection dataset, achieving good performance in detecting both wired and wireless table cells.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
</details>

<details>
<summary><b>Text Detection Module (Required):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
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
<td>82.56</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td>The server-side text detection model of PP-OCRv4, with higher accuracy, suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td>The mobile text detection model of PP-OCRv4, with higher efficiency, suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Recognition Module Model (Required):</b></summary>


<details><summary> 👉Full Model List</summary>

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
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>80.5</td>
<td>SVTRv2 is a server-side recognition model developed by the OpenOCR team at Fudan University’s FVL Lab. It won first place in the OCR End-to-End Recognition task of the PaddleOCR Model Challenge, improving end-to-end accuracy on Benchmark A by 6% compared to PP-OCRv4.</td>
</tr>
</table>

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
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>48.8</td>
<td>RepSVTR is a mobile text recognition model based on SVTRv2. It won first place in the OCR End-to-End Recognition task of the PaddleOCR Model Challenge, improving accuracy on Benchmark B by 2.5% over PP-OCRv4 with comparable inference speed.</td>
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
</details>

<details>
<summary><b>Text Line Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th>
<th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Pretrained Model</a></td>
<td>98.85</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.96</td>
<td>A text line classification model based on PP-LCNet_x0_25, containing two categories: 0 degrees and 180 degrees</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Formula Recognition Module (Optional):</b></summary>
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
<summary><b>Seal Text Detection Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Pretrained Model</a></td>
<td>98.40</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td>Server-side seal text detection model based on PP-OCRv4, offering higher accuracy and suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Pretrained Model</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td>Mobile-side seal text detection model based on PP-OCRv4, offering higher efficiency and suitable for edge-side deployment</td>
</tr>
</tbody>
</table>
</details>
</details>

<details>
<summary><b>Chart Parsing Module: </b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Model parameter size（B）</th>
<th>Model Storage Size (GB)</th>
<th>Model Score </th>
<th>Description</th>
</tr>
<tr>
<td>PP-Chart2Table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.tar">Inference Model</a></td>
<td>0.58</td>
<td>1.4</td>
<th>75.98</th>
<td>PP-Chart2Table is a self-developed multimodal model by the PaddlePaddle team, focusing on chart parsing, demonstrating outstanding performance in both Chinese and English chart parsing tasks. The team adopted a carefully designed data generation strategy, constructing a high-quality multimodal dataset of nearly 700,000 entries covering common chart types like pie charts, bar charts, stacked area charts, and various application scenarios. They also designed a two-stage training method, utilizing large model distillation to fully leverage massive unlabeled OOD data. In internal business tests in both Chinese and English scenarios, PP-Chart2Table not only achieved the SOTA level among models of the same parameter scale but also reached accuracy comparable to 7B parameter scale VLM models in critical scenarios.</td>
</tr>
</table>
</details>

<details>
<summary><b>Test Environment Description:</b></summary>
  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
          <li><strong>Test Dataset: </strong>
                        <ul>
                         <li>Document Image Orientation Classification Module: A self-built dataset using PaddleX, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Region Detection Model: A self-built layout detection dataset using PaddleOCR, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Table Structure Recognition Model: A self-built English table recognition dataset using PaddleX.</li>
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a>.</li>
                          <li>English Recognition Model: A self-built English dataset using PaddleX.</li>
                          <li>Multilingual Recognition Model: A self-built multilingual dataset using PaddleX.</li>
                          <li>Text Line Orientation Classification Model: A self-built dataset using PaddleX, covering various scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Seal Text Detection Model: A self-built dataset using PaddleX, containing 500 images of circular seal textures.</li>
                        </ul>
                </li>
              <li><strong>Hardware Configuration: </strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environments: Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
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

## 2. Quick Start

Before using the PP-StructureV3 pipeline locally, please make sure you have completed the installation of the wheel package according to the [installation guide](../installation.en.md). After installation, you can use it via command line or Python integration.

Please note: If you encounter issues such as the program becoming unresponsive, unexpected program termination, running out of memory resources, or extremely slow inference during execution, please try adjusting the configuration according to the documentation, such as disabling unnecessary features or using lighter-weight models.

### 2.1 Command Line Usage

Use a single command to quickly experience the PP-StructureV3 pipeline:

```bash
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png

# Use --use_doc_orientation_classify to enable document orientation classification
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_doc_orientation_classify True

# Use --use_doc_unwarping to enable document unwarping module
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_doc_unwarping True

# Use --use_textline_orientation to enable text line orientation classification
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_textline_orientation False

# Use --device to specify GPU for inference
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --device gpu
```

<details><summary><b>Command line supports more parameters. Click to expand for detailed parameter descriptions</b></summary>
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
<td>Data to be predicted. Required.
.e.g., local path to image or PDF file: <code>/root/data/img.jpg</code>; <b>URL</b>, e.g., online image or PDF: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">example</a>; <b>local directory</b>: directory containing images to predict, e.g., <code>/root/data/</code> (currently, directories with PDFs are not supported; PDFs must be specified by file path).
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Path to save inference results. If not set, results will not be saved locally.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout detection model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
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
<td>Whether to use Non-Maximum Suppression (NMS) as post-processing for layout detection. If not set, the parameter will default to the value initialized in the pipeline, which is set to <code>True</code> by default.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Unclip ratio for detected boxes in layout detection model. Any float > <code>0</code>. If not set, the default is <code>1.0</code>.
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The merging mode for the detection boxes output by the model in layout detection.
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
<td><code>chart_recognition_model_name</code></td>
<td>Name of the chart parsing model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>Directory path of the chart parsing model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>Batch size for the chart parsing model. If not set, the default batch size is <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>Name of the region detection model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>Directory path of the region detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the document unwarping model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the document unwarping model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If not set, the default model will be used.</td>
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
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limitation for text detection. Any integer > <code>0</code>. If not set, the default value will be <code>960</code>.
</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of the image side length limit for text detection.
supports <code>min</code> and <code>max</code>; <code>min</code> means ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> means the longest side does not exceed <code>limit_side_len</code>. If not set, the default value will be <code>max</code>.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels with scores above this value in the probability map are considered text.Any float > <code>0</code>
. If not set, the default is <code>0.3</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Box threshold. A bounding box is considered text if the average score of pixels inside is greater than this value.
Any float > <code>0</code>. If not set, the default is <code>0.6</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Expansion ratio for text detection. The higher the value, the larger the expansion area.
any float > <code>0</code>. If not set, the default is <code>2.0</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>Name of the text line orientation model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>Directory of the text line orientation model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>Batch size for the text line orientation model. If not set, the default is <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>Directory of the text recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>Batch size for text recognition. If not set, the default is <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Score threshold for text recognition. Only results above this value will be kept.
Any float > <code>0</code>. If not set, the default is <code>0.0</code> (no threshold).
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>Name of the table classification model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>Directory of the table classification model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>Name of the wired table structure recognition model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>Directory of the wired table structure recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>Name of the wireless table structure recognition model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>Directory of the wireless table structure recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>Name of the wired table cell detection model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>Directory of the wired table cell detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>Name of the wireless table cell detection model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>Directory of the wireless table cell detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_name</code></td>
<td>Name of the wireless table orientation classification model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_dir</code></td>
<td>Directory of the table orientation classification model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>Name of the seal text detection model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>Directory of the seal text detection model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
Any integer > <code>0</code>. If not set, the default is <code>736</code>.
</td>
<td><code>int</code></td>
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
any float > <code>0</code>. If not set, the default is <code>0.2</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Box threshold. Boxes with average pixel scores above this value are considered text regions.
any float > <code>0</code>. If not set, the default is <code>0.6</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. Higher value means larger expansion area.Any float > <code>0</code>. If not set, the default is <code>0.5</code>.
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>Name of the seal text recognition model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>Directory of the seal text recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>Batch size for seal text recognition. If not set, the default is <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Recognition score threshold. Text results above this value will be kept. Any float > <code>0</code>. If not set, the default is <code>0.0</code> (no threshold).
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>Name of the formula recognition model. If not set, the default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>Directory of the formula recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>Batch size of the formula recognition model. If not set, the default is <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use document orientation classification module.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use document unwarping module.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to load and use the text line orientation classification module. If not set, the default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to load and use seal text recognition subpipeline. If not set, the default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to load and use table recognition subpipeline. If not set, the default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to load and use formula recognition subpipeline. If not set, the default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart parsing module.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to load and use the document region detection pipeline. If not set, the default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device for inference. You can specify a device ID:
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> means using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> means GPU 0</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> means NPU 0</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> means XPU 0</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> means MLU 0</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> means DCU 0</li>
</ul>If not set, the pipeline initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use the Paddle Inference TensorRT subgraph engine. If the model does not support acceleration through TensorRT, setting this flag will not enable acceleration.<br/>
For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.<br/>
For Paddle with CUDA version 12.6, the compatible TensorRT version is 10.x (x>=5), and it is recommended to install TensorRT 10.5.0.18.
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computation precision, e.g., fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
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
<td>Number of threads to use when inferring on CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>Path to the PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>
<br />

The inference result will be printed in the terminal. The default output of the PP-StructureV3 pipeline is as follows:

<details><summary> 👉Click to expand</summary>
<pre>
<code>
{'res': {'input_path': '/root/.paddlex/predict_input/pp_structure_v3_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 0}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9848763942718506, 'coordinate': [743.2788696289062, 777.3158569335938, 1115.24755859375, 1067.84228515625]}, {'cls_id': 2, 'label': 'text', 'score': 0.9827454686164856, 'coordinate': [1137.95556640625, 1127.66943359375, 1524, 1367.6356201171875]}, {'cls_id': 1, 'label': 'image', 'score': 0.9813530445098877, 'coordinate': [755.2349243164062, 184.64149475097656, 1523.7294921875, 684.6146392822266]}, {'cls_id': 2, 'label': 'text', 'score': 0.980336606502533, 'coordinate': [350.7603759765625, 1148.5648193359375, 706.8020629882812, 1367.00341796875]}, {'cls_id': 2, 'label': 'text', 'score': 0.9798877239227295, 'coordinate': [1147.3890380859375, 802.6549072265625, 1523.9051513671875, 994.9046630859375]}, {'cls_id': 2, 'label': 'text', 'score': 0.9724758863449097, 'coordinate': [741.2205810546875, 1074.2657470703125, 1110.120849609375, 1191.2010498046875]}, {'cls_id': 2, 'label': 'text', 'score': 0.9724437594413757, 'coordinate': [355.6563415527344, 899.6616821289062, 710.9073486328125, 1042.1270751953125]}, {'cls_id': 2, 'label': 'text', 'score': 0.9723313450813293, 'coordinate': [0, 181.92404174804688, 334.43384313583374, 330.294677734375]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720360636711121, 'coordinate': [356.7376403808594, 753.35302734375, 714.37841796875, 892.6129760742188]}, {'cls_id': 2, 'label': 'text', 'score': 0.9711183905601501, 'coordinate': [1144.5242919921875, 1001.2548217773438, 1524, 1120.6578369140625]}, {'cls_id': 2, 'label': 'text', 'score': 0.9707457423210144, 'coordinate': [0, 849.873291015625, 325.0664693713188, 1067.2911376953125]}, {'cls_id': 2, 'label': 'text', 'score': 0.9700680375099182, 'coordinate': [363.04437255859375, 289.2635498046875, 719.1571655273438, 427.5818786621094]}, {'cls_id': 2, 'label': 'text', 'score': 0.9693533182144165, 'coordinate': [359.4466857910156, 606.0006103515625, 717.9885864257812, 746.55126953125]}, {'cls_id': 2, 'label': 'text', 'score': 0.9682930111885071, 'coordinate': [0.050221771001815796, 1073.1942138671875, 323.85799154639244, 1191.3121337890625]}, {'cls_id': 2, 'label': 'text', 'score': 0.9649553894996643, 'coordinate': [0.7939082384109497, 1198.5465087890625, 321.2581721544266, 1317.218017578125]}, {'cls_id': 2, 'label': 'text', 'score': 0.9644040465354919, 'coordinate': [0, 337.225830078125, 332.2462143301964, 428.298583984375]}, {'cls_id': 2, 'label': 'text', 'score': 0.9637495279312134, 'coordinate': [365.5925598144531, 188.2151336669922, 718.556640625, 283.7483215332031]}, {'cls_id': 2, 'label': 'text', 'score': 0.9603620767593384, 'coordinate': [355.30633544921875, 1048.5457763671875, 708.771484375, 1141.828369140625]}, {'cls_id': 2, 'label': 'text', 'score': 0.9508902430534363, 'coordinate': [361.0450744628906, 530.7780151367188, 719.6325073242188, 599.1027221679688]}, {'cls_id': 2, 'label': 'text', 'score': 0.9459834694862366, 'coordinate': [0.035085976123809814, 532.7417602539062, 330.5401824116707, 772.7175903320312]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9400503635406494, 'coordinate': [760.1524658203125, 1214.560791015625, 1085.24853515625, 1274.7890625]}, {'cls_id': 2, 'label': 'text', 'score': 0.9341079592704773, 'coordinate': [1.025873064994812, 777.8804931640625, 326.99016749858856, 844.8532104492188]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9259933233261108, 'coordinate': [0.11050379276275635, 450.3547058105469, 311.77746546268463, 510.5243835449219]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9208691716194153, 'coordinate': [380.79510498046875, 447.859130859375, 698.1744384765625, 509.0489807128906]}, {'cls_id': 2, 'label': 'text', 'score': 0.8683002591133118, 'coordinate': [1149.1656494140625, 778.3809814453125, 1339.960205078125, 796.5060424804688]}, {'cls_id': 2, 'label': 'text', 'score': 0.8455104231834412, 'coordinate': [561.3448486328125, 140.87547302246094, 915.4432983398438, 162.76724243164062]}, {'cls_id': 11, 'label': 'doc_title', 'score': 0.735536515712738, 'coordinate': [76.71978759765625, 0, 1400.4561157226562, 98.32131713628769]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.7187536954879761, 'coordinate': [790.4249267578125, 704.4551391601562, 1509.9013671875, 747.6876831054688]}, {'cls_id': 2, 'label': 'text', 'score': 0.6218013167381287, 'coordinate': [737.427001953125, 1296.2047119140625, 1104.2994384765625, 1368]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': True}, 'dt_polys': array([[[  77,    0],
        ...,
        [  76,   98]],

       ...,

       [[1142, 1350],
        ...,
        [1142, 1367]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([0, ..., 0]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者沈小晓任彦', '黄培照', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '以下简称"厄特孔院")举办“喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年竣工，建成后将为厄', '益深厚。', '特孔院提供全新的办学场地。', '学好中文，我们的', '□', '在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '中的每一个角落"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '昌边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新：“中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起，我们欢迎你…"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜禾', '小的仅有6岁。"尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和”“禅”“山"等汉字。“这件文物证', '交的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明，很早以前我们就通过海上丝绸之路进行', '中文，在2017年第十届“汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。”北红海', '半代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '本优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话：“这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '软曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？"“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。”', '软善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。”', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·', '亚14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发用', '露娅对记者说：“这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '印象深刻。“中国博物馆不仅有许多保存完好', '“共同向世界展示非', '中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥”比赛中获得一等奖。莉迪亚说：“学', '的文物，还充分运用先进科技手段进行展示', '一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，“', '了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰，希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终木', '学好中文，我们的未来不是梦！”', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜒曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '中贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99875408, ..., 0.98324996]), 'rec_polys': array([[[  77,    0],
        ...,
        [  76,   98]],

       ...,

       [[1142, 1350],
        ...,
        [1142, 1367]]], dtype=int16), 'rec_boxes': array([[  76, ...,  103],
       ...,
       [1142, ..., 1367]], dtype=int16)}}}
</code></pre></details>

For explanation of the result parameters, refer to [2.2 Python Script Integration](#222-python-script-integration).

<b>Note:</b> Due to the large size of the default model in the pipeline, the inference speed may be slow. You can refer to the model list in Section 1 to replace it with a faster model.

### 2.2 Python Script Integration

The command line method is for quick testing and visualization. In actual projects, you usually need to integrate the model via code. You can perform pipeline inference with just a few lines of code as shown below:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
# ocr = PPStructureV3(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# ocr = PPStructureV3(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# ocr = PPStructureV3(use_textline_orientation=True) # Use use_textline_orientation to enable/disable textline orientation classification model
# ocr = PPStructureV3(device="gpu") # Use device to specify GPU for model inference
output = pipeline.predict("./pp_structure_v3_demo.png")
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

For PDF files, each page will be processed individually and generate a separate Markdown file. If you want to convert the entire PDF to a single Markdown file, use the following method:

```python
from pathlib import Path
from paddleocr import PPStructureV3

input_file = "./your_pdf_file.pdf"
output_path = Path("./output")

pipeline = PPStructureV3()
output = pipeline.predict(input=input_file)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)

```

**Note:**

- The default text recognition model used by PP-StructureV3 is a **Chinese-English recognition model**, which has limited accuracy for purely English texts. For English-only scenarios, you can set the `text_recognition_model_name` parameter to an English model such as `en_PP-OCRv4_mobile_rec` to achieve better recognition performance. For other languages, refer to the model list above and select the appropriate language recognition model for replacement.

- In the example code, the parameters `use_doc_orientation_classify`, `use_doc_unwarping`, and `use_textline_orientation` are all set to `False` by default. These indicate that document orientation classification, document image unwarping, and textline orientation classification are disabled. You can manually set them to `True` if needed.

The above Python script performs the following steps:

<details><summary>(1) Instantiate <code>PPStructureV3</code> to create the pipeline object. The parameter descriptions are as follows:</summary>

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
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model.
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
<td>Whether to use Non-Maximum Suppression (NMS) as post-processing for layout detection. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is set to <code>True</code> by default.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion ratio for the bounding boxes from the layout detection model.
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
<td>Filtering method for overlapping boxes in layout detection.
<ul>
<li><b>str</b>: Options include <code>large</code>, <code>small</code>, and <code>union</code> to retain the larger box, smaller box, or both;</li>
<li><b>dict</b>: A dictionary with <b>int</b> keys representing <code>cls_id</code>, and <b>str</b> values, e.g., <code>{0: "large", 2: "small"}</code> means using different modes for different classes;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value <code>large</code>.</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td>Name of the chart parsing model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>Directory path of the chart parsing model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>Batch size for the chart parsing model. If set to <code>None</code>, the default is <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>Name of the region detection model for sub-modules in document layout. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>Directory path of the region detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the document unwarping model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the document unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>Directory path of the text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limitation for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>960</code>.</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side is no less than <code>det_limit_side_len</code>, while <code>max</code> ensures the longest side is no greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>max</code>.</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels in the output probability map with scores above this value are considered as text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value of <code>0.3</code>.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Bounding box threshold. If the average score of all pixels inside the box exceeds this threshold, it is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value of <code>0.6</code>.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Expansion ratio for text detection. The larger the value, the more the text region is expanded.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value of <code>2.0</code>.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>Name of the textline orientation model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>Directory path of the textline orientation model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>Batch size for the textline orientation model. If set to <code>None</code>, the default batch size is <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>Directory path of the text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size is <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Score threshold for text recognition. Only results with scores above this threshold will be retained.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>0.0</code> (no threshold).</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>Name of the table classification model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>Directory path of the table classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>Name of the wired table structure recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>Directory path of the wired table structure recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>Name of the wireless table structure recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>Directory path of the wireless table structure recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>Name of the wired table cell detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>Directory path of the wired table cell detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>Name of the wireless table cell detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>Directory path of the wireless table cell detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_name</code></td>
<td>Name of the wireless table orientation classification model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_dir</code></td>
<td>Directory of the table orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>Name of the seal text detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>Directory path of the seal text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>736</code>.</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Limit type for seal text detection image side length.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side is no less than <code>det_limit_side_len</code>, while <code>max</code> ensures the longest side is no greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>min</code>.</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels with scores greater than this value in the probability map are considered text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.2</code>.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Bounding box threshold. If the average score of all pixels inside a detection box exceeds this threshold, it is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.6</code>.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. The larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.5</code>.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>Name of the seal text recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>Directory path of the seal text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>Batch size for the seal text recognition model. If set to <code>None</code>, the default value is <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Score threshold for seal text recognition. Text results with scores above this threshold will be retained.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.0</code> (no threshold).</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>Name of the formula recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>Directory path of the formula recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>Batch size for the formula recognition model. If set to <code>None</code>, the default value is <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to enable the document orientation classification module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to enable the document image unwarping module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to enable seal text recognition subpipeline. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to enable table recognition subpipeline. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to enable formula recognition subpipeline. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to load and use the chart parsing module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to use the document region detection pipeline. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device used for inference. Supports specifying device ID:
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> means using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> means using GPU 0;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> means using NPU 0;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> means using XPU 0;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> means using MLU 0;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> means using DCU 0;</li>
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
For Paddle with CUDA version 12.6, the compatible TensorRT version is 10.x (x>=5), and it is recommended to install TensorRT 10.5.0.18.
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
<td>Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.</td>
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
<td>Number of threads used for inference on CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>Path to the PaddleX pipeline configuration file.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>(2) Call the <code>predict()</code> method of the PP-StructureV3 pipeline object for inference. This method returns a result list. The pipeline also provides a <code>predict_iter()</code> method. Both methods accept the same parameters and return the same type of results. The only difference is that <code>predict_iter()</code> returns a <code>generator</code> that allows incremental processing and retrieval of prediction results, which is useful for handling large datasets or saving memory. Choose the method that fits your needs. Below are the parameters of the <code>predict()</code> method:</summary>

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
<td>Input data to be predicted. Required. Supports multiple types:
<ul>
<li><b>Python Var</b>: Image data represented as <code>numpy.ndarray</code>;</li>
<li><b>str</b>: Local path to image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL</b> to image or PDF, e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">example</a>; <b>directory</b> containing image files, e.g., <code>/root/data/</code> (directories with PDFs are not supported, use full file path for PDFs);</li>
<li><b>list</b>: Elements can be any of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"].</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use document orientation classification during inference. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use document image unwarping during inference. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use textline orientation classification during inference. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal text recognition sub-pipeline during inference. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-pipeline during inference. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to use the formula recognition sub-pipeline during inference. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart parsing module. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to use the document region detection pipeline. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
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
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>Whether to enable direct conversion of wired table cell detection results to HTML. If enabled, HTML will be constructed directly based on the geometric relationship of wired table cell detection results.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_wireless_table_cells_trans_to_html</code></td>
<td>Whether to enable direct conversion of wireless table cell detection results to HTML. If enabled, HTML will be constructed directly based on the geometric relationship of wireless table cell detection results.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td>Whether to enable table orientation classification. When enabled, it can correct the orientation and correctly complete table recognition if the table in the image is rotated by 90/180/270 degrees.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td>Whether to enable OCR within cell segmentation. When enabled, OCR detection results will be segmented and re-recognized based on cell prediction results to avoid text loss.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td>Whether to enable end-to-end wired table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td>Whether to enable end-to-end wireless table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</table>
</details>

<details><summary>(3) Process the prediction results: each prediction result corresponds to a Result object, which supports printing, saving as image, or saving as a <code>json</code> file:</summary>

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
<td>Whether to format output as indented <code>JSON</code>.</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level to beautify the <code>JSON</code> output. Only effective when <code>format_json=True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When <code>True</code>, all non-ASCII characters are escaped. When <code>False</code>, original characters are retained. Only effective when <code>format_json=True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If a directory, the filename will be based on the input type.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level for beautified <code>JSON</code> output. Only effective when <code>format_json=True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. Only effective when <code>format_json=True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save intermediate visualization results as PNG image files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>Save each page of an image or PDF file as a markdown file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save tables in the file as HTML format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save tables in the file as XLSX format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>concatenate_markdown_pages()</code></td>
<td>Concatenate multiple markdown pages into a single document</td>
<td><code>markdown_list</code></td>
<td><code>list</code></td>
<td>List of markdown data for each page.</td>
<td>Returns the merged markdown text and image list.</td>
</tr>
</table>

- Calling `print()` will print the result to the terminal. Explanation of the printed content:
    - `input_path`: `(str)` Input path of the image or PDF to be predicted

    - `page_index`: `(Union[int, None])` If input is a PDF, indicates the page number; otherwise `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters configured for the pipeline

        - `use_doc_preprocessor`: `(bool)` Whether to enable document preprocessor sub-pipeline
        - `use_seal_recognition`: `(bool)` Whether to enable seal text recognition sub-pipeline
        - `use_table_recognition`: `(bool)` Whether to enable table recognition sub-pipeline
        - `use_formula_recognition`: `(bool)` Whether to enable formula recognition sub-pipeline

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` Document preprocessing result dictionary, only exists if `use_doc_preprocessor=True`
        - `input_path`: `(str)` Image path accepted by document preprocessor, `None` if input is `numpy.ndarray`
        - `page_index`: `None` since input is `numpy.ndarray`
        - `model_settings`: `(Dict[str, bool])` Model configuration for the document preprocessor
          - `use_doc_orientation_classify`: `(bool)` Whether to enable document orientation classification
          - `use_doc_unwarping`: `(bool)` Whether to enable image unwarping
        - `angle`: `(int)` Predicted angle result if orientation classification is enabled

    - `parsing_res_list`: `(List[Dict])` List of parsed results, each item is a dictionary in reading order
        - `block_bbox`: `(np.ndarray)` Bounding box of the layout block
        - `block_label`: `(str)` Block label such as `text`, `table`
        - `block_content`: `(str)` Content within the layout block
        - `seg_start_flag`: `(bool)` Whether the block starts a paragraph
        - `seg_end_flag`: `(bool)` Whether the block ends a paragraph
        - `sub_label`: `(str)` Sub-label of the block, e.g., `title_text`
        - `sub_index`: `(int)` Sub-index of the block, used for markdown reconstruction
        - `index`: `(int)` Index of the block, used for layout sorting






    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` Dictionary of global OCR results
      - `input_path`: `(Union[str, None])` OCR sub-pipeline input path; `None` if input is `numpy.ndarray`
      - `page_index`: `None` since input is `numpy.ndarray`
      - `model_settings`: `(Dict)` OCR model configuration
      - `dt_polys`: `(List[numpy.ndarray])` List of polygons for text detection. Each box is a numpy array with shape (4, 2), dtype int16
      - `dt_scores`: `(List[float])` Confidence scores for detection boxes
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Text detection module parameters
        - `limit_side_len`: `(int)` Side length limit for image preprocessing
        - `limit_type`: `(str)` Limit processing method
        - `thresh`: `(float)` Threshold for text pixel classification
        - `box_thresh`: `(float)` Threshold for text detection boxes
        - `unclip_ratio`: `(float)` Unclip ratio for expanding boxes
        - `text_type`: `(str)` Text detection type, currently fixed as "general"

      - `text_type`: `(str)` Text detection type, currently fixed as "general"
      - `textline_orientation_angles`: `(List[int])` Orientation classification results for text lines
      - `text_rec_score_thresh`: `(float)` Threshold for text recognition filtering
      - `rec_texts`: `(List[str])` Recognized texts filtered by score threshold
      - `rec_scores`: `(List[float])` Recognition scores filtered by threshold
      - `rec_polys`: `(List[numpy.ndarray])` Filtered detection boxes, same format as `dt_polys`

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of formula recognition results
        - `rec_formula`: `(str)` Recognized formula string
        - `rec_polys`: `(numpy.ndarray)` Bounding box for the formula, shape (4, 2), dtype int16
        - `formula_region_id`: `(int)` Region ID of the formula

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of seal text recognition results
        - `input_path`: `(str)` Input path for the seal image
        - `page_index`: `None` since input is `numpy.ndarray`
        - `model_settings`: `(Dict)` Model configuration for seal text recognition
        - `dt_polys`: `(List[numpy.ndarray])` Seal detection boxes, same format as `dt_polys`
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Detection parameters, same as above
        - `text_type`: `(str)` Detection type, currently fixed as "seal"
        - `text_rec_score_thresh`: `(float)` Score threshold for recognition
        - `rec_texts`: `(List[str])` Recognized texts filtered by score
        - `rec_scores`: `(List[float])` Recognition scores filtered by threshold
        - `rec_polys`: `(List[numpy.ndarray])` Filtered seal boxes, same format as `dt_polys`
        - `rec_boxes`: `(numpy.ndarray)` Rectangle boxes, shape (n, 4), dtype int16

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of table recognition results
        - `cell_box_list`: `(List[numpy.ndarray])` Bounding boxes of table cells
        - `pred_html`: `(str)` Table as an HTML string
        - `table_ocr_pred`: `(dict)` OCR results for the table
            - `rec_polys`: `(List[numpy.ndarray])` Detected cell boxes
            - `rec_texts`: `(List[str])` Recognized texts for cells
            - `rec_scores`: `(List[float])` Confidence scores for cell recognition
            - `rec_boxes`: `(numpy.ndarray)` Rectangle boxes for detection, shape (n, 4), dtype int16

- Calling `save_to_json()` saves the above content to the specified `save_path`. If it’s a directory, the saved path will be `save_path/{your_img_basename}_res.json`. If it’s a file, it saves directly. Numpy arrays are converted to lists since JSON doesn't support them.
- Calling `save_to_img()` saves visual results to the specified `save_path`. If a directory, various visualizations such as layout detection, OCR, and reading order are saved. If a file, only the last image is saved and others are overwritten.
- Calling `save_to_markdown()` saves converted markdown files to `save_path/{your_img_basename}.md`. For PDF input, it's recommended to specify a directory to avoid file overwriting.
- Calling `concatenate_markdown_pages()` merges multi-page markdown results from the `PP-StructureV3 pipeline` into a single document and returns the merged content.

Additionally, you can access the prediction results and visual images through the following attributes:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>json</code></td>
<td>Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get visualized image results as a <code>dict</code></td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"><code>markdown</code></td>
<td rowspan="3">Get markdown results as a <code>dict</code></td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>
- The `json` attribute returns the prediction result as a dictionary, which is consistent with the content saved using the `save_to_json()` method.
- The `img` attribute returns the prediction result as a dictionary. The keys include `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, each corresponding to a visualized `Image.Image` object for layout detection, OCR, text paragraph, formula, table, and seal results. If optional modules are not used, the dictionary only contains `layout_det_res`.
- The `markdown` attribute returns the prediction result as a dictionary. The keys include `markdown_texts`, `markdown_images`, and `page_continuation_flags`, where the values represent the markdown text, displayed images (`Image.Image` objects), and a boolean tuple indicating whether the first and last elements of the current page are paragraph boundaries.

</details>

## 3. Development Integration / Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration or deployment.

If you want to directly use the pipeline in your Python project, refer to the example code in [2.2 Python script mode](#22-python脚本方式集成).

In addition, PaddleOCR provides two other deployment options described in detail below:

🚀 High-Performance Inference: In production environments, many applications have strict performance requirements (especially response speed) to ensure system efficiency and smooth user experience. PaddleOCR offers a high-performance inference option that deeply optimizes model inference and pre/post-processing for significant end-to-end acceleration. For detailed high-performance inference workflow, refer to [High Performance Inference](../deployment/high_performance_inference.en.md).

☁️ Service Deployment: Service-based deployment is common in production. It encapsulates the inference logic as a service, allowing clients to access it via network requests to obtain results. For detailed instructions on service deployment, refer to [Service Deployment](../deployment/serving.en.md).

Below is the API reference and multi-language service invocation examples for basic service deployment:

<details><summary>API Reference</summary>
<p>Main operations provided by the service:</p>
<ul>
<li>HTTP method: POST</li>
<li>Request and response bodies are both JSON objects.</li>
<li>When the request is successful, the response status code is <code>200</code>, and the response body contains:</li>
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
<td>UUID of the request</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code, fixed to <code>0</code></td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message, fixed to <code>"Success"</code></td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request fails, the response body includes:</li>
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
<td>UUID of the request</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code, same as HTTP status code</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message</td>
</tr>
</tbody>
</table>
<p>Main operation provided:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform layout parsing.</p>
<p><code>POST /layout-parsing</code></p>
<ul>
<li>Request body parameters:</li>
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
<td>URL of image or PDF file accessible to the server, or base64-encoded file content. By default, only the first 10 pages of a PDF are processed.<br />To remove this limit, add the following to the pipeline config:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre>
</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code>｜<code>null</code></td>
<td>File type. <code>0</code> for PDF, <code>1</code> for image. If omitted, the type is inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_doc_orientation_classify</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_doc_unwarping</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_textline_orientation</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_seal_recognition</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_table_recognition</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_formula_recognition</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>layout_threshold</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>layout_nms</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>Refer to the <code>layout_unclip_ratio</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>Refer to the <code>layout_merge_bboxes_mode</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Refer to the <code>text_det_limit_side_len</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refer to the <code>text_det_limit_type</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_thresh</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_box_thresh</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_unclip_ratio</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_rec_score_thresh</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Refer to the <code>seal_det_limit_side_len</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refer to the <code>seal_det_limit_type</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>seal_det_thresh</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>seal_det_box_thresh</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>seal_det_unclip_ratio</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>seal_rec_score_thresh</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableCellsOcrResults</code></td>
<td><code>boolean</code></td>
<td>Refer to the <code>use_table_cells_ocr_results</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Refer to the <code>use_e2e_wired_table_rec_model</code> parameter in the pipeline’s <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Refer to the <code>use_e2e_wireless_table_rec_model</code> parameter in the pipeline’s <code>predict</code> method.</td>
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
<li>When the request is successful, the <code>result</code> field of the response contains the following attributes:</li>
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
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>Layout parsing results. The array length is 1 (for image input) or the number of processed pages (for PDF input). For PDF input, each element corresponds to one processed page.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>layoutParsingResults</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field from the JSON output of the pipeline’s <code>predict</code> method, with <code>input_path</code> and <code>page_index</code> removed.</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>Markdown result.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Refer to the pipeline’s <code>img</code> attribute. Images are JPEG encoded in Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Input image. JPEG encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>The <code>markdown</code> object has the following attributes:</p>
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
<td><code>text</code></td>
<td><code>string</code></td>
<td>Markdown text.</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>Key-value pairs of image relative paths and Base64-encoded image content.</td>
</tr>
<tr>
<td><code>isStart</code></td>
<td><code>boolean</code></td>
<td>Whether the first element on the current page is the start of a paragraph.</td>
</tr>
<tr>
<td><code>isEnd</code></td>
<td><code>boolean</code></td>
<td>Whether the last element on the current page is the end of a paragraph.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-language Service Call Examples</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests
import pathlib

API_URL = "http://localhost:8080/layout-parsing" # Service URL

image_path = "./demo.jpg"

# Encode the local image with Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64-encoded file content or file URL
    "fileType": 1, # file type, 1 represents image file
}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
print("\nDetected layout elements:")
for i, res in enumerate(result["layoutParsingResults"]):
    print(res["prunedResult"])
    md_dir = pathlib.Path(f"markdown_{i}")
    md_dir.mkdir(exist_ok=True)
    (md_dir / "doc.md").write_text(res["markdown"]["text"])
    for img_path, img in res["markdown"]["images"].items():
        img_path = md_dir / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base64.b64decode(img))
    print(f"Markdown document saved at {md_dir / 'doc.md'}")
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost", 8080);

    const std::string filePath = "./demo.jpg";

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::string bufferStr(buffer.data(), static_cast<size_t>(size));
    std::string encodedFile = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/layout-parsing", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result.contains("layoutParsingResults")) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        const auto& results = result["layoutParsingResults"];
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];

            if (res.contains("prunedResult")) {
                std::cout << "Layout result [" << i << "]: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImages") && res["outputImages"].is_object()) {
                for (auto& [imgName, imgBase64] : res["outputImages"].items()) {
                    std::string outputPath = imgName + "_" + std::to_string(i) + ".jpg";
                    std::string decodedImage = base64::from_base64(imgBase64.get<std::string>());

                    std::ofstream outFile(outputPath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(decodedImage.c_str(), decodedImage.size());
                        outFile.close();
                        std::cout << "Saved image: " << outputPath << std::endl;
                    } else {
                        std::cerr << "Failed to save image: " << outputPath << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "Request failed." << std::endl;
        if (response) {
            std::cerr << "HTTP status: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/layout-parsing";
        String imagePath = "./demo.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String base64Image = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("file", base64Image);
        payload.put("fileType", 1);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(JSON, payload.toString());

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode root = objectMapper.readTree(responseBody);
                JsonNode result = root.get("result");

                JsonNode layoutParsingResults = result.get("layoutParsingResults");
                for (int i = 0; i < layoutParsingResults.size(); i++) {
                    JsonNode item = layoutParsingResults.get(i);
                    int finalI = i;
                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    JsonNode outputImages = item.get("outputImages");
                    outputImages.fieldNames().forEachRemaining(imgName -> {
                        try {
                            String imgBase64 = outputImages.get(imgName).asText();
                            byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                            String imgPath = imgName + "_" + finalI + ".jpg";
                            try (FileOutputStream fos = new FileOutputStream(imgPath)) {
                                fos.write(imgBytes);
                                System.out.println("Saved image: " + imgPath);
                            }
                        } catch (IOException e) {
                            System.err.println("Failed to save image: " + e.getMessage());
                        }
                    });
                }
            } else {
                System.err.println("Request failed with HTTP code: " + response.code());
            }
        }
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
    "path/filepath"
)

func main() {
    API_URL := "http://localhost:8080/layout-parsing"
    filePath := "./demo.jpg"

    fileBytes, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fileData := base64.StdEncoding.EncodeToString(fileBytes)

    payload := map[string]interface{}{
        "file":     fileData,
        "fileType": 1,
    }
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Printf("Error marshaling payload: %v\n", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Printf("Error creating request: %v\n", err)
        return
    }
    req.Header.Set("Content-Type", "application/json")

    res, err := client.Do(req)
    if err != nil {
        fmt.Printf("Error sending request: %v\n", err)
        return
    }
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        fmt.Printf("Unexpected status code: %d\n", res.StatusCode)
        return
    }

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Printf("Error reading response: %v\n", err)
        return
    }

    type Markdown struct {
        Text   string            `json:"text"`
        Images map[string]string `json:"images"`
    }

    type LayoutResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        Markdown     Markdown               `json:"markdown"`
        OutputImages map[string]string      `json:"outputImages"`
        InputImage   *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            LayoutParsingResults []LayoutResult `json:"layoutParsingResults"`
            DataInfo             interface{}    `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error parsing response: %v\n", err)
        return
    }

    for i, res := range respData.Result.LayoutParsingResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        mdDir := fmt.Sprintf("markdown_%d", i)
        os.MkdirAll(mdDir, 0755)
        mdFile := filepath.Join(mdDir, "doc.md")
        if err := os.WriteFile(mdFile, []byte(res.Markdown.Text), 0644); err != nil {
            fmt.Printf("Error writing markdown file: %v\n", err)
        } else {
            fmt.Printf("Markdown document saved at %s\n", mdFile)
        }

        for path, imgBase64 := range res.Markdown.Images {
            fullPath := filepath.Join(mdDir, path)
            os.MkdirAll(filepath.Dir(fullPath), 0755)
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding markdown image: %v\n", err)
                continue
            }
            if err := os.WriteFile(fullPath, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving markdown image: %v\n", err)
            }
        }

        for name, imgBase64 := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding output image %s: %v\n", name, err)
                continue
            }
            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving output image %s: %v\n", filename, err)
            } else {
                fmt.Printf("Output image saved at %s\n", filename)
            }
        }
    }
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
    static readonly string API_URL = "http://localhost:8080/layout-parsing";
    static readonly string inputFilePath = "./demo.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);

        var payload = new JObject
        {
            { "file", fileData },
            { "fileType", 1 }
        };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        JArray layoutParsingResults = (JArray)jsonResponse["result"]["layoutParsingResults"];
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
                        byte[] imageBytes = Convert.FromBase64String(base64Img);
                        File.WriteAllBytes(imgPath, imageBytes);
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/layout-parsing';
const imagePath = './demo.jpg';
const fileType = 1;

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: fileType
};

axios.post(API_URL, payload)
  .then(response => {
    const results = response.data.result.layoutParsingResults;
    results.forEach((res, index) => {
      console.log(`\n[${index}] prunedResult:`);
      console.log(res.prunedResult);

      const outputImages = res.outputImages;
      if (outputImages) {
        Object.entries(outputImages).forEach(([imgName, base64Img]) => {
          const imgPath = `${imgName}_${index}.jpg`;
          fs.writeFileSync(imgPath, Buffer.from(base64Img, 'base64'));
          console.log(`Output image saved at ${imgPath}`);
        });
      } else {
        console.log(`[${index}] No outputImages.`);
      }
    });
  })
  .catch(error => {
    console.error('Error during API request:', error.message || error);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/layout-parsing";
$image_path = "./demo.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["layoutParsingResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $img_base64) {
            $output_image_path = "{$img_name}_{$i}.jpg";
            file_put_contents($output_image_path, base64_decode($img_base64));
            echo "Output image saved at $output_image_path\n";
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

## 4. Secondary Development

If the default model weights provided by the PP-StructureV3 pipeline do not meet your accuracy or speed requirements in your scenario, you can try **fine-tuning the existing model** using **your own domain-specific or application-specific data** to improve the performance of the PP-StructureV3 pipeline for your use case.

### 4.1 Model Fine-tuning

Since the PP-StructureV3 pipeline contains multiple modules, unsatisfactory results may originate from any individual module. You can analyze the problematic cases with poor extraction performance, visualize the images, identify the specific module causing the issue, and then refer to the fine-tuning tutorials linked in the table below to perform model fine-tuning.

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
<td>Inaccurate layout detection, such as missing seals or tables</td>
<td>Layout Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate table structure recognition</td>
<td>Table Structure Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/table_structure_recognition.html#4-secondary-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate formula recognition</td>
<td>Formula Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/formula_recognition.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missing seal text detection</td>
<td>Seal Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/seal_text_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missing text detection</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_detection.html#4-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect text recognition results</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html#v-secondary-development">Link</a></td>
</tr>
<tr>
<td>Incorrect correction of vertical or rotated text lines</td>
<td>Text Line Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect correction of full image orientation</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported yet</td>
</tr>
</tbody>
</table>

### 4.2 Model Deployment

Once you have completed fine-tuning with your private dataset, you will obtain the local model weights. You can then use these fine-tuned weights by customizing the pipeline configuration file.

1. Export the pipeline configuration file

You can call the `export_paddlex_config_to_yaml` method of the PPStructureV3 object in PaddleOCR to export the current pipeline configuration as a YAML file:

```Python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
pipeline.export_paddlex_config_to_yaml("PP-StructureV3.yaml")
```

2. Modify the configuration file
After obtaining the default pipeline configuration file, replace the corresponding path in the configuration with the local path of your fine-tuned model weights. For example:
```yaml
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout_plus-L
    model_dir: null # Replace with the path to the fine-tuned layout detection model weights
......
SubPipelines:
  GeneralOCR:
    pipeline_name: OCR
    text_type: general
    use_doc_preprocessor: False
    use_textline_orientation: False
    SubModules:
      TextDetection:
        module_name: text_detection
        model_name: PP-OCRv5_server_det
        model_dir: null # Replace with the path to the fine-tuned text detection model weights
        limit_side_len: 960
        limit_type: max
        max_side_limit: 4000
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 1.5

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv5_server_rec
        model_dir: null # Replace with the path to the fine-tuned text recognition model weights
        batch_size: 1
        score_thresh: 0
......
```

The pipeline configuration file not only includes parameters supported by the PaddleOCR CLI and Python API but also allows for more advanced configurations. For more details, refer to the corresponding pipeline usage tutorial in the [PaddleX Pipeline Usage Overview](https://paddlepaddle.github.io/PaddleX/3.0/en/pipeline_usage/pipeline_develop_guide.html), and adjust the configurations as needed based on your requirements.

1. Load the pipeline configuration file via CLI

After modifying the configuration file, specify the updated pipeline configuration path using the `--paddlex_config` parameter in the command line. PaddleOCR will load its content as the pipeline configuration. Example:

```bash
paddleocr pp_structurev3 --paddlex_config PP-StructureV3.yaml ...
```

4. Load the pipeline configuration file via Python API
When initializing the pipeline object, you can pass the PaddleX pipeline configuration file path or a configuration dictionary using the `paddlex_config` parameter. PaddleOCR will load its content as the pipeline configuration. Example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
```
