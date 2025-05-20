---
comments: true
---

# PP-StructureV3 Production Line User Guide

## 1. Introduction to PP-StructureV3 Production Line

Layout analysis is a technique used to extract structured information from document images. It is primarily used to convert complex document layouts into machine-readable data formats. This technology has broad applications in document management, information extraction, and data digitization. Layout analysis combines Optical Character Recognition (OCR), image processing, and machine learning algorithms to identify and extract text blocks, titles, paragraphs, images, tables, and other layout elements from documents. This process generally includes three main steps: layout analysis, element analysis, and data formatting. The final result is structured document data, which enhances the efficiency and accuracy of data processing. <b>PP-StructureV3 improves upon the general layout analysis v1 production line by enhancing layout region detection, table recognition, and formula recognition. It also adds capabilities such as multi-column reading order recovery and result conversion to Markdown files. It performs excellently across various document types and can handle complex document data.<b>  This production line also provides flexible service deployment options, supporting invocation using multiple programming languages on various hardware. In addition, it offers secondary development capabilities, allowing you to train and fine-tune models on your own dataset and integrate the trained models seamlessly.

<b>PP-StructureV3 includes the following six modules. Each module can be independently trained and inferred, and contains multiple models. Click the corresponding module for more documentation.<b>

- [Layout Detection Module](../module_usage/layout_detection.en.md)
- [General OCR Sub-line](./OCR.en.md)
- [Document Image Preprocessing Sub-line](./doc_preprocessor.en.md) (Optional)
- [Table Recognition Sub-line](./table_recognition_v2.en.md) (Optional)
- [Seal Recognition Sub-line](./seal_recognition.en.md) (Optional)
- [Formula Recognition Sub-line](./formula_recognition.en.md) (Optional)

In this production line, you can choose the model to use based on the benchmark data below.

<details>
<summary><b>Document Image Orientation Classification Module:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Pretrained Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>Document image classification model based on PP-LCNet_x1_0, supporting four categories: 0°, 90°, 180°, 270°</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Image Rectification Module:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>CER</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Pretrained Model</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>High-precision text image rectification model</td>
</tr>
</tbody>
</table>

<p><b>Layout Detection Module Models:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Pretrained Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>High-precision layout region detection model based on RT-DETR-L, trained on a custom dataset including English/Chinese papers, magazines, contracts, books, exams, and research reports</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Pretrained Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>Balanced accuracy and efficiency layout region detection model based on PicoDet-L, trained on a custom dataset including English/Chinese papers, magazines, contracts, books, exams, and research reports</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Pretrained Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>High-efficiency layout region detection model based on PicoDet-S, trained on a custom dataset including English/Chinese papers, magazines, contracts, books, exams, and research reports</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Table Structure Recognition Module:</b></summary>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Pretrained Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
<td rowspan="2">SLANeXt series is a new generation table structure recognition model developed by the PaddlePaddle Vision Team. Compared with SLANet and SLANet_plus, SLANeXt focuses on recognizing table structures. It provides separately trained weights for wired and wireless tables, significantly improving recognition performance, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Pretrained Model</a></td>
</tr>
</table>

<p><b>Table Classification Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Top1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Pretrained Model</a></td>
<td>94.2</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>6.6M</td>
</tr>
</table>

<p><b>Table Cell Detection Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Pretrained Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. Based on RT-DETR-L, the PaddlePaddle Vision Team pre-trained the model on a custom table cell detection dataset, achieving good performance for both wired and wireless tables.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Pretrained Model</a></td>
</tr>
</table>
</details>

<details>
<summary><b>Text Detection Module:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Pretrained Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>371.65 / 371.65</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Pretrained Model</a></td>
<td>79.0</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Pretrained Model</a></td>
<td>69.2</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Pretrained Model</a></td>
<td>63.8</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Recognition Module:</b></summary>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>86.38</td>
<td> - </td>
<td> - </td>
<td>205 M</td>
<td>PP-OCRv5_server_rec is a new-generation text recognition model. It efficiently and accurately supports four major languages: Simplified Chinese, Traditional Chinese, English, and Japanese, as well as complex scenarios including handwriting, vertical text, pinyin, and rare characters. It ensures recognition quality while maintaining inference speed and model robustness for various document understanding tasks.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>81.29</td>
<td> - </td>
<td> - </td>
<td>136 M</td>
<td>PP-OCRv5_mobile_rec is a new-generation text recognition model. It efficiently and accurately supports four major languages: Simplified Chinese, Traditional Chinese, English, and Japanese, as well as complex scenarios including handwriting, vertical text, pinyin, and rare characters. It ensures recognition quality while maintaining inference speed and model robustness for various document understanding tasks.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>91 M</td>
<td>PP-OCRv4_server_rec_doc is based on PP-OCRv4_server_rec and further trained on mixed data from Chinese documents and PP-OCR datasets. It adds recognition support for Traditional Chinese, Japanese, and special symbols, covering over 15,000 characters. It improves both document-related and general text recognition accuracy.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>11 M</td>
<td>Lightweight recognition model of PP-OCRv4 with high inference efficiency, suitable for deployment on various hardware including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>85.19</td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>87 M</td>
<td>Server-side model of PP-OCRv4 with high recognition accuracy, deployable on various types of servers.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>7.3 M</td>
<td>Ultra-lightweight English recognition model trained on PP-OCRv4, supporting English and number recognition.</td>
</tr>
</table>

> ❗ The above are the <b>6 core models</b> primarily supported by the text recognition module. This module supports a total of <b>20 full models</b>, including multiple multilingual recognition models. The complete model list is as follows:





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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td> - </td>
<td> - </td>
<td>205 M</td>
<td>PP-OCRv5_server_rec is a new-generation text recognition model. It efficiently and accurately supports four major languages: Simplified Chinese, Traditional Chinese, English, and Japanese, as well as handwriting, vertical text, pinyin, and rare characters, offering robust and efficient support for document understanding.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td> - </td>
<td> - </td>
<td>136 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>91 M</td>
<td>Based on PP-OCRv4_server_rec, trained on additional Chinese documents and PP-OCR mixed data. It supports over 15,000 characters including Traditional Chinese, Japanese, and special symbols, enhancing both document-specific and general text recognition accuracy.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>11 M</td>
<td>Lightweight model of PP-OCRv4 with high inference efficiency, suitable for deployment on various edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>85.19</td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>87 M</td>
<td>Server-side model of PP-OCRv4 with high recognition accuracy, suitable for deployment on various servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>75.43</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>11 M</td>
<td>Lightweight model of PP-OCRv3 with high inference efficiency, suitable for deployment on various edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td>SVTRv2 is a server-side recognition model developed by the OpenOCR team at Fudan University’s FVL Lab. It won first place in the OCR End-to-End Recognition task of the PaddleOCR Model Challenge, improving end-to-end accuracy on Benchmark A by 6% compared to PP-OCRv4.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>Ultra-lightweight English recognition model trained on PP-OCRv4, supporting English and number recognition.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>An ultra-lightweight Korean text recognition model trained based on PP-OCRv3, supporting Korean and digits recognition</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>An ultra-lightweight Japanese text recognition model trained based on PP-OCRv3, supporting Japanese and digits recognition</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>An ultra-lightweight Traditional Chinese text recognition model trained based on PP-OCRv3, supporting Traditional Chinese and digits recognition</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>An ultra-lightweight Telugu text recognition model trained based on PP-OCRv3, supporting Telugu and digits recognition</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>An ultra-lightweight Kannada text recognition model trained based on PP-OCRv3, supporting Kannada and digits recognition</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>An ultra-lightweight Tamil text recognition model trained based on PP-OCRv3, supporting Tamil and digits recognition</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>An ultra-lightweight Latin text recognition model trained based on PP-OCRv3, supporting Latin and digits recognition</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>An ultra-lightweight Arabic script recognition model trained based on PP-OCRv3, supporting Arabic script and digits recognition</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>An ultra-lightweight Cyrillic script recognition model trained based on PP-OCRv3, supporting Cyrillic script and digits recognition</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Pretrained Model</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>A text line classification model based on PP-LCNet_x0_25, containing two categories: 0 degrees and 180 degrees</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Formula Recognition Module:</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Avg-BLEU (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Pretrained Model</a></td>
<td>86.13</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.4 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder, and is trained on a dataset of one million formulas including simple, complex, scanned, and handwritten formulas, significantly improving the accuracy of real-world formula recognition.</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Pretrained Model</a></td>
<td>87.12</td>
<td>202.25/-</td>
<td>-/-</td>
<td>167.9 M</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddleOCR team. It supports the recognition of 50,000 common LaTeX source code tokens. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone and applies parallel masking and model distillation to significantly improve inference speed while maintaining high accuracy. It is suitable for scenarios such as simple printed formulas and cross-line printed formulas. The PP-FormulaNet-L version is based on Vary_VIT_B and is trained on a large-scale formula dataset. It shows significant improvement in recognizing complex formulas and is suitable for printed and handwritten formula scenarios.</td>

</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Pretrained Model</a></td>
<td>92.13</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>535.2 M</td>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>71.63</td>
<td>-/-</td>
<td>-/-</td>
<td>89.7 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone and a transformer as the decoder to significantly improve the accuracy of formula recognition.</td>
</tr>
</table>
</details>

<details>
<summary><b>Seal Text Detection Module:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Normal / High Performance]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Pretrained Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>Server-side seal text detection model based on PP-OCRv4, offering higher accuracy and suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Pretrained Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>Mobile-side seal text detection model based on PP-OCRv4, offering higher efficiency and suitable for edge-side deployment</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Test Environment Description:</b></summary>

<ul>
  <li><b>Performance Test Environment</b>
    <ul>
      <li><strong>Test Datasets:</strong>
        <ul>
          <li>Document image orientation classification model: PaddleX internal dataset covering documents, certificates, etc., containing 1000 images.</li>
          <li>Text image correction model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>.</li>
          <li>Layout analysis model: PaddleOCR internal layout dataset with 10k images of documents like papers, magazines, reports, etc.</li>
          <li>Table structure recognition model: PaddleX internal English table dataset.</li>
          <li>Text detection model: PaddleOCR internal Chinese dataset covering street view, web images, documents, handwriting; detection contains 500 images.</li>
          <li>Chinese recognition model: PaddleOCR internal Chinese dataset covering street view, web images, documents, handwriting; recognition contains 11k images.</li>
          <li>ch_SVTRv2_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Model Challenge - Task 1: End-to-end OCR Recognition</a> Track A Evaluation Set.</li>
          <li>ch_RepSVTR_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Model Challenge - Task 1: End-to-end OCR Recognition</a> Track B Evaluation Set.</li>
          <li>English recognition model: PaddleX internal English dataset.</li>
          <li>Multilingual recognition model: PaddleX internal multilingual dataset.</li>
          <li>Text line orientation classification model: PaddleX internal dataset covering documents, certificates, etc., containing 1000 images.</li>
          <li>Seal text detection model: PaddleX internal dataset containing 500 round seal images.</li>
        </ul>
      </li>
      <li><strong>Hardware Configuration:</strong>
        <ul>
          <li>GPU: NVIDIA Tesla T4</li>
          <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
          <li>Other environment: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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
<td>Optimal combination of prior precision types and acceleration strategies</td>
<td>FP32 precision / 8 threads</td>
<td>Optimally selected backend (Paddle/OpenVINO/TRT etc.)</td>
</tr>
</tbody>
</table>

</details>

<br />
<b>If you care more about model accuracy, please choose a model with higher accuracy; if you prioritize inference speed, please choose a model with faster inference; if you care about storage size, please choose a model with smaller size.</b>

## 2. Quick Start

Before using the PP-StructureV3 pipeline locally, please make sure you have completed the installation of the wheel package according to the [installation guide](../installation.en.md). After installation, you can use it via command line or Python integration.

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
<td>Data to be predicted. Required. Supports multiple input types.
<ul>
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: e.g., local path to image or PDF file: <code>/root/data/img.jpg</code>; <b>URL</b>, e.g., online image or PDF: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">example</a>; <b>local directory</b>: directory containing images to predict, e.g., <code>/root/data/</code> (currently, directories with PDFs are not supported; PDFs must be specified by file path)</li>
<li><b>List</b>: list elements must be one of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Path to save inference results. If set to <code>None</code>, results will not be saved locally.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model.
<ul>
<li><b>float</b>: any value between <code>0-1</code></li>
<li><b>dict</b>: <code>{0:0.1}</code>, where key is class ID and value is the threshold for that class</li>
<li><b>None</b>: if set to <code>None</code>, the default value is used, which is <code>0.5</code></li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to apply NMS post-processing for layout detection model.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Unclip ratio for detected boxes in layout detection model.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>Tuple[float,float]</b>: separate ratios for width and height</li>
<li><b>dict</b>: key is <b>int</b> (class ID), value is <b>tuple</b>, e.g., <code>{0: (1.1, 2.0)}</code> means class 0 boxes will be expanded 1.1x in width, 2.0x in height</li>
<li><b>None</b>: if set to <code>None</code>, default is <code>1.0</code></li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merge mode for overlapping boxes in layout detection.
<ul>
<li><b>str</b>: <code>large</code>, <code>small</code>, <code>union</code>, for keeping larger box, smaller box, or both</li>
<li><b>dict</b>: key is <b>int</b> (class ID), value is <b>str</b>, e.g., <code>{0: "large", 2: "small"}</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>large</code></li>
</ul>
</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td>Name of the chart recognition model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>Directory path of the chart recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>Batch size for the chart recognition model. If set to <code>None</code>, the default batch size is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>Name of the region detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>Directory path of the region detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the document unwarping model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the document unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>Directory path of the text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Maximum side length limit for text detection.
<ul>
<li><b>int</b>: any integer > <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the default value will be <code>960</code>;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>
<ul>
<li><b>str</b>: supports <code>min</code> and <code>max</code>; <code>min</code> means ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> means the longest side does not exceed <code>limit_side_len</code></li>
<li><b>None</b>: if set to <code>None</code>, the default value will be <code>max</code>.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels with scores above this value in the probability map are considered text.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.3</code></li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Box threshold. A bounding box is considered text if the average score of pixels inside is greater than this value.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.6</code></li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Expansion ratio for text detection. The higher the value, the larger the expansion area.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>2.0</code></li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>Name of the text line orientation model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>Directory of the text line orientation model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>Batch size for the text line orientation model. If set to <code>None</code>, default is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>Directory of the text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>Batch size for text recognition. If set to <code>None</code>, default is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Score threshold for text recognition. Only results above this value will be kept.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.0</code> (no threshold)</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>Name of the table classification model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>Directory of the table classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>Name of the wired table structure recognition model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>Directory of the wired table structure recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>Name of the wireless table structure recognition model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>Directory of the wireless table structure recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>Name of the wired table cell detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>Directory of the wired table cell detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>Name of the wireless table cell detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>Directory of the wireless table cell detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>Name of the seal text detection model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>Directory of the seal text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
<ul>
<li><b>int</b>: any integer > <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the default is <code>736</code>;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Limit type for image side in seal text detection.
<ul>
<li><b>str</b>: supports <code>min</code> and <code>max</code>; <code>min</code> ensures shortest side ≥ <code>det_limit_side_len</code>, <code>max</code> ensures longest side ≤ <code>limit_side_len</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>min</code>;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Pixel threshold. Pixels with scores above this value in the probability map are considered text.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.2</code></li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Box threshold. Boxes with average pixel scores above this value are considered text regions.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.6</code></li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. Higher value means larger expansion area.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.5</code></li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>Name of the seal text recognition model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>Directory of the seal text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>Batch size for seal text recognition. If set to <code>None</code>, default is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Recognition score threshold. Text results above this value will be kept.
<ul>
<li><b>float</b>: any float > <code>0</code></li>
<li><b>None</b>: if set to <code>None</code>, default is <code>0.0</code> (no threshold)</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>Name of the formula recognition model. If set to <code>None</code>, the default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>Directory of the formula recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>Batch size of the formula recognition model. If set to <code>None</code>, the default is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to enable document orientation classification. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to enable document unwarping. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to enable seal recognition subpipeline. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to enable table recognition subpipeline. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to enable formula recognition subpipeline. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to enable chart recognition model. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to enable region detection submodule for document images. If set to <code>None</code>, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device for inference. You can specify a device ID.
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code></li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> means GPU 0</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> means NPU 0</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> means XPU 0</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> means MLU 0</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> means DCU 0</li>
<li><b>None</b>: If set to <code>None</code>, GPU 0 will be used by default if available; otherwise, CPU will be used.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high performance inference.</td>
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
<td>Minimum subgraph size for optimizing subgraph execution.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computation precision, e.g., fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN. If set to <code>None</code>, enabled by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
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
<td><code>None</code></td>
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
output = pipeline.predict("./pp_structure_v3_demo.png")

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
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model.
<ul>
<li><b>float</b>: Any float between <code>0-1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> where the key is the class ID and the value is the threshold for that class;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>0.5</code>;</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing for the layout detection model.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion ratio for the bounding boxes from the layout detection model.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: Expansion ratios in horizontal and vertical directions;</li>
<li><b>dict</b>: A dictionary with <b>int</b> keys representing <code>cls_id</code>, and <b>tuple</b> values, e.g., <code>{0: (1.1, 2.0)}</code> means width is expanded 1.1× and height 2.0× for class 0 boxes;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>1.0</code>;</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Filtering method for overlapping boxes in layout detection.
<ul>
<li><b>str</b>: Options include <code>large</code>, <code>small</code>, and <code>union</code> to retain the larger box, smaller box, or both;</li>
<li><b>dict</b>: A dictionary with <b>int</b> keys representing <code>cls_id</code>, and <b>str</b> values, e.g., <code>{0: "large", 2: "small"}</code> means using different modes for different classes;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value <code>large</code>;</li>
</ul>
</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td>Name of the chart recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>Directory path of the chart recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>Batch size for the chart recognition model. If set to <code>None</code>, the default is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>Name of the region detection model for sub-modules in document layout. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>Directory path of the region detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the document unwarping model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the document unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>Directory path of the text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Maximum side length limit for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>960</code>;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side is no less than <code>det_limit_side_len</code>, while <code>max</code> ensures the longest side is no greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>max</code>;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels in the output probability map with scores above this value are considered as text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value of <code>0.3</code>;</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Bounding box threshold. If the average score of all pixels inside the box exceeds this threshold, it is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value of <code>0.6</code>;</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Expansion ratio for text detection. The larger the value, the more the text region is expanded.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default value of <code>2.0</code>;</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>Name of the textline orientation model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>Directory path of the textline orientation model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>Batch size for the textline orientation model. If set to <code>None</code>, the default batch size is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>Directory path of the text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Score threshold for text recognition. Only results with scores above this threshold will be retained.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, uses the pipeline default of <code>0.0</code> (no threshold);</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>Name of the table classification model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>Directory path of the table classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>Name of the wired table structure recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>Directory path of the wired table structure recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>Name of the wireless table structure recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>Directory path of the wireless table structure recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>Name of the wired table cell detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>Directory path of the wired table cell detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>Name of the wireless table cell detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>Directory path of the wireless table cell detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>Name of the seal text detection model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>Directory path of the seal text detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>736</code>;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Limit type for seal text detection image side length.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures the shortest side is no less than <code>det_limit_side_len</code>, while <code>max</code> ensures the longest side is no greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>min</code>;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Pixel threshold for detection. Pixels with scores greater than this value in the probability map are considered text pixels.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.2</code>;</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Bounding box threshold. If the average score of all pixels inside a detection box exceeds this threshold, it is considered a text region.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.6</code>;</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal text detection. The larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.5</code>;</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>Name of the seal text recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>Directory path of the seal text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>Batch size for the seal text recognition model. If set to <code>None</code>, the default value is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Score threshold for seal text recognition. Text results with scores above this threshold will be retained.
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value is <code>0.0</code> (no threshold);</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>Name of the formula recognition model. If set to <code>None</code>, the pipeline default model is used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>Directory path of the formula recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>Batch size for the formula recognition model. If set to <code>None</code>, the default value is <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to enable the document orientation classification module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to enable the document image unwarping module. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to enable the chart recognition model. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to enable the region detection model for document layout. If set to <code>None</code>, the default value is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device used for inference. Supports specifying device ID.
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> means using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> means using GPU 0;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> means using NPU 0;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> means using XPU 0;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> means using MLU 0;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> means using DCU 0;</li>
<li><b>None</b>: If set to <code>None</code>, GPU 0 will be used by default. If GPU is not available, CPU will be used;</li>
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
<td>Whether to use TensorRT for accelerated inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>Minimum subgraph size used to optimize model subgraph computation.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computation precision, e.g., fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN acceleration. If set to <code>None</code>, MKL-DNN is enabled by default.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
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
<td><code>str</code></td>
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
<li><b>Python Var</b>: Image data represented as <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path to image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL</b> to image or PDF, e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">example</a>; <b>directory</b> containing image files, e.g., <code>/root/data/</code> (directories with PDFs are not supported, use full file path for PDFs)</li>
<li><b>List</b>: Elements can be any of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use document orientation classification during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use document image unwarping during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use textline orientation classification during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal recognition sub-pipeline during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-pipeline during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to use the formula recognition sub-pipeline during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Same as the parameter used during initialization.</td>
<td><code>float</code></td>
<td><code>None</code></td>
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
<td>Whether to format output as indented <code>JSON</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level to beautify the <code>JSON</code> output. Only effective when <code>format_json=True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When <code>True</code>, all non-ASCII characters are escaped. When <code>False</code>, original characters are retained. Only effective when <code>format_json=True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If a directory, the filename will be based on the input type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level for beautified <code>JSON</code> output. Only effective when <code>format_json=True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. Only effective when <code>format_json=True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save intermediate visualization results as PNG image files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>Save each page of an image or PDF file as a markdown file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save tables in the file as HTML format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save tables in the file as XLSX format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>concatenate_markdown_pages()</code></td>
<td>Concatenate multiple markdown pages into a single document</td>
<td><code>markdown_list</code></td>
<td><code>list</code></td>
<td>List of markdown data for each page</td>
<td>Returns the merged markdown text and image list</td>
</tr>
</table>

- Calling `print()` will print the result to the terminal. Explanation of the printed content:
    - `input_path`: `(str)` Input path of the image or PDF to be predicted

    - `page_index`: `(Union[int, None])` If input is a PDF, indicates the page number; otherwise `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters configured for the pipeline

        - `use_doc_preprocessor`: `(bool)` Whether to enable document preprocessor sub-pipeline
        - `use_seal_recognition`: `(bool)` Whether to enable seal recognition sub-pipeline
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

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of seal recognition results
        - `input_path`: `(str)` Input path for the seal image
        - `page_index`: `None` since input is `numpy.ndarray`
        - `model_settings`: `(Dict)` Model configuration for seal recognition
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
<td>Key-value pairs of image relative paths and base64-encoded image content.</td>
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

# Encode the local image to Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64-encoded file content or file URL
    "fileType": 1, # File type, 1 indicates image file
}

# Call the API
response = requests.post(API_URL, json=payload)

# Handle the response data
assert response.status_code == 200
result = response.json()["result"]
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
<td><a href="../module_usage/layout_detection.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate table structure recognition</td>
<td>Table Structure Recognition Module</td>
<td><a href="../module_usage/layout_detection.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate formula recognition</td>
<td>Formula Recognition Module</td>
<td><a href="../module_usage/formula_recognition.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missing seal text detection</td>
<td>Seal Text Detection Module</td>
<td><a href="../module_usage/seal_text_detection.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missing text detection</td>
<td>Text Detection Module</td>
<td><a href="../module_usage/text_detection.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect text recognition results</td>
<td>Text Recognition Module</td>
<td><a href="../module_usage/text_recognition.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect correction of vertical or rotated text lines</td>
<td>Text Line Orientation Classification Module</td>
<td><a href="../module_usage/text_line_orientation_classification.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect correction of full image orientation</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="../module_usage/doc_img_orientation_classification.en.md#iv-custom-development">Link</a></td>
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
When initializing the pipeline object, you can pass the PaddleX pipeline configuration file path or a configuration dictionary using the paddlex_config parameter. PaddleOCR will load its content as the pipeline configuration. Example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
```
