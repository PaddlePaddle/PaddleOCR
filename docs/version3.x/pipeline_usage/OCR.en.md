---
comments: true
---

# General OCR Pipeline Usage Guide

## 1. OCR Pipeline Introduction

OCR is a technology that converts text from images into editable text. It is widely used in fields such as document digitization, information extraction, and data processing. OCR can recognize printed text, handwritten text, and even certain types of fonts and symbols.

The general OCR pipeline is used to solve text recognition tasks by extracting text information from images and outputting it in text form. This pipeline supports the use of PP-OCRv3, PP-OCRv4, and PP-OCRv5 models, with the default model being the PP-OCRv5_mobile model released by PaddleOCR 3.0, which improves by 13 percentage points over PP-OCRv4_mobile in various scenarios.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/01.png"/>

<b>The General OCR Pipeline consists of the following 5 modules. Each module can be independently trained and inferred, and includes multiple models. For detailed information, click the corresponding module to view its documentation.</b>

- [Document Image Orientation Classification Module](../module_usage/doc_img_orientation_classification.md) (Optional)
- [Text Image Unwarping Module](../module_usage/text_image_unwarping.md) (Optional)
- [Text Line Orientation Classification Module](../module_usage/text_line_orientation_classification.md) (Optional)
- [Text Detection Module](../module_usage/text_detection.md)
- [Text Recognition Module](../module_usage/text_recognition.md)

In this pipeline, you can select models based on the benchmark test data provided below.

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
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
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
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>30.3</td>
<td>High-precision Text Image Unwarping model.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Detection Module:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Training Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>371.65 / 371.65</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
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
<th>Model</th><th>Model Download Links</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>86.38</td>
<td> 8.45/2.36 </td>
<td> 122.69/122.69 </td>
<td>81 M</td>
<td rowspan="2">PP-OCRv5_rec is a next-generation text recognition model. It aims to efficiently and accurately support the recognition of four major languages—Simplified Chinese, Traditional Chinese, English, and Japanese—as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters using a single model. While maintaining recognition performance, it balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>81.29</td>
<td> 1.46/5.43 </td>
<td> 5.32/91.79 </td>
<td>16 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>91 M</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data, building upon PP-OCRv4_server_rec. It enhances the recognition capabilities for some Traditional Chinese characters, Japanese characters, and special symbols, supporting over 15,000 characters. In addition to improving document-related text recognition, it also enhances general text recognition capabilities.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>11 M</td>
<td>A lightweight recognition model of PP-OCRv4 with high inference efficiency, suitable for deployment on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>87 M</td>
<td>The server-side model of PP-OCRv4, offering high inference accuracy and deployable on various servers.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>7.3 M</td>
<td>An ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model, supporting English and numeric character recognition.</td>
</tr>
</table>

> ❗ The above section lists the <b>6 core models</b> that are primarily supported by the text recognition module. In total, the module supports <b>20 comprehensive models</b>, including multiple multilingual text recognition models. Below is the complete list of models:

<details><summary> 👉Details of the Model List</summary>

* <b>PP-OCRv5 Multi-Scenario Models</b>

<table>
<tr>
<th>Model</th><th>Model Download Links</th>
<th>Avg Accuracy for Chinese Recognition (%)</th>
<th>Avg Accuracy for English Recognition (%)</th>
<th>Avg Accuracy for Traditional Chinese Recognition (%)</th>
<th>Avg Accuracy for Japanese Recognition (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td> 8.45/2.36 </td>
<td> 122.69/122.69 </td>
<td>81 M</td>
<td rowspan="2">PP-OCRv5_rec is a next-generation text recognition model. It aims to efficiently and accurately support the recognition of four major languages—Simplified Chinese, Traditional Chinese, English, and Japanese—as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters using a single model. While maintaining recognition performance, it balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td> 1.46/5.43 </td>
<td> 5.32/91.79 </td>
<td>16 M</td>
</tr>
</table>

* <b> Chinese Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Training Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>91 M</td>
<td>PP-OCRv4_server_rec_doc is built upon PP-OCRv4_server_rec and trained on mixed data including more Chinese document data and PP-OCR training data. It enhances recognition of traditional Chinese characters, Japanese, and special symbols, supporting 15,000+ characters. It improves both document-specific and general text recognition capabilities.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>11 M</td>
<td>Lightweight recognition model of PP-OCRv4 with high inference efficiency, deployable on various hardware devices including edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>87 M</td>
<td>Server-side model of PP-OCRv4 with high inference accuracy, deployable on various server platforms</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>75.43</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>11 M</td>
<td>Lightweight recognition model of PP-OCRv3 with high inference efficiency, deployable on various hardware devices including edge devices</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team from Fudan University Vision and Learning Lab (FVL). It won first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition, improving end-to-end recognition accuracy by 6% compared to PP-OCRv4 on List A.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">RepSVTR is a mobile text recognition model based on SVTRv2. It won first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition, improving end-to-end recognition accuracy by 2.5% compared to PP-OCRv4 on List B while maintaining comparable inference speed.</td>
</tr>
</table>

* <b> English Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>Ultra-lightweight English recognition model based on PP-OCRv4, supporting English and digit recognition</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
<td>Ultra-lightweight English recognition model based on PP-OCRv3, supporting English and digit recognition</td>
</tr>
</table>


* <b>Multilingual Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>Ultra-lightweight Korean recognition model based on PP-OCRv3, supporting Korean and numeric recognition</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>Ultra-lightweight Japanese recognition model based on PP-OCRv3, supporting Japanese and numeric recognition</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>Ultra-lightweight Traditional Chinese recognition model based on PP-OCRv3, supporting Traditional Chinese and numeric recognition</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>Ultra-lightweight Telugu recognition model based on PP-OCRv3, supporting Telugu and numeric recognition</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>Ultra-lightweight Kannada recognition model based on PP-OCRv3, supporting Kannada and numeric recognition</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>Ultra-lightweight Tamil recognition model based on PP-OCRv3, supporting Tamil and numeric recognition</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>Ultra-lightweight Latin recognition model based on PP-OCRv3, supporting Latin and numeric recognition</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>Ultra-lightweight Arabic recognition model based on PP-OCRv3, supporting Arabic and numeric recognition</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>Ultra-lightweight Cyrillic recognition model based on PP-OCRv3, supporting Cyrillic and numeric recognition</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
<td>Ultra-lightweight Devanagari recognition model based on PP-OCRv3, supporting Devanagari and numeric recognition</td>
</tr>
</table>
</details>
</details>

<details>
<summary><strong>Test Environment Details:</strong></summary>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
            <li><strong>Test Datasets:
             </strong>
                <ul>
                  <li>Document Image Orientation Classification Model: PaddleX in-house dataset covering ID cards and documents, with 1,000 images.</li>
                  <li>Text Image Correction Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>.</li>
                  <li>Text Detection Model: PaddleOCR in-house Chinese dataset covering street views, web images, documents, and handwriting, with 500 images for detection.</li>
                  <li>Chinese Recognition Model: PaddleOCR in-house Chinese dataset covering street views, web images, documents, and handwriting, with 11,000 images for recognition.</li>
                  <li>ch_SVTRv2_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Challenge - Task 1: OCR End-to-End Recognition</a> A-set evaluation data.</li>
                  <li>ch_RepSVTR_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Challenge - Task 1: OCR End-to-End Recognition</a> B-set evaluation data.</li>
                  <li>English Recognition Model: PaddleX in-house English dataset.</li>
                  <li>Multilingual Recognition Model: PaddleX in-house multilingual dataset.</li>
                  <li>Text Line Orientation Classification Model: PaddleX in-house dataset covering ID cards and documents, with 1,000 images.</li>
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
            <th>Acceleration Techniques</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Standard Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Optimal backend selection (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

</details>

<br />
<b>If you prioritize model accuracy, choose models with higher accuracy; if inference speed is critical, select faster models; if model size matters, opt for smaller models.</b>

## 2. Quick Start  

Before using the general OCR pipeline locally, ensure you have installed the wheel package by following the [Installation Guide](../installation.en.md). Once installed, you can experience OCR via the command line or Python integration.  

### 2.1 Command Line  

Run a single command to quickly test the OCR pipeline:  

```bash  
# Default: Uses PP-OCRv5 model  
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --save_path ./output \
    --device gpu:0 

# Use PP-OCRv4 model by --ocr_version PP-OCRv4
paddleocr ocr -i ./general_ocr_002.png --ocr_version PP-OCRv4
```  

<details><summary><b>Command line supports more parameter settings. Click to expand for detailed instructions on command line parameters.</b></summary>
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
<td>Data to be predicted, supporting multiple input types (required).
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">Example</a>; <b>Local directory</b>, which must contain images to be predicted, such as the local path: <code>/root/data/</code> (currently, predicting PDFs in a directory is not supported; PDFs need to specify the exact file path)</li>
<li><b>List</b>: List elements must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Path to save inference result files. If set to <code>None</code>, inference results will not be saved locally.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the production line default model will be used.</td>
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
<td>Name of the text image unwarping model. If set to <code>None</code>, the production line default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the text image unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If set to <code>None</code>, the production line default model will be used.</td>
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
<td><code>text_line_orientation_model_name</code></td>
<td>Name of the text line orientation model. If set to <code>None</code>, the production line default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_line_orientation_model_dir</code></td>
<td>Directory path of the text line orientation model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_line_orientation_batch_size</code></td>
<td>Batch size for the text line orientation model. If set to <code>None</code>, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the production line default model will be used.</td>
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
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the text image unwarping function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Maximum side length limit for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>; </li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>960</code>) will be used; </li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of side length limit for text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> means ensuring the shortest side of the image is not smaller than <code>det_limit_side_len</code>, and <code>max</code> means ensuring the longest side of the image is not larger than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>max</code>) will be used; </li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for text detection. In the output probability map, pixels with scores higher than this threshold will be considered text pixels.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.3</code>) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Text detection box threshold. If the average score of all pixels within the detected result boundary is higher than this threshold, the result will be considered a text region.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.6</code>) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion coefficient. This method is used to expand the text region—the larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>2.0</code>) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_input_shape</code></td>
<td>Input shape for text detection.</td>
<td><code>tuple</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold. Text results with scores higher than this threshold will be retained.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.0</code>, i.e., no threshold) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_input_shape</code></td>
<td>Input shape for text recognition.</td>
<td><code>tuple</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>lang</code></td>
<td>OCR model for a specified language.
<ul>
<li><b>ch</b>: Chinese;
<li><b>en</b>: English;
<li><b>korean</b>: Korean;
<li><b>japan</b>: Japanese;
<li><b>chinese_cht</b>: Traditional Chinese;
<li><b>te</b>: Telugu;
<li><b>ka</b>: Kannada;
<li><b>ta</b>: Tamil;
<li><b>None</b>: If set to <code>None</code>, <code>ch</code> will be used by default; </li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>ocr_version</code></td>
<td>OCR version.
<ul>
<li><b>PP-OCRv5</b>: Use <code>PP-OCRv5</code> series models;
<li><b>PP-OCRv4</b>: Use <code>PP-OCRv4</code> series models;
<li><b>PP-OCRv3</b>: Use <code>PP-OCRv3</code> series models;
<li><b>None</b>: If set to <code>None</code>, <code>PP-OCRv5</code> series models will be used by default; </li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_model_dir</code></td>
<td>Deprecated. Please use <code>text_detection_model_dir</code> instead. Directory path of the text detection model. If set to None, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_limit_side_len</code></td>
<td>Deprecated. Please use <code>text_det_limit_side_len</code> instead. Maximum side length limit for text detection.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_limit_type</code></td>
<td>Deprecated. Please use <code>text_det_limit_type</code> instead. Type of side length limit for text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> means ensuring the shortest side of the image is not smaller than <code>det_limit_side_len</code>, and <code>max</code> means ensuring the longest side of the image is not larger than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>max</code>) will be used; </li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_db_thresh</code></td>
<td>Deprecated. Please use <code>text_det_thresh</code> instead. Pixel threshold for text detection. In the output probability map, pixels with scores higher than this threshold will be considered text pixels.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.3</code>) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_db_box_thresh</code></td>
<td>Deprecated. Please use <code>text_det_box_thresh</code> instead. Text detection box threshold. If the average score of all pixels within the detected result boundary is higher than this threshold, the result will be considered a text region.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.6</code>) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_db_unclip_ratio</code></td>
<td>Deprecated. Please use <code>text_det_unclip_ratio</code> instead. Text detection expansion coefficient. This method is used to expand the text region—the larger the value, the larger the expanded area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>2.0</code>) will be used</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>rec_model_dir</code></td>
<td>Deprecated. Please use <code>text_recognition_model_dir</code> instead. Directory path of the text recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>rec_batch_num</code></td>
<td>Deprecated. Please use <code>text_recognition_batch_size</code> instead. Batch size for the text recognition model. If set to <code>None</code>, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_angle_cls</code></td>
<td>Deprecated. Please use <code>use_textline_orientation</code> instead. Whether to use the text line orientation function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cls_model_dir</code></td>
<td>Deprecated. Please use <code>text_line_orientation_model_dir</code> instead. Directory path of the text line orientation model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cls_batch_num</code></td>
<td>Deprecated. Please use <code>text_line_orientation_batch_size</code> instead. Batch size for the text line orientation model. If set to <code>None</code>, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device for inference. Supports specifying a specific card number.
<ul>
<li><b>CPU</b>: <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used;</li>
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
<td>Minimum subgraph size for optimizing model subgraph computation.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computational precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.
</td>
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
<td>Path to the PaddleX production line configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />

Results are printed to the terminal:  

```bash
{'res': {'input_path': './general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[  3,  10],
        ...,
        [  4,  30]],

       ...,

       [[ 99, 456],
        ...,
        [ 99, 479]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.997700', '', 'Cm', '登机牌', 'BOARDING', 'PASS', 'CLASS', '序号SERIAL NO.', '座位号', 'SEAT NO.', '航班FLIGHT', '日期DATE', '舱位', '', 'W', '035', '12F', 'MU2379', '03DEc', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKT NO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭 GATESCL0SE10MINUTESBEFOREDEPARTURETIME'], 'rec_scores': array([0.67634439, ..., 0.97416091]), 'rec_polys': array([[[  3,  10],
        ...,
        [  4,  30]],

       ...,

       [[ 99, 456],
        ...,
        [ 99, 479]]], dtype=int16), 'rec_boxes': array([[  3, ...,  30],
       ...,
       [ 99, ..., 479]], dtype=int16)}}
```

If `save_path` is specified, the visualization results will be saved under `save_path`. The visualization output is shown below:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/03.png"/>

### 2.2 Python Script Integration  

The command-line method is for quick testing. For project integration, you can achieve OCR inference with just a few lines of code:  

```python  
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
)
# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # Uses other PP-OCR versions via version parameter
# ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # Switch to PP-OCRv5_server models
result = ocr.predict("./general_ocr_002.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")  
```  

In the above Python script, the following steps are performed:

<details><summary>(1) Instantiate the OCR production line object via <code>PaddleOCR()</code>, with specific parameter descriptions as follows:</summary>

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
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the production line's default model will be used.</td>
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
<td>Name of the text image unwarping model. If set to <code>None</code>, the production line's default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the text image unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>Name of the text detection model. If set to <code>None</code>, the production line's default model will be used.</td>
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
<td><code>text_line_orientation_model_name</code></td>
<td>Name of the text line orientation model. If set to <code>None</code>, the production line's default model will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_line_orientation_model_dir</code></td>
<td>Directory path of the text line orientation model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_line_orientation_batch_size</code></td>
<td>Batch size for the text line orientation model. If set to <code>None</code>, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the production line's default model will be used.</td>
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
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size will be <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the text image unwarping function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation function. If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>True</code>) will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Maximum side length limit for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>960</code>) will be used;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of side length limit for text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> means ensuring the shortest side of the image is not smaller than <code>det_limit_side_len</code>, and <code>max</code> means ensuring the longest side of the image is not larger than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (initialized to <code>max</code>) will be used;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Pixel threshold for text detection. Pixels with scores higher than this threshold in the output probability map will be considered text pixels.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.3</code>) will be used;</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Box threshold for text detection. A detection result will be considered a text region if the average score of all pixels within the bounding box is higher than this threshold.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.6</code>) will be used;</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Dilation coefficient for text detection. This method is used to dilate the text region, and the larger this value, the larger the dilated area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>2.0</code>) will be used;</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_input_shape</code></td>
<td>Input shape for text detection.</td>
<td><code>tuple</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Recognition score threshold for text. Text results with scores higher than this threshold will be retained.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter (<code>0.0</code>, i.e., no threshold) will be used;</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_input_shape</code></td>
<td>Input shape for text recognition.</td>
<td><code>tuple</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>lang</code></td>
<td>OCR model language to use.
<ul>
<li><b>ch</b>: Chinese;</li>
<li><b>en</b>: English;</li>
<li><b>korean</b>: Korean;</li>
<li><b>japan</b>: Japanese;</li>
<li><b>chinese_cht</b>: Traditional Chinese;</li>
<li><b>te</b>: Telugu;</li>
<li><b>ka</b>: Kannada;</li>
<li><b>ta</b>: Tamil;</li>
<li><b>None</b>: If set to <code>None</code>, <code>ch</code> will be used by default;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>ocr_version</code></td>
<td>OCR version.
<ul>
<li><b>PP-OCRv5</b>: Use <code>PP-OCRv5</code> series models;</li>
<li><b>PP-OCRv4</b>: Use <code>PP-OCRv4</code> series models;</li>
<li><b>PP-OCRv3</b>: Use <code>PP-OCRv3</code> series models;</li>
<li><b>None</b>: If set to <code>None</code>, <code>PP-OCRv5</code> series models will be used by default;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device for inference. Supports specifying a specific card number.
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> for CPU inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> for inference on the 1st GPU;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> for inference on the 1st NPU;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> for inference on the 1st XPU;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> for inference on the 1st MLU;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> for inference on the 1st DCU;</li>
<li><b>None</b>: If set to <code>None</code>, the production line's initialized value for this parameter will be used. During initialization, it will give priority to the local GPU 0 device; if not available, the CPU device will be used;</li>
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
<td>Minimum subgraph size for optimizing subgraph computation.</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computational precision, such as fp32, fp16.</td>
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
<td>Number of threads used for CPU inference.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>Path to the PaddleX production line configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(2) Invoke the <code>predict()</code> method of the OCR production line object for inference prediction, which returns a results list. Additionally, the production line provides the <code>predict_iter()</code> method. Both methods are completely consistent in parameter acceptance and result return, except that <code>predict_iter()</code> returns a <code>generator</code>, which can process and obtain prediction results incrementally, suitable for handling large datasets or scenarios where memory saving is desired. You can choose to use either of these two methods according to actual needs. The following are the parameters and descriptions of the <code>predict()</code> method:</summary>

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
<td>Data to be predicted, supporting multiple input types, required.
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">example</a>; <b>local directory</b>, which needs to contain images to be predicted, such as the local path: <code>/root/data/</code> (currently, predicting PDF files in the directory is not supported; PDF files need to specify the specific file path)</li>
<li><b>List</b>: List elements must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The same as the parameter during instantiation.</td>
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
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<td><code>text_det_limit_side_len</code></td>
<td>The same as the parameter during instantiation.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<td><code>text_det_limit_type</code></td>
<td>The same as the parameter during instantiation.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<td><code>text_det_thresh</code></td>
<td>The same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<td><code>text_det_box_thresh</code></td>
<td>The same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<td><code>text_det_unclip_ratio</code></td>
<td>The same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<td><code>text_rec_score_thresh</code></td>
<td>The same as the parameter during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</table>
</details>

<details><summary>(3) Process the prediction results. The prediction result of each sample is a corresponding Result object, which supports operations of printing, saving as an image, and saving as a <code>json</code> file:</summary>

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
<td rowspan="3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the results as a json-formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path to save. When it is a directory, the saved file name will be consistent with the input file type name</td>
<td>No default</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image-formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path to save, supporting directory or file path</td>
<td>No default</td>
</tr>
</table>

<ul>
    <li>Calling the <code>print()</code> method will print the results to the terminal. The content printed to the terminal is explained as follows:
        <ul>
            <li><code>input_path</code>: <code>(str)</code> Input path of the image to be predicted</li>
            <li><code>page_index</code>: <code>(Union[int, None])</code> If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is <code>None</code></li>
            <li><code>model_settings</code>: <code>(Dict[str, bool])</code> Model parameters configured for the production line
                <ul>
                    <li><code>use_doc_preprocessor</code>: <code>(bool)</code> Control whether to enable the document preprocessing sub-production line</li>
                    <li><code>use_textline_orientation</code>: <code>(bool)</code> Control whether to enable the text line orientation classification function</li>
                </ul>
            </li>
            <li><code>doc_preprocessor_res</code>: <code>(Dict[str, Union[str, Dict[str, bool], int]])</code> Output results of the document preprocessing sub-production line. Only exists when <code>use_doc_preprocessor=True</code>
                <ul>
                    <li><code>input_path</code>: <code>(Union[str, None])</code> Image path accepted by the image preprocessing sub-production line. When the input is <code>numpy.ndarray</code>, it is saved as <code>None</code></li>
                    <li><code>model_settings</code>: <code>(Dict)</code> Model configuration parameters of the preprocessing sub-production line
                        <ul>
                            <li><code>use_doc_orientation_classify</code>: <code>(bool)</code> Control whether to enable document orientation classification</li>
                            <li><code>use_doc_unwarping</code>: <code>(bool)</code> Control whether to enable text image unwarping</li>
                        </ul>
                    </li>
                    <li><code>angle</code>: <code>(int)</code> Prediction result of document orientation classification. When enabled, the values are [0,1,2,3], corresponding to [0°,90°,180°,270°]; when disabled, it is -1</li>
                </ul>
            </li>
            <li><code>dt_polys</code>: <code>(List[numpy.ndarray])</code> List of text detection polygon boxes. Each detection box is represented by a numpy array of 4 vertex coordinates, with the array shape being (4, 2) and the data type being int16</li>
            <li><code>dt_scores</code>: <code>(List[float])</code> List of confidence scores for text detection boxes</li>
            <li><code>text_det_params</code>: <code>(Dict[str, Dict[str, int, float]])</code> Configuration parameters for the text detection module
                <ul>
                    <li><code>limit_side_len</code>: <code>(int)</code> Side length limit value during image preprocessing</li>
                    <li><code>limit_type</code>: <code>(str)</code> Processing method for side length limits</li>
                    <li><code>thresh</code>: <code>(float)</code> Confidence threshold for text pixel classification</li>
                    <li><code>box_thresh</code>: <code>(float)</code> Confidence threshold for text detection boxes</li>
                    <li><code>unclip_ratio</code>: <code>(float)</code> Dilation coefficient for text detection boxes</li>
                    <li><code>text_type</code>: <code>(str)</code> Type of text detection, currently fixed as "general"</li>
                </ul>
            </li>
            <li><code>textline_orientation_angles</code>: <code>(List[int])</code> Prediction results of text line orientation classification. When enabled, actual angle values are returned (e.g., [0,0,1]); when disabled, [-1,-1,-1] is returned</li>
            <li><code>text_rec_score_thresh</code>: <code>(float)</code> Filtering threshold for text recognition results</li>
            <li><code>rec_texts</code>: <code>(List[str])</code> List of text recognition results, containing only texts with confidence scores exceeding <code>text_rec_score_thresh</code></li>
            <li><code>rec_scores</code>: <code>(List[float])</code> List of text recognition confidence scores, filtered by <code>text_rec_score_thresh</code></li>
            <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> List of text detection boxes filtered by confidence, in the same format as <code>dt_polys</code></li>
            <li><code>rec_boxes</code>: <code>(numpy.ndarray)</code> Array of rectangular bounding boxes for detection boxes, with shape (n, 4) and dtype int16. Each row represents the [x_min, y_min, x_max, y_max] coordinates of a rectangular box, where (x_min, y_min) is the top-left coordinate and (x_max, y_max) is the bottom-right coordinate</li>
        </ul>
    </li>
    <li>Calling the <code>save_to_json()</code> method will save the above content to the specified <code>save_path</code>. If a directory is specified, the save path will be <code>save_path/{your_img_basename}_res.json</code>. If a file is specified, it will be saved directly to that file. Since json files do not support saving numpy arrays, <code>numpy.array</code> types will be converted to list form.</li>
    <li>Calling the <code>save_to_img()</code> method will save the visualization results to the specified <code>save_path</code>. If a directory is specified, the save path will be <code>save_path/{your_img_basename}_ocr_res_img.{your_img_extension}</code>. If a file is specified, it will be saved directly to that file. (The production line usually generates many result images, so it is not recommended to directly specify a specific file path, as multiple images will be overwritten, leaving only the last one.)</li>
</ul>

<p>Additionally, you can also obtain the visualized image with results and prediction results through attributes, as follows:</p>

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

<ul>
    <li>The prediction results obtained by the <code>json</code> attribute are in dict format, and the content is consistent with that saved by calling the <code>save_to_json()</code> method.</li>
    <li>The <code>img</code> attribute returns a dictionary-type result. The keys are <code>ocr_res_img</code> and <code>preprocessed_img</code>, with corresponding values being two <code>Image.Image</code> objects: one for displaying the visualized image of OCR results and the other for displaying the visualized image of image preprocessing. If the image preprocessing submodule is not used, only <code>ocr_res_img</code> will be included in the dictionary.</li>
</ul>

</details>

## 3. Development Integration/Deployment

If the general OCR pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general OCR pipeline directly in your Python project, you can refer to the sample code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleOCR provides two other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In real-world production environments, many applications have stringent performance requirements (especially for response speed) to ensure system efficiency and smooth user experience. To address this, PaddleOCR offers high-performance inference capabilities, which deeply optimize model inference and pre/post-processing to achieve significant end-to-end speed improvements. For detailed high-performance inference workflows, refer to the [High-Performance Inference Guide](../deployment/high_performance_inference.en.md).

☁️ **Service Deployment**: Service deployment is a common form of deployment in production environments. By encapsulating inference functionality as a service, clients can access these services via network requests to obtain inference results. For detailed pipeline service deployment workflows, refer to the [Service Deployment Guide](../deployment/serving.en.md).

Below are the API reference for basic service deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body has the following attributes:</li>
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
<td>UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request fails, the response body has the following attributes:</li>
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
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain OCR results for an image.</p>
<p><code>POST /ocr</code></p>
<ul>
<li>The request body has the following attributes:</li>
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
<td>A server-accessible URL to an image or PDF file, or the Base64-encoded content of such a file. By default, for PDF files with more than 10 pages, only the first 10 pages are processed.<br /> To remove the page limit, add the following configuration to the pipeline config file:
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
<td>File type. <code>0</code> for PDF, <code>1</code> for image. If omitted, the type is inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_doc_orientation_classify</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_doc_unwarping</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_textline_orientation</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Refer to the <code>text_det_limit_side_len</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refer to the <code>text_det_limit_type</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_thresh</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_box_thresh</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_unclip_ratio</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_rec_score_thresh</code> parameter in the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successful, the <code>result</code> in the response body has the following attributes:</li>
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
<td><code>ocrResults</code></td>
<td><code>object</code></td>
<td>OCR results. The array length is 1 (for image input) or the number of processed document pages (for PDF input). For PDF input, each element represents the result for a corresponding page.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Input data information.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>ocrResults</code> is an <code>object</code> with the following attributes:</p>
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
<td>A simplified version of the <code>res</code> field in the JSON output of the pipeline object's <code>predict</code> method, excluding <code>input_path</code> and <code>page_index</code>.</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>OCR result image with detected text regions highlighted. JPEG format, Base64-encoded.</td>
</tr>
<tr>
<td><code>docPreprocessingImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Visualization of preprocessing results. JPEG format, Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Input image. JPEG format, Base64-encoded.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-Language Service Call Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests

API_URL = "http://localhost:8080/ocr"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["ocrResults"]):
    print(res["prunedResult"])
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output image saved at {ocr_img_path}")
</code></pre></details>
</details>

## 4. Custom Development
If the default model weights provided by the General OCR Pipeline do not meet your expectations in terms of accuracy or speed for your specific scenario, you can leverage your own domain-specific or application-specific data to further fine-tune the existing models, thereby improving the recognition performance of the General OCR Pipeline in your use case.

### 4.1 Model Fine-Tuning
The general OCR pipeline consists of multiple modules. If the pipeline's performance does not meet expectations, the issue may stem from any of these modules. You can analyze poorly recognized images to identify the problematic module and refer to the corresponding fine-tuning tutorials in the table below for adjustments.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Module to Fine-Tune</th>
<th>Fine-Tuning Reference</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate whole-image rotation correction</td>
<td>Document orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text image unwarping module</td>
<td>Fine-tuning not supported</td>
</tr>
<tr>
<td>Inaccurate textline rotation correction</td>
<td>Textline orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Text detection misses</td>
<td>Text detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_detection.html#4-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect text recognition</td>
<td>Text recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html#v-secondary-development">Link</a></td>
</tr>
</tbody>
</table>

### 4.2 Model Deployment  

After you complete fine-tuning training using a private dataset, you can obtain a local model weight file. You can then use the fine-tuned model weights by specifying the local model save path through parameters or by customizing the production line configuration file.

#### 4.2.1 Specify the local model path through parameters

When initializing the production line object, specify the local model path through parameters. Take the usage of the weights after fine-tuning the text detection model as an example, as follows:

Command line mode:

```bash
# Specify the local model path via --text_detection_model_dir
paddleocr ocr -i ./general_ocr_002.png --text_detection_model_dir your_det_model_path

# PP-OCRv5_mobile_det model is used as the default text detection model. If you do not fine-tune this model, modify the model name by using --text_detection_model_name
paddleocr ocr -i ./general_ocr_002.png --text_detection_model_name PP-OCRv5_server_det --text_detection_model_dir your_v5_server_det_model_path
```

Script mode：

```python

from paddleocr import PaddleOCR

#  Specify the local model path via text_detection_model_dir
pipeline = PaddleOCR(text_detection_model_dir="./your_det_model_path")

# PP-OCRv5_mobile_det model is used as the default text detection model. If you do not fine-tune this model, modify the model name by using text_detection_model_name
# pipeline = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_detection_model_dir="./your_v5_server_det_model_path")

```

#### 4.2.2 Specify the local model path through the configuration file

1.Obtain the production line configuration file

Call the `export_paddlex_config_to_yaml` method of the **General OCR Pipeline** object in PaddleOCR to export the current pipeline configuration as a YAML file:  

```Python  
from paddleocr import PaddleOCR  

pipeline = PaddleOCR()  
pipeline.export_paddlex_config_to_yaml("PaddleOCR.yaml")  
```  

2.Modify the Configuration File  

After obtaining the default pipeline configuration file, replace the paths of the default model weights with the local paths of your fine-tuned model weights. For example:  

```yaml  
......  
SubModules:  
  TextDetection:  
    box_thresh: 0.6  
    limit_side_len: 960  
    limit_type: max  
    max_side_limit: 4000  
    model_dir: null # Replace with the path to your fine-tuned text detection model weights  
    model_name: PP-OCRv5_server_det  # If the name of the fine-tuned model is different from the default model name, please modify it here as well
    module_name: text_detection  
    thresh: 0.3  
    unclip_ratio: 1.5  
  TextLineOrientation:  
    batch_size: 6  
    model_dir: null  # Replace with the path to your fine-tuned text LineOrientation model weights  
    model_name: PP-LCNet_x0_25_textline_ori  # If the name of the fine-tuned model is different from the default model name, please modify it here as well
    module_name: textline_orientation  
  TextRecognition:  
    batch_size: 6  
    model_dir: null # Replace with the path to your fine-tuned text recognition model weights  
    model_name: PP-OCRv5_server_rec  # If the name of the fine-tuned model is different from the default model name, please modify it here as well
    module_name: text_recognition  
    score_thresh: 0.0  
......  
```  

The pipeline configuration file includes not only the parameters supported by the PaddleOCR CLI and Python API but also advanced configurations. For detailed instructions, refer to the [PaddleX Pipeline Usage Overview](https://paddlepaddle.github.io/PaddleX/3.0/en/pipeline_usage/pipeline_develop_guide.html) and adjust the configurations as needed.  

3.Load the Configuration File in CLI  

After modifying the configuration file, specify its path using the `--paddlex_config` parameter in the command line. PaddleOCR will read the file and apply the configurations. Example:  

```bash  
paddleocr ocr --paddlex_config PaddleOCR.yaml ...  
```  

4.Load the Configuration File in Python API  

When initializing the pipeline object, pass the path of the PaddleX pipeline configuration file or a configuration dictionary via the `paddlex_config` parameter. PaddleOCR will read and apply the configurations. Example:  

```python  
from paddleocr import PaddleOCR  

pipeline = PaddleOCR(paddlex_config="PaddleOCR.yaml")  
```
