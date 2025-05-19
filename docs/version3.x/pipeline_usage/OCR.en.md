---
comments: true
---

# General OCR Pipeline Usage Guide

## 1. OCR Pipeline Introduction

OCR (Optical Character Recognition) is a technology that converts text in images into editable text. It is widely used in document digitization, information extraction, and data processing. OCR can recognize printed text, handwritten text, and even certain types of fonts and symbols.

The General OCR Pipeline is designed to solve text recognition tasks by extracting text information from images and outputting it in text format. This pipeline integrates the industry-renowned PP-OCRv3 and PP-OCRv4 end-to-end OCR systems, supporting recognition for over 80 languages. Additionally, it includes functionalities for image orientation correction and distortion correction. Based on this pipeline, millisecond-level accurate text prediction can be achieved on CPUs, covering various scenarios such as general, manufacturing, finance, and transportation. The pipeline also offers flexible service-oriented deployment options, supporting calls in multiple programming languages across various hardware platforms. Furthermore, it provides secondary development capabilities, allowing you to fine-tune models on your own datasets, with trained models seamlessly integrable.

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
<summary><b>Text Image Unwar'p Module (Optional):</b></summary>
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
<td>- / -</td>
<td>- / -</td>
<td>101</td>
<td>Server-side text detection model for PP-OCRv5, offering higher accuracy, suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>- / -</td>
<td>- / -</td>
<td>4.7</td>
<td>Mobile-side text detection model for PP-OCRv5, offering higher efficiency, suitable for deployment on edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>Server-side text detection model for PP-OCRv4, offering higher accuracy, suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>63.8</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>Mobile-side text detection model for PP-OCRv4, offering higher efficiency, suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Recognition Module:</b></summary>
<table>
<tr>
<th>Model</th><th>Download Links</th>
<th>Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Training Model</a></td>
<td>86.38</td>
<td> - </td>
<td> - </td>
<td>205</td>
<td>PP-OCRv5_server_rec is a next-generation text recognition model designed to efficiently and accurately support Simplified Chinese, Traditional Chinese, English, and Japanese, as well as complex scenarios like handwriting, vertical text, pinyin, and rare characters. It balances recognition performance with inference speed and robustness, providing reliable support for document understanding across diverse scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>81.29</td>
<td> - </td>
<td> - </td>
<td>128</td>
<td>PP-OCRv5_mobile_rec is a next-generation lightweight text recognition model optimized for efficiency and accuracy across Simplified Chinese, Traditional Chinese, English, and Japanese, including complex scenarios like handwriting and vertical text. It delivers robust performance while maintaining fast inference speeds.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Training Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>181</td>
<td>PP-OCRv4_server_rec_doc is trained on a hybrid dataset of Chinese document data and PP-OCR training data, enhancing recognition for Traditional Chinese, Japanese, and special characters. It supports 15,000+ characters and improves both document-specific and general text recognition.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>88</td>
<td>PP-OCRv4's lightweight recognition model, optimized for fast inference on edge devices and various hardware platforms.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>151</td>
<td>PP-OCRv4's server-side model, delivering high accuracy for deployment on various servers.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>66</td>
<td>An ultra-lightweight English recognition model based on PP-OCRv4, supporting English and numeric characters.</td>
</tr>
</table>


> ❗ The above table highlights <b>6 core models</b> from the text recognition module, which includes <b>10 full models</b> in total, covering multiple multilingual recognition models. For the complete list:

<details><summary> 👉 Full Model Details</summary>

* <b>PP-OCRv5 Multi-Scene Models</b>

<table>
<tr>
<th>Model</th><th>Download Links</th>
<th>Chinese Accuracy(%)</th>
<th>English Accuracy(%)</th>
<th>Traditional Chinese Accuracy(%)</th>
<th>Japanese Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Training Model</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td> - </td>
<td> - </td>
<td>205</td>
<td>PP-OCRv5_server_rec is a next-generation text recognition model supporting Simplified Chinese, Traditional Chinese, English, and Japanese, including complex scenarios like handwriting and vertical text.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td> - </td>
<td> - </td>
<td>128</td>
<td>PP-OCRv5_mobile_rec is a lightweight version optimized for efficiency and accuracy across multiple languages and scenarios.</td>
</tr>
</table>

* <b>Chinese Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Links</th>
<th>Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Training Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>181</td>
<td>Enhanced for document text recognition, supporting 15,000+ characters including Traditional Chinese and Japanese.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>88</td>
<td>Lightweight model optimized for edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>151</td>
<td>High-accuracy server-side model.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>75.43</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>138</td>
<td>Lightweight PP-OCRv3 model for edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Links</th>
<th>Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>126</td>
<td rowspan="1">
SVTRv2, developed by FVL's OpenOCR team, won first prize in the PaddleOCR Algorithm Challenge, improving end-to-end recognition accuracy by 6% over PP-OCRv4.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Links</th>
<th>Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>70</td>
<td rowspan="1">RepSVTR, a mobile-optimized version of SVTRv2, won first prize in the PaddleOCR Challenge, improving accuracy by 2.5% over PP-OCRv4 with comparable speed.</td>
</tr>
</table>

* <b>English Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Links</th>
<th>Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High-Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>66</td>
<td>Ultra-lightweight English recognition model supporting English and numeric characters.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>85</td>
<td>PP-OCRv3-based English recognition model.</td>
</tr>
</table>

* <b>Multilingual Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>114 M</td>
<td>Ultra-lightweight Korean recognition model based on PP-OCRv3, supporting Korean and numeric characters</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>120 M </td>
<td>Ultra-lightweight Japanese recognition model based on PP-OCRv3, supporting Japanese and numeric characters</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>152 M </td>
<td>Ultra-lightweight Traditional Chinese recognition model based on PP-OCRv3, supporting Traditional Chinese and numeric characters</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>85 M </td>
<td>Ultra-lightweight Telugu recognition model based on PP-OCRv3, supporting Telugu and numeric characters</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>85 M </td>
<td>Ultra-lightweight Kannada recognition model based on PP-OCRv3, supporting Kannada and numeric characters</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>85 M </td>
<td>Ultra-lightweight Tamil recognition model based on PP-OCRv3, supporting Tamil and numeric characters</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>85 M</td>
<td>Ultra-lightweight Latin recognition model based on PP-OCRv3, supporting Latin and numeric characters</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>85 M</td>
<td>Ultra-lightweight Arabic script recognition model based on PP-OCRv3, supporting Arabic script and numeric characters</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>85 M  </td>
<td>Ultra-lightweight Cyrillic script recognition model based on PP-OCRv3, supporting Cyrillic script and numeric characters</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>85 M</td>
<td>Ultra-lightweight Devanagari script recognition model based on PP-OCRv3, supporting Devanagari script and numeric characters</td>
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

<details><summary><b>More command-line parameters available. Click to expand for details.</b></summary>  
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
<td><code>doc_orientation_classify_model_name</code></td>  
<td>Name of the document orientation classification model. If <code>None</code>, the default pipeline model is used.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>doc_orientation_classify_model_dir</code></td>  
<td>Directory path of the document orientation classification model. If <code>None</code>, the official model is downloaded.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>doc_unwarping_model_name</code></td>  
<td>Name of the text image correction model. If <code>None</code>, the default pipeline model is used.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>doc_unwarping_model_dir</code></td>  
<td>Directory path of the text image correction model. If <code>None</code>, the official model is downloaded.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_detection_model_name</code></td>  
<td>Name of the text detection model. If <code>None</code>, the default pipeline model is used.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_detection_model_dir</code></td>  
<td>Directory path of the text detection model. If <code>None</code>, the official model is downloaded.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_line_orientation_model_name</code></td>  
<td>Name of the text line orientation model. If <code>None</code>, the default pipeline model is used.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_line_orientation_model_dir</code></td>  
<td>Directory path of the text line orientation model. If <code>None</code>, the official model is downloaded.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_line_orientation_batch_size</code></td>  
<td>Batch size for the text line orientation model. If <code>None</code>, defaults to <code>1</code>.</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_recognition_model_name</code></td>  
<td>Name of the text recognition model. If <code>None</code>, the default pipeline model is used.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_recognition_model_dir</code></td>  
<td>Directory path of the text recognition model. If <code>None</code>, the official model is downloaded.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_recognition_batch_size</code></td>  
<td>Batch size for the text recognition model. If <code>None</code>, defaults to <code>1</code>.</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_doc_orientation_classify</code></td>  
<td>Whether to enable document orientation classification. If <code>None</code>, defaults to pipeline initialization value (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_doc_unwarping</code></td>  
<td>Whether to enable text image correction. If <code>None</code>, defaults to pipeline initialization value (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_textline_orientation</code></td>  
<td>Whether to enable text line orientation classification. If <code>None</code>, defaults to pipeline initialization value (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_limit_side_len</code></td>  
<td>Maximum side length limit for text detection.  
<ul>  
<li><b>int</b>: Any integer > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization value (<code>960</code>).</li>  
</ul>  
</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_limit_type</code></td>  
<td>Side length limit type for text detection.  
<ul>  
<li><b>str</b>: Supports <code>min</code> (ensures shortest side ≥ <code>det_limit_side_len</code>) or <code>max</code> (ensures longest side ≤ <code>limit_side_len</code>);</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization value (<code>max</code>).</li>  
</ul>  
</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_thresh</code></td>  
<td>Pixel threshold for text detection. Pixels with scores > this threshold are considered text.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization value (<code>0.3</code>).</li>  
</ul>  
</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_box_thresh</code></td>  
<td>Box threshold for text detection. Detected regions with average scores > this threshold are retained.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization value (<code>0.6</code>).</li>  
</ul>  
</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_unclip_ratio</code></td>  
<td>Expansion ratio for text detection. Larger values expand text regions more.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization value (<code>2.0</code>).</li>  
</ul>  
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
<td>Score threshold for text recognition. Results with scores > this threshold are retained.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization value (<code>0.0</code>, no threshold).</li>  
</ul>  
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
<td>Specifies the OCR model language.  
<ul>  
<li><b>ch</b>: Chinese;</li>  
<li><b>en</b>: English;</li>  
<li><b>korean</b>: Korean;</li>  
<li><b>japan</b>: Japanese;</li>  
<li><b>chinese_cht</b>: Traditional Chinese;</li>  
<li><b>te</b>: Telugu;</li>  
<li><b>ka</b>: Kannada;</li>  
<li><b>ta</b>: Tamil;</li>  
<li><b>None</b>: If <code>None</code>, defaults to <code>ch</code>.</li>  
</ul>  
</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>ocr_version</code></td>  
<td>OCR model version.  
<ul>  
<li><b>PP-OCRv5</b>: Uses PP-OCRv5 models;</li>  
<li><b>PP-OCRv4</b>: Uses PP-OCRv4 models;</li>  
<li><b>PP-OCRv3</b>: Uses PP-OCRv3 models;</li>  
<li><b>None</b>: If <code>None</code>, defaults to PP-OCRv5 models.</li>  
</ul>  
</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>input</code></td>  
<td>Input data (required). Supports:  
<ul>  
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> image data;</li>  
<li><b>str</b>: Local file path (e.g., <code>/root/data/img.jpg</code>), URL (e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">example</a>), or directory (e.g., <code>/root/data/</code>);</li>  
<li><b>List</b>: List of inputs, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>.</li>  
</ul>  
</td>  
<td><code>Python Var|str|list</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>save_path</code></td>  
<td>Path to save inference results. If <code>None</code>, results are not saved locally.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>device</code></td>  
<td>Device for inference. Supports:  
<ul>  
<li><b>CPU</b>: <code>cpu</code>;</li>  
<li><b>GPU</b>: <code>gpu:0</code> (first GPU);</li>  
<li><b>NPU</b>: <code>npu:0</code>;</li>  
<li><b>XPU</b>: <code>xpu:0</code>;</li>  
<li><b>MLU</b>: <code>mlu:0</code>;</li>  
<li><b>DCU</b>: <code>dcu:0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to GPU 0 (if available) or CPU.</li>  
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
<td>Whether to use TensorRT for acceleration.</td>  
<td><code>bool</code></td>  
<td><code>False</code></td>  
</tr>  
<tr>  
<td><code>min_subgraph_size</code></td>  
<td>Minimum subgraph size for model optimization.</td>  
<td><code>int</code></td>  
<td><code>3</code></td>  
</tr>  
<tr>  
<td><code>precision</code></td>  
<td>Computation precision (e.g., <code>fp32</code>, <code>fp16</code>).</td>  
<td><code>str</code></td>  
<td><code>fp32</code></td>  
</tr>  
<tr>  
<td><code>enable_mkldnn</code></td>  
<td>Whether to enable MKL-DNN acceleration. If <code>None</code>, enabled by default.</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>cpu_threads</code></td>  
<td>Number of CPU threads for inference.</td>  
<td><code>int</code></td>  
<td><code>8</code></td>  
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

## 2.2 Python Script Integration  

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
result = ocr.predict("./general_ocr_002.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")  
```  

The Python script above performs the following steps:  

<details><summary>(1) Initialize the OCR pipeline with <code>PaddleOCR()</code>. Parameter details:</summary>  

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
<td><code>doc_orientation_classify_model_name</code></td>  
<td>Name of the document orientation model. If <code>None</code>, uses the default pipeline model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>doc_orientation_classify_model_dir</code></td>  
<td>Directory path of the document orientation model. If <code>None</code>, downloads the official model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>doc_unwarping_model_name</code></td>  
<td>Name of the text image correction model. If <code>None</code>, uses the default pipeline model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>doc_unwarping_model_dir</code></td>  
<td>Directory path of the text image correction model. If <code>None</code>, downloads the official model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_detection_model_name</code></td>  
<td>Name of the text detection model. If <code>None</code>, uses the default pipeline model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_detection_model_dir</code></td>  
<td>Directory path of the text detection model. If <code>None</code>, downloads the official model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_line_orientation_model_name</code></td>  
<td>Name of the text line orientation model. If <code>None</code>, uses the default pipeline model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_line_orientation_model_dir</code></td>  
<td>Directory path of the text line orientation model. If <code>None</code>, downloads the official model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_line_orientation_batch_size</code></td>  
<td>Batch size for the text line orientation model. If <code>None</code>, defaults to <code>1</code>.</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_recognition_model_name</code></td>  
<td>Name of the text recognition model. If <code>None</code>, uses the default pipeline model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_recognition_model_dir</code></td>  
<td>Directory path of the text recognition model. If <code>None</code>, downloads the official model.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_recognition_batch_size</code></td>  
<td>Batch size for the text recognition model. If <code>None</code>, defaults to <code>1</code>.</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_doc_orientation_classify</code></td>  
<td>Whether to enable document orientation classification. If <code>None</code>, defaults to pipeline initialization (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_doc_unwarping</code></td>  
<td>Whether to enable text image correction. If <code>None</code>, defaults to pipeline initialization (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_textline_orientation</code></td>  
<td>Whether to enable text line orientation classification. If <code>None</code>, defaults to pipeline initialization (<code>True</code>).</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_limit_side_len</code></td>  
<td>Maximum side length limit for text detection.  
<ul>  
<li><b>int</b>: Any integer > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization (<code>960</code>).</li>  
</ul>  
</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_limit_type</code></td>  
<td>Side length limit type for text detection.  
<ul>  
<li><b>str</b>: Supports <code>min</code> (ensures shortest side ≥ <code>det_limit_side_len</code>) or <code>max</code> (ensures longest side ≤ <code>limit_side_len</code>);</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization (<code>max</code>).</li>  
</ul>  
</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_thresh</code></td>  
<td>Pixel threshold for text detection. Pixels with scores > this threshold are considered text.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization (<code>0.3</code>).</li>  
</ul>  
</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_box_thresh</code></td>  
<td>Box threshold for text detection. Detected regions with average scores > this threshold are retained.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization (<code>0.6</code>).</li>  
</ul>  
</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>text_det_unclip_ratio</code></td>  
<td>Expansion ratio for text detection. Larger values expand text regions more.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization (<code>2.0</code>).</li>  
</ul>  
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
<td>Score threshold for text recognition. Results with scores > this threshold are retained.  
<ul>  
<li><b>float</b>: Any float > <code>0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to pipeline initialization (<code>0.0</code>, no threshold).</li>  
</ul>  
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
<td>Specifies the OCR model language.  
<ul>  
<li><b>ch</b>: Chinese;</li>  
<li><b>en</b>: English;</li>  
<li><b>korean</b>: Korean;</li>  
<li><b>japan</b>: Japanese;</li>  
<li><b>chinese_cht</b>: Traditional Chinese;</li>  
<li><b>te</b>: Telugu;</li>  
<li><b>ka</b>: Kannada;</li>  
<li><b>ta</b>: Tamil;</li>  
<li><b>None</b>: If <code>None</code>, defaults to <code>ch</code>.</li>  
</ul>  
</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>ocr_version</code></td>  
<td>OCR model version.  
<ul>  
<li><b>PP-OCRv5</b>: Uses PP-OCRv5 models;</li>  
<li><b>PP-OCRv4</b>: Uses PP-OCRv4 models;</li>  
<li><b>PP-OCRv3</b>: Uses PP-OCRv3 models;</li>  
<li><b>None</b>: If <code>None</code>, defaults to PP-OCRv5 models.</li>  
</ul>  
</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>device</code></td>  
<td>Device for inference. Supports:  
<ul>  
<li><b>CPU</b>: <code>cpu</code>;</li>  
<li><b>GPU</b>: <code>gpu:0</code> (first GPU);</li>  
<li><b>NPU</b>: <code>npu:0</code>;</li>  
<li><b>XPU</b>: <code>xpu:0</code>;</li>  
<li><b>MLU</b>: <code>mlu:0</code>;</li>  
<li><b>DCU</b>: <code>dcu:0</code>;</li>  
<li><b>None</b>: If <code>None</code>, defaults to GPU 0 (if available) or CPU.</li>  
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
<td>Whether to use TensorRT for acceleration.</td>  
<td><code>bool</code></td>  
<td><code>False</code></td>  
</tr>  
<tr>  
<td><code>min_subgraph_size</code></td>  
<td>Minimum subgraph size for model optimization.</td>  
<td><code>int</code></td>  
<td><code>3</code></td>  
</tr>  
<tr>  
<td><code>precision</code></td>  
<td>Computation precision (e.g., <code>fp32</code>, <code>fp16</code>).</td>  
<td><code>str</code></td>  
<td><code>fp32</code></td>  
</tr>  
<tr>  
<td><code>enable_mkldnn</code></td>  
<td>Whether to enable MKL-DNN acceleration. If <code>None</code>, enabled by default.</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>cpu_threads</code></td>  
<td>Number of CPU threads for inference.</td>  
<td><code>int</code></td>  
<td><code>8</code></td>  
</tr>  
</tbody>  
</table>  
</details>  

<details><summary>(2) Call the <code>predict()</code> method for inference. Alternatively, <code>predict_iter()</code> returns a generator for memory-efficient batch processing. Parameters:</summary>  

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
<td>Input data (required). Supports:  
<ul>  
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> image data;</li>  
<li><b>str</b>: Local file path (e.g., <code>/root/data/img.jpg</code>), URL (e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">example</a>), or directory (e.g., <code>/root/data/</code>);</li>  
<li><b>List</b>: List of inputs, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>.</li>  
</ul>  
</td>  
<td><code>Python Var|str|list</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>device</code></td>  
<td>Same as initialization.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_doc_orientation_classify</code></td>  
<td>Whether to enable document orientation classification during inference.</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<tr>  
<td><code>use_doc_unwarping</code></td>  
<td>Whether to enable text image correction during inference.</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>use_textline_orientation</code></td>  
<td>Whether to enable text line orientation classification during inference.</td>  
<td><code>bool</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>text_det_limit_side_len</code></td>  
<td>Same as initialization.</td>  
<td><code>int</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>text_det_limit_type</code></td>  
<td>Same as initialization.</td>  
<td><code>str</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>text_det_thresh</code></td>  
<td>Same as initialization.</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>text_det_box_thresh</code></td>  
<td>Same as initialization.</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>text_det_unclip_ratio</code></td>  
<td>Same as initialization.</td>  
<td><code>float</code></td>  
<td><code>None</code></td>  
</tr>  
<td><code>text_rec_score_thresh</code></td>
<td>Same as initialization.</td>  
<td><code>float</code></td>
<td><code>None</code></td>
</table>
</details>

<details><summary>(3) Processing prediction results: Each sample's prediction result is a corresponding Result object, supporting printing, saving as images, and saving as <code>json</code> files:</summary>

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Explanation</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format output with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level for prettifying <code>JSON</code> output (only when <code>format_json=True</code>)</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-<code>ASCII</code> characters to <code>Unicode</code> (only when <code>format_json=True</code>)</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Output file path (uses input filename when directory specified)</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Indentation level for prettifying <code>JSON</code> output (only when <code>format_json=True</code>)</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Whether to escape non-<code>ASCII</code> characters (only when <code>format_json=True</code>)</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Output path (supports directory or file path)</td>
<td>None</td>
</tr>
</table>

- The <code>print()</code> method outputs results to terminal with the following structure:

    - <code>input_path</code>: <code>(str)</code> Input image path

    - <code>page_index</code>: <code>(Union[int, None])</code> PDF page number (if input is PDF), otherwise <code>None</code>

    - <code>model_settings</code>: <code>(Dict[str, bool])</code> Pipeline configuration
        - <code>use_doc_preprocessor</code>: <code>(bool)</code> Whether document preprocessing is enabled
        - <code>use_textline_orientation</code>: <code>(bool)</code> Whether text line orientation classification is enabled

    - <code>doc_preprocessor_res</code>: <code>(Dict[str, Union[str, Dict[str, bool], int]])</code> Document preprocessing results (only when <code>use_doc_preprocessor=True</code>)
        - <code>input_path</code>: <code>(Union[str, None])</code> Preprocessor input path (<code>None</code> for <code>numpy.ndarray</code> input)
        - <code>model_settings</code>: <code>(Dict)</code> Preprocessor configuration
            - <code>use_doc_orientation_classify</code>: <code>(bool)</code> Whether document orientation classification is enabled
            - <code>use_doc_unwarping</code>: <code>(bool)</code> Whether text image correction is enabled
        - <code>angle</code>: <code>(int)</code> Document orientation prediction (0-3 for 0°,90°,180°,270°; -1 if disabled)

    - <code>dt_polys</code>: <code>(List[numpy.ndarray])</code> Text detection polygons (4 vertices per box, shape=(4,2), dtype=int16)

    - <code>dt_scores</code>: <code>(List[float])</code> Text detection confidence scores

    - <code>text_det_params</code>: <code>(Dict[str, Dict[str, int, float]])</code> Text detection parameters
        - <code>limit_side_len</code>: <code>(int)</code> Image side length limit
        - <code>limit_type</code>: <code>(str)</code> Length limit handling method
        - <code>thresh</code>: <code>(float)</code> Text pixel classification threshold
        - <code>box_thresh</code>: <code>(float)</code> Detection box confidence threshold
        - <code>unclip_ratio</code>: <code>(float)</code> Text region expansion ratio
        - <code>text_type</code>: <code>(str)</code> Fixed as "general"

    - <code>textline_orientation_angles</code>: <code>(List[int])</code> Text line orientation predictions (actual angles when enabled, [-1,-1,-1] when disabled)

    - <code>text_rec_score_thresh</code>: <code>(float)</code> Text recognition score threshold

    - <code>rec_texts</code>: <code>(List[str])</code> Recognized texts (filtered by <code>text_rec_score_thresh</code>)

    - <code>rec_scores</code>: <code>(List[float])</code> Recognition confidence scores (filtered)

    - <code>rec_polys</code>: <code>(List[numpy.ndarray])</code> Filtered detection polygons (same format as <code>dt_polys</code>)

    - <code>rec_boxes</code>: <code>(numpy.ndarray)</code> Rectangular bounding boxes (shape=(n,4), dtype=int16) with [x_min, y_min, x_max, y_max] coordinates

- <code>save_to_json()</code> saves results to specified <code>save_path</code>:
  - Directory: saves as <code>save_path/{your_img_basename}_res.json</code>
  - File: saves directly to specified path
  - Note: Converts <code>numpy.array</code> to lists since JSON doesn't support numpy arrays

- <code>save_to_img()</code> saves visualization results:
  - Directory: saves as <code>save_path/{your_img_basename}_ocr_res_img.{your_img_extension}</code>
  - File: saves directly (not recommended for multiple images to avoid overwriting)

* Additionally, results with visualizations and predictions can be obtained through the following attributes:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Retrieves prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Retrieves visualized images in <code>dict</code> format</td>
</tr>
</table>

- The `json` attribute returns prediction results as a dict, with content identical to what's saved by the `save_to_json()` method.
- The `img` attribute returns prediction results as a dictionary containing two `Image.Image` objects under keys `ocr_res_img` (OCR result visualization) and `preprocessed_img` (preprocessing visualization). If the image preprocessing submodule isn't used, only `ocr_res_img` will be present.

</details>

## 3. Development Integration/Deployment

If the general OCR pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general OCR pipeline directly in your Python project, you can refer to the sample code in [2.2 Python Script Integration](#22-python-script-intergration).

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
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text image unwarping module</td>
<td>Fine-tuning not supported</td>
</tr>
<tr>
<td>Inaccurate textline rotation correction</td>
<td>Textline orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/textline_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Text detection misses</td>
<td>Text detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_detection.html">Link</a></td>
</tr>
<tr>
<td>Incorrect text recognition</td>
<td>Text recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html">Link</a></td>
</tr>
</tbody>
</table>
