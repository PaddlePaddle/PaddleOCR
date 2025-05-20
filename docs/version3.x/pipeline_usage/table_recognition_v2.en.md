---
comments: true
---

# General Table Recognition v2 Production Line User Guide

## 1. Introduction to General Table Recognition v2 Production Line

Table recognition is a technology that automatically identifies and extracts table content and its structure from documents or images. It is widely used in fields such as data entry, information retrieval, and document analysis. By using computer vision and machine learning algorithms, table recognition can convert complex table information into an editable format, making it easier for users to further process and analyze data.

The General Table Recognition v2 Production Line (PP-TableMagic) is designed to tackle table recognition tasks, identifying tables in images and outputting them in HTML format. Unlike the original General Table Recognition Production Line, this version introduces two new modules: table classification and table cell detection. By adopting a <b>multi-model pipeline combining "table classification + table structure recognition + cell detection"</b>, it achieves better end-to-end table recognition performance compared to the previous version. Based on this, the General Table Recognition v2 Production Line <b>natively supports targeted model fine-tuning</b>, allowing developers to customize it to varying degrees for satisfactory performance in different application scenarios. <b>Furthermore, the General Table Recognition v2 Production Line also supports end-to-end table structure recognition models (e.g., SLANet, SLANet_plus, etc.) and allows independent configuration for wired and wireless table recognition methods, enabling developers to freely select and combine the best table recognition solutions.</b>

This production line is applicable in a variety of fields, including general, manufacturing, finance, and transportation. It also provides flexible service deployment options, supporting multiple programming languages on various hardware. Additionally, it offers capabilities for secondary development, allowing you to train and fine-tune your own datasets based on this production line, with the trained models seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/01.png"/>

<b>The General Table Recognition Production Line v2 includes the following 8 modules. Each module can be trained and inferred independently and contains multiple models. For detailed information, please click on the corresponding module to view the documentation.</b>

- [Table Structure Recognition Module](../module_usage/table_structure_recognition.md)
- [Table Classification Module](../module_usage/table_classification.md)
- [Table Cell Detection Module](../module_usage/table_cells_detection.md)
- [Text Detection Module](../module_usage/text_detection.md)
- [Text Recognition Module](../module_usage/text_recognition.md)
- [Layout Region Detection Module](../module_usage/layout_detection.md) (optional)
- [Document Image Orientation Classification Module](../module_usage/doc_img_orientation_classification.md) (optional)
- [Text Image Unwarping Module](../module_usage/text_image_unwarping.md) (optional)

In this production line, you can choose the models to use based on the benchmark data below.

<details>
<summary> <b>Table Structure Recognition Module Models:</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td rowspan="1">SLANet is a table structure recognition model developed by Baidu PaddlePaddle's vision team. This model significantly improves the accuracy and inference speed of table structure recognition by using a CPU-friendly lightweight backbone network PP-LCNet, a high-low feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structure and location information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td rowspan="1">SLANet_plus is an enhanced version of the SLANet table structure recognition model developed by Baidu PaddlePaddle's vision team. Compared to SLANet, SLANet_plus significantly improves the recognition capabilities for wireless tables and complex tables, while reducing the model's sensitivity to table positioning accuracy, allowing for accurate recognition even if the table is slightly misaligned.</td>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
<td rowspan="2">The SLANeXt series is a new generation of table structure recognition models developed by Baidu PaddlePaddle's vision team. Compared to SLANet and SLANet_plus, SLANeXt focuses on recognizing table structures and has been trained with dedicated weights for recognizing wired and wireless tables, significantly enhancing recognition capabilities across both types, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
</details>

<details>
<summary> <b>Table Classification Module Models:</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
<td>94.2</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>6.6M</td>
</tr>
</table>
</details>

<details>
<summary> <b>Table Cell Detection Module Models:</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The Baidu PaddlePaddle vision team based RT-DETR-L as the base model, completing pre-training on a self-built table cell detection dataset, achieving good performance in detecting both wired and wireless table cells.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
</details>

<details>
<summary> <b>Text Detection Module Models:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
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

> ❗ The above section lists the **6 core models** that are primarily supported by the text recognition module. In total, the module supports **20 comprehensive models**, including multiple multilingual text recognition models. Below is the complete list of models:

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

* <b>Chinese Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Training Model</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>181 M</td>
<td>PP-OCRv4_server_rec_doc is based on PP-OCRv4_server_rec, trained with a mix of more Chinese document data and PP-OCR training data, increasing the recognition capabilities for some Traditional Chinese, Japanese, and special characters, supporting recognition of over 15,000 characters. In addition to improving the document-related text recognition capabilities, it also enhances general text recognition capabilities.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>88 M</td>
<td>PP-OCRv4's lightweight recognition model has high inference efficiency and can be deployed on various hardware, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>151 M</td>
<td>PP-OCRv4's server-side model has high inference accuracy and can be deployed on various servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>75.43</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>138 M</td>
<td>PP-OCRv3's lightweight recognition model has high inference efficiency and can be deployed on various hardware, including edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>126 M</td>
<td rowspan="1">SVTRv2 is a server-side text recognition model developed by the OpenOCR team at Fudan University's Vision and Learning Laboratory (FVL). It won first place in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task, achieving a 6% improvement in end-to-end recognition accuracy compared to PP-OCRv4.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>70 M</td>
<td rowspan="1">RepSVTR is a mobile text recognition model based on SVTRv2, which won first place in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task, achieving a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4, while maintaining the same inference speed.</td>
</tr>
</table>

* <b>English Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>66 M</td>
<td>This ultra-lightweight English recognition model is trained based on the PP-OCRv4 recognition model, supporting English and digit recognition.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>85 M </td>
<td>This ultra-lightweight English recognition model is trained based on the PP-OCRv3 recognition model, supporting English and digit recognition.</td>
</tr>
</table>

* <b>Multilingual Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
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
<td>This ultra-lightweight Korean recognition model is trained based on the PP-OCRv3 recognition model, supporting Korean and digit recognition.</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>120 M </td>
<td>This ultra-lightweight Japanese recognition model is trained based on the PP-OCRv3 recognition model, supporting Japanese and digit recognition.</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>152 M </td>
<td>This ultra-lightweight Traditional Chinese recognition model is trained based on the PP-OCRv3 recognition model, supporting Traditional Chinese and digit recognition.</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>85 M </td>
<td>This ultra-lightweight Telugu recognition model is trained based on the PP-OCRv3 recognition model, supporting Telugu and digit recognition.</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>85 M </td>
<td>This ultra-lightweight Kannada recognition model is trained based on the PP-OCRv3 recognition model, supporting Kannada and digit recognition.</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>85 M </td>
<td>This ultra-lightweight Tamil recognition model is trained based on the PP-OCRv3 recognition model, supporting Tamil and digit recognition.</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>85 M</td>
<td>This ultra-lightweight Latin recognition model is trained based on the PP-OCRv3 recognition model, supporting Latin and digit recognition.</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>85 M</td>
<td>This ultra-lightweight Arabic alphabet recognition model is trained based on the PP-OCRv3 recognition model, supporting Arabic letters and digit recognition.</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>85 M  </td>
<td>This ultra-lightweight Slavic alphabet recognition model is trained based on the PP-OCRv3 recognition model, supporting Slavic letters and digit recognition.</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>85 M</td>
<td>This ultra-lightweight Devanagari alphabet recognition model is trained based on the PP-OCRv3 recognition model, supporting Devanagari letters and digit recognition.</td>
</tr>
</table>
</details>
</details>

<details>
<summary> <b>Layout Region Detection Module Models:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
<td>This higher-precision layout region localization model is trained based on RT-DETR-L on a self-built dataset that includes Chinese and English papers, multi-column magazines, newspapers, PPTs, contracts, books, exam papers, research reports, ancient texts, Japanese documents, and vertical text documents.</td>
</tr>
<tr>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>This high-precision layout region localization model is trained based on RT-DETR-L on a self-built dataset that includes Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>This layout region localization model balances accuracy and efficiency, trained based on PicoDet-L on a self-built dataset that includes Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>This highly efficient layout region localization model is trained based on PicoDet-S on a self-built dataset that includes Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</td>
</tr>
</tbody>
</table>

> ❗ The above lists the <b>4 core models</b> that are key to the layout detection module. The module supports a total of <b>12 complete models</b>, including multiple pre-defined models for different categories. The complete model list is as follows:
<details><summary> 👉 Model List Details</summary>
* <b>Table Layout Detection Models</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>This high-efficiency layout region localization model is trained based on PicoDet-1x on a self-built dataset, capable of locating tables as one type of region.</td>
</tr>
</tbody></table>

* <b>3-Class Layout Detection Models, Including Tables, Images, and Stamps</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>This high-efficiency layout region localization model is trained using a self-built dataset of Chinese and English papers, magazines, and research reports based on the lightweight model PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>This layout region localization model balances efficiency and accuracy, trained using a self-built dataset of Chinese and English papers, magazines, and research reports based on the model PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>This high-precision layout region localization model is trained using a self-built dataset of Chinese and English papers, magazines, and research reports based on the model RT-DETR-H.</td>
</tr>
</tbody>
</table>

* <b>5-Class English Document Region Detection Models, Including Text, Titles, Tables, Images, and Lists</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>This high-efficiency English document layout region localization model is trained on the PubLayNet dataset.</td>
</tr>
</tbody></table>
</b>

* <b>17-Class Region Detection Models, Including 17 Common Layout Categories: Title, Image, Text, Number, Abstract, Content, Chart Title, Formula, Table, Table Title, References, Document Title, Footnote, Header, Algorithm, Footer, and Stamp</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>This high-efficiency layout region localization model is trained using a self-built dataset of Chinese and English papers, magazines, and research reports based on the lightweight model PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>This layout region localization model balances efficiency and accuracy, trained using a self-built dataset of Chinese and English papers, magazines, and research reports based on the model PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>This high-precision layout region localization model is trained using a self-built dataset of Chinese and English papers, magazines, and research reports based on the model RT-DETR-H.</td>
</tr>
</tbody>
</table>
</details>
</details>

<details>
<summary> <b>Text Image Unwarping Module Models (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>MS-SSIM (%)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>High-precision text image correction model.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>Document Image Orientation Classification Module Models (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
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
<td>Based on PP-LCNet_x1_0, this document image classification model includes four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>Testing Environment Information:</b></summary>

  <ul>
      <li><b>Performance Testing Environment</b>
          <ul>
            <li><strong>Test Dataset:
             </strong>
                <ul>
                  <li>Document Image Orientation Classification Model: A self-built dataset by PaddleX, covering multiple scenarios including identification documents and documents, containing 1,000 images.</li>
                  <li>Layout Region Detection Model: A self-built layout region detection dataset by PaddleOCR, containing 500 common document type images, including Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</li>
                  <li>Table Layout Detection Model: A self-built layout table region detection dataset by PaddleOCR, containing 7,835 images with tables in Chinese and English papers.</li>
                  <li>3-Class Layout Detection Model: A self-built layout region detection dataset by PaddleOCR, containing 1,154 common document type images including Chinese and English papers, magazines, and research reports.</li>
                  <li>5-Class English Document Region Detection Model: The evaluation dataset from <a href="https://developer.ibm.com/exchanges/data/all/publaynet">PubLayNet</a>, containing 11,245 images of English documents.</li>
                  <li>17-Class Region Detection Model: A self-built layout region detection dataset by PaddleOCR, containing 892 common document type images including Chinese and English papers, magazines, and research reports.</li>
                  <li>Table Structure Recognition Model: A high-difficulty Chinese table recognition dataset built internally by PaddleX.</li>
                  <li>Table Cell Detection Model: An evaluation set built internally by PaddleX.</li>
                  <li>Table Classification Model: An evaluation set built internally by PaddleX.</li>
                  <li>Text Detection Model: A Chinese dataset built by PaddleOCR, covering multiple scenarios including street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                  <li>Chinese Recognition Model: A Chinese dataset built by PaddleOCR, covering multiple scenarios including street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                  <li>ch_SVTRv2_rec: The evaluation set from <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> A-list.</li>
                  <li>ch_RepSVTR_rec: The evaluation set from <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> B-list.</li>
                  <li>English Recognition Model: An English dataset built internally by PaddleX.</li>
                  <li>Multilingual Recognition Model: A multilingual dataset built internally by PaddleX.</li>
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
      <li><b>Inference Mode Information</b></li>
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
            <td>Regular Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of prior precision type and acceleration strategy</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Optimal backend (Paddle/OpenVINO/TRT, etc.) selected based on prior knowledge</td>
        </tr>
    </tbody>
</table>

</details>
</details>

<br />
<b>If you prioritize model accuracy, please choose models with higher accuracy; if you care more about inference speed, please select models with faster inference speeds; if you focus on model storage size, please choose models with smaller storage volumes.</b>

## 2. Quick Start

Before using the table structure recognition V2 pipeline locally, please ensure that you have completed the installation of the wheel package according to the [installation guide](../installation.md). After installation, you can experience it locally using the command line or Python integration.

### 2.1 Command Line Experience

A single command allows you to quickly experience the effects of the table_recognition_v2 pipeline:

```bash
paddleocr table_recognition_v2 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition_v2.jpg

# Specify whether to use the document orientation classification model with --use_doc_orientation_classify
paddleocr table_recognition_v2 -i ./general_formula_recognition_001.png --use_doc_orientation_classify True

# Specify whether to use the text image unwarping module with --use_doc_unwarping
paddleocr table_recognition_v2 -i ./general_formula_recognition_001.png --use_doc_unwarping True

# Specify the device to use GPU for model inference with --device
paddleocr table_recognition_v2 -i ./general_formula_recognition_001.png --device gpu
```

<details><summary><b>More command line parameters are supported. Click to expand for detailed descriptions of the command line parameters</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types, required.
<ul>
<li><b>Python Var</b>: For example, image data represented as <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path to image files or PDF files: <code>/root/data/img.jpg</code>; <b>as URL links</b>, such as network URLs for image files or PDF files: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">example</a>; <b>as local directories</b>, the directory must contain images to be predicted, such as local path: <code>/root/data/</code> (currently, predictions do not support directories that contain PDF files; the PDF file must be specified to the specific file path).</li>
<li><b>List</b>: The elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
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
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td><code>table_classification_model_name</code></td>
<td>Name of the table classification model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the wired table structure recognition model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the wireless table structure recognition model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the wired table cell detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>Name of the wireless table cell detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the text image unwarping model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the text detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Image side length limit for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of the image side length limit for text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, while <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold. In the output probability map, only pixels with a score greater than this threshold will be considered text pixels.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.3</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold. When the average score of all pixels within the detection result box is greater than this threshold, the result is considered a text area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.6</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion coefficient. This method expands the text area; the larger this value, the larger the expanded area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>2.0</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold. Text results with a score greater than this threshold will be retained.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.0</code>. That is, no threshold is set.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load the document orientation classification module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load the text image unwarping module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to load the layout detection module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_ocr_model</code></td>
<td>Whether to load the OCR module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying a specific card number.
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which prioritizes using the local GPU device 0; if not available, it will use the CPU device.</li>
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
<td>Computation precision, such as fp32, fp16.</td>
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
<td>Number of threads to use for inference on the CPU.</td>
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

The running results will be printed to the terminal. The default configuration of the table_recognition_v2 pipeline's running results is as follows:

```
{'res': {'input_path': '/root/.paddlex/predict_input/table_recognition_v2.jpg', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True, 'use_ocr_model': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 18, 'label': 'chart', 'score': 0.6778535842895508, 'coordinate': [0, 0, 1281.0206, 585.5999]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[  4, 301],
        ...,
        [  4, 334]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.4, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1]), 'text_rec_score_thresh': 0, 'rec_texts': ['其'], 'rec_scores': array([0.97335929]), 'rec_polys': array([[[  4, 301],
        ...,
        [  4, 334]]], dtype=int16), 'rec_boxes': array([[  4, ..., 334]], dtype=int16)}, 'table_res_list': []}}
```

The visualization results are saved under `save_path`, and the visualization results are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/02.jpg">

### 2.2 Python Script Integration

The command line method is designed for quick experience and viewing effects. Generally, in a project, it is often necessary to integrate through code. You can complete quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddleocr import TableRecognitionPipelineV2

pipeline = TableRecognitionPipelineV2()
# ocr = TableRecognitionPipelineV2(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model with use_doc_orientation_classify
# ocr = TableRecognitionPipelineV2(use_doc_unwarping=True) # Specify whether to use the text image unwarping module with use_doc_unwarping
# ocr = TableRecognitionPipelineV2(device="gpu") # Specify the device to use GPU for model inference
output = pipeline.predict("./general_formula_recognition_001.png")
for res in output:
    res.print() ## Print the predicted structured output
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps are performed:

(1) Instantiate the general table recognition V2 pipeline object using `TableRecognitionPipelineV2()`. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td><code>table_classification_model_name</code></td>
<td>Name of the table classification model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the wired table structure recognition model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the wireless table structure recognition model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the wired table cell detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>Name of the wireless table cell detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the text image unwarping model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Name of the text detection model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Image side length limit for text detection.
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of the image side length limit for text detection.
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, while <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold. In the output probability map, only pixels with a score greater than this threshold will be considered text pixels.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.3</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold. When the average score of all pixels within the detection result box is greater than this threshold, the result is considered a text area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.6</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion coefficient. This method expands the text area; the larger this value, the larger the expanded area.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>2.0</code>.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>Name of the text recognition model. If set to <code>None</code>, the default model of the pipeline will be used.</td>
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
<td>Batch size for the text recognition model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold. Text results with a score greater than this threshold will be retained.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.0</code>. That is, no threshold is set.</li>
</ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load the document orientation classification module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load the text image unwarping module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to load the layout detection module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_ocr_model</code></td>
<td>Whether to load the OCR module. If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying a specific card number.
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which prioritizes using the local GPU device 0; if not available, it will use the CPU device.</li>
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
<td>Computation precision, such as fp32, fp16.</td>
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
<td>Number of threads to use for inference on the CPU.</td>
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

(2) Call the `predict()` method of the general table recognition V2 pipeline object to perform inference prediction, which returns a result list.

Additionally, the pipeline also provides the `predict_iter()` method. Both methods accept the same parameters and return results in the same way; the difference is that `predict_iter()` returns a `generator`, allowing for gradual processing and retrieval of prediction results, suitable for handling large datasets or for scenarios where memory savings are desired. You can choose to use either method based on your actual needs.

The parameters and descriptions of the `predict()` method are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types, required.
<ul>
<li><b>Python Var</b>: For example, image data represented as <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path to image files or PDF files: <code>/root/data/img.jpg</code>; <b>as URL links</b>, such as network URLs for image files or PDF files: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition_v2.jpg">example</a>; <b>as local directories</b>, the directory must contain images to be predicted, such as local path: <code>/root/data/</code> (currently, predictions do not support directories that contain PDF files; the PDF file must be specified to the specific file path).</li>
<li><b>List</b>: The elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Same as the parameters during instantiation.</td>
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
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to use the layout detection module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_ocr_model</code></td>
<td>Whether to use the <code>ocr</code> model during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Same as the parameters during instantiation.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Same as the parameters during instantiation.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Same as the parameters during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Same as the parameters during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Same as the parameters during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Same as the parameters during instantiation.</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td>Whether to use the wired end-to-end table recognition mode during inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td>Whether to use the wireless end-to-end table recognition mode during inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>Whether to use the wired table cell detection result direct-to-HTML mode during inference. If enabled, it directly constructs the HTML based on the geometric relationships of the wired table cell detection results.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_wireless_table_cells_trans_to_html</code></td>
<td>Whether to use the wireless table cell detection result direct-to-HTML mode during inference. If enabled, it directly constructs the HTML based on the geometric relationships of the wireless table cell detection results.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td>Whether to use the table orientation classification mode during inference. If enabled, it can correct the direction and correctly complete table recognition when the table in the image has 90/180/270-degree rotation.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td>Whether to use the cell-split OCR mode during inference. If enabled, it will split and re-recognize OCR detection results based on the cell prediction results to avoid missing text.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is a corresponding Result object, which supports printing, saving as an image, saving as an `xlsx` file, saving as an `HTML` file, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> keeps the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a json format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When it is a directory, the saved file will be named the same as the input file type.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> keeps the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supporting directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save results as an xlsx format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supporting directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save results as an html format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supporting directory or file path</td>
<td>None</td>
</tr>
</tbody>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates which page of the PDF it is; otherwise, it is `None`.

    - `model_settings`: `(Dict[str, bool])` Configuration parameters required by the pipeline.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout area detection sub-pipeline.
        - `use_ocr_model`: `(bool)` Controls whether to enable the OCR sub-pipeline.
    - `layout_det_res`: `(Dict[str, Union[List[numpy.ndarray], List[float]]])` Output results of the layout detection sub-module. Only exists when `use_layout_detection=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the layout detection area module, saved as `None` when the input is `numpy.ndarray`.
        - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates which page of the PDF it is; otherwise, it is `None`.
        - `boxes`: `(List[Dict])` List of detection boxes for the layout seal area. Each element in the list contains the following fields:
            - `cls_id`: `(int)` The category ID of the detection box.
            - `score`: `(float)` The confidence of the detection box.
            - `coordinate`: `(List[float])` The coordinates of the four vertices of the detection box, in the order of x1, y1, x2, y2, indicating the x coordinate and y coordinate of the top left corner and the bottom right corner.
    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` Output results of the document preprocessing sub-pipeline. Only exists when `use_doc_preprocessor=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the image preprocessing sub-pipeline, saved as `None` when the input is `numpy.ndarray`.
        - `model_settings`: `(Dict)` Configuration parameters for the preprocessing sub-pipeline.
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification.
            - `use_doc_unwarping`: `(bool)` Controls whether to enable text image unwarping.
        - `angle`: `(int)` Prediction result of the document orientation classification. When enabled, the values are [0,1,2,3], corresponding to [0°,90°,180°,270°]; when not enabled, it is -1.

    - `dt_polys`: `(List[numpy.ndarray])` List of polygon boxes for text detection. Each detection box is represented by a numpy array consisting of 4 vertex coordinates, with an array shape of (4, 2) and data type of int16.

    - `dt_scores`: `(List[float])` List of confidence scores for the text detection boxes.

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` Side length limit value during image preprocessing.
        - `limit_type`: `(str)` Processing method for side length limits.
        - `thresh`: `(float)` Confidence threshold for classifying text pixels.
        - `box_thresh`: `(float)` Confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` Expansion coefficient for text detection boxes.
        - `text_type`: `(str)` The type of text detection, currently fixed as "general".

    - `text_rec_score_thresh`: `(float)` Filtering threshold for text recognition results.

    - `rec_texts`: `(List[str])` List of text recognition results, including only the text with confidence exceeding `text_rec_score_thresh`.

    - `rec_scores`: `(List[float])` List of confidence scores for text recognition, filtered by `text_rec_score_thresh`.

    - `rec_polys`: `(List[numpy.ndarray])` List of text detection boxes that have been filtered by confidence, with the same format as `dt_polys`.

    - `rec_boxes`: `(numpy.ndarray)` Array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and dtype of int16. Each row represents the coordinates of a rectangular box as [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top left corner coordinate, and (x_max, y_max) is the bottom right corner coordinate.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`; if a file is specified, it will be saved directly to that file. Since json files do not support saving numpy arrays, the `numpy.array` types will be converted to list format.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`; if a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten and only the last image will be retained.)
- Calling the `save_to_html()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_table_1.html`; if a file is specified, it will be saved directly to that file. In the general table recognition V2 pipeline, the HTML format of the table in the image will be written to the specified HTML file.
- Calling the `save_to_xlsx()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.xlsx`; if a file is specified, it will be saved directly to that file. In the general table recognition V2 pipeline, the Excel format of the table in the image will be written to the specified xlsx file.

* Additionally, it also supports obtaining visualization images and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get visualization images in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is of dict type, and the relevant content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type data. The keys are `table_res_img`, `ocr_res_img`, `layout_res_img`, and `preprocessed_img`, and the corresponding values are four `Image.Image` objects, in the order of: visualization image of the table recognition result, visualization image of the OCR result, visualization image of the layout area detection result, and visualization image of the image preprocessing. If a certain sub-module is not used, the corresponding result image will not be included in the dictionary.

## 3. Development Integration/Deployment

If the model can meet your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the model directly in your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleOCR provides two other deployment methods, which are described in detail below:

🚀 High-Performance Inference: In actual production environments, many applications have strict performance criteria (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. To address this, PaddleOCR provides high-performance inference capabilities aimed at optimizing model inference and preprocessing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to [High-Performance Inference](../deployment/high_performance_inference.md).

☁️ Service-Oriented Deployment: Service-oriented deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services via network requests to obtain inference results. For detailed service-oriented deployment procedures, please refer to [Serving](../deployment/serving.md).

Below is the API reference for basic service-oriented deployment and examples of multilingual service calls:

<details><summary>API Reference</summary>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li>HTTP request method is POST.</li>
<li>Both request and response bodies are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the properties of the response body are as follows:</li>
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
<td>Request UUID.</td>
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
<li>When the request processing is unsuccessful, the properties of the response body are as follows:</li>
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
<td>Request UUID.</td>
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
<p>The main operation provided by the service is as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Locate and identify tables in the image.</p>
<p><code>POST /table-recognition</code></p>
<ul>
<li>The properties of the request body are as follows:</li>
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
<td>URL of an accessible image file or PDF file on the server, or the Base64 encoded result of the content of the above types of files. By default, for PDF files with more than 10 pages, only the first 10 pages will be processed.<br /> To lift the page limit, please add the following configuration in the model configuration file:
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
<td>File type. <code>0</code> represents a PDF file, <code>1</code> represents an image file. If this property is not present in the request body, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the <code>use_doc_orientation_classify</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the <code>use_doc_unwarping</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the <code>use_layout_detection</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrModel</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the <code>use_ocr_model</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the <code>layout_threshold</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the <code>layout_nms</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>Please refer to the <code>layout_unclip_ratio</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>layout_merge_bboxes_mode</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the <code>text_det_limit_side_len</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>text_det_limit_type</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the <code>text_det_thresh</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the <code>text_det_box_thresh</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the <code>text_det_unclip_ratio</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the <code>text_rec_score_thresh</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableCellsOcrResults</code></td>
<td><code>boolean</code></td>
<td>Please refer to the <code>use_table_cells_ocr_results</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Please refer to the <code>use_e2e_wired_table_rec_model</code> parameter description in the <code>predict</code> method of the model object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Please refer to the <code>use_e2e_wireless_table_rec_model</code> parameter description in the <code>predict</code> method of the model object.</td>
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
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tableRecResults</code></td>
<td><code>object</code></td>
<td>Table recognition results. The length of the array is 1 (for image input) or the actual number of processed document pages (for PDF input). For PDF input, each element in the array represents the result of each processed page in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Input data information.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tableRecResults</code> is an <code>object</code> with the following properties:</p>
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
<td>A simplified version of the JSON representation of the result generated by the <code>predict</code> method of the model object, where the <code>input_path</code> and <code>page_index</code> fields are removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Refer to the <code>img</code> property description of the model prediction results. The images are in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Input image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multilingual Service Call Examples</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/table-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["tableRecResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>
</details>
<br/>

## 4. Secondary Development

If the default model weights provided by the General Table Recognition v2 model are not satisfactory in terms of accuracy or speed for your scenario, you can try further <b>fine-tuning</b> the existing model using <b>your own domain-specific or application scenario data</b> to improve the recognition performance of the General Table Recognition v2 model in your scenario.

Since the General Table Recognition v2 model consists of several modules, if the model's performance is not as expected, it may be due to any of these modules. You can analyze images with poor recognition performance to determine which module is problematic and refer to the corresponding fine-tuning tutorial links in the following table for model fine-tuning.

<table>
<thead>
<tr>
<th>Situation</th>
<th>Fine-Tuning Module</th>
<th>Fine-Tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Table classification error</td>
<td>Table Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html#iv-secondary-development">Link</a></td>
</tr>
<tr>
<td>Table cell location error</td>
<td>Table Cell Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_cells_detection.html#iv-secondary-development">Link</a></td>
</tr>
<tr>
<td>Table structure recognition error</td>
<td>Table Structure Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/table_structure_recognition.html#4-secondary-development">Link</a></td>
</tr>
<tr>
<td>Failed to detect the area where the table is located</td>
<td>Layout Area Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Text detection missed</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_detection.html#4-custom-development">Link</a></td>
</tr>
<tr>
<td>Incorrect text content</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html#v-secondary-development">Link</a></td>
</tr>
<tr>
<td>Overall image rotation/table rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Image distortion correction is inaccurate</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>
