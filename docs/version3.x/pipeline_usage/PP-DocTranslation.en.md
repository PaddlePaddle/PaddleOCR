---
comments: true
---

# PP-DocTranslation Pipeline Usage Tutorial

## 1. Introduction to PP-DocTranslation Pipeline

PP-DocTranslation is a document intelligent translation solution provided by PaddlePaddle. It integrates advanced general layout analysis technology and large language model (LLM) capabilities to offer you efficient document intelligent translation services. This solution can accurately identify and extract various elements within documents, including text blocks, headings, paragraphs, images, tables, and other complex layout structures, and on this basis, achieve high-quality multilingual translation. PP-DocTranslation supports mutual translation among multiple mainstream languages, particularly excelling in handling documents with complex layouts and strong contextual dependencies, striving to deliver precise, natural, fluent, and professional translation results. This pipeline also provides flexible serving options, supporting the use of multiple programming languages on various hardware. Moreover, it offers the capability for secondary development, allowing you to train and fine-tune models on your own datasets based on this pipeline, and the trained models can also be seamlessly integrated.

<b>The PP-DocTranslation pipeline uses the PP-StructureV3 sub-pipeline, and thus has all the functions of the PP-StructureV3 pipeline. For more information on the functions and usage details of the PP-StructureV3 pipeline, you can click on the [PP-StructureV3 Pipeline Documentation](./PP-StructureV3.en.md) page.</b>

In this pipeline, you can select the model to use based on the benchmark data below.

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<details><summary> 👉Model List Details</summary>
<p><b>Document Image Orientation Classification Module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (M)</th>
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
<td>A document image classification model based on PP-LCNet_x1_0 with four classes: 0°, 90°, 180°, and 270°</td>
</tr>
</tbody>
</table>

<p><b>Text Image Unwarping Module:</b></p>
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
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Pretrained Model</a></td>
<td>0.179</td>
<td>30.3</td>
<td>High-accuracy text image unwarping model</td>
</tr>
</tbody>
</table>

<p><b>Layout Detection Module Models:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Pretrained Model</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01</td>
<td>High-accuracy layout detection model based on RT-DETR-L, trained on a custom dataset covering scenarios like Chinese/English papers, multi-column magazines, newspapers, PPTs, contracts, books, exams, research reports, ancient books, Japanese documents, and vertical text documents</td>
</tr>
<tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Pretrained Model</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>High-accuracy layout detection model based on RT-DETR-L, trained on a custom dataset covering papers, magazines, contracts, books, exams, and research reports</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Pretrained Model</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>Balanced accuracy-efficiency layout detection model based on PicoDet-L, trained on a custom dataset covering papers, magazines, contracts, books, exams, and research reports</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Pretrained Model</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>High-efficiency layout detection model based on PicoDet-S, trained on a custom dataset for papers, magazines, contracts, books, exams, and research reports</td>
</tr>
</tbody>
</table>

<p><b>Table Structure Recognition Module:</b></p>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Pretrained Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">85.92 / 85.92</td>
<td rowspan="2">- / 501.66</td>
<td rowspan="2">351M</td>
<td rowspan="2">SLANeXt series is a next-generation table structure recognition model developed by Baidu PaddlePaddle Vision Team. Compared with SLANet and SLANet_plus, SLANeXt focuses on recognizing table structures, with dedicated weights for wired and wireless tables, significantly improving performance especially for wired tables.</td>
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
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Pretrained Model</a></td>
<td>94.2</td>
<td>2.62 / 0.60</td>
<td>3.17 / 1.14</td>
<td>6.6M</td>
</tr>
</table>

<p><b>Table Cell Detection Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Pretrained Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">33.47 / 27.02</td>
<td rowspan="2">402.55 / 256.56</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. Baidu PaddlePaddle Vision Team used RT-DETR-L as the base and pre-trained on a custom table cell detection dataset, achieving strong performance on both wired and wireless tables.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Pretrained Model</a></td>
</tr>
</table>

<p><b>Text Detection Module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Pretrained Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>383.15 / 383.15</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model, higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Pretrained Model</a></td>
<td>79.0</td>
<td>10.67 / 6.36</td>
<td>57.77 / 28.15</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model, more efficient, suitable for edge device deployment</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Pretrained Model</a></td>
<td>69.2</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model, higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Pretrained Model</a></td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model, more efficient, suitable for edge device deployment</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">Pretrained Model</a></td>
<td>Accuracy similar to PP-OCRv4_mobile_det</td>
<td>9.90 / 3.60</td>
<td>41.93 / 20.76</td>
<td>2.1</td>
<td>PP-OCRv3 mobile-side text detection model, more efficient, suitable for edge device deployment</td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">Pretrained Model</a></td>
<td>Accuracy similar to PP-OCRv4_server_det</td>
<td>119.50 / 75.00</td>
<td>379.35 / 318.35</td>
<td>102.1</td>
<td>PP-OCRv3 server-side text detection model, higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
</tbody>
</table>

<p><b>Text Recognition Module Models:</b></p>

* <b>Chinese Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Training Model</a></td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec is a next-generation text recognition model. It aims to efficiently and accurately support four major languages—Simplified Chinese, Traditional Chinese, English, and Japanese—as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters. While maintaining recognition performance, it balances inference speed and model robustness, providing efficient and precise technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>81.29</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Training Model</a></td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>74.7</td>
<td>PP-OCRv4_server_rec_doc is trained on a mix of more Chinese document data and PP-OCR training data, based on PP-OCRv4_server_rec. It enhances recognition capabilities for Traditional Chinese, Japanese, and special characters, supporting 15,000+ characters. In addition to improving document-related text recognition, it also enhances general text recognition.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.6</td>
<td>The lightweight recognition model of PP-OCRv4, with high inference efficiency, deployable on various hardware devices including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>80.61 </td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>71.2</td>
<td>The server-side model of PP-OCRv4, with high inference accuracy, deployable on various servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>72.96</td>
<td>3.89 / 1.16</td>
<td>8.72 / 3.56</td>
<td>9.2</td>
<td>The lightweight recognition model of PP-OCRv3, with high inference efficiency, deployable on various hardware devices including edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>73.9</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team from Fudan University's Vision and Learning Lab (FVL). It won first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition, achieving a 6% improvement in end-to-end recognition accuracy over PP-OCRv4 on the A榜 leaderboard.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>22.1</td>
<td rowspan="1">RepSVTR is a mobile text recognition model based on SVTRv2. It won first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition, achieving a 2.5% improvement in end-to-end recognition accuracy over PP-OCRv4 on the B榜 leaderboard, with comparable inference speed.</td>
</tr>
</table>

* <b>English Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>6.8</td>
<td>An ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model, supporting English and numeric recognition.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>70.69</td>
<td>3.56 / 0.78</td>
<td>8.44 / 5.78</td>
<td>7.8 M </td>
<td>An ultra-lightweight English recognition model trained based on the PP-OCRv3 recognition model, supporting English and numeric recognition.</td>
</tr>
</table>

* <b>Multilingual Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>60.21</td>
<td>3.73 / 0.98</td>
<td>8.76 / 2.91</td>
<td>8.6</td>
<td>An ultra-lightweight Korean recognition model trained based on the PP-OCRv3 recognition model, supporting Korean and numeric recognition.</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>45.69</td>
<td>3.86 / 1.01</td>
<td>8.62 / 2.92</td>
<td>8.8 M </td>
<td>An ultra-lightweight Japanese recognition model trained based on the PP-OCRv3 recognition model, supporting Japanese and numeric recognition.</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>82.06</td>
<td>3.90 / 1.16</td>
<td>9.24 / 3.18</td>
<td>9.7 M </td>
<td>An ultra-lightweight Traditional Chinese recognition model trained based on the PP-OCRv3 recognition model, supporting Traditional Chinese and numeric recognition.</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>95.88</td>
<td>3.59 / 0.81</td>
<td>8.28 / 6.21</td>
<td>7.8 M </td>
<td>An ultra-lightweight Telugu recognition model trained based on the PP-OCRv3 recognition model, supporting Telugu and numeric recognition.</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.96</td>
<td>3.49 / 0.89</td>
<td>8.63 / 2.77</td>
<td>8.0 M </td>
<td>An ultra-lightweight Kannada recognition model trained based on the PP-OCRv3 recognition model, supporting Kannada and numeric recognition.</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.83</td>
<td>3.49 / 0.86</td>
<td>8.35 / 3.41</td>
<td>8.0 M </td>
<td>An ultra-lightweight Tamil recognition model trained based on the PP-OCRv3 recognition model, supporting Tamil and numeric recognition.</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>76.93</td>
<td>3.53 / 0.78</td>
<td>8.50 / 6.83</td>
<td>7.8</td>
<td>An ultra-lightweight Latin recognition model trained based on the PP-OCRv3 recognition model, supporting Latin and numeric recognition.</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>73.55</td>
<td>3.60 / 0.83</td>
<td>8.44 / 4.69</td>
<td>7.8</td>
<td>An ultra-lightweight Arabic script recognition model trained based on the PP-OCRv3 recognition model, supporting Arabic script and numeric recognition.</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>94.28</td>
<td>3.56 / 0.79</td>
<td>8.22 / 2.76</td>
<td>7.9 M  </td>
<td>An ultra-lightweight Cyrillic script recognition model trained based on the PP-OCRv3 recognition model, supporting Cyrillic script and numeric recognition.</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>96.44</td>
<td>3.60 / 0.78</td>
<td>6.95 / 2.87</td>
<td>7.9</td>
<td>An ultra-lightweight Devanagari script recognition model trained based on the PP-OCRv3 recognition model, supporting Devanagari script and numeric recognition.</td>
</tr>
</table>

<p><b>Text Line Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Download Link</th>
<th>Top-1 Acc(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>95.54</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.32</td>
<td>A text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees.</td>
</tr>
</tbody>
</table>

<p><b>Formula Recognition Module:</b></p>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg-BLEU(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<td>UniMERNet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>86.13</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.4 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. Trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, it significantly improves recognition accuracy for real-world scenarios.</td>
<td>PP-FormulaNet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.12</td>
<td>1311.84 / 1311.84</td>
<td>- / 8288.07</td>
<td>167.9</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by Baidu's PaddlePaddle Vision team, supporting 50,000 common LaTeX vocabulary items. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone and employs techniques like parallel masking and model distillation to significantly improve inference speed while maintaining high recognition accuracy, suitable for simple printed formulas, cross-line simple printed formulas, etc. The PP-FormulaNet-L version is based on Vary_VIT_B as its backbone and is trained on a large-scale formula dataset, showing significant improvement in complex formula recognition compared to PP-FormulaNet-S, suitable for simple printed formulas, complex printed formulas, handwritten formulas, etc.</td>

</tr>
<td>PP-FormulaNet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>92.13</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>535.2</td>
<td>LaTeX_OCR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>71.63</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>89.7</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. By using Hybrid ViT as the backbone and transformer as the decoder, it significantly improves the accuracy of formula recognition.</td>
</tr>
</tbody>
</table>

<p><b>Seal Text Recognition Module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Download Link</th>
<th>Detection Hmean(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.21</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td>The server-side seal text detection model of PP-OCRv4, with higher accuracy, suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.47</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.6</td>
<td>The mobile-side seal text detection model of PP-OCRv4, with higher efficiency, suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>

<strong>Testing Environment Description:</strong>

  <ul>
      <li><b>Performance Testing Environment</b>
          <ul>
            <li><strong>Test Datasets:
             </strong>
                <ul>
                  <li>Document Image Orientation Classification Model: A dataset built by PaddleX, covering multiple scenarios such as IDs and documents, containing 1,000 images.</li>
                  <li>Text Image Unwarping Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>.</li>
                  <li>Layout Detection Model: A layout analysis dataset built by PaddleOCR, containing 10,000 common document-type images such as Chinese and English papers, magazines, and reports.</li>
                  <li>PP-DocLayout_plus-L: A layout detection dataset built by PaddleOCR, containing 1,300 document-type images such as Chinese and English papers, magazines, newspapers, reports, PPTs, exams, and textbooks.</li>
                  <li>Table Structure Recognition Model: An internal English table recognition dataset built by PaddleX.</li>
                  <li>Text Detection Model: A Chinese dataset built by PaddleOCR, covering street views, web images, documents, and handwriting, with 500 images for detection.</li>
                  <li>Chinese Recognition Model: A Chinese dataset built by PaddleOCR, covering street views, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                  <li>ch_SVTRv2_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition</a> A榜 evaluation set.</li>
                  <li>ch_RepSVTR_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition</a> B榜 evaluation set.</li>
                  <li>English Recognition Model: An English dataset built by PaddleX.</li>
                  <li>Multilingual Recognition Model: A multilingual dataset built by PaddleX.</li>
                  <li>Text Line Orientation Classification Model: A dataset built by PaddleX, covering multiple scenarios such as IDs and documents, containing 1,000 images.</li>
                  <li>Seal Text Recognition Model: A dataset built by PaddleX, containing 500 circular seal images.</li>
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
            <td>Optimal combination of precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Optimal backend selection (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

</details>

## 2. Quick Start

Before using the PP-DocTranslation pipeline locally, please ensure that you have completed the installation of the wheel package according to the [Installation Tutorial](../installation.en.md). If you prefer to install dependencies selectively, please refer to the relevant instructions in the installation documentation. The corresponding dependency group for this pipeline is `trans`.

Please note: If you encounter issues such as the program becoming unresponsive, unexpected program termination, running out of memory resources, or extremely slow inference during execution, please try adjusting the configuration according to the documentation, such as disabling unnecessary features or using lighter-weight models.

Before use, you need to prepare the API key for a large language model, which supports the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or local large model services that comply with the OpenAI interface standards.

### 2.1 Experience via Command Line

You can download the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) and quickly experience the pipeline effect with a single command:

```bash
paddleocr pp_doctranslation -i vehicle_certificate-1.png --target_language en --qianfan_api_key your_api_key
```

<details><summary><b>Command line supports more parameter settings. Click to expand for detailed description of command line parameters</b></summary>
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
<td><b>Meaning:</b>Data to be predicted, required. <br/>
<b>Description:</b>
For example, local path of image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">example</a>; <b>local directory</b>, the directory must contain images to be predicted, such as local path: <code>/root/data/</code> (currently does not support PDF files in the directory, PDF files need to specify the exact file path).
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td><b>Meaning:</b>Specifies the path to save the inference result files. <br/>
<b>Description:</b>
If not set, inference results will not be saved locally.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>target_language</code></td>
<td><b>Meaning:</b>Target language (ISO 639-1 language code).</td>
<td><code>str</code></td>
<td><code>zh</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td><b>Meaning:</b>Model name for layout detection.<br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td><b>Meaning:</b>Directory path of the layout detection model.<br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td><b>Meaning:</b>Score threshold for layout model. <br/>
<b>Description:</b>Any float between <code>0-1</code>. If not set, the pipeline initialized value will be used, default initialized as <code>0.5</code>.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td><b>Meaning:</b>Whether to use post-processing NMS in layout detection. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default initialized as <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td><b>Meaning:</b>Expansion coefficient for detection boxes in layout detection model. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value will be used, default initialized as <code>1.0</code>.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td><b>Meaning:</b>Mode for merging detection boxes output by the layout detection model. <br/>
<b>Description:</b> 
<ul>
<li><b>large</b>: when set to large, among overlapping boxes, only the largest outer box is kept and the overlapping inner boxes are deleted;</li>
<li><b>small</b>: when set to small, among overlapping boxes, only the smaller inner boxes are kept and the overlapping outer boxes are deleted;</li>
<li><b>union</b>: no box filtering, both inner and outer boxes are kept;</li>
</ul>If not set, the pipeline initialized value will be used, default initialized as <code>large</code>.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td><b>Meaning:</b>Model name for chart parsing. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for chart parsing model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for chart parsing model. <br/>
<b>Description:</b> 
If not set, batch size defaults to <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td><b>Meaning:</b>Model name for region detection. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td><b>Meaning:</b>Directory path for region detection model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td><b>Meaning:</b>Model name for document orientation classification. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td><b>Meaning:</b>Directory path for document orientation classification model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td><b>Meaning:</b>Model name for text image unwarping.<br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td><b>Meaning:</b>Directory path for text image unwarping model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td><b>Meaning:</b>Model name for text detection. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td><b>Meaning:</b>Directory path for text detection model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td><b>Meaning:</b>Image side length limit for text detection. <br/>
<b>Description:</b> 
Any integer greater than <code>0</code>. If not set, the pipeline initialized value will be used, default initialized as <code>960</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td><b>Meaning:</b>Type of image side length limit for text detection. <br/>
<b>Description:</b> 
Supports <code>min</code> and <code>max</code>. <code>min</code> means ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> means ensuring the longest side of the image is not greater than <code>limit_side_len</code>. If not set, the pipeline initialized value will be used, default initialized as <code>max</code>.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td><b>Meaning:</b>Detection pixel threshold. In the output probability map, pixels with score greater than this threshold are considered text pixels. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.3</code> will be used by default.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td><b>Meaning:</b>Detection box threshold. If the average score of all pixels within the detected bounding box is greater than this threshold, the result is considered a text region. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.6</code> will be used by default.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td><b>Meaning:</b>Text detection expansion coefficient, used to expand text regions. The larger the value, the larger the expansion area. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>2.0</code> will be used by default.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td><b>Meaning:</b>Model name for textline orientation. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td><b>Meaning:</b>Directory path for textline orientation model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td><b>Meaning:</b>Batch size for textline orientation model. <br/>
<b>Description:</b> 
If not set, batch size defaults to <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td><b>Meaning:</b>Model name for text recognition. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for text recognition model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for text recognition model. <br/>
<b>Description:</b> 
If not set, batch size defaults to <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td><b>Meaning:</b>Text recognition threshold. Text results with scores greater than this threshold will be kept. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.0</code> will be used, meaning no threshold.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td><b>Meaning:</b>Model name for table classification. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td><b>Meaning:</b>Directory path for table classification model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td><b>Meaning:</b>Model name for wired table structure recognition. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for wired table structure recognition model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td><b>Meaning:</b>Model name for wireless table structure recognition. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for wireless table structure recognition model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td><b>Meaning:</b>Model name for wired table cells detection. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td><b>Meaning:</b>Directory path for wired table cells detection model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td><b>Meaning:</b>Model name for wireless table cells detection. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td><b>Meaning:</b>Directory path for wireless table cells detection model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_name</code></td>
<td><b>Meaning:</b>Model name for table orientation classification. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_dir</code></td>
<td><b>Meaning:</b>Directory path for table orientation classification model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td><b>Meaning:</b>Model name for seal text detection. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td><b>Meaning:</b>Directory path for seal text detection model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td><b>Meaning:</b>Image side length limit for seal text detection. <br/>
<b>Description:</b> 
Any integer greater than <code>0</code>. If not set, the pipeline initialized value will be used, default initialized as <code>736</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td><b>Meaning:</b>Type of image side length limit for seal text detection. <br/>
<b>Description:</b> 
Supports <code>min</code> and <code>max</code>. <code>min</code> means ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> means ensuring the longest side is not greater than <code>limit_side_len</code>. If not set, the pipeline initialized value will be used, default initialized as <code>min</code>.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td><b>Meaning:</b>Detection pixel threshold. In the output probability map, pixels with score greater than this threshold are considered text pixels. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.2</code> will be used by default.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td><b>Meaning:</b>Detection box threshold. If the average score of all pixels within the detected bounding box is greater than this threshold, the result is considered a text region. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.6</code> will be used by default.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td><b>Meaning:</b>Expansion coefficient for seal text detection. This method is used to expand the text region; the larger the value, the larger the expansion area. <br/>
<b>Description:</b> 
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.5</code> will be used by default.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td><b>Meaning:</b>Model name for seal text recognition. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for seal text recognition model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for seal text recognition model. <br/>
<b>Description:</b> 
If not set, batch size defaults to <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td><b>Meaning:</b>Text recognition threshold. Text results with scores greater than this threshold will be kept. <br/>
<b>Description:</b> 
</b>
Any float greater than <code>0</code>. If not set, the pipeline initialized value <code>0.0</code> will be used, meaning no threshold.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td><b>Meaning:</b>Model name for formula recognition. <br/>
<b>Description:</b> 
If not set, the pipeline default model will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for formula recognition model. <br/>
<b>Description:</b> 
If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size of the formula recognition model. <br/>
<b>Description:</b> 
If not set, the batch size defaults to <code>1</code>.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td><b>Meaning:</b>Whether to load and use the document orientation classification module. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>False</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td><b>Meaning:</b>Whether to load and use the text image unwarping module. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>False</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td><b>Meaning:</b>Whether to load and use the text line orientation classification module. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the seal text recognition sub-pipeline. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the table recognition sub-pipeline. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the formula recognition sub-pipeline. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the chart parsing module. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>False</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td><b>Meaning:</b>Whether to load and use the region detection module. <br/>
<b>Description:</b> 
If not set, the pipeline initialized value will be used, default is <code>True</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>qianfan_api_key</code></td>
<td><b>Meaning:</b>API key for the Qianfan platform.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td><b>Meaning:</b>Device used for inference.<br/>
<b>Description:</b> 
Supports specifying exact card number:
<ul>
<li><b>CPU</b>: e.g. <code>cpu</code> means using CPU for inference;</li>
<li><b>GPU</b>: e.g. <code>gpu:0</code> means using GPU #1 for inference;</li>
<li><b>NPU</b>: e.g. <code>npu:0</code> means using NPU #1 for inference;</li>
<li><b>XPU</b>: e.g. <code>xpu:0</code> means using XPU #1 for inference;</li>
<li><b>MLU</b>: e.g. <code>mlu:0</code> means using MLU #1 for inference;</li>
<li><b>DCU</b>: e.g. <code>dcu:0</code> means using DCU #1 for inference;</li>
<li><b>MetaX GPU</b>: e.g. <code>metax_gpu:0</code> means using MetaX GPU #1 for inference;</li>
<li><b>Iluvatar GPU</b>: e.g. <code>iluvatar_gpu:0</code> means using Iluvatar GPU #1 for inference;</li>
</ul>If not set, the pipeline initialized value will be used. At initialization, the local GPU device #0 will be preferred, if none, CPU device will be used.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td><b>Meaning:</b>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td><b>Meaning:</b>Whether to enable the TensorRT subgraph engine of Paddle Inference. <br/>
<b>Description:</b> 
If the model does not support acceleration by TensorRT, enabling this flag will not enable acceleration.<br/>
For PaddlePaddle with CUDA 11.8, compatible TensorRT version is 8.x (x≥6), recommended TensorRT version is 8.6.1.6.<br/>
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td><b>Meaning:</b>Computation precision, e.g. fp32, fp16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td><b>Meaning:</b>Whether to enable MKL-DNN accelerated inference. 
<b>Description:</b> 
If MKL-DNN is unavailable or the model does not support acceleration via MKL-DNN, enabling this flag will not enable acceleration.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td><b>Meaning:</b>MKL-DNN cache capacity.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td><b>Meaning:</b>Number of threads used for inference on CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td><b>Meaning:</b>Path to PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>
<br />

The execution results will be printed to the terminal.

### 2.2 Integration via Python Script

The command-line method is for quickly experiencing and viewing the results. Generally, in projects, integration via code is often required. You can download the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) and use the following sample code for inference:

```python
from paddleocr import PPDocTranslation

# Create a translation pipeline
pipeline = PPDocTranslation()

# Document path
input_path = "document_sample.pdf"

# Output directory
output_path = "./output"

# Large model configuration
chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

if input_path.lower().endswith(".md"):
    # Read markdown documents, supporting passing in directories and url links with the .md suffix
    ori_md_info_list = pipeline.load_from_markdown(input_path)
else:
    # Use PP-StructureV3 to perform layout parsing on PDF/image documents to obtain markdown information
    visual_predict_res = pipeline.visual_predict(
        input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
use_table_recognition=True,
    )

    ori_md_info_list = []
    for res in visual_predict_res:
        layout_parsing_result = res["layout_parsing_result"]
        ori_md_info_list.append(layout_parsing_result.markdown)
        layout_parsing_result.save_to_img(output_path)
        layout_parsing_result.save_to_markdown(output_path)

    # Concatenate the markdown information of multi-page documents into a single markdown file, and save the merged original markdown text
    if input_path.lower().endswith(".pdf"):
        ori_md_info = pipeline.concatenate_markdown_pages(ori_md_info_list)
        ori_md_info.save_to_markdown(output_path)

# Perform document translation (target language: English)
tgt_md_info_list = pipeline.translate(
    ori_md_info_list=ori_md_info_list,
    target_language="en",
    chunk_size=5000,
    chat_bot_config=chat_bot_config,
)
# Save the translation results
for tgt_md_info in tgt_md_info_list:
    tgt_md_info.save_to_markdown(output_path)
```

After executing the above code, you will obtain the parsed results of the original document to be translated, the Markdown file of the original text to be translated, and the Markdown file of the translated document, all saved in the <code>output</code> directory.

The process, API description, and output description of PP-DocTranslation prediction are as follows:

<details><summary>(1) Instantiate the PP-DocTranslation pipeline object by calling <code>PPDocTranslation</code>.</summary>

Relevant parameter descriptions are as follows:

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
<td><b>Meaning:</b>The model name for layout detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the layout detection model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td><b>Meaning:</b>Score threshold for the layout model.<br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: Any float between <code>0-1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code>, where the key is the class ID and the value is the threshold for that class;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value will be used, defaulting to <code>0.5</code>.</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td><b>Meaning:</b>Whether to use post-processing NMS for layout detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's initialized value will be used, defaulting to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td><b>Meaning:</b>Expansion coefficient for detection boxes in the layout detection model. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: Expansion coefficients in horizontal and vertical directions respectively;</li>
<li><b>dict</b>: Keys are <b>int</b> representing <code>cls_id</code>, values are <b>tuple</b>, e.g. <code>{0: (1.1, 2.0)}</code>, meaning for class 0 detection boxes, center remains unchanged, width expanded by 1.1 times, height expanded by 2.0 times;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value will be used, defaulting to <code>1.0</code>.</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td><b>Meaning:</b>Overlap box filtering method for layout detection. <br/>
<b>Description:</b> 
<ul>
<li><b>str</b>: <code>large</code>, <code>small</code>, <code>union</code>, indicating whether to keep the larger box, smaller box, or both during overlap filtering;</li>
<li><b>dict</b>: Keys are <b>int</b> <code>cls_id</code>, values are <b>str</b>, e.g. <code>{0: "large", 2: "small"}</code>, meaning use "large" mode for class 0 boxes and "small" mode for class 2 boxes;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value will be used, defaulting to <code>large</code>.</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td><b>Meaning:</b>The model name for chart parsing. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the chart parsing model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for the chart parsing model. <br/>
<b>Description:</b> 
If set to <code>None</code>, batch size defaults to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td><b>Meaning:</b>The model name for region detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the region detection model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td><b>Meaning:</b>The model name for document orientation classification. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the document orientation classification model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td><b>Meaning:</b>The model name for text image unwarping. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the text image unwarping model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td><b>Meaning:</b>The model name for text detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the text detection model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td><b>Meaning:</b>Image side length limit for text detection. <br/>
<b>Description:</b> 
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value will be used, defaulting to <code>960</code>.</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td><b>Meaning:</b>Type of image side length limit for text detection. <br/>
<b>Description:</b> 
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> means ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> means ensuring the longest side is not greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value will be used, defaulting to <code>max</code>.</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td><b>Meaning:</b>Pixel threshold for detection; pixels in the output probability map with scores above this threshold are considered text pixels. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value of <code>0.3</code> will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td><b>Meaning:</b>Detection box threshold; when the average score of all pixels inside a detected box exceeds this threshold, it is considered a text region. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value of <code>0.6</code> will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td><b>Meaning:</b>Expansion coefficient for text detection; this method expands the text region, and the larger the value, the larger the expansion area. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value of <code>2.0</code> will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td><b>Meaning:</b>The model name for text line orientation classification. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the text line orientation model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td><b>Meaning:</b>Batch size for the text line orientation model. <br/>
<b>Description:</b> 
If set to <code>None</code>, batch size defaults to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td><b>Meaning:</b>The model name for text recognition. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the text recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for the text recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, batch size defaults to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td><b>Meaning:</b>Text recognition threshold; text results with scores greater than this threshold will be retained. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: Any float greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the pipeline's initialized value of <code>0.0</code> (no threshold) will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td><b>Meaning:</b>The model name for table classification. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the table classification model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td><b>Meaning:</b>The model name for wired table structure recognition. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the wired table structure recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td><b>Meaning:</b>The model name for wireless table structure recognition. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the wireless table structure recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td><b>Meaning:</b>The model name for wired table cell detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the wired table cell detection model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td><b>Meaning:</b>The model name for wireless table cell detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the wireless table cell detection model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_name</code></td>
<td><b>Meaning:</b>The model name for table orientation classification. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the table orientation classification model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td><b>Meaning:</b>The model name for seal text detection. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline's default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td><b>Meaning:</b>The directory path of the seal text detection model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td><b>Meaning:</b>Image side length limit for seal text detection. <br/>
<b>Description:</b> 
<ul>
<li><b>int</b>: any integer greater than <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the parameter value initialized by the pipeline will be used, with a default initialization of <code>736</code>.</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td><b>Meaning:</b>Type of image side length limit for seal text detection. <br/>
<b>Description:</b> 
<ul>
<li><b>str</b>: supports <code>min</code> and <code>max</code>, where <code>min</code> ensures the shortest image side is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures the longest image side is not greater than <code>limit_side_len</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the parameter value initialized by the pipeline will be used, with a default initialization of <code>min</code>.</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td><b>Meaning:</b>Detection pixel threshold. In the output probability map, pixels with scores above this threshold are considered text pixels. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: any floating number greater than <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the pipeline default parameter value <code>0.2</code> will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td><b>Meaning:</b>Detection box threshold. When the average score of all pixels within the detected bounding box is greater than this threshold, the result is considered a text region. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: any floating number greater than <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the pipeline default parameter value <code>0.6</code> will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td><b>Meaning:</b>Expansion coefficient for seal text detection. This method expands the text region; the larger the value, the larger the expansion area. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: any floating number greater than <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the pipeline default parameter value <code>0.5</code> will be used.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td><b>Meaning:</b>Name of the seal text recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for the seal text recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for the seal text recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the batch size defaults to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td><b>Meaning:</b>Seal text recognition threshold. Text results with scores above this threshold will be retained. <br/>
<b>Description:</b> 
<ul>
<li><b>float</b>: any floating number greater than <code>0</code>;</li>
<li><b>None</b>: if set to <code>None</code>, the pipeline default parameter value <code>0.0</code> will be used, meaning no threshold is set.</li>
</ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td><b>Meaning:</b>Name of the formula recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td><b>Meaning:</b>Directory path for the formula recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td><b>Meaning:</b>Batch size for the formula recognition model. <br/>
<b>Description:</b> 
If set to <code>None</code>, the batch size defaults to <code>1</code>.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td><b>Meaning:</b>Whether to load and use the document orientation classification module. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>False</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load and use the text image unwarping module. 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>False</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to load and use the text line orientation classification module. If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the seal text recognition sub-pipeline. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the table recognition sub-pipeline. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the formula recognition sub-pipeline. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td><b>Meaning:</b>Whether to load and use the chart parsing module. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>False</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td><b>Meaning:</b>Whether to load and use the document region detection module. <br/>
<b>Description:</b> 
If set to <code>None</code>, the pipeline initialized parameter value will be used, defaulting to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td><b>Meaning:</b>Large language model configuration information. <br/>
<b>Description:</b>
The configuration content is the following dict:
<pre><code>{
"module_name": "chat_bot",
"model_name": "ernie-3.5-8k",
"base_url": "https://qianfan.baidubce.com/v2",
"api_type": "openai",
"api_key": "api_key"  # Please set this to the actual API key
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td><b>Meaning:</b>Device used for inference. <br/>
<b>Description:</b> 
Supports specifying a specific card number:
<ul>
<li><b>CPU</b>: e.g. <code>cpu</code> means using CPU for inference;</li>
<li><b>GPU</b>: e.g. <code>gpu:0</code> means using the first GPU for inference;</li>
<li><b>NPU</b>: e.g. <code>npu:0</code> means using the first NPU for inference;</li>
<li><b>XPU</b>: e.g. <code>xpu:0</code> means using the first XPU for inference;</li>
<li><b>MLU</b>: e.g. <code>mlu:0</code> means using the first MLU for inference;</li>
<li><b>DCU</b>: e.g. <code>dcu:0</code> means using the first DCU for inference;</li>
<li><b>MetaX GPU</b>: e.g. <code>metax_gpu:0</code> means using the first MetaX GPU for inference;</li>
<li><b>Iluvatar GPU</b>: e.g. <code>iluvatar_gpu:0</code> means using the first Iluvatar GPU for inference;</li>
<li><b>None</b>: if set to <code>None</code>, initialization will prioritize using the local GPU device 0; if unavailable, CPU will be used.</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td><b>Meaning:</b>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td><b>Meaning:</b>Whether to enable Paddle Inference’s TensorRT subgraph engine. <br/>
<b>Description:</b> 
If the model does not support acceleration via TensorRT, enabling this flag will have no effect.<br/>
For Paddle with CUDA 11.8, the compatible TensorRT version is 8.x (x≥6), recommended installation is TensorRT 8.6.1.6.<br/>
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td><b>Meaning:</b>Computation precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td><b>Meaning:</b>Whether to enable MKL-DNN accelerated inference. <br/>
<b>Description:</b> 
If MKL-DNN is unavailable or the model does not support acceleration via MKL-DNN, enabling this flag will have no effect.
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td><b>Meaning:</b>MKL-DNN cache capacity.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td><b>Meaning:</b>Number of threads used during inference on CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td><b>Meaning:</b>Path to the PaddleX pipeline configuration file.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>(2) Call the <code>visual_predict()</code> method of the PP-DocTranslation pipeline object to obtain visual prediction results. This method returns a list of results. Additionally, the pipeline provides a <code>visual_predict_iter()</code> method. Both methods accept the same parameters and return the same results, but <code>visual_predict_iter()</code> returns a <code>generator</code>, which can process and retrieve prediction results step-by-step, suitable for large datasets or memory-saving scenarios. You can choose either method according to your actual needs. Below are the parameters of the <code>visual_predict()</code> method and their descriptions:</summary>

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
<td><b>Meaning:</b>Data to be predicted, supports multiple input types, required.<br/>
<b>Description:</b> 
<ul>
  <li><b>Python Var</b>: image data such as <code>numpy.ndarray</code>;</li>
  <li><b>str</b>: local path of image or PDF files, e.g. <code>/root/data/img.jpg</code>; <b>URL link</b>: network URL of image or PDF files, e.g. <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">example</a>; <b>local directory</b>: directory containing images to be predicted, e.g. <code>/root/data/</code> (currently does not support PDFs in directories, PDF files need to specify exact file path);</li>
  <li><b>list</b>: list elements must be one of the above types, e.g. <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td><b>Meaning:</b>Whether to use the document orientation classification module during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td><b>Meaning:</b>Whether to use the text image unwarping module during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td><b>Meaning:</b>Whether to use the text line orientation classification module during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td><b>Meaning:</b>Whether to use the seal text recognition sub-pipeline during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td><b>Meaning:</b>Whether to use the table recognition sub-pipeline during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td><b>Meaning:</b>Whether to use the formula recognition sub-pipeline during inference.<br/>
<b>Description:</b> 
 Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td><b>Meaning:</b>Whether to use the chart parsing module during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td><b>Meaning:</b>Whether to use the document layout detection module during inference. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td><b>Meaning:</b>Parameter meaning is basically the same as the instantiated parameter. <br/>
<b>Description:</b> 
Setting to <code>None</code> means using the instantiated parameter, otherwise this parameter has higher priority.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td><b>Meaning:</b>Whether to enable direct conversion of wired table cell detection results to HTML.<br/> 
<b>Description:</b> 
When enabled, HTML is constructed directly based on the geometric relations of wired table cell detection results.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_wireless_table_cells_trans_to_html</code></td>
<td><b>Meaning:</b>Whether to enable direct conversion of wireless table cell detection results to HTML. <br/>
<b>Description:</b> 
When enabled, HTML is constructed directly based on the geometric relations of wireless table cell detection results.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td><b>Meaning:</b>Whether to enable table orientation classification. <br/>
<b>Description:</b> 
When enabled, tables with 90/180/270 degree rotations in images can be corrected in orientation and correctly recognized.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td><b>Meaning:</b>Whether to enable OCR segmentation by table cells. <br/>
<b>Description:</b> 
When enabled, OCR detection results are segmented and re-recognized based on cell prediction results to avoid missing text.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td><b>Meaning:</b>Whether to enable end-to-end wired table recognition mode. <br/>
<b>Description:</b> 
When enabled, the cell detection model is not used, only the table structure recognition model is used.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td><b>Meaning:</b>Whether to enable end-to-end wireless table recognition mode. <br/>
<b>Description:</b> 
When enabled, the cell detection model is not used, only the table structure recognition model is used.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</table>
</details>

<details><summary>(3) Processing visual prediction results: Each sample's prediction result is a corresponding Result object, supporting operations such as printing, saving as images, and saving as <code>json</code> files:</summary>

<table>

<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; if <code>False</code>, original characters are preserved. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path for saving. If a directory is specified, the saved file name matches the input file type name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; if <code>False</code>, original characters are preserved. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save visualized images from intermediate modules as PNG format images</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path for saving, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>Save each page of image or PDF files as separate markdown files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path for saving, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save tables in the file as HTML format files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path for saving, supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save tables in the file as XLSX format files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File path for saving, supports directory or file path</td>
<td>None</td>
</tr>
</table>

<ul>
<li>Calling the<code>print()</code> method will print the results to the terminal, with the following explanation of printed content:
    <ol start="1" type="1">
    <li><code>input_path</code>: <code>(str)</code> Input path of the image or PDF to be predicted</li>
    <li><code>page_index</code>: <code>(Union[int, None])</code> If the input is a PDF, this indicates the current page number; otherwise <code>None</code></li>
    <li><code>model_settings</code>: <code>(Dict[str, bool])</code> Model parameters configured for the pipeline
        <ol>
        <li><code>use_doc_preprocessor</code>: <code>(bool)</code> Controls whether to enable the document preprocessing sub-pipeline</li>
        <li><code>use_general_ocr</code>: <code>(bool)</code> Controls whether to enable the OCR sub-pipeline</li>
        <li><code>use_seal_recognition</code>: <code>(bool)</code> Controls whether to enable the seal recognition sub-pipeline</li>
        <li><code>use_table_recognition</code>: <code>(bool)</code> Controls whether to enable the table recognition sub-pipeline</li>
        <li><code>use_formula_recognition</code>: <code>(bool)</code> Controls whether to enable the formula recognition sub-pipeline</li>
        </ol>
    </li>
    <li><code>doc_preprocessor_res</code>: <code>(Dict[str, Union[List[float], str]])</code> Document preprocessing result dictionary, present only when <code>use_doc_preprocessor=True</code>
        <ol>
        <li><code>input_path</code>: <code>(str)</code> Input path of document preprocessing sub-pipeline, when input is <code>numpy.ndarray</code>, saved as <code>None</code>, here it is <code>None</code></li>
        <li><code>page_index</code>: <code>(None)</code> Page index of document preprocessing sub-pipeline, when input is <code>numpy.ndarray</code>, saved as <code>None</code>, here it is <code>None</code></li>
        <li><code>model_settings</code>: <code>(Dict[str, bool])</code> Model configuration parameters of document preprocessing sub-pipeline
          <ul>
          <li><code>use_doc_orientation_classify</code>: <code>(bool)</code> Controls whether to enable the document image orientation classification sub-pipeline</li>
          <li><code>use_doc_unwarping</code>: <code>(bool)</code> Controls whether to enable the document image unwarping sub-pipeline</li>
          </ul>
        <li><code>angle</code>: <code>(int)</code> Document image orientation classification sub-pipeline prediction result, returns actual angle value when enabled</li>
        </ul>
    </li>
    <li><code>parsing_res_list</code>: <code>(List[Dict])</code> List of parsing results, each element is a dictionary, the order of the list is the reading order after parsing.
        <ol>
        <li><code>block_bbox</code>: <code>(np.ndarray)</code> Bounding box of layout area, shape (4, 2), dtype int16; each row represents one rectangle</li>
        <li><code>block_label</code>: <code>(str)</code> Label of layout area, such as <code>text</code>, <code>table</code> etc.</li>
        <li><code>block_content</code>: <code>(str)</code> Content of layout area, such as text, table, formula etc.</li>
        <li><code>block_id</code>: <code>(int)</code> Index of layout area, used to display layout sorting results.</li>
        <li><code>block_order</code> <code>(int)</code> Order of layout area, used to display layout reading order, default value is <code>None</code> for non-ordered parts.</li>
        </ol>
    </li>
    <li><code>overall_ocr_res</code>: <code>(Dict[str, Union[List[str], List[float], numpy.ndarray]])</code> Global OCR result dictionary, present only when <code>use_general_ocr=True</code>
        <ol>
        <li><code>input_path</code>: <code>(Union[str, None])</code> Input path of OCR sub-pipeline, when input is <code>numpy.ndarray</code>, saved as <code>None</code></li>
        <li><code>page_index</code>: <code>(None)</code> Page index of OCR sub-pipeline, when input is <code>numpy.ndarray</code>, saved as <code>None</code></li>
        <li><code>model_settings</code>: <code>(Dict)</code> Model configuration parameters of OCR sub-pipeline</li>
        <li><code>dt_polys</code>: <code>(List[numpy.ndarray])</code> List of text detection polygons. Each detection box is represented by a numpy array of shape (4, 2), dtype int16, which contains the coordinates of the 4 vertices of the rectangle</li> 
        <li><code>dt_scores</code>: <code>(List[float])</code> List of confidence scores for text detection boxes</li>    
        <li><code>text_det_params</code>: <code>(Dict[str, Dict[str, int, float]])</code> Configuration parameters of the text detection module
            <ol>
            <li><code>limit_side_len</code>: <code>(int)</code> Side length limit value for image preprocessing</li>
            <li><code>limit_type</code>: <code>(str)</code> Side length limit handling method</li>
            </ol>
        <li><code>thresh</code>: <code>(float)</code> Text pixel classification confidence threshold</li>
        <li><code>box_thresh</code>: <code>(float)</code> Confidence threshold for text detection boxes</li>
        <li><code>unclip_ratio</code>: <code>(float)</code> Expansion factor for text detection boxes</li>
        <li><code>text_type</code>: <code>(str)</code> Type of text detection, currently fixed as "general"</li>
        <li><code>textline_orientation_angles</code>: <code>(List[int])</code> Text line direction classification prediction results. Returns actual angle value when enabled, such as [0,0,1]</li>
        <li><code>text_rec_score_thresh</code>: <code>(float)</code> Text recognition result filtering threshold</li>
        <li><code>rec_texts</code>: <code>(List[str])</code> Text recognition result list, only contains texts with confidence higher than <code>text_rec_score_thresh</code></li>
        <li><code>rec_scores</code>: <code>(List[float])</code> Text recognition confidence list, filtered by <code>text_rec_score_thresh</code></li>
        <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> List of text detection boxes filtered by confidence, format same as <code>dt_polys</code></li>
         </ol>  
    </li>
    <li><code>formula_res_list</code>: <code>(List[Dict[str, Union[numpy.ndarray, List[float], str]]])</code> List of formula recognition results, each element is a dictionary
        <ol>
        <li><code>rec_formula</code>: <code>(str)</code> Formula recognition result</li>
        <li><code>rec_polys</code>: <code>(numpy.ndarray)</code> Formula detection boxes, shape (4, 2), dtype int16</li>
        <li><code>formula_region_id</code>: <code>(int)</code> Region id of formula, used to display formula sorting results.</li>
        </ol>
    </li>
    <li><code>seal_res_list</code>: <code>(List[Dict[str, Union[numpy.ndarray, List[float], str]]])</code> List of seal recognition results, each element is a dictionary
        <ol>
        <li><code>input_path</code>: <code>(str)</code> Input path of seal image</li>
        <li><code>page_index</code>: <code>(None)</code> Page index of seal image, when input is <code>numpy.ndarray</code>, saved as <code>None</code></li>        
        <li><code>model_settings</code>: <code>(Dict)</code> Model configuration parameters of seal recognition sub-pipeline</li>
        <li><code>dt_polys</code>: <code>(List[numpy.ndarray])</code> Seal detection boxes list, format same as <code>dt_polys</code></li>
        <li><code>text_det_params</code>: <code>(Dict[str, Dict[str, int, float]])</code> Configuration parameters of seal detection module, same as <code>text_det_params</code></li>
        <li><code>text_type</code>: <code>(str)</code> Type of seal detection, currently fixed as "seal"</li>
        <li><code>text_rec_score_thresh</code>: <code>(float)</code> Seal recognition result filtering threshold</li>
        <li><code>rec_texts</code>: <code>(List[str])</code> List of seal recognition results, only contains texts with confidence higher than <code>text_rec_score_thresh</code></li>
        <li><code>rec_scores</code>: <code>(List[float])</code> List of confidence scores for seal recognition results, filtered by <code>text_rec_score_thresh</code></li>
        <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> List of seal detection boxes filtered by confidence, format same as <code>dt_polys</code></li>
        <li> <code>rec_boxes</code>: <code>(numpy.ndarray)</code> Rectangle bounding boxes of seal detection boxes, shape (n, 4), dtype int16. Each row represents a rectangle</li>
        </ol>
    </li>
    <li><code>table_res_list</code>: <code>(List[Dict[str, Union[numpy.ndarray, List[float], str]]])</code> List of table recognition results, each element is a dictionary
        <ol>
        <li><code>table_id</code>: <code>(int)</code> Index of table, used to display table sorting results.</li>
        <li><code>cell_box_list</code>: <code>(List[numpy.ndarray])</code> List of table cell detection boxes, format same as <code>dt_polys</code></li>
        <li><code>pred_html</code>: <code>(str)</code> HTML format string of table</li>
        <li><code>table_ocr_pred</code>: <code>(dict)</code> OCR recognition results of table
            <ul>
            <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> List of table cell detection boxes, format same as <code>dt_polys</code></li>
            <li><code>rec_texts</code>: <code>(List[str])</code> List of table cell recognition results</li>
            <li><code>rec_scores</code>: <code>(List[float])</code> List of confidence scores for table cell recognition results</li>
            <li><code>rec_boxes</code>: <code>(numpy.ndarray)</code> Rectangle bounding boxes of table cell detection boxes, shape (n, 4), dtype int16. Each row represents a rectangle
            </ul>
            </li>
        </li>
    </li>
    </ol>
</li>
<li>Calling the<code>save_to_json()</code> method will save the above content to the specified <code>save_path</code> .If a directory is specified, the saved path will be <code>save_path/{your_img_basename}_res.json</code>，If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, all <code>numpy.array</code> types will be converted to list format.</li>
<li>Calling the <code>save_to_img()</code> method will save visualization results to the specified <code>save_path</code>. If a directory is specified, it will save layout detection visual images, global OCR visual images, layout reading order visual images, etc. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly, or multiple images will be overwritten, leaving only the last image.)</li>
<li>Calling the <code>save_to_markdown()</code> method will save the converted Markdown files to the specified <code>save_path</code>. The saved file path will be <code>save_path/{your_img_basename}.md</code>. If the input is a PDF file, it is recommended to specify a directory directly, otherwise multiple markdown files will be overwritten.</li>
<li>Calling the <code>concatenate_markdown_pages()</code> method merges the multi-page Markdown contents output by the PP-DocTranslation pipeline <code>markdown_list</code> into a single complete document and returns the merged Markdown content.</li>
</ul>
</details>

<details><summary>(4) Call the <code>translate()</code> method to perform document translation. This method returns the original and translated markdown content as a markdown object, which can be saved locally by executing the <code>save_to_markdown()</code> method for the desired parts. Below are the relevant parameters of the <code>translate()</code> method:</summary>

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
<td><code>ori_md_info_list</code></td>
<td>List of original Markdown data containing content to be translated. Must be a list of dictionaries, each representing a document block</td>
<td><code>List[Dict]</code></td>
<td></td>
</tr>
<tr>
<td><code>target_language</code></td>
<td>Target language (ISO 639-1 language code, e.g. <code>"en"</code>/<code>"ja"</code>/<code>"fr"</code>)</td>
<td><code>str</code></td>
<td><code>"zh"</code></td>
</tr>
<tr>
<td><code>chunk_size</code></td>
<td>Character count threshold for chunked translation processing</td>
<td><code>int</code></td>
<td><code>5000</code></td>
</tr>
<tr>
<td><code>task_description</code></td>
<td>Custom task description prompt</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>output_format</code></td>
<td>Specified output format requirements, e.g. "preserve original Markdown structure"</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>rules_str</code></td>
<td>Custom translation rule description</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>few_shot_demo_text_content</code></td>
<td>Few-shot learning example text content</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>few_shot_demo_key_value_list</code></td>
<td>Structured few-shot example data in key-value pairs, can include professional terminology glossary</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>glossary</code></td>
<td>Professional terminology glossary for translation</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>llm_request_interval</code></td>
<td>Interval in seconds between requests to the large language model. This parameter helps prevent too frequent calls to the LLM.</td>
<td><code>float</code></td>
<td><code>0.0</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>Large language model configuration. Setting to <code>None</code> uses instantiation parameters; otherwise, this parameter takes priority.</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

## 3. Development Integration/Deployment

If the pipeline can meet your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the sample code in [2.2 Python Script Approach](#22-python脚本方式集成).

In addition, PaddleOCR also offers two other deployment methods, detailed as follows:

🚀 High-Performance Inference: In real-world production environments, many applications have stringent performance criteria (especially response speed) for deployment strategies to ensure efficient system operation and a smooth user experience. To this end, PaddleOCR provides high-performance inference capabilities, aiming to deeply optimize model inference and pre/post-processing, achieving significant acceleration in the end-to-end process. For detailed information on the high-performance inference process, please refer to [High-Performance Inference](../deployment/high_performance_inference.en.md).

☁️ Serving: Serving is a common deployment form in real-world production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. For detailed information on the pipeline serving process, please refer to [Serving](../deployment/serving.en.md).

Below are the API references for basic serving and examples of multi-language service invocation:

<details><summary>API Reference</summary>
<p>Main operations provided by the serving:</p>
<ul>
<li>HTTP request method is POST.</li>
<li>Both request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body has the following properties:</li>
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
<li>When the request is not successful, the response body has the following properties:</li>
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
<td>Error code. Same as response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>Main operations provided by the serving are as follows:</p>
<ul>
<li><b><code>analyzeImages</code></b></li>
</ul>
<p>Use computer vision models to analyze images, obtaining OCR, table recognition results, etc.</p>
<p><code>POST /doctrans-visual</code></p>
<ul>
<li>Request body properties are as follows:</li>
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
<td>URL of image or PDF file accessible by the server, or Base64 encoding of such file contents. By default, for PDF files over 10 pages, only the first 10 pages are processed.<br /> To remove the page limit, add the following configuration in the pipeline config file:
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
<td>File type. <code>0</code> means PDF, <code>1</code> means image file. If not present in the request, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_doc_orientation_classify</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_doc_unwarping</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_textline_orientation</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_seal_recognition</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_table_recognition</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_formula_recognition</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_chart_recognition</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useRegionDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>use_region_detection</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>object</code> | <code>null</code></td>
<td>See the <code>layout_threshold</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the <code>layout_nms</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>See the <code>layout_unclip_ratio</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>See the <code>layout_merge_bboxes_mode</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>See the <code>text_det_limit_side_len</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the <code>text_det_limit_type</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>text_det_thresh</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>text_det_box_thresh</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>text_det_unclip_ratio</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>text_rec_score_thresh</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>See the <code>seal_det_limit_side_len</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the <code>seal_det_limit_type</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>seal_det_thresh</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>seal_det_box_thresh</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>seal_det_unclip_ratio</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the <code>seal_rec_score_thresh</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWiredTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>See the <code>use_wired_table_cells_trans_to_html</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWirelessTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>See the <code>use_wireless_table_cells_trans_to_html</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableOrientationClassify</code></td>
<td><code>boolean</code></td>
<td>See the <code>use_table_orientation_classify</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrResultsWithTableCells</code></td>
<td><code>boolean</code></td>
<td>See the <code>use_ocr_results_with_table_cells</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>See the <code>use_e2e_wired_table_rec_model</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>See the <code>use_e2e_wireless_table_rec_model</code> parameter description in the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Whether to return visualization result images and intermediate images during processing.
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>If <code>true</code> is passed: return images.</li>
<li>If <code>false</code> is passed: do not return images.</li>
<li>If this parameter is not provided in the request body or <code>null</code> is passed: follow the pipeline config file setting <code>Serving.visualize</code>.</li>
</ul>
<br/>For example, add the following field in the pipeline config file:<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
By default, images will not be returned; the <code>visualize</code> parameter in the request body can override this default behavior. If neither the request body nor the config file sets it (or the request body passes <code>null</code> and the config file does not set it), images will be returned by default.
</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the response body's <code>result</code> has the following properties:</li>
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
<td>Layout parsing results. The array length is 1 (for image input) or equals the actual number of processed pages (for PDF input). For PDF input, each element corresponds to the result of each processed page in order.</td>
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
<td>Simplified version of the <code>res</code> field in the JSON representation of the <code>layout_parsing_result</code> generated by the pipeline object's <code>visual_predict</code> method, with <code>input_path</code> and <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>Markdown result.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the <code>img</code> property description in the pipeline prediction results. Images are in JPEG format and Base64 encoded.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Input image. JPEG format, Base64 encoded.</td>
</tr>
</tbody>
</table>
<p><code>markdown</code> is an <code>object</code> with the following properties:</p>
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
<td><code>text</code></td>
<td><code>string</code></td>
<td>Markdown text.</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>Key-value pairs of Markdown image relative paths and Base64 encoded images.</td>
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
</table>
<ul>
<li><b><code>translate</code></b></li>
</ul>
<p>Use a large model to translate documents.</p>
<p><code>POST /doctrans-translate</code></p>
<ul>
<li>Request body properties are as follows:</li>
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
<td><code>markdownList</code></td>
<td><code>array</code></td>
<td>List of Markdown to be translated. Can be obtained from the results of the <code>analyzeImages</code> operation. The <code>images</code> attribute will not be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>targetLanguage</code></td>
<td><code>string</code></td>
<td>Please refer to the <code>target_language</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>chunkSize</code></td>
<td><code>integer</code></td>
<td>Please refer to the <code>chunk_size</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>taskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>task_description</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>outputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>output_format</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>rulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>rules_str</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>fewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>few_shot_demo_text_content</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>fewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the <code>few_shot_demo_key_value_list</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>glossary</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the <code>glossary</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmRequestInterval</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the <code>llm_request_interval</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>chatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the <code>chat_bot_config</code> parameter description in the <code>translate</code> method of the pipeline object.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> in the response body has the following attributes:</li>
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
<td><code>translationResults</code></td>
<td><code>array</code></td>
<td>Translation results.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>translationResults</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>language</code></td>
<td><code>string</code></td>
<td>Target language.</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>Markdown result. Object definition is consistent with the <code>markdown</code> returned by the <code>analyzeImages</code> operation.</td>
</tr>
</tbody>
</table>

<li><b>Note: </b></li>Including sensitive parameters such as the API key for large model calls in the request body may pose security risks. If not necessary, set these parameters in the configuration file and do not pass them during the request.<br/><br/>
</details>
<details><summary>Examples of multi-language service invocation</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import pathlib
import pprint
import sys

import requests


API_BASE_URL = "http://127.0.0.1:8080"

file_path = "./demo.jpg"
target_language = "en"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data,
    "fileType": 1,
}
resp_visual = requests.post(url=f"{API_BASE_URL}/doctrans-visual", json=payload)
if resp_visual.status_code != 200:
    print(
        f"Request to doctrans-visual failed with status code {resp_visual.status_code}."
    )
    pprint.pp(resp_visual.json())
    sys.exit(1)
result_visual = resp_visual.json()["result"]

markdown_list = []
for i, res in enumerate(result_visual["layoutParsingResults"]):
    md_dir = pathlib.Path(f"markdown_{i}")
    md_dir.mkdir(exist_ok=True)
    (md_dir / "doc.md")
write_text(res["markdown"]["text"])
    for img_path, img in res["markdown"]["images"].items():
        img_path = md_dir / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base64.b64decode(img))
    print(f"The Markdown document to be translated is saved at {md_dir / 'doc.md'}")
    del res["markdown"]["images"]
    markdown_list.append(res["markdown"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")

payload = {
    "markdownList": markdown_list,
"targetLanguage": target_language,
}
resp_translate = requests.post(url=f"{API_BASE_URL}/doctrans-translate", json=payload)
if resp_translate.status_code != 200:
    print(
        f"Request to doctrans-translate failed with status code {resp_translate.status_code}."
    )
    pprint.pprint(resp_translate.json())  # Corrected 'pp' to 'pprint' for proper function call
    sys.exit(1)
result_translate = resp_translate.json()["result"]

for i, res in enumerate(result_translate["translationResults"]):
    md_dir = pathlib.Path(f"markdown_{i}")
    (md_dir / "doc_translated.md").write_text(res["markdown"]["text"])
    print(f"Translated markdown document saved at {md_dir / 'doc_translated.md'}")</code></pre></details>
</details>
<br/>

## 4. Secondary Development
If the default model weights provided by the PP-DocTranslation pipeline do not meet your accuracy or speed requirements in your scenario, you can try to use<b>your own data from specific domains or application scenarios</b>to further<b>fine-tune</b>the existing model to improve the recognition effect in your scenario.

### 4.1 Model Fine-tuning
Since the PP-DocTranslation pipeline contains several modules, if the performance of the model pipeline does not meet expectations, the issue may originate from any one of these modules. You can analyze cases with poor extraction results, use visualized images to determine which module has the problem, and refer to the corresponding fine-tuning tutorial links in the following table to fine-tune the model.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-tuning module</th>
<th>Fine-tuning reference link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate detection of layout areas, such as failure to detect seals and tables</td>
<td>Layout detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html#_5">Link</a></td>
</tr>
<tr>
<td>Inaccurate recognition of table structures</td>
<td>Table structure recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/table_structure_recognition.html#_5">Link</a></td>
</tr>
<tr>
<td>Inaccurate recognition of formulas</td>
<td>Formula recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/formula_recognition.html#_5">Link</a></td>
</tr>
<tr>
<td>Omission in detecting seal texts</td>
<td>Seal text detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/seal_text_detection.html#_5">Link</a></td>
</tr>
<tr>
<td>Omission in detecting texts</td>
<td>Text detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/text_detection.html#_5">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/text_recognition.html#_5">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of vertical or rotated text lines</td>
<td>Text line orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#_5">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of whole image rotation</td>
<td>Document image orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#_5">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of image distortion</td>
<td>Text image unwarping module</td>
<td>Fine-tuning is temporarily not supported</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After completing fine-tuning training with your private dataset, you can obtain a local model weight file. Then, you can use the fine-tuned model weights by customizing the pipeline configuration file.

1. Obtain the pipeline configuration file

You can call the `export_paddlex_config_to_yaml` method of the PP-DocTranslation pipeline object in PaddleOCR to export the current pipeline configuration to a YAML file:

```Python
from paddleocr import PPDocTranslation

pipeline = PPDocTranslation()
pipeline.export_paddlex_config_to_yaml("PP-DocTranslation.yaml")
```

2. Modify the configuration file

After obtaining the default pipeline configuration file, replace the local path of the fine-tuned model weights with the corresponding location in the pipeline configuration file. For example,

```yaml
......
SubModules:
    TextDetection:
    module_name: text_detection
    model_name: PP-OCRv5_server_det
    model_dir: null # Replace with the path to the weights of the fine-tuned text detection model
    limit_side_len: 960
    limit_type: max
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 1.5

    TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_server_rec
    model_dir: null # Replace with the path to the weights of the fine-tuned text recognition model
    batch_size: 1
            score_thresh: 0
......
```

The pipeline configuration file not only includes parameters supported by PaddleOCR CLI and Python API but also allows for more advanced configurations. Detailed information can be found in the corresponding pipeline usage tutorial in the [Overview of PaddleX Model Pipeline Usage](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/pipeline_develop_guide.html). Refer to the detailed instructions therein and adjust the configurations according to your needs.

3. Load the pipeline configuration file in CLI

After modifying the configuration file, specify the path to the modified pipeline configuration file using the `--paddlex_config` parameter in the command line. PaddleOCR will then read its contents as the pipeline configuration. Here is an example:

```bash
paddleocr pp_doctranslation --paddlex_config PP-DocTranslation.yaml ...
```

4. Load the pipeline configuration file in the Python API

When initializing the pipeline object, you can pass the path of the PaddleX pipeline configuration file or a configuration dict through the `paddlex_config` parameter, and PaddleOCR will read its content as the pipeline configuration. The example is as follows:

```python
from paddleocr import PPDocTranslation

pipeline = PPDocTranslation(paddlex_config="PP-DocTranslation.yaml")
```
