---
comments: true
---

# PP-StructureV3 Pipeline Tutorial

## 1. Introduction to PP-StructureV3 pipeline
Layout parsing is a technology that extracts structured information from document images, primarily used to convert complex document layouts into machine-readable data formats. This technology is widely applied in document management, information extraction, and data digitization. By combining Optical Character Recognition (OCR), image processing, and machine learning algorithms, layout parsing can identify and extract text blocks, headings, paragraphs, images, tables, and other layout elements from documents. The process typically involves three main steps: layout detection, element analysis, and data formatting, ultimately generating structured document data to improve the efficiency and accuracy of data processing. <b>The PP-StructureV3 pipeline, based on the v1 pipeline, enhances the capabilities of layout region detection, table recognition, and formula recognition, adds chart understanding capability, and the ability to restore multi-column reading order and convert results into Markdown files. It performs excellently on various document data and can handle more complex document data.</b> This pipeline also provides flexible serving deployment options, supporting the use of multiple programming languages on various hardware. Moreover, this pipeline offers the capability for custom development; you can train and optimize models on your own dataset based on this pipeline, and the trained models can be seamlessly integrated.


<b>The PP-StructureV3 pipeline includes a mandatory layout region analysis module and a general OCR sub-pipeline,</b> as well as optional sub-pipelines for document image preprocessing, table recognition, seal recognition, and formula recognition.

<b>If you prioritize model accuracy, choose a high-accuracy model; if you prioritize model inference speed, choose a faster inference model; if you prioritize model storage size, choose a smaller storage model.</b>
<details><summary>👉 Model List Details</summary>
<p><b>Document Image Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, containing four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>

<p><b>Text Image Rectification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>A high-precision text image rectification model.</td>
</tr>
</tbody>
</table>

<p><b>Layout Detection Module Model (Required):</b></p>
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
<td>A high-precision layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exam papers, and research reports, based on RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>A layout region detection model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exam papers, and research reports, based on PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exam papers, and research reports, based on PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>86.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>A high-efficiency layout region detection model trained on the PubLayNet dataset, based on PicoDet-1x, capable of locating five types of regions: text, title, table, image, and list.</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>95.7</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset, based on PicoDet-1x, capable of locating one type of region: table.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>87.1</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, and research reports, based on the lightweight PicoDet-S model, with three categories: table, image, and seal.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>70.3</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, and research reports, based on the lightweight PicoDet-S model, with 17 common layout categories: paragraph title, image, text, number, abstract, content, chart title, formula, table, table title, reference, document title, footnote, header, algorithm, footer, and seal.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.3</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, and research reports, based on PicoDet-L, with three categories: table, image, and seal.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>79.9</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, and research reports, based on PicoDet-L, with 17 common layout categories: paragraph title, image, text, number, abstract, content, chart title, formula, table, table title, reference, document title, footnote, header, algorithm, footer, and seal.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.9</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout region localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H, with 3 categories: table, image, and seal.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>92.6</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout region localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H, with 17 common layout categories: paragraph title, image, text, number, abstract, content, chart title, formula, table, table title, references, document title, footnote, header, algorithm, footer, and seal.</td>
</tr>
</tbody>
</table>

<p><b>Table Structure Recognition Module (Optional):</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td>SLANet is a table structure recognition model independently developed by the Baidu PaddlePaddle Vision Team. This model significantly improves the accuracy and inference speed of table structure recognition by using a lightweight backbone network PP-LCNet that is friendly to CPUs, a high-low feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structure and position information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td>SLANet_plus is the enhanced version of the SLANet table structure recognition model independently developed by the Baidu PaddlePaddle Vision Team. Compared to SLANet, SLANet_plus has significantly improved the ability to recognize wireless and complex tables and reduced the model's sensitivity to table positioning accuracy. Even if there is a deviation in table positioning, it can still recognize accurately.</td>
</tr>
</table>

<p><b>Text Detection Module (Required):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
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

<p><b>Text Recognition Module Model (Required):</b></p>

* <b>Chinese Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the recognition capabilities for some traditional Chinese characters, Japanese, and special characters. The number of recognizable characters is over 15,000. In addition to the improvement in document-related text recognition, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>The lightweight recognition model of PP-OCRv4 has high inference efficiency and can be deployed on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>The server-side model of PP-OCRv4 offers high inference accuracy and can be deployed on various types of servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>72.96</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td>PP-OCRv3’s lightweight recognition model is designed for high inference efficiency and can be deployed on a variety of hardware devices, including edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server text recognition model developed by the OpenOCR team of Fudan University's Visual and Learning Laboratory (FVL). It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the A list is 6% higher than that of PP-OCRv4.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">    The RepSVTR text recognition model is a mobile text recognition model based on SVTRv2. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the B list is 2.5% higher than that of PP-OCRv4, with the same inference speed.</td>
</tr>
</table>

* <b>English Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model supports the recognition of English and numbers.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv3 recognition model supports the recognition of English and numbers.</td>
</tr>
</table>

* <b>Multilingual Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>The ultra-lightweight Korean recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Korean and numbers. </td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>The ultra-lightweight Japanese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Japanese and numbers.</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>The ultra-lightweight Traditional Chinese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Traditional Chinese and numbers.</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>The ultra-lightweight Telugu recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Telugu and numbers.</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>The ultra-lightweight Kannada recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Kannada and numbers.</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>The ultra-lightweight Tamil recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Tamil and numbers.</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>The ultra-lightweight Latin recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Latin script and numbers.</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>The ultra-lightweight Arabic script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Arabic script and numbers.</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>
The ultra-lightweight cyrillic alphabet recognition model trained based on the PP-OCRv3 recognition model supports the recognition of cyrillic letters and numbers.</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M  </td>
<td>The ultra-lightweight Devanagari script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Devanagari script and numbers.</td>
</tr>
</table>

<p><b>Text Line Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>A text line classification model based on PP-LCNet_x0_25, with two categories: 0 degrees and 180 degrees.</td>
</tr>
</tbody>
</table>

<p><b>Formula Recognition Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>BLEU Score</th>
<th>Normed Edit Distance</th>
<th>ExpRate (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>2047.13 / 2047.13</td>
<td>10582.73 / 10582.73</td>
<td>89.7 M</td>
</tr>
</tbody>
</table>

<p><b>Seal Text Detection Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>The PP-OCRv4 server seal text detection model offers higher precision and is suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>The PP-OCRv4 mobile seal text detection model provides higher efficiency and is suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>

<p><b>Text Image Rectification Module Model:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>MS-SSIM (%)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>High-precision text image rectification model</td>
</tr>
</tbody>
</table>
<p><b>The precision metrics of the model are measured from the <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>.</b></p>
<p><b>Document Image Orientation Classification Module Model:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>The document image classification model based on PP-LCNet_x1_0 includes four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
          <li><strong>Test Dataset：</strong>
                        <ul>
                         <li>Document Image Orientation Classification Module: A self-built dataset using PaddleOCR, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Region Detection Model: A self-built layout detection dataset using PaddleOCR, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Table Structure Recognition Model: A self-built English table recognition dataset using PaddleX.</li>
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a>.</li>
                          <li>English Recognition Model: A self-built English dataset using PaddleX.</li>
                          <li>Multilingual Recognition Model: A self-built multilingual dataset using PaddleX.</li>
                          <li>Text Line Orientation Classification Model: A self-built dataset using PaddleOCR, covering various scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Seal Text Detection Model: A self-built dataset using PaddleOCR, containing 500 images of circular seal textures.</li>
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

## 2. Quick Start
All the model pipelines provided by PaddleX can be quickly experienced. You can use the command line or Python on your local machine to experience the effect of the PP-StructureV3 pipeline.

Before using the PP-StructureV3 pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ocr`.

> When performing GPU inference, the default configuration may use more than 16 GB of VRAM. Please ensure that your GPU has sufficient memory. To reduce VRAM usage, you can modify the configuration file as described below to disable unnecessary features.

### 2.1 Experiencing via Command Line

You can quickly experience the PP-StructureV3 pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png) and replace `--input` with the local path to perform prediction.

```
paddlex --pipeline PP-StructureV3 \
        --input pp_structure_v3_demo.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --use_e2e_wireless_table_rec_model True \
        --save_path ./output \
        --device gpu:0
```

The parameter description can be found in [2.2.2 Python Script Integration](#222-python-script-integration). Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to [Pipeline Parallel Inference](../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices).

After running, the result will be printed to the terminal, as follows:

<details><summary>👉Click to Expand</summary>
<pre><code>

{'res': {'input_path': 'pp_structure_v3_demo.png', 'model_settings': {'use_doc_preprocessor': False, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9853514432907104, 'coordinate': [770.9531, 776.6814, 1122.6057, 1058.7322]}, {'cls_id': 1, 'label': 'image', 'score': 0.9848673939704895, 'coordinate': [775.7434, 202.27979, 1502.8113, 686.02136]}, {'cls_id': 2, 'label': 'text', 'score': 0.983731746673584, 'coordinate': [1152.3197, 1113.3275, 1503.3029, 1346.586]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832221865653992, 'coordinate': [1152.5602, 801.431, 1503.8436, 986.3563]}, {'cls_id': 2, 'label': 'text', 'score': 0.9829439520835876, 'coordinate': [9.549545, 849.5713, 359.1173, 1058.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9811657667160034, 'coordinate': [389.58298, 1137.2659, 740.66235, 1346.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9775941371917725, 'coordinate': [9.1302185, 201.85, 359.0409, 339.05692]}, {'cls_id': 2, 'label': 'text', 'score': 0.9750366806983948, 'coordinate': [389.71454, 752.96924, 740.544, 889.92456]}, {'cls_id': 2, 'label': 'text', 'score': 0.9738152027130127, 'coordinate': [389.94565, 298.55988, 740.5585, 435.5124]}, {'cls_id': 2, 'label': 'text', 'score': 0.9737328290939331, 'coordinate': [771.50256, 1065.4697, 1122.2582, 1178.7324]}, {'cls_id': 2, 'label': 'text', 'score': 0.9728517532348633, 'coordinate': [1152.5154, 993.3312, 1503.2349, 1106.327]}, {'cls_id': 2, 'label': 'text', 'score': 0.9725610017776489, 'coordinate': [9.372787, 1185.823, 359.31738, 1298.7227]}, {'cls_id': 2, 'label': 'text', 'score': 0.9724331498146057, 'coordinate': [389.62848, 610.7389, 740.83234, 746.2377]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720287322998047, 'coordinate': [389.29898, 897.0936, 741.41516, 1034.6616]}, {'cls_id': 2, 'label': 'text', 'score': 0.9713053703308105, 'coordinate': [10.323685, 1065.4663, 359.6786, 1178.8872]}, {'cls_id': 2, 'label': 'text', 'score': 0.9689728021621704, 'coordinate': [9.336395, 537.6609, 359.2901, 652.1881]}, {'cls_id': 2, 'label': 'text', 'score': 0.9684857130050659, 'coordinate': [10.7608185, 345.95068, 358.93616, 434.64087]}, {'cls_id': 2, 'label': 'text', 'score': 0.9681928753852844, 'coordinate': [9.674866, 658.89075, 359.56528, 770.4319]}, {'cls_id': 2, 'label': 'text', 'score': 0.9634978175163269, 'coordinate': [770.9464, 1281.1785, 1122.6522, 1346.7156]}, {'cls_id': 2, 'label': 'text', 'score': 0.96304851770401, 'coordinate': [390.0113, 201.28055, 740.1684, 291.53073]}, {'cls_id': 2, 'label': 'text', 'score': 0.962053120136261, 'coordinate': [391.21393, 1040.952, 740.5046, 1130.32]}, {'cls_id': 2, 'label': 'text', 'score': 0.9565253853797913, 'coordinate': [10.113251, 777.1482, 359.439, 842.437]}, {'cls_id': 2, 'label': 'text', 'score': 0.9497362375259399, 'coordinate': [390.31357, 537.86285, 740.47595, 603.9285]}, {'cls_id': 2, 'label': 'text', 'score': 0.9371236562728882, 'coordinate': [10.2034, 1305.9753, 359.5958, 1346.7295]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9338151216506958, 'coordinate': [791.6062, 1200.8479, 1103.3257, 1259.9324]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9326773285865784, 'coordinate': [408.0737, 457.37024, 718.9509, 516.63464]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9274250864982605, 'coordinate': [29.448685, 456.6762, 340.99194, 515.6999]}, {'cls_id': 2, 'label': 'text', 'score': 0.8742568492889404, 'coordinate': [1154.7095, 777.3624, 1330.3086, 794.5853]}, {'cls_id': 2, 'label': 'text', 'score': 0.8442489504814148, 'coordinate': [586.49316, 160.15454, 927.468, 179.64203]}, {'cls_id': 11, 'label': 'doc_title', 'score': 0.8332607746124268, 'coordinate': [133.80017, 37.41908, 1380.8601, 124.1429]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6770150661468506, 'coordinate': [812.1718, 705.1199, 1484.6973, 747.1692]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[ 133,   35],
        ...,
        [ 133,  131]],

       ...,

       [[1154, 1323],
        ...,
        [1152, 1355]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者', '沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等,曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院")举办"喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年峻工，建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落…"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新："中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起我们欢迎你"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。”尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万""和""禅"“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明,很早以前我们就通过海上丝绸之路进行', '习中文,在2017年第十届"汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。"这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话："这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？""非常想！我想', '育等领域的发展，中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示："每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。"', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说："这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '“共同向世界展示非', '印象深刻。“中国博物馆不仅有许多保存完好', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说："学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，"厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰,希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！"', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜓曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99943757, ..., 0.98181838]), 'rec_polys': array([[[ 133,   35],
        ...,
        [ 133,  131]],

       ...,

       [[1154, 1323],
        ...,
        [1152, 1355]]], dtype=int16), 'rec_boxes': array([[ 133, ...,  131],
       ...,
       [1152, ..., 1359]], dtype=int16)}, 'text_paragraphs_ocr_res': {'rec_polys': array([[[ 133,   35],
        ...,
        [ 133,  131]],

       ...,

       [[1154, 1323],
        ...,
        [1152, 1355]]], dtype=int16), 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者', '沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等,曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院")举办"喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年峻工，建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落…"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新："中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起我们欢迎你"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。”尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万""和""禅"“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明,很早以前我们就通过海上丝绸之路进行', '习中文,在2017年第十届"汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。"这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话："这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？""非常想！我想', '育等领域的发展，中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示："每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。"', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说："这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '“共同向世界展示非', '印象深刻。“中国博物馆不仅有许多保存完好', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说："学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，"厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰,希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！"', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜓曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99943757, ..., 0.98181838]), 'rec_boxes': array([[ 133, ...,  131],
       ...,
       [1152, ..., 1359]], dtype=int16)}}}

</code></pre></details>

The result parameter description can be found in the result interpretation in [2.2.2 Python Script Integration](#222-python-script-integration).

<b>Note:</b> Since the default model of the pipeline is relatively large, the inference speed may be slow. You can refer to the model list in Section 1 and replace it with a model that has faster inference speed.

### 2.2 Python Script Integration
Just a few lines of code can complete the quick inference of the pipeline. Taking the PP-StructureV3 pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-StructureV3")

output = pipeline.predict(
    input="./pp_structure_v3_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the structured JSON result of the current image
    res.save_to_markdown(save_path="output") ## Save the result of the current image in Markdown format
```
If it is a PDF file, each page of the PDF will be processed separately, and each page will have its own corresponding Markdown file. If you want to convert the entire PDF file into a Markdown file, it is recommended to run it in the following way:

```python
from pathlib import Path
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-StructureV3")

input_file = "./your_pdf_file.pdf"
output_path = Path("./output")

output = pipeline.predict(
    input=input_file,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

markdown_texts = ""
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

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

- The default text recognition model used by PP-StructureV3 is the **Chinese-English recognition model**. Limited recognition capability for purely English text. For fully English scenarios, you can modify the `model_name` under the `TextRecognition` configuration item in the [PP-StructureV3 configuration file](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-rc/paddlex/configs/pipelines/PP-StructureV3.yaml) to `en_PP-OCRv4_mobile_rec` or other English recognition models for better recognition results. For other language scenarios, you can also refer to the model list mentioned earlier and choose the corresponding language recognition model for replacement.

- In the example code, the parameters `use_doc_orientation_classify`, `use_doc_unwarping`, and `use_textline_orientation` are all set to False by default. These parameters respectively control the document orientation classification, document unwarping, and text line orientation classification functions. If you need to use these features, you can manually set them to True.

- PP-StructureV3 provides flexible parameter configuration, allowing you to adjust parameters for layout detection, text detection, text recognition, etc., based on the characteristics of the document for better results. For more detailed configurations, please refer to the [PP-StructureV3 configuration file](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-rc/paddlex/configs/pipelines/PP-StructureV3.yaml).

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` instance to create a pipeline object. The specific parameter descriptions are as follows:

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
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>The path to the pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline. It supports specifying the specific GPU card number, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu". Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to <a href="../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices">Pipeline Parallel Inference</a>.</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to <code>None</code>, the setting from the configuration file or <code>config</code> will be used.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the pipeline object to perform inference prediction. This method will return a `generator`. Below are the parameters and their descriptions for the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image file or PDF file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the web URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">example</a>; <b>Local directory</b>, the directory should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with an exact file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Production inference device</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: <code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: <code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: <code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: <code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: <code>dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used. During initialization, the local GPU device 0 will be prioritized; if unavailable, the CPU device will be used;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document unwarping module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_general_ocr</code></td>
<td>Whether to use the OCR sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal recognition sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to use the formula recognition sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart recognition sub-production line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to use the document region detection production line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Layout model score threshold</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number between <code>0</code> and <code>1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> where the key is the category ID and the value is the threshold for that category;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>0.5</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether the layout area detection model uses NMS post-processing</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion ratio of the detection box for the layout area detection model</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: The expansion ratios in the horizontal and vertical directions, respectively;</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors, e.g., <code>{0: (1.1, 2.0)}</code> means cls_id 0 expanding the width by 1.1 times and the height by 2.0 times while keeping the center unchanged</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Overlap box filtering method for layout area detection</td>
<td><code>str|dict|None</code></td>
<td>
<ul>
<li><b>str</b>: <code>large</code>, <code>small</code>, <code>union</code>, representing whether to retain the larger box, the smaller box, or both during overlap box filtering;</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes, e.g., <code>{0: "large", 2: "small"}</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>large</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limit for text detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of image side length limit for text detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>max</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold. In the output probability map, only pixels with scores greater than this threshold will be considered as text pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, which is <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold. The detection result will be considered as a text area only if the average score of all pixels within the bounding box is greater than this threshold.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, which is <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion ratio. This method is used to expand the text area. The larger this value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, which is <code>2.0</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold, text results with scores greater than this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.0</code>. That is, no threshold is set</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter, initialized as <code>960</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Type of image side length limit for seal detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, <code>min</code> indicates that the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> indicates that the longest side of the image is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter, initialized as <code>max</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Detection pixel threshold, in the output probability map, pixels with scores greater than this threshold will be considered as seal pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.3</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Detection box threshold, within the detection result bounding box, if the average score of all pixels is greater than this threshold, the result will be considered as a seal area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.6</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal detection, this method is used to expand the text area, the larger this value, the larger the expanded area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>2.0</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Seal recognition threshold, text results with scores greater than this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.0</code>. That is, no threshold is set</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>Whether to enable direct conversion of wired table cell detection results to HTML. Default is False. If enabled, HTML will be constructed directly based on the geometric relationship of wired table cell detection results.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
<td><code>False</code></td>
</td>
</tr>
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>Whether to enable direct conversion of wireless table cell detection results to HTML. Default is False. If enabled, HTML will be constructed directly based on the geometric relationship of wireless table cell detection results.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
<td><code>False</code></td>
</td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td>Whether to enable table orientation classification. When enabled, it can correct the orientation and correctly complete table recognition if the table in the image is rotated by 90/180/270 degrees.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td>Whether to enable OCR within cell segmentation. When enabled, OCR detection results will be segmented and re-recognized based on cell prediction results to avoid text loss.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td>Whether to enable end-to-end wired table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td>Whether to enable end-to-end wireless table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
</table>

(3) Process the prediction results: The prediction result of each sample is a corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file will have the same name as the input file</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the visualization images of each module in PNG format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>Saves each page of the image or PDF file as a markdown formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the table in the file as an HTML file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the table in the file as an XLSX file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates which page of the PDF it is; otherwise, it is `None`.

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring the pipeline.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessor sub-line.
        - `use_general_ocr`: `(bool)` Controls whether to enable the OCR sub-line.
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal recognition sub-line.
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition sub-line.
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition sub-line.

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, where each element is a dictionary. The order of the list is the reading order after parsing.
        - `layout_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `{label}`: `(str)` The key is the label of the layout area, such as `text`, `table`, etc., and the content is the content within the layout area.
        - `layout`: `(str)` The layout type, such as `double`, `single`, etc.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results
      -  `input_path`: `(Union[str, None])` The image path accepted by the image OCR sub-line. When the input is a `numpy.ndarray`, it is saved as `None`.
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR sub-line.
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with a shape of (4, 2) and a data type of int16.
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes.
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` The side length limit value during image preprocessing.
        - `limit_type`: `(str)` The processing method for side length limits.
        - `thresh`: `(float)` The confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` The expansion ratio for text detection boxes.
        - `text_type`: `(str)` The type of text detection, currently fixed as "general".

      - `text_type`: `(str)` The type of text detection, currently fixed as "general".
      - `textline_orientation_angles`: `(List[int])` The prediction results for text line orientation classification. When enabled, it returns the actual angle values (e.g., [0,0,1]).
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results.
      - `rec_texts`: `(List[str])` A list of text recognition results, containing only texts with confidence scores above `text_rec_score_thresh`.
      - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, filtered by `text_rec_score_thresh`.
      - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes filtered by confidence score, in the same format as `dt_polys`.

    - `text_paragraphs_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` OCR results for paragraphs, excluding paragraphs of layout types such as tables, seals, and formulas.
        - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes, in the same format as `dt_polys`.
        - `rec_texts`: `(List[str])` A list of text recognition results.
        - `rec_scores`: `(List[float])` A list of confidence scores for text recognition results.
        - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and a dtype of int16. Each row represents a rectangle.

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of formula recognition results, where each element is a dictionary.
        - `rec_formula`: `(str)` The result of formula recognition.
        - `rec_polys`: `(numpy.ndarray)` The detection box for the formula, with a shape of (4, 2) and a dtype of int16.
        - `formula_region_id`: `(int)` The region ID where the formula is located.

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of seal recognition results, where each element is a dictionary.
        - `input_path`: `(str)` The input path of the seal image.
        - `model_settings`: `(Dict)` Model configuration parameters for the seal recognition sub-line.
        - `dt_polys`: `(List[numpy.ndarray])` A list of seal detection boxes, in the same format as `dt_polys`.
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the seal detection module, with the same specific parameter meanings as above.
        - `text_type`: `(str)` The type of seal detection, currently fixed as "seal".
        - `text_rec_score_thresh`: `(float)` The filtering threshold for seal recognition results.
        - `rec_texts`: `(List[str])` A list of seal recognition results, containing only texts with confidence scores above `text_rec_score_thresh`.
        - `rec_scores`: `(List[float])` A list of confidence scores for seal recognition, filtered by `text_rec_score_thresh`.
        - `rec_polys`: `(List[numpy.ndarray])` A list of seal detection boxes filtered by confidence score, in the same format as `dt_polys`.
        - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and a dtype of int16. Each row represents a rectangle.

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of table recognition results, where each element is a dictionary.
        - `cell_box_list`: `(List[numpy.ndarray])` A list of bounding boxes for table cells.
        - `pred_html`: `(str)` The HTML format string of the table.
        - `table_ocr_pred`: `(dict)` The OCR recognition results of the table.
            - `rec_polys`: `(List[numpy.ndarray])` A list of detection boxes for cells.
            - `rec_texts`: `(List[str])` The recognition results of cells.
            - `rec_scores`: `(List[float])` The recognition confidence scores of cells.
            - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and a dtype of int16. Each row represents a rectangle.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving NumPy arrays, `numpy.array` types will be converted to lists.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, it will save images such as layout detection visualization, global OCR visualization, and layout reading order visualization. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly; otherwise, multiple images will be overwritten, and only the last image will be retained.)
- Calling the `save_to_markdown()` method will save the converted Markdown file to the specified `save_path`. The save path will be `save_path/{your_img_basename}.md`. If the input is a PDF file, it is recommended to specify a directory directly; otherwise, multiple Markdown files will be overwritten.

In addition, it also supports obtaining visualized images with results and prediction results through attributes, as follows:
<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format</td>
</tr>
<thead>
<tr>
<th>Property</th>
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
<td rowspan="2">Get the visualization image in <code>dict</code> format</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"><code>markdown</code></td>
<td rowspan="3">Get the markdown result in <code>dict</code> format</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>

- The prediction result obtained through the `json` attribute is data of the dict type, and its content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of the dictionary type. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, and the corresponding values are `Image.Image` objects: used to display the visualized images of layout detection, OCR, OCR text paragraphs, formulas, tables, and seal results. If the optional module is not used, the dictionary only contains `layout_det_res`.
- - The `markdown` property returns the prediction result as a dictionary. The keys are `markdown_texts` and `markdown_images`, corresponding to markdown text and images for display in Markdown (`Image.Image` objects).

In addition, you can obtain the PP-StructureV3 pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config PP-StructureV3 --save_path ./my_path
```

If you have obtained the configuration file, you can customize the PP-StructureV3 pipeline configuration. You just need to modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/PP-StructureV3.yaml")
output = pipeline.predict(
    input="./pp_structure_v3_demo.png",,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the structured JSON result of the current image
    res.save_to_markdown(save_path="output") ## Save the result of the current image in Markdown format
```

<b>Note:</b> The parameters in the configuration file are the pipeline initialization parameters. If you wish to change the initialization parameters of the PP-StructureV3 pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline directly into your Python project, you can refer to the example code in [2.2 Python Script Method](#22-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving Deployment</b>: Serving Deployment is a common form of deployment in actual production environments. By encapsulating the inference functionality into a service, clients can access these services through network requests to obtain inference results. PaddleX supports various serving deployment solutions for pipelines. For detailed procedures, please refer to the [PaddleX Serving Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below is the API reference for basic serving deployment and examples of service calls in multiple languages:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>The UUID of the request.</td>
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
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
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
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform layout parsing.</p>
<p><code>POST /layout-parsing</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
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
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the above file types. By default, for PDF files exceeding 10 pages, only the content of the first 10 pages will be processed.<br />
To remove the page limit, please add the following configuration to the pipeline configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre></td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code>｜<code>null</code></td>
<td>File type. <code>0</code> represents a PDF file, and <code>1</code> represents an image file. If this attribute is missing from the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_textline_orientation</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_seal_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_table_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_formula_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_chart_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useRegionDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_region_detection</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_side_len</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_type</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_box_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_rec_score_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_side_len</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_type</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_box_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_rec_score_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWiredTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_wired_table_cells_trans_to_html</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWirelessTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_wireless_table_cells_trans_to_html</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableOrientationClassify</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_table_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrResultsWithTableCells</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_ocr_results_with_table_cells</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_e2e_wired_table_rec_model</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_e2e_wireless_table_rec_model</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
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
<td>The layout parsing results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
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
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the <code>predict</code> method of the pipeline object, with the <code>input_path</code> and the <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>The Markdown result.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute of the result of the pipeline prediction. The images are in JPEG format and are Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
<p><code>markdown</code> is an <code>object</code> with the following attributes:</p>
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
<td>The Markdown text.</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>A key-value pair of relative paths of Markdown images and Base64-encoded images.</td>
</tr>
<tr>
<td><code>isStart</code></td>
<td><code>boolean</code></td>
<td>Whether the first element on the current page is the start of a segment.</td>
</tr>
<tr>
<td><code>isEnd</code></td>
<td><code>boolean</code></td>
<td>Whether the last element on the current page is the end of a segment.</td>
</tr>
</tbody>
</table></details>

<details><summary>Multi-language Service Call Example</summary>
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
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing the device to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model into your pipeline and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the PP-StructureV3 pipeline do not meet your requirements in terms of accuracy or speed, you can try to fine-tune the existing model using your own domain-specific or application-specific data to improve the recognition performance of the PP-StructureV3 pipeline in your scenario.


### 4.1 Model Fine-Tuning
Since the PP-StructureV3 pipeline includes several modules, the unsatisfactory performance of the pipeline may originate from any one of these modules. You can analyze the cases with poor extraction results, identify which module is problematic through visualizing the images, and refer to the corresponding fine-tuning tutorial links in the table below to fine-tune the model.

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
<td>Missed detection of seal text</td>
<td>Seal Text Detection Module</td>
<td><a href="../module_usage/seal_text_detection.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Missed detection of text</td>
<td>Text Detection Module</td>
<td><a href="../module_usage/text_detection.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="../module_usage/text_recognition.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of vertical or rotated text lines</td>
<td>Text Line Orientation Classification Module</td>
<td><a href="../module_usage/text_line_orientation_classification.en.md#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of whole-image rotation</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="../module_usage/doc_img_orientation_classification.en.md#iv-custom-development">Link</a></td>
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
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
pipeline.export_paddlex_config_to_yaml("PP-StructureV3.yaml")
```

2. Editing Pipeline Configuration Files

Replace the local directory of the fine-tuned model weights to the corresponding position in the pipeline configuration file. For example:

```yaml
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout_plus-L
    model_dir: null # Replace with the fine-tuned layout detection model weights directory
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
        model_dir: null # Replace with the fine-tuned text detection model weights directory
        limit_side_len: 960
        limit_type: max
        max_side_limit: 4000
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 1.5

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv5_server_rec
        model_dir: null # Replace with the fine-tuned text recognition model weights directory
        batch_size: 1
        score_thresh: 0
......
```

The exported PaddleX pipeline configuration file not only includes parameters supported by PaddleOCR's CLI and Python API but also allows for more advanced settings. Please refer to the corresponding pipeline usage tutorials in [PaddleX Pipeline Usage Overview](https://paddlepaddle.github.io/PaddleX/3.0/en/pipeline_usage/pipeline_develop_guide.html) for detailed instructions on adjusting various configurations according to your needs.


3. Loading Pipeline Configuration Files in CLI

By specifying the path to the PaddleX pipeline configuration file using the `--paddlex_config` parameter, PaddleOCR will read its contents as the configuration for inference. Here is an example:

```bash
paddleocr pp_structurev3 --paddlex_config PP-StructureV3.yaml ...
```

4. Loading Pipeline Configuration Files in Python API

When initializing the pipeline object, you can pass the path to the PaddleX pipeline configuration file or a configuration dictionary through the `paddlex_config` parameter, and PaddleOCR will use it as the configuration for inference. Here is an example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
```
