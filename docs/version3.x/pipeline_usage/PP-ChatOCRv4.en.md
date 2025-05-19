# PP-ChatOCRv4-doc Pipeline Tutorial

## 1. Introduction to PP-ChatOCRv4-doc Pipeline
PP-ChatOCRv4-doc is a unique document and image intelligent analysis solution from PaddlePaddle, combining LLM, MLLM, and OCR technologies to address complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. Integrated with ERNIE Bot, it fuses massive data and knowledge, achieving high accuracy and wide applicability. This pipeline also provides flexible service deployment options, supporting deployment on various hardware. Furthermore, it offers custom development capabilities, allowing you to train and fine-tune models on your own datasets, with seamless integration of trained models.

<img src="https://github.com/user-attachments/assets/0870cdec-1909-4247-9004-d9efb4ab9635">

The Document Scene Information Extraction v4 pipeline includes modules for **Layout Region Detection**, **Table Structure Recognition**, **Table Classification**, **Table Cell Localization**, **Text Detection**, **Text Recognition**, **Seal Text Detection**, **Text Image Rectification**, and **Document Image Orientation Classification**. The relevant models are integrated as sub-pipelines, and you can view the model configurations of different modules through the [pipeline configuration](../../../../paddlex/configs/pipelines/PP-ChatOCRv4-doc.yaml).

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b> Benchmarks for some models are as follows:

<details><summary> 👉Model List Details</summary>
<p><b>Table Structure Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td>SLANet_plus is an enhanced version of SLANet, the table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves the recognition ability for wireless and complex tables and reduces the model's sensitivity to the accuracy of table positioning, enabling more accurate recognition even with offset table positioning.</td>
</tr>
</table>

<p><b>Layout Detection Module Models</b>:</p>
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
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 500 common document-type images of Chinese and English papers, magazines, contracts, books, exams, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

> ❗ The above list includes the <b>3 core models</b> that are key supported by the text recognition module. The module actually supports a total of <b>11 full models</b>, including several predefined models with different categories. The complete model list is as follows:

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
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout table area detection dataset by PaddleOCR, containing 7835 Chinese and English document images with tables. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

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
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 1154 common document images of Chinese and English papers, magazines, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

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
<b>Note: The evaluation dataset for the above precision metrics is the [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) dataset, containing 11245 English document images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

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

<p><b>Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
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

<p><b>Text Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.20</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4 is the next version of Baidu PaddlePaddle's self-developed text recognition model PP-OCRv3. By introducing data augmentation schemes and GTC-NRTR guidance branches, it further improves text recognition accuracy without compromising inference speed. The model offers both server (server) and mobile (mobile) versions to meet industrial needs in different scenarios.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>79.20</td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 8.08</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 5.93</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">
The RepSVTR text recognition model is a mobile-oriented text recognition model based on SVTRv2. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the B-list, while maintaining similar inference speed.
</td>
</tr>
</table>

<p><b>Formula Recognition Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>BLEU Score</th>
<th>Normed Edit Distance</th>
<th>ExpRate (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size</th>
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

<p><b>Seal Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                    <li><strong>Test Dataset：</strong>
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
The pre-trained pipelines provided by PaddleX allow for quick experience of their effects. You can locally use Python to experience the effects of the PP-ChatOCRv4-doc pipeline.

### 2.1 Local Experience
Before using the PP-ChatOCRv4-doc pipeline locally, ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ie`.

Before performing model inference, you first need to prepare the API key for the large language model. PP-ChatOCRv4 supports large model services on the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or the locally deployed standard OpenAI interface. If using the Baidu Cloud Qianfan Platform, refer to [Authentication and Authorization](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps_en) to obtain the API key. If using a locally deployed large model service, refer to the [PaddleNLP Large Model Deployment Documentation](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm) for deployment of the dialogue interface and vectorization interface for large models, and fill in the corresponding `base_url` and `api_key`. If you need to use a multimodal large model for data fusion, refer to the OpenAI service deployment in the [PaddleMIX Model Documentation](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee) for multimodal large model deployment, and fill in the corresponding `base_url` and `api_key`.

After updating the configuration file, you can complete quick inference using just a few lines of Python code. You can use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) for testing:

**Note**: If local deployment of a multimodal large model is restricted due to the local environment, you can comment out the lines containing the `mllm` variable in the code and only use the large language model for information extraction.

```python
from paddlex import create_pipeline

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
    "model_name": "PP-DocBee",
    "base_url": "http://172.0.0.1:8080/v1/chat/completions",  # your local mllm service url
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

pipeline = create_pipeline(pipeline="PP-ChatOCRv4-doc", initial_predictor=False)

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
    key_list=["驾驶室准乘人数"],
    mllm_chat_bot_config=mllm_chat_bot_config,
)
mllm_predict_info = mllm_predict_res["mllm_res"]
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)

```

After running, the output result is as follows:

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

PP-ChatOCRv4 Prediction Process, API Description, and Output Description:

<details><summary>(1) Instantiate the PP-ChatOCRv4 Pipeline Object by Calling the <code>create_pipeline</code> Method.</summary>

The following are the parameter descriptions:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline inference. Supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
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
<tr>
<td><code>initial_predictor</code></td>
<td>Whether to initialize the inference module (if <code>False</code>, it will be initialized when the relevant inference module is used for the first time).</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(2) Call the <code>visual_predict()</code> Method of the PP-ChatOCRv4 Pipeline Object to Obtain Visual Prediction Results. This method returns a generator.</summary>

The following are the parameters and descriptions of the `visual_predict()` method:

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
<tbody>
<tr>
<td><code>input</code></td>
<td>The data to be predicted, supporting multiple input types, required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data.</li>
  <li><b>str</b>: Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories, PDF files need to be specified to the specific file path).</li>
  <li><b>List</b>: List elements need to be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline inference.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: Such as <code>cpu</code> to use CPU for inference;</li>
  <li><b>GPU</b>: Such as <code>gpu:0</code> to use the first GPU for inference;</li>
  <li><b>NPU</b>: Such as <code>npu:0</code> to use the first NPU for inference;</li>
  <li><b>XPU</b>: Such as <code>xpu:0</code> to use the first XPU for inference;</li>
  <li><b>MLU</b>: Such as <code>mlu:0</code> to use the first MLU for inference;</li>
  <li><b>DCU</b>: Such as <code>dcu:0</code> to use the first DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline. During initialization, it will prioritize using the local GPU 0 device, and if not available, it will use the CPU device;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document distortion correction module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_general_ocr</code></td>
<td>Whether to use the OCR sub-pipeline.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal recognition sub-pipeline.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-pipeline.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>The score threshold for the layout model.</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
  <li><b>float</b>: Any floating-point number between <code>0-1</code>;</li>
  <li><b>dict</b>: <code>{0:0.1}</code> where the key is the category ID and the value is the threshold for that category;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.5</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>The expansion coefficient for layout detection.</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td>
<ul>
  <li><b>float</b>: Any floating-point number greater than <code>0</code>;</li>
  <li><b>Tuple[float,float]</b>: The expansion coefficients in the horizontal and vertical directions, respectively;</li>
  <li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors for each category.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The method for filtering overlapping bounding boxes.</td>
<td><code>str|dict|None</code></td>
<td>
<ul>
  <li><b>str</b>: large, small, union. Respectively representing retaining the larger box, smaller box, or both when overlapping boxes are filtered.</li>
  <li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes for each category.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>large</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>The side length limit for text detection images.</td>
<td><code>int|None</code></td>
<td>
<ul>
  <li><b>int</b>: Any integer greater than <code>0</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>The type of side length limit for text detection images.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>The pixel threshold for detection. In the output probability map, pixel points with scores greater than this threshold will be considered as text pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>The bounding box threshold for detection. When the average score of all pixel points within the detection result bounding box is greater than this threshold, the result will be considered as a text region.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>The expansion coefficient for text detection. This method is used to expand the text region, and the larger the value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>2.0</code>.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>The text recognition threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.0</code>. I.e., no threshold is set.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>The side length limit for seal detection images.</td>
<td><code>int|None</code></td>
<td>
<ul>
  <li><b>int</b>: Any integer greater than <code>0</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>The type of side length limit for seal detection images.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>The pixel threshold for detection. In the output probability map, pixel points with scores greater than this threshold will be considered as seal pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>The bounding box threshold for detection. When the average score of all pixel points within the detection result bounding box is greater than this threshold, the result will be considered as a seal region.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>The expansion coefficient for seal detection. This method is used to expand the seal region, and the larger the value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>2.0</code>.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>The seal recognition threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.0</code>. I.e., no threshold is set.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(3) Process the Visual Prediction Results.</summary>

The prediction result for each sample is of `dict` type, containing two fields: `visual_info` and `layout_parsing_result`. You can obtain visual information through `visual_info` (including `normal_text_dict`, `table_text_list`, `table_html_list`, etc.), and place the information for each sample into the `visual_info_list` list, which will be fed into the large language model later.

Of course, you can also obtain the layout parsing results through `layout_parsing_result`, which includes tables, text, images, and other content contained in the document or image. It supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td>Whether to format the output content with JSON indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output JSON data for better readability, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-ASCII characters to Unicode. When set to <code>True</code>, all non-ASCII characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Saves the result as a json file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When it is a directory, the saved file name is consistent with the input file type</td>
<td>N/A</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output JSON data for better readability, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-ASCII characters to Unicode. When set to <code>True</code>, all non-ASCII characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Saves the visual images of each intermediate module in png format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supports directory or file path</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Saves the tables in the file as html files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supports directory or file path</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Saves the tables in the file as xlsx files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supports directory or file path</td>
<td>N/A</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters required for the pipeline

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing pipeline
        - `use_general_ocr`: `(bool)` Controls whether to enable the OCR pipeline
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal recognition pipeline
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition pipeline
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition pipeline

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, each element is a dictionary, and the list order is the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results
      - `input_path`: `(Union[str, None])` The image path accepted by the OCR pipeline, when the input is `numpy.ndarray`, it is saved as `None`
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR pipeline
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with a shape of (4, 2) and a data type of int16
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module
        - `limit_side_len`: `(int)` The side length limit for image preprocessing
        - `limit_type`: `(str)` The processing method for the side length limit
        - `thresh`: `(float)` The confidence threshold for text pixel classification
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes
        - `unclip_ratio`: `(float)` The inflation coefficient for text detection boxes
        - `text_type`: `(str)` The type of text detection, currently fixed as "general"

      - `text_type`: `(str)` The type of text detection, currently fixed as "general"
      - `textline_orientation_angles`: `(List[int])` The prediction results of text line orientation classification. When enabled, it returns actual angle values (e.g., [0,0,1])
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results
      - `rec_texts`: `(List[str])` A list of text recognition results, only including texts with confidence exceeding `text_rec_score```markdown
- Calling the `save_to_json()` method will save the aforementioned content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list form.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (Production pipelines often involve numerous result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten, leaving only the last one.)

In addition, it is also supported to obtain visualized images with results and prediction results through attributes, as detailed below:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Obtain prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Obtain visualized images in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is data of type `dict`, with content consistent with that saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of type `dict`. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, with corresponding values being `Image.Image` objects: used for displaying visualized images of layout detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, only `layout_det_res` will be included in the dictionary.
</details>

<details><summary>(4) Call the <code>build_vector()</code> method of the PP-ChatOCRv4 pipeline object to construct vectors for text content.</summary>

Below are the parameters and their descriptions for the `build_vector()` method:

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
<td><code>visual_info</code></td>
<td>Visual information, which can be a dictionary containing visual information or a list composed of such dictionaries</td>
<td><code>list|dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Minimum number of characters</td>
<td><code>int</code></td>
<td>
A positive integer greater than 0, determined based on the token length supported by the large language model
</td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>block_size</code></td>
<td>Chunk size for establishing a vector library for long text</td>
<td><code>int</code></td>
<td>
A positive integer greater than 0, determined based on the token length supported by the large language model
</td>
<td><code>300</code></td>
</tr>
<tr>
<td><code>flag_save_bytes_vector</code></td>
<td>Whether to save text as a binary file</td>
<td><code>bool</code></td>
<td>
<code>True|False</code>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model, referring to the "LLM_Retriever" field in the configuration file</td>
<td><code>dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
</table>

This method returns a dictionary containing visual text information, with the following content:

- `flag_save_bytes_vector`: `(bool)` Whether the result is saved as a binary file
- `flag_too_short_text`: `(bool)` Whether the text length is less than the minimum number of characters
- `vector`: `(str|list)` Binary content or text content of the text, depending on the values of `flag_save_bytes_vector` and `min_characters`. If `flag_save_bytes_vector=True` and the text length is greater than or equal to the minimum number of characters, binary content is returned; otherwise, the original text is returned.
</details>

<details><summary>(5) Call the <code>mllm_pred()</code> method of the PP-ChatOCRv4 pipeline object to obtain multimodal large model extraction results.</summary>

Below are the parameters and their descriptions for the `mllm_pred()` method:

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
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required</td>
<td><code>Python Var|str</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data</li>
  <li><b>str</b>: Local path of an image file or a single-page PDF file, e.g., <code>/root/data/img.jpg</code>; <b>or URL link</b>, such as the network URL of an image file or a single-page PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>key_list</code></td>
<td>A single key or a list of keys used to extract information</td>
<td><code>Union[str, List[str]]</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_chat_bot_config</code></td>
<td>Configuration parameters for the multimodal large model, referring to the "MLLM_Chat" field in the configuration file</td>
<td><code>dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>(6) Call the <code>chat()</code> method of the PP-ChatOCRv4 pipeline object to extract key information.</summary>

Below are the parameters and their descriptions for the `chat()` method:

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
<tbody>
<tr>
<td><code>key_list</code></td>
<td>A single key or a list of keys used to extract information</td>
<td><code>Union[str, List[str]]</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>visual_info</code></td>
<td>Visual information results</td>
<td><code>List[dict]</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_vector_retrieval</code></td>
<td>Whether to use vector retrieval</td>
<td><code>bool</code></td>
<td><code>True|False</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>vector_info</code></td>
<td>Vector information for retrieval</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Minimum number of characters required</td>
<td><code>int</code></td>
<td>A positive integer greater than 0</td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>text_task_description</code></td>
<td>Description of the text task</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_output_format</code></td>
<td>Output format of the text result</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rules_str</code></td>
<td>Rules for generating text results</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_text_content</code></td>
<td>Text content for few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_key_value_list</code></td>
<td>Key-value list for few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_task_description</code></td>
<td>Description of the table task</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_output_format</code></td>
<td>Output format of the table result</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_rules_str</code></td>
<td>Rules for generating table results</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_text_content</code></td>
<td>Text content for table few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_key_value_list</code></td>
<td>Key-value list for table few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_predict_info</code></td>
<td>Results from the multimodal large language model</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_integration_strategy</code></td>
<td>Integration strategy for multimodal large language model and large language model data, supporting the use of either alone or the fusion of both results</td>
<td><code>str</code></td>
<td><code>"integration"</code></td>
<td><code>"integration", "llm_only", and "mllm_only"</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>Configuration information for the large language model, with content referring to the "LLM_Chat" field in the pipeline configuration file</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model, with content referring to the "LLM_Retriever" field in the configuration file</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

This method will print the results to the terminal. The content printed to the terminal is explained as follows:
  - `chat_res`: `(dict)` The result of information extraction, which is a dictionary containing the keys to be extracted and their corresponding values.

</details>

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the sample code in [2.2 Local Experience](#22-local-experience).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed instructions on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ **Serving**: Serving is a common deployment form in actual production environments. By encapsulating the inference functionality as a service, clients can access these services through network requests to obtain inference results. PaddleX supports multiple serving solutions for pipelines. For detailed instructions on serving, please refer to the [PaddleX Serving Guide](../../../pipeline_deploy/serving.md).

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
<td>A simplified version of the <code>res</code> field in the JSON representation of the results generated by the pipeline's <code>visual_predict</code> method, with the <code>input_path</code> and the <code>page_index</code> fields removed.</td>
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
<td><code>integer</code> | <code>null</code></td>
<td>Minimum data length to enable the vector database.</td>
<td>No</td>
</tr>
<tr>
<td><code>blockSize</code></td>
<td><code>int</code> | <code>null</code></td>
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
<td><code>boolean</code> | <code>null</code></td>
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
<td>Minimum data length to enable the vector database.</td>
<td>No</td>
</tr>
<tr>
<td><code>textTaskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
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
<td><code>string</code> | <code>null</code></td>
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


API_BASE_URL = "http://0.0.0.0:8080"

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
</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method where computing and data processing functions are placed on the user's device itself. The device can directly process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions on edge deployment, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md).
You can choose an appropriate deployment method for your pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Custom Development
