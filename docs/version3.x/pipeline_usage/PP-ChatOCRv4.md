---
comments: true
---

# PP-ChatOCRv4 产线使用教程

## 1. PP-ChatOCRv4 产线介绍

PP-ChatOCRv4 是飞桨特色的文档和图像智能分析解决方案，结合了 LLM、MLLM 和 OCR 技术，一站式解决版面分析、生僻字、多页 pdf、表格、印章文本识别等常见的复杂文档信息抽取难点问题，结合文心大模型将海量数据和知识相融合，准确率高且应用广泛。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上部署。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4.png" width="80%"/>
</div>

<b>PP-ChatOCRv4 产线中包含以下9个模块。每个模块均可独立进行训练和推理，并包含多个模型。有关详细信息，请点击相应模块以查看文档。</b>

- [文档图像方向分类模块](../module_usage/doc_img_orientation_classification.md)（可选）
- [文本图像矫正模块](../module_usage/text_image_unwarping.md)（可选）
- [版面区域检测模块](../module_usage/layout_detection.md)
- [表格结构识别模块](../module_usage/table_structure_recognition.md)（可选）
- [文本检测模块](../module_usage/text_detection.md)
- [文本识别模块](../module_usage/text_recognition.md)
- [文本行方向分类模块](../module_usage/textline_orientation_classification.md)（可选）
- [公式识别模块](../module_usage/formula_recognition.md)（可选）
- [印章文本检测模块](../module_usage/seal_text_detection.md)（可选）

在本产线中，您可以根据下方的基准测试数据选择使用的模型。

> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

<details>
<summary> <b>文档图像方向分类模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>文本图像矫正模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>CER </th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>版面区域检测模块模型：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>基于RT-DETR-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的高精度版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">训练模型</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>基于PicoDet-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的精度效率平衡的版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">训练模型</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>基于PicoDet-S在中英文论文、杂志、合同、书本、试卷和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet_layout_1x</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>97.8</td>
<td>9.62 / 6.75</td>
<td>26.96 / 12.77</td>
<td>7.4</td>
<td>基于PicoDet-1x在PubLayNet数据集训练的高效率版面区域定位模型，可定位包含文字、标题、表格、图片以及列表这5类区域</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>97.5</td>
<td>9.57 / 6.63</td>
<td>27.66 / 16.75</td>
<td>7.4</td>
<td>基于PicoDet-1x在自建数据集训练的高效率版面区域定位模型，可定位包含表格这1类区域</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>88.2</td>
<td>8.43 / 3.44</td>
<td>17.60 / 6.51</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>8.80 / 3.62</td>
<td>17.51 / 6.35</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>表格结构识别模块（可选）：</b></summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>精度（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>SLANet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td>
<td>59.52</td>
<td>23.96 / 21.75</td>
<td>- / 43.12</td>
<td>6.9</td>
<td>SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络 PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块 SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
</tr>
<tr>
<td>SLANet_plus</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
<td>63.69</td>
<td>23.43 / 22.16</td>
<td>- / 41.80</td>
<td>6.9</td>
<td>SLANet_plus 是百度飞桨视觉团队自研的表格结构识别模型SLANet的增强版。相较于SLANet，SLANet_plus 对无线表、复杂表格的识别能力得到了大幅提升，并降低了模型对表格定位准确性的敏感度，即使表格定位出现偏移，也能够较准确地进行识别。</td>
</tr>
</table>
</details>

<details>
<summary> <b>文本检测模块：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">训练模型</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>383.15 / 383.15</td>
<td>84.3</td>
<td>PP-OCRv5 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>79.0</td>
<td>10.67 / 6.36</td>
<td>57.77 / 28.15</td>
<td>4.7</td>
<td>PP-OCRv5 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>69.2</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>文本识别模块：</b></summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">训练模型</a></td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>81.29</td>
<td> 1.46/5.43 </td>
<td> 5.32/91.79 </td>
<td>16</td>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">训练模型</a></td>
<td>86.58</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>181</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>85.19</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>173</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td>基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
</table>

> ❗ 以上列出的是文本识别模块重点支持的<b>6个核心模型</b>，该模块总共支持<b>20个全量模型</b>，包含多个多语言文本识别模型，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

* <b>PP-OCRv5 多场景模型</b>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>中文识别 Avg Accuracy(%)</th>
<th>英文识别 Avg Accuracy(%)</th>
<th>繁体中文识别 Avg Accuracy(%)</th>
<th>日文识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">训练模型</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_server_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
</tr>
</table>

* <b>中文识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">训练模型</a></td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>182</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>85.19</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>173</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>72.96</td>
<td>3.89 / 1.16</td>
<td>8.72 / 3.56</td>
<td>10.3</td>
<td>PP-OCRv3的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
</table>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>80.5</td>
<td rowspan="1">
SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。
</td>
</tr>
</table>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>48.8</td>
<td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
</tr>
</table>

* <b>英文识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td> 70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td>基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>70.69</td>
<td>3.56 / 0.78</td>
<td>8.44 / 5.78</td>
<td>17.3</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
</table>


* <b>多语言识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>60.21</td>
<td>3.73 / 0.98</td>
<td>8.76 / 2.91</td>
<td>9.6</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量韩文识别模型，支持韩文、数字识别</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>45.69</td>
<td>3.86 / 1.01</td>
<td>8.62 / 2.92</td>
<td>9.8</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量日文识别模型，支持日文、数字识别</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>82.06</td>
<td>3.90 / 1.16</td>
<td>9.24 / 3.18</td>
<td>10.8</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量繁体中文识别模型，支持繁体中文、数字识别</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>95.88</td>
<td>3.59 / 0.81</td>
<td>8.28 / 6.21</td>
<td>8.7</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰卢固文识别模型，支持泰卢固文、数字识别</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>96.96</td>
<td>3.49 / 0.89</td>
<td>8.63 / 2.77</td>
<td>17.4</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量卡纳达文识别模型，支持卡纳达文、数字识别</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>76.83</td>
<td>3.49 / 0.86</td>
<td>8.35 / 3.41</td>
<td>8.7</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰米尔文识别模型，支持泰米尔文、数字识别</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>76.93</td>
<td>3.53 / 0.78</td>
<td>8.50 / 6.83</td>
<td>8.7</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量拉丁文识别模型，支持拉丁文、数字识别</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>73.55</td>
<td>3.60 / 0.83</td>
<td>8.44 / 4.69</td>
<td>17.3</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量阿拉伯字母识别模型，支持阿拉伯字母、数字识别</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>94.28</td>
<td>3.56 / 0.79</td>
<td>8.22 / 2.76</td>
<td>8.7</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量斯拉夫字母识别模型，支持斯拉夫字母、数字识别</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>96.44</td>
<td>3.60 / 0.78</td>
<td>6.95 / 2.87</td>
<td>8.7</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量梵文字母识别模型，支持梵文字母、数字识别</td>
</tr>
</table>
</details>
</details>

<details>
<summary> <b>文本行方向分类模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th>
<th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">训练模型</a></td>
<td>98.85</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.96</td>
<td>基于PP-LCNet_x0_25的文本行分类模型，含有两个类别，即0度，180度</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>公式识别模块（可选）：</b></summary>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>UniMERNet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">训练模型</a></td>
<td>85.91</td>
<td>43.50</td>
<td>1311.84 / 1311.84</td>
<td>- / 8288.07</td>
<td>1530</td>
<td>UniMERNet是由上海AI Lab研发的一款公式识别模型。该模型采用Donut Swin作为编码器，MBartDecoder作为解码器，并通过在包含简单公式、复杂公式、扫描捕捉公式和手写公式在内的一百万数据集上进行训练，大幅提升了模型对真实场景公式的识别准确率</td>
</tr>
<tr>
<td>PP-FormulaNet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">训练模型</a></td>
<td>87.00</td>
<td>45.71</td>
<td>182.25 / 182.25</td>
<td>- / 254.39</td>
<td>224</td>
<td rowspan="2">PP-FormulaNet 是由百度飞桨视觉团队开发的一款先进的公式识别模型，支持5万个常见LateX源码词汇的识别。PP-FormulaNet-S 版本采用了 PP-HGNetV2-B4 作为其骨干网络，通过并行掩码和模型蒸馏等技术，大幅提升了模型的推理速度，同时保持了较高的识别精度，适用于简单印刷公式、跨行简单印刷公式等场景。而 PP-FormulaNet-L 版本则基于 Vary_VIT_B 作为骨干网络，并在大规模公式数据集上进行了深入训练，在复杂公式的识别方面，相较于PP-FormulaNet-S表现出显著的提升，适用于简单印刷公式、复杂印刷公式、手写公式等场景。 </td>

</tr>
<td>PP-FormulaNet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">训练模型</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1482.03 / 1482.03</td>
<td>- / 3131.54</td>
<td>695</td>
</tr>
<td>PP-FormulaNet_plus-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">训练模型</a></td>
<td>88.71</td>
<td>53.32</td>
<td>179.20 / 179.20</td>
<td>- / 260.99</td>
<td>248</td>
<td rowspan="3">PP-FormulaNet_plus 是百度飞桨视觉团队在 PP-FormulaNet 的基础上开发的增强版公式识别模型。与原版相比，PP-FormulaNet_plus 在训练中使用了更为丰富的公式数据集，包括中文学位论文、专业书籍、教材试卷以及数学期刊等多种来源。这一扩展显著提升了模型的识别能力。

其中，PP-FormulaNet_plus-M 和 PP-FormulaNet_plus-L 模型新增了对中文公式的支持，并将公式的最大预测 token 数从 1024 扩大至 2560，大幅提升了对复杂公式的识别性能。同时，PP-FormulaNet_plus-S 模型则专注于增强英文公式的识别能力。通过这些改进，PP-FormulaNet_plus 系列模型在处理复杂多样的公式识别任务时表现更加出色。 </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">训练模型</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1040.27 / 1040.27</td>
<td>- / 1615.80</td>
<td>592</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">训练模型</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1476.07 / 1476.07</td>
<td>- / 3125.58</td>
<td>698</td>
</tr>
<tr>
<td>LaTeX_OCR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>99</td>
<td>LaTeX-OCR是一种基于自回归大模型的公式识别算法，通过采用 Hybrid ViT 作为骨干网络，transformer作为解码器，显著提升了公式识别的准确性。</td>
</tr>
</table>
</details>

<details>
<summary> <b>印章文本检测模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td>
<td>98.40</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td>PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td>PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署</td>
</tr>
</tbody>
</table>
</details>
</details>

<details>
<summary> <b>测试环境说明：</b></summary>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
                       <li><strong>测试数据集：
             </strong>
                <ul>
                  <li>文档图像方向分类模型：PaddleOCR 自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li>文本图像矫正模型：<a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>。 </li>
                  <li>版面区域检测模型：PaddleOCR 自建的版面区域分析数据集，包含中英文论文、杂志和研报等常见的 1w 张文档类型图片。 </li>
                  <li> 表格结构识别模型：内部自建的英文表格识别数据集。</li>
                  <li> 文本检测模型：PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。</li>
                  <li>中文识别模型： PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。 </li>
                  <li>ch_SVTRv2_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜评估集。</li>
                  <li>ch_RepSVTR_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜评估集。</li>
                  <li>英文识别模型：PaddleOCR 自建的英文数据集。</li>
                  <li>多语言识别模型：PaddleOCR 自建的多语种数据集。</li>
                  <li>文本行方向分类模型：PaddleOCR 自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li>印章文本检测模型：PaddleOCR 自建的数据集，包含500张圆形印章图像。</li>
                </ul>
             </li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>软件环境：</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle 3.0.0 / paddleocr 3.0.3</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>推理模式说明</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>模式</th>
            <th>GPU配置</th>
            <th>CPU配置</th>
            <th>加速技术组合</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>常规模式</td>
            <td>FP32精度 / 无TRT加速</td>
            <td>FP32精度 / 8线程</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>高性能模式</td>
            <td>选择先验精度类型和加速策略的最优组合</td>
            <td>FP32精度 / 8线程</td>
            <td>选择先验最优后端（Paddle/OpenVINO/TRT等）</td>
        </tr>
    </tbody>
</table>
</details>

<br />
<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>

## 2. 快速开始

在本地使用 PP-ChatOCRv4 产线前，请确保您已经按照[安装教程](../installation.md)完成了wheel包安装。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `ie`。

**请注意，如果在执行过程中遇到程序失去响应、程序异常退出、内存资源耗尽、推理速度极慢等问题，请尝试参考文档调整配置，例如关闭不需要使用的功能或使用更轻量的模型。**

在进行模型推理之前，首先需要准备大语言模型的 api_key，PP-ChatOCRv4 支持在[百度云千帆平台](https://console.bce.baidu.com/qianfan/ais/console/onlineService)或者本地部署的标准 OpenAI 接口大模型服务。如果使用百度云千帆平台，可以参考[认证鉴权](https://cloud.baidu.com/doc/qianfan-api/s/ym9chdsy5) 获取 api_key。如果使用本地部署的大模型服务，可以参考[PaddleNLP大模型部署文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)进行大模型部署对话接口部署和向量化接口部署，并填写对应的 base_url 和 api_key 即可。如果需要使用多模态大模型进行数据融合，可以参考[PaddleMIX模型文档](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)中的OpenAI服务部署进行多模态大模型部署，并填写对应的 base_url 和 api_key 即可。

**注:** 如果因本地环境限制无法在本地部署多模态大模型，可以将代码中的含有“mllm”变量的行注释掉，仅使用大语言模型完成信息抽取。

### 2.1 命令行方式体验

可以下载 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png)，使用一行命令即可快速体验产线效果：

```bash
paddleocr pp_chatocrv4_doc -i vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key

# 通过 --invoke_mllm 和 --pp_docbee_base_url 使用多模态大模型
paddleocr pp_chatocrv4_doc -i vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --invoke_mllm True --pp_docbee_base_url http://127.0.0.1:8080/
```

<details><summary><b>命令行支持更多参数设置，点击展开以查看命令行参数的详细说明</b></summary>
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>待预测数据，必填。如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>keys</code></td>
<td>用于信息提取的键。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径。如果不设置，推理结果将不会保存到本地。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>invoke_mllm</code></td>
<td>是否加载并使用多模态大模型。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>用于版面区域检测的模型名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>文本检测模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>文本检测模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的batch size。如果不设置，将默认设置batch size为<code>1</code>。</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_name</code></td>
<td>表格结构识别模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_dir</code></td>
<td>表格结构识别模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>印章文本检测模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>印章文本检测模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>印章文本识别模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>印章文本识别模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>印章文本识别模型的batch size。如果不设置，将默认设置batch size为<code>1</code>。</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向分类模块。如果不设置，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否加载并使用印章文本识别子产线。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否加载并使用表格识别子产线。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。
<code>0-1</code> 之间的任意浮点数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>0.5</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面检测是否使用后处理NMS。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数。任意大于 <code>0</code>  浮点数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>1.0</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面检测中模型输出的检测框的合并处理模式。
<ul>
<li><b>large</b>，设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框；</li>
<li><b>small</b>，设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框；</li>
<li><b>union</b>，不进行框的过滤处理，内外框都保留；</li>
</ul>如果不设置，将使用产线初始化的该参数值，默认初始化为<code>large</code>。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的图像边长限制。大于 <code>0</code> 的任意整数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>960</code>。
</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的边长度限制类型。支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
</ul>如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>max</code>。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值。输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。大于 <code>0</code> 的任意浮点数。如果不设置，将默认使用产线初始化的该参数值 <code>0.3</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>0.6</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。大于 <code>0</code> 的任意浮点数。如果不设置，将默认使用产线初始化的该参数值 <code>2.0</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。大于 <code>0</code> 的任意浮点数。如果不设置，将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>印章文本检测的图像边长限制。大于 <code>0</code> 的任意整数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>736</code>。
</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>印章文本检测的图像边长限制类型。支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code>。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>min</code>。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。大于 <code>0</code> 的任意浮点数。如果不设置，将默认使用产线初始化的该参数值 <code>0.2</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。大于 <code>0</code> 的任意浮点数。如果不设置，将默认使用产线初始化的该参数值 <code>0.6</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。大于 <code>0</code> 的任意浮点数
。如果不设置，将使用产线初始化的该参数值，默认为 <code>0.5</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>印章文本识别阈值，得分大于该阈值的文本结果会被保留。大于 <code>0</code> 的任意浮点数
。如果不设置，将使用产线初始化的该参数值，默认为<code>0.0</code>，即不设阈值。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>qianfan_api_key</code></td>
<td>千帆平台的API key。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>pp_docbee_base_url</code></td>
<td>多模态大模型服务的URL。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号：
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
</ul>如果不设置，将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>是否启用高性能推理。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>是否启用 Paddle Inference 的 TensorRT 子图引擎。如果模型不支持通过 TensorRT 加速，即使设置了此标志，也不会使用加速。<br/>
对于 CUDA 11.8 版本的飞桨，兼容的 TensorRT 版本为 8.x（x>=6），建议安装 TensorRT 8.6.1.6。<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN 缓存容量。
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上进行推理时使用的线程数。</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>
<br />

运行结果会被打印到终端上，运行结果如下：

```bash
驾驶室准乘人数 2
```

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，可以下载 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png)，使用如下示例代码进行推理：

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

运行后，输出结果如下：

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

PP-ChatOCRv4 预测的流程、API说明、产出说明如下：

<details><summary>（1）调用 <code>PPChatOCRv4Doc</code> 实例化PP-ChatOCRv4产线对象。</summary>

相关参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>用于版面区域检测的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>文本检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>文本检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的batch size。如果设置为<code>None</code>，将默认设置batch size为<code>1</code>。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_name</code></td>
<td>表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_structure_recognition_model_dir</code></td>
<td>表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>印章文本检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>印章文本检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_name</code></td>
<td>印章文本识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>印章文本识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>印章文本识别模型的batch size。如果设置为<code>None</code>，将默认设置batch size为<code>1</code>。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向分类模块. 如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否加载并使用印章文本识别子产线。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否加载并使用表格识别子产线。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。
<ul>
<li><b>float</b>：<code>0-1</code> 之间的任意浮点数；</li>
<li><b>dict</b>： <code>{0:0.1}</code> key为类别ID，value为该类别的阈值；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>0.5</code>。</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面检测是否使用后处理NMS。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数。
<ul>
<li><b>float</b>：任意大于 <code>0</code>  浮点数；</li>
<li><b>Tuple[float,float]</b>：在横纵两个方向各自的扩张系数；</li>
<li><b>dict</b>，dict的key为<b>int</b>类型，代表<code>cls_id</code>，value为<b>tuple</b>类型，如<code>{0: (1.1，2.0)}</code>，表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>1.0</code>。</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测的重叠框过滤方式。
<ul>
<li><b>str</b>：<code>large</code>，<code>small</code>，<code>union</code>，分别表示重叠框过滤时选择保留大框，小框还是同时保留；</li>
<li><b>dict</b>，dict的key为<b>int</b>类型，代表<code>cls_id</code>，value为<b>str</b>类型，如<code>{0: "large"，2: "small"}</code>，表示对第0类别检测框使用large模式，对第2类别检测框使用small模式；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>large</code>。</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的图像边长限制。
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>960</code>。</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的边长度限制类型。
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>max</code>。</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.3</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.6</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>2.0</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.0</code>，即不设阈值。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>印章文本检测的图像边长限制。
<ul>
<li><b>int</b>：大于<code>0</code>的任意整数；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>736</code>。</li>
</ul>
</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>印章文本检测的图像边长限制类型。
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>min</code>。</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.2</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.6</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.5</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>印章文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认为<code>0.0</code>，即不设阈值。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>向量检索大模型配置参数。配置内容为如下dict：
<pre><code>{
"module_name": "retriever",
"model_name": "embedding-v1",
"base_url": "https://qianfan.baidubce.com/v2",
"api_type": "qianfan",
"api_key": "api_key"  # 请将此设置为实际的API密钥
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_chat_bot_config</code></td>
<td>多模态大模型配置参数。配置内容为如下dict：
<pre><code>{
"module_name": "chat_bot",
"model_name": "PP-DocBee",
"base_url": "http://127.0.0.1:8080/", # 请将此设置为实际的多模态大模型服务的url
"api_type": "openai",
"api_key": "api_key"  # 请将此设置为实际的API密钥
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>大语言模型配置信息。配置内容为如下dict：
<pre><code>{
"module_name": "chat_bot",
"model_name": "ernie-3.5-8k",
"base_url": "https://qianfan.baidubce.com/v2",
"api_type": "openai",
"api_key": "api_key"  # 请将此设置为实际的API密钥
}</code></pre>
</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号：
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为<code>None</code>，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备。</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>是否启用高性能推理。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>是否启用 Paddle Inference 的 TensorRT 子图引擎。如果模型不支持通过 TensorRT 加速，即使设置了此标志，也不会使用加速。<br/>
对于 CUDA 11.8 版本的飞桨，兼容的 TensorRT 版本为 8.x（x>=6），建议安装 TensorRT 8.6.1.6。<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN 缓存容量。
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上进行推理时使用的线程数。</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>（2）调用 PP-ChatOCRv4 产线对象的 <code>visual_predict()</code> 方法获取视觉预测结果，该方法会返回一个结果列表。另外，产线还提供了 <code>visual_predict_iter()</code> 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 <code>visual_predict_iter()</code> 返回的是一个 <code>generator</code>，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。以下是 <code>visual_predict()</code> 方法的参数及其说明：</summary>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填。
<ul>
  <li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据；</li>
  <li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)；</li>
  <li><b>list</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code>。</li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否在推理时使用文档方向分类模块。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否在推理时使用文本图像矫正模块。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向分类模块。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否在推理时使用印章文本识别子产线。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否在推理时使用表格识别子产线。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
</table>
</details>

<details><summary>（3）对视觉预测结果进行处理。</summary>

每个样本的预测结果均为 `dict` 类型，包含 `visual_info` 和 `layout_parsing_result` 两个字段。通过 `visual_info` 得到视觉信息（包含 `normal_text_dict`、`table_text_list`、`table_html_list` 等信息），并将每个样本的信息放到 `visual_info_list` 列表中，该列表的内容会在之后送入大语言模型中。

当然，您也可以通过 `layout_parsing_result` 获取版面解析的结果，该结果包含文件或图片中包含的表格、文字、图片等内容，且支持打印、保存为图片、保存为`json`文件的操作:

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
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化。</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效。</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效。</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致。</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效。</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效。</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将中间各个模块的可视化图像保存在png格式的图像。</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>将文件中的表格保存为html格式的文件。</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>将文件中的表格保存为xlsx格式的文件。</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：
    - `input_path`: `(str)` 待预测图像的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_seal_recognition`: `(bool)` 控制是否启用印章文本识别子产线
        - `use_table_recognition`: `(bool)` 控制是否启用表格识别子产线
        - `use_formula_recognition`: `(bool)` 控制是否启用公式识别子产线

    - `parsing_res_list`: `(List[Dict])` 解析结果的列表，每个元素为一个dict，列表顺序为解析后的阅读顺序。
        - `block_bbox`: `(np.ndarray)` 版面区域的边界框。
        - `block_label`: `(str)` 版面区域的标签，例如`text`, `table`等。
        - `block_content`: `(str)` 内容为版面区域内的内容。

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` 全局 OCR 结果的dict
      -  `input_path`: `(Union[str, None])` 图像OCR子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
      - `model_settings`: `(Dict)` OCR子产线的模型配置参数
      - `dt_polys`: `(List[numpy.ndarray])` 文本检测的多边形框列表。每个检测框由4个顶点坐标构成的numpy数组表示，数组shape为(4, 2)，数据类型为int16
      - `dt_scores`: `(List[float])` 文本检测框的置信度列表
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 文本检测模块的配置参数
        - `limit_side_len`: `(int)` 图像预处理时的边长限制值
        - `limit_type`: `(str)` 边长限制的处理方式
        - `thresh`: `(float)` 文本像素分类的置信度阈值
        - `box_thresh`: `(float)` 文本检测框的置信度阈值
        - `unclip_ratio`: `(float)` 文本检测框的膨胀系数
        - `text_type`: `(str)` 文本检测的类型，当前固定为"general"

      - `text_type`: `(str)` 文本检测的类型，当前固定为"general"
      - `textline_orientation_angles`: `(List[int])` 文本行方向分类的预测结果。启用时返回实际角度值（如[0,0,1]
      - `text_rec_score_thresh`: `(float)` 文本识别结果的过滤阈值
      - `rec_texts`: `(List[str])` 文本识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本
      - `rec_scores`: `(List[float])` 文本识别的置信度列表，已按`text_rec_score_thresh`过滤
      - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的文本检测框列表，格式同`dt_polys`

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 公式识别结果列表，每个元素为一个dict
        - `rec_formula`: `(str)` 公式识别结果
        - `rec_polys`: `(numpy.ndarray)` 公式检测框，shape为(4, 2)，dtype为int16
        - `formula_region_id`: `(int)` 公式所在的区域编号

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 印章文本识别结果列表，每个元素为一个dict
        - `input_path`: `(str)` 印章图像的输入路径
        - `model_settings`: `(Dict)` 印章文本识别子产线的模型配置参数
        - `dt_polys`: `(List[numpy.ndarray])` 印章检测框列表，格式同`dt_polys`
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 印章检测模块的配置参数, 具体参数含义同上
        - `text_type`: `(str)` 印章检测的类型，当前固定为"seal"
        - `text_rec_score_thresh`: `(float)` 印章文本识别结果的过滤阈值
        - `rec_texts`: `(List[str])` 印章文本识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本
        - `rec_scores`: `(List[float])` 印章文本识别的置信度列表，已按`text_rec_score_thresh`过滤
        - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的印章检测框列表，格式同`dt_polys`
        - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 表格识别结果列表，每个元素为一个dict
        - `cell_box_list`: `(List[numpy.ndarray])` 表格单元格的边界框列表
        - `pred_html`: `(str)` 表格的HTML格式字符串
        - `table_ocr_pred`: `(dict)` 表格的OCR识别结果
            - `rec_polys`: `(List[numpy.ndarray])` 单元格的检测框列表
            - `rec_texts`: `(List[str])` 单元格的识别结果
            - `rec_scores`: `(List[float])` 单元格的识别置信度
            - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：
<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果。</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化图像。</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个dict类型的数据。其中，键分别为 `layout_det_res`、`overall_ocr_res`、`text_paragraphs_ocr_res`、`formula_res_region1`、`table_cell_img` 和 `seal_res_region1`，对应的值是 `Image.Image` 对象：分别用于显示版面区域检测、OCR、OCR文本段落、公式、表格和印章结果的可视化图像。如果没有使用可选模块，则dict中只包含 `layout_det_res`。
</details>

<details><summary>（4）调用PP-ChatOCRv4的产线对象的 <code>build_vector()</code> 方法，对文本内容进行向量构建。</summary>

以下是 `build_vector()` 方法的参数及其说明：


<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>visual_info</code></td>
<td>视觉信息，可以是包含视觉信息的dict，或者由这些dict组成的列表。</td>
<td><code>list|dict</code></td>
<td></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>最小字符数量。为大于0的正整数，可以根据大语言模型支持的token长度来决定。</td>
<td><code>int</code></td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>block_size</code></td>
<td>长文本建立向量库时分块大小。为大于0的正整数，可以根据大语言模型支持的token长度来决定。</td>
<td><code>int</code></td>
<td><code>300</code></td>
</tr>
<tr>
<td><code>flag_save_bytes_vector</code></td>
<td>文字是否保存为二进制文件。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>向量检索大模型配置参数，参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</table>
该方法会返回一个包含视觉文本信息的dict，dict的内容如下：

- `flag_save_bytes_vector`：`(bool)`是否将结果保存为二进制文件
- `flag_too_short_text`：`(bool)`是否文本长度小于最小字符数量
- `vector`: `(str|list)` 文本的二进制内容或者文本内容，取决于`flag_save_bytes_vector`和`min_characters`的值，如果`flag_save_bytes_vector=True`且文本长度大于等于最小字符数量，则返回二进制内容；否则返回原始的文本。
</details>

<details><summary>（5）调用PP-ChatOCRv4的产线对象的 <code>mllm_pred()</code> 方法，获取多模态大模型抽取结果。</summary>

以下是 `mllm_pred()` 方法的参数及其说明：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填。
<ul>
  <li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据；</li>
  <li><b>str</b>：如图像文件或者单页PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或单页PDF文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png">示例</a>。</li>
</ul>
</td>
<td><code>Python Var|str</code></td>
<td></td>
</tr>
<tr>
<td><code>key_list</code></td>
<td>用于提取信息的单个键或键列表。</td>
<td><code>Union[str, List[str]]</code></td>
<td></td>
</tr>
<tr>
<td><code>mllm_chat_bot_config</code></td>
<td>多模态大模型配置参数，参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>（6）调用PP-ChatOCRv4的产线对象的 <code>chat()</code> 方法，对关键信息进行抽取。</summary>

以下是 `chat()` 方法的参数及其说明：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>key_list</code></td>
<td>用于提取信息的单个键或键列表。</td>
<td><code>Union[str, List[str]]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>visual_info</code></td>
<td>视觉信息结果。</td>
<td><code>List[dict]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_vector_retrieval</code></td>
<td>是否使用向量检索。</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>vector_info</code></td>
<td>用于检索的向量信息。</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>所需的最小字符数。为大于0的正整数。</td>
<td><code>int</code></td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>text_task_description</code></td>
<td>文本任务的描述。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_output_format</code></td>
<td>文本结果的输出格式。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rules_str</code></td>
<td>生成文本结果的规则。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_text_content</code></td>
<td>用于少样本演示的文本内容。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_key_value_list</code></td>
<td>用于少样本演示的键值列表。/td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_task_description</code></td>
<td>表任务的描述。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_output_format</code></td>
<td>表结果的输出格式。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_rules_str</code></td>
<td>生成表结果的规则。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_text_content</code></td>
<td>表少样本演示的文本内容。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_key_value_list</code></td>
<td>表少样本演示的键值列表。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_predict_info</code></td>
<td>多模态大模型结果。</td>
<td><code>dict|None</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
<td><code>mllm_integration_strategy</code></td>
<td>多模态大模型和大语言模型数据融合策略，支持单独使用其中一个或者融合两者结果。可选："integration", "llm_only" and "mllm_only"。</td>
<td><code>str</code></td>
<td><code>"integration"</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>大语言模型配置信息，参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>向量检索大模型配置参数，参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

该方法会将结果打印到终端，打印到终端的内容解释如下：
  - `chat_res`: `(dict)` 提取信息的结果，是一个dict，包含了待抽取的键和对应的值。

</details>

## 3. 开发集成/部署

如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleOCR 也提供了其他两种部署方式，详细说明如下：

🚀 高性能推理：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleOCR 提供高性能推理功能，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[高性能推理](../deployment/high_performance_inference.md)。

☁️ 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。详细的产线服务化部署流程请参考[服务化部署](../deployment/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>
<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>analyzeImages</code></b></li>
</ul>
<p>使用计算机视觉模型对图像进行分析，获得OCR、表格识别结果等，并提取图像中的关键信息。</p>
<p><code>POST /chatocr-visual</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。默认对于超过10页的PDF文件，只有前10页的内容会被处理。<br /> 要解除页数限制，请在产线配置文件中添加以下配置：
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre>
</td>
<td>是</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则将根据URL推断文件类型。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>use_doc_orientation_classify</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>use_doc_unwarping</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>use_seal_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>use_table_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>layout_threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>layout_nms</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>layout_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>layout_merge_bboxes_mode</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>text_det_limit_side_len</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>text_det_limit_type</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>text_det_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>text_det_box_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>text_det_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>text_rec_score_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>seal_det_limit_side_len</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>seal_det_limit_type</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>seal_det_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>seal_det_box_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>seal_det_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>visual_predict</code> 方法的 <code>seal_rec_score_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>是否返回可视化结果图以及处理过程中的中间图像等。
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>传入 <code>true</code>：返回图像。</li>
<li>传入 <code>false</code>：不返回图像。</li>
<li>若请求体中未提供该参数或传入 <code>null</code>：遵循产线配置文件<code>Serving.visualize</code> 的设置。</li>
</ul>
<br/>例如，在产线配置文件中添加如下字段：<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
将默认不返回图像，通过请求体中的<code>visualize</code>参数可以覆盖默认行为。如果请求体和配置文件中均未设置（或请求体传入<code>null</code>、配置文件中未设置），则默认返回图像。
</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>使用计算机视觉模型得到的分析结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>array</code></td>
<td>图像中的关键信息，可用作其他操作的输入。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>layoutParsingResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>产线对象的 <code>visual_predict</code> 方法生成的 <code>layout_parsing_result</code> 的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 和 <code>page_index</code> 字段。</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>参见产线视觉预测结果的 <code>img</code> 属性说明。</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>buildVectorStore</code></b></li>
</ul>
<p>构建向量数据库。</p>
<p><code>POST /chatocr-vector</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>visualInfo</code></td>
<td><code>array</code></td>
<td>图像中的关键信息。由<code>analyzeImages</code>操作提供。</td>
<td>是</td>
</tr>
<tr>
<td><code>minCharacters</code></td>
<td><code>integer</code></td>
<td>请参阅产线对象中 <code>build_vector</code> 方法的 <code>min_characters</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>blockSize</code></td>
<td><code>integer</code></td>
<td>请参阅产线对象中 <code>build_vector</code> 方法的 <code>block_size</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>retrieverConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>build_vector</code> 方法的 <code>retriever_config</code> 参数相关说明。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>vectorInfo</code></td>
<td><code>object</code></td>
<td>向量数据库序列化结果，可用作其他操作的输入。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>invokeMLLM</code></b></li>
</ul>
<p>调用多模态大模型。</p>
<p><code>POST /chatocr-mllm</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>keyList</code></td>
<td><code>array</code></td>
<td>键列表。</td>
<td>是</td>
</tr>
<tr>
<td><code>mllmChatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>mllm_pred</code> 方法的 <code>mllm_chat_bot_config</code> 参数相关说明。</td>
<td>否</td>
</tr>
</tbody>
<table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>mllmPredictInfo</code></td>
<td><code>object</code></td>
<td>多模态大模型调用结果。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>chat</code></b></li>
</ul>
<p>与大语言模型交互，利用大语言模型提炼关键信息。</p>
<p><code>POST /chatocr-chat</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>keyList</code></td>
<td><code>array</code></td>
<td>键列表。</td>
<td>是</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>图像中的关键信息。由<code>analyzeImages</code>操作提供。</td>
<td>是</td>
</tr>
<tr>
<td><code>useVectorRetrieval</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>use_vector_retrieval</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>vectorInfo</code></td>
<td><code>object</code> | <code>null</code></td>
<td>向量数据库序列化结果。由<code>buildVectorStore</code>操作提供。</td>
<td>否</td>
</tr>
<tr>
<td><code>minCharacters</code></td>
<td><code>integer</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>min_characters</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textTaskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>text_task_description</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textOutputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>text_output_format</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textRulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>text_rules_str</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textFewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>text_few_shot_demo_text_content</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textFewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>text_few_shot_demo_key_value_list</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>tableTaskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>table_task_description</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>tableOutputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>table_output_format</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>tableRulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>table_rules_str</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>tableFewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>table_few_shot_demo_text_content</code> 参数相关说明。</td>
</tr>
<tr>
<td><code>tableFewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>table_few_shot_demo_key_value_list</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>mllmPredictInfo</code></td>
<td><code>object</code> | <code>null</code></td>
<td>多模态大模型调用结果。由<code>invokeMllm</code>操作提供。</td>
<td>否</td>
</tr>
<tr>
<td><code>mllmIntegrationStrategy</code></td>
<td><code>string</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>mllm_integration_strategy</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>chatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>chat_bot_config</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>retrieverConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>chat</code> 方法的 <code>retriever_config</code> 参数相关说明。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>chatResult</code></td>
<td><code>object</code></td>
<td>关键信息抽取结果。</td>
</tr>
</tbody>
</table>
<li><b>注意：</b></li>
在请求体中包含大模型调用的API key等敏感参数可能存在安全风险。如无必要，请在配置文件中设置这些参数，在请求时不传递。
<br/><br/>
</details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">
# 此脚本只展示了图片的用例，其他文件类型的调用请查看API参考来调整

import base64
import pprint
import sys
import requests


API_BASE_URL = "http://127.0.0.1:8080"

image_path = "./demo.jpg"
keys = ["姓名"]

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
    json keys = { "合格证编号" };

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
        String[] keys = {"合格证编号"};

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
    keys := []string{"合格证编号"}

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
    static readonly string[] keys = { "合格证编号" };

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
const keys = ['合格证编号'];

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
$keys = ["合格证编号"];

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

## 4. 二次开发
如果 PP-ChatOCRv4 产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升在您的场景中的识别效果。

### 4.1 模型微调
由于 PP-ChatOCRv4 产线包含若干模块，模型产线的效果如果不及预期，可能来自于其中任何一个模块。您可以对提取效果差的 case 进行分析，通过可视化图像，确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。


<table>
  <thead>
    <tr>
      <th>情形</th>
      <th>微调模块</th>
      <th>微调参考链接</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>版面区域检测不准，如印章、表格未检出等</td>
      <td>版面区域检测模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>表格结构识别不准</td>
      <td>表格结构识别</td>
      <td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/table_structure_recognition.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>印章文本存在漏检</td>
      <td>印章文本检测模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/seal_text_detection.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>文本存在漏检</td>
      <td>文本检测模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/text_detection.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>文本内容都不准</td>
      <td>文本识别模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/text_recognition.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>垂直或者旋转文本行矫正不准</td>
      <td>文本行方向分类模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>整图旋转矫正不准</td>
      <td>文档图像方向分类模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#_5">链接</a></td>
    </tr>
    <tr>
      <td>图像扭曲矫正不准</td>
      <td>文本图像矫正模块</td>
      <td>暂不支持微调</td>
    </tr>
  </tbody>
</table>


### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件，然后可以通过自定义产线配置文件的方式，使用微调后的模型权重。

1. 获取产线配置文件

可调用 PaddleOCR 中 PPChatOCRv4 产线对象的 `export_paddlex_config_to_yaml` 方法，将当前产线配置导出为 YAML 文件：

```Python
from paddleocr import PPChatOCRv4Doc

pipeline = PPChatOCRv4Doc()
pipeline.export_paddlex_config_to_yaml("PP-ChatOCRv4-doc.yaml")
```

2. 修改配置文件

在得到默认的产线配置文件后，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可。例如

```yaml
......
SubModules:
    TextDetection:
    module_name: text_detection
    model_name: PP-OCRv5_server_det
    model_dir: null # 替换为微调后的文本检测模型权重路径
    limit_side_len: 960
    limit_type: max
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 1.5

    TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_server_rec
    model_dir: null # 替换为微调后的文本检测模型权重路径
    batch_size: 1
            score_thresh: 0
......
```

在产线配置文件中，不仅包含 PaddleOCR CLI 和 Python API 支持的参数，还可进行更多高级配置，具体信息可在 [PaddleX模型产线使用概览](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/pipeline_develop_guide.html) 中找到对应的产线使用教程，参考其中的详细说明，根据需求调整各项配置。

3. 在 CLI 中加载产线配置文件

在修改完成配置文件后，通过命令行的 `--paddlex_config` 参数指定修改后的产线配置文件的路径，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```bash
paddleocr pp_chatocrv4_doc --paddlex_config PP-ChatOCRv4-doc.yaml ...
```

4. 在 Python API 中加载产线配置文件

初始化产线对象时，可通过 `paddlex_config` 参数传入 PaddleX 产线配置文件路径或配置dict，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```python
from paddleocr import PPChatOCRv4Doc

pipeline = PPChatOCRv4Doc(paddlex_config="PP-ChatOCRv4-doc.yaml")
```
