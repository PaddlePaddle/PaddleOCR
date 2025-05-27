---
comments: true
---

# 印章文本识别产线使用教程

## 1. 印章文本识别产线介绍
印章文本识别是一种自动从文档或图像中提取和识别印章内容的技术，印章文本的识别是文档处理的一部分，在很多场景都有用途，例如合同比对，出入库审核以及发票报销审核等场景。

印章文本识别产线用于识别印章的文本内容，提取印章图像中的文字信息以文本形式输出，本产线集成了业界知名的 PP-OCRv4 的端到端 OCR 串联系统，支持弯曲印章文本的检测和识别。同时，本产线集成了可选的版面区域定位模块，可以在整个文档中准确定位印章所在的版面位置。此外也增加可选的文档图像的方向矫正和扭曲矫正功能。基于本产线，可实现 CPU 上毫秒级的文本内容精准预测。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/PP-ChatOCRv3_doc_seal/01.png"/>

<b>印章文本识别产线中包含以下5个模块。每个模块均可独立进行训练和推理，并包含多个模型。有关详细信息，请点击相应模块以查看文档。</b>

- [印章文本检测模块](../module_usage/seal_text_detection.md)
- [文本识别模块](../module_usage/text_recognition.md)
- [版面区域检测模块](../module_usage/layout_detection.md)（可选）
- [文档图像方向分类模块](../module_usage/doc_img_orientation_classification.md) （可选）
- [文本图像矫正模块](../module_usage/text_image_unwarping.md) （可选）

在本产线中，您可以根据下方的基准测试数据选择使用的模型。

<details>
<summary> <b>版面区域检测模块（可选）：</b></summary>

* <b>版面检测模型，包含20个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、表格、图和表标题（图标题、表格标题和图表标题）、印章、图表、侧栏文本和参考文献内容</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">训练模型</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位模型</td>
</tr>
<tr>
</tbody>
</table>

* <b>版面检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>基于RT-DETR-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的高精度版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">训练模型</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>基于PicoDet-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的精度效率平衡的版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">训练模型</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>基于PicoDet-S在中英文论文、杂志、合同、书本、试卷和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
</tbody>
</table>


>❗ 以上列出的是版面检测模块重点支持的<b>4个核心模型</b>，该模块总共支持<b>13个全量模型</b>，包含多个预定义了不同类别的模型，其中包含印章类别的模型有9个，除上述3个核心模型外，其余模型列表如下：


<details><summary> 👉模型列表详情</summary>

* <b>3类版面检测模型，包含表格、图像、印章</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody></table>

* <b>17类区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody>
</table>
</details>
</details>

<details>
<summary> <b>文档图像方向分类模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
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
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary> <b>印章文本检测模块：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td>
<td>98.40</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td>
<td>96.36</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署</td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">训练模型</a></td>
<td>86.38</td>
<td> 8.45/2.36 </td>
<td> 122.69/122.69 </td>
<td>81 M</td>
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>81.29</td>
<td> 1.46/5.43 </td>
<td> 5.32/91.79 </td>
<td>16 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">训练模型</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>181 M</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>88 M</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>151 M</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>66 M</td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">训练模型</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td> 1.46/5.43 </td>
<td> 5.32/91.79 </td>
<td>81 M</td>
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td> 1.46/5.43 </td>
<td> 5.32/91.79 </td>
<td>16 M</td>
</tr>
</table>

* <b>中文识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">训练模型</a></td>
<td>86.58</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>91 M</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>83.28</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>11 M</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>85.19 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>87 M</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>75.43</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>11 M</td>
<td>PP-OCRv3的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
</table>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量韩文识别模型，支持韩文、数字识别</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量日文识别模型，支持日文、数字识别</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量繁体中文识别模型，支持繁体中文、数字识别</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰卢固文识别模型，支持泰卢固文、数字识别</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量卡纳达文识别模型，支持卡纳达文、数字识别</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰米尔文识别模型，支持泰米尔文、数字识别</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量拉丁文识别模型，支持拉丁文、数字识别</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量阿拉伯字母识别模型，支持阿拉伯字母、数字识别</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量斯拉夫字母识别模型，支持斯拉夫字母、数字识别</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量梵文字母识别模型，支持梵文字母、数字识别</td>
</tr>
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
                  <li>文档图像方向分类模型：自建的内部数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li>文本图像矫正模型：<a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>。</li>
                   <li>版面区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志、合同、书本、试卷和研报等常见的 500 张文档类型图片。</li>
                    <li>3类版面检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 1154 张文档类型图片。</li>
                  <li> 17类区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。</li>
                  <li>文本检测模型：PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。</li>
                   <li>中文识别模型： PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。</li>
                    <li>ch_SVTRv2_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜评估集。</li>
                  <li>ch_RepSVTR_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜评估集。</li>
                  <li>英文识别模型：自建的内部英文数据集。</li>
                   <li>多语言识别模型：自建的内部多语种数据集。</li>
                    <li>文本行方向分类模型：自建的内部数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                   <li>印章文本检测模型：自建的内部数据集，包含500张圆形印章图像。</li>
                </ul>
             </li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>其他环境：Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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
<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

## 2. 快速开始

在本地使用印章文本识别产线前，请确保您已经按照[安装教程](../installation.md)完成了wheel包安装。安装完成后，可以在本地使用命令行体验或 Python 集成。

### 2.1 命令行方式体验

一行命令即可快速体验 seal_recognition 产线效果：

```bash
paddleocr seal_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False

# 通过 --device 指定模型推理时使用 GPU
paddleocr seal_recognition -i ./seal_text_det.png --device gpu
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
<td>待预测数据，支持多种输入类型，必填。
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径。如果设置为<code>None</code>, 推理结果将不会保存到本地。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>版面检测模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面检测模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>印章文本检测模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>印章文本检测模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的批处理大小。如果设置为<code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载文档方向分类模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载文本图像矫正模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否加载版面检测模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面检测置信度阈值，得分大于该阈值才会被输出。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>dict</b>：key是int类别id， value是大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.5</code></li></li></li></ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面检测是否使用后处理NMS。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>检测框的边长缩放倍数。
<ul>
<li><b>float</b>, 大于0的浮点数，如 1.1 , 表示将模型输出的检测框中心不变，宽和高都扩张1.1倍</li>
<li><b>列表</b>, 如 [1.2, 1.5] , 表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为1.0</li>
</ul>
</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面检测中模型输出的检测框的合并处理模式。
<ul>
<li><b>large</b>, 设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>small</b>, 设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>union</b>, 不进行框的过滤处理，内外框都保留</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>large</code></li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>印章文本检测的图像边长限制。
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>736</code>；</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>印章文本检测的图像边长限制类型。
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>min</code>；</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.2</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.5</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号。
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>str</code></td>
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
<td>是否使用 TensorRT 进行推理加速。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>最小子图大小，用于优化模型子图的计算。</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速库。如果设置为<code>None</code>, 将默认启用。
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
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
<br />

运行结果会被打印到终端上，默认配置的 seal_recognition 产线的运行结果如下：

```bash
{'res': {'input_path': './seal_text_det.png', 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 16, 'label': 'seal', 'score': 0.975529670715332, 'coordinate': [6.191284, 0.16680908, 634.39325, 628.85345]}]}, 'seal_res_list': [{'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[320,  38],
       ...,
       [315,  38]]), array([[461, 347],
       ...,
       [456, 346]]), array([[439, 445],
       ...,
       [434, 444]]), array([[158, 468],
       ...,
       [154, 466]])], 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.2, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 0.5}, 'text_type': 'seal', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['天津君和缘商贸有限公司', '发票专用章', '吗繁物', '5263647368706'], 'rec_scores': array([0.99340463, ..., 0.9916274 ]), 'rec_polys': [array([[320,  38],
       ...,
       [315,  38]]), array([[461, 347],
       ...,
       [456, 346]]), array([[439, 445],
       ...,
       [434, 444]]), array([[158, 468],
       ...,
       [154, 466]])], 'rec_boxes': array([], dtype=float64)}]}}
```

可视化结果保存在`save_path`下，其中印章OCR的可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/04.png"/>

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddleocr import SealRecognition

pipeline = SealRecognition(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
)
# ocr = SealRecognition(device="gpu") # 通过 device 指定模型推理时使用 GPU
output = pipeline.predict("./seal_text_det.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `SealRecognition()` 实例化 印章文本识别 产线对象，具体参数说明如下：

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
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>版面检测模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面检测模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>印章文本检测模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>印章文本检测模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的批处理大小。如果设置为<code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载文档方向分类模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载文本图像矫正模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否加载版面检测模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面检测置信度阈值，得分大于该阈值才会被输出。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>dict</b>：key是int类别id， value是大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.5</code></li></li></li></ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面检测是否使用后处理NMS。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>检测框的边长缩放倍数。
<ul>
<li><b>float</b>, 大于0的浮点数，如 1.1 , 表示将模型输出的检测框中心不变，宽和高都扩张1.1倍</li>
<li><b>列表</b>, 如 [1.2, 1.5] , 表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为1.0</li>
</ul>
</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面检测中模型输出的检测框的合并处理模式。
<ul>
<li><b>large</b>, 设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>small</b>, 设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>union</b>, 不进行框的过滤处理，内外框都保留</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>large</code></li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>印章文本检测的图像边长限制。
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>736</code>；</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>印章文本检测的图像边长限制类型。
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>min</code>；</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.2</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.5</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号。
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>str</code></td>
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
<td>是否使用 TensorRT 进行推理加速。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>最小子图大小，用于优化模型子图的计算。</td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速库。如果设置为<code>None</code>, 将默认启用。
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
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

（2）调用 印章文本识别 产线对象的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。

另外，产线还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。

以下是 `predict()` 方法的参数及其说明：

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
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>与实例化时的参数相同。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否在推理时使用文档方向分类模块。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否在推理时使用文本图像矫正模块。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否在推理时使用版面检测模块。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>与实例化时的参数相同。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>与实例化时的参数相同。</td>
<td><code>string</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>与实例化时的参数相同。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>与实例化时的参数相同。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
</tr></table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_layout_detection`: `(bool)` 控制是否启用版面检测子模块
    - `layout_det_res`: `(Dict[str, Union[List[numpy.ndarray], List[float]]])` 版面检测子模块的输出结果。仅当`use_layout_detection=True`时存在
        - `input_path`: `(Union[str, None])` 版面检测区域模块接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
        - `boxes`: `(List[Dict])` 版面印章区域的检测框列表，每个列表中的元素，包含以下字段
            - `cls_id`: `(int)` 检测框的印章类别id
            - `score`: `(float)` 检测框的置信度
            - `coordinate`: `(List[float])` 检测框的四个顶点坐标，顺序为x1,y1,x2,y2表示左上角的x坐标，左上角的y坐标，右下角x坐标，右下角的y坐标
    - `seal_res_list`: `List[Dict]` 印章文本识别的结果列表，每个元素包含以下字段
        - `input_path`: `(Union[str, None])` 印章文本识别产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
        - `model_settings`: `(Dict[str, bool])` 印章文本识别产线的模型配置参数
          - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
          - `use_textline_orientation`: `(bool)` 控制是否启用文本行方向分类子模块
    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` 文档预处理子产线的输出结果。仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(Union[str, None])` 图像预处理子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `model_settings`: `(Dict)` 预处理子产线的模型配置参数
            - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档方向分类
            - `use_doc_unwarping`: `(bool)` 控制是否启用文本图像矫正
        - `angle`: `(int)` 文档方向分类的预测结果。启用时取值为[0,1,2,3]，分别对应[0°,90°,180°,270°]；未启用时为-1

    - `dt_polys`: `(List[numpy.ndarray])` 印章文本检测的多边形框列表。每个检测框由多个顶点坐标构成的numpy数组表示，数组shape为(n, 2)

    - `dt_scores`: `(List[float])` 文本检测框的置信度列表

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 文本检测模块的配置参数
        - `limit_side_len`: `(int)` 图像预处理时的边长限制值
        - `limit_type`: `(str)` 边长限制的处理方式
        - `thresh`: `(float)` 文本像素分类的置信度阈值
        - `box_thresh`: `(float)` 文本检测框的置信度阈值
        - `unclip_ratio`: `(float)` 文本检测框的膨胀系数
        - `text_type`: `(str)` 印章文本检测的类型，当前固定为"seal"


    - `text_rec_score_thresh`: `(float)` 文本识别结果的过滤阈值

    - `rec_texts`: `(List[str])` 文本识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本

    - `rec_scores`: `(List[float])` 文本识别的置信度列表，已按`text_rec_score_thresh`过滤

    - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的文本检测框列表，格式同`dt_polys`

    - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，印章识别产线为空数组

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_seal_res_region1.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `layout_det_res` 、  `seal_res_region1`和 `preprocessed_img`，对应的值是三个 `Image.Image` 对象：一个用于显示版面检测可视化，一个用于显示印章文本识别结果的可视化图像，另一个用于展示图像预处理的可视化图像。如果没有使用图像预处理子模块，则字典中不包含preprocessed_img，如果没有使用版面区域检测模块，则字典中不包含layout_det_res。

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
<li><b><code>infer</code></b></li>
</ul>
<p>获取印章文本识别结果。</p>
<p><code>POST /seal-recognition</code></p>
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
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_doc_orientation_classify</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_doc_unwarping</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_layout_detection</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_nms</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_merge_bboxes_mode</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>seal_det_limit_side_len</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>seal_det_limit_type</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>seal_det_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>seal_det_box_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>seal_det_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>seal_rec_score_thresh</code> 参数相关说明。</td>
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
<td><code>sealRecResults</code></td>
<td><code>object</code></td>
<td>印章文本识别结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>sealRecResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td>产线对象的 <code>predict</code> 方法生成结果的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 和 <code>page_index</code> 字段。</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>参见产线预测结果的 <code>img</code> 属性说明。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table></details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/seal-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["sealRecResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>
</details>
<br/>

## 4. 二次开发
如果印章文本识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升印章文本识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于印章文本识别产线包含若干模块，模型产线的效果如果不及预期，可能来自于其中任何一个模块。您可以对识别效果差的图片进行分析，进而确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。

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
<td>印章位置检测不准或未检出</td>
<td>版面检测模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html#_5">链接</a></td>
</tr>
<tr>
<td>印章文本存在漏检</td>
<td>印章文本检测模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/seal_text_detection.html#_5">链接</a></td>
</tr>
<tr>
<td>文本内容不准</td>
<td>文本识别模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/text_recognition.html#_5">链接</a></td>
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

当您使用私有数据集完成微调训练后，可获得本地模型权重文件，然后可以通过参数指定本地模型保存路径的方式，或者通过自定义产线配置文件的方式，使用微调后的模型权重。

#### 4.2.1 通过参数指定本地模型路径

在初始化产线对象时，通过参数指定本地模型路径。以文本检测模型微调后的权重的使用方法为例，示例如下：

命令行方式:

```bash
# 通过 --doc_orientation_classify_model_dir 指定本地模型路径
paddleocr seal_recognition -i ./seal_text_det.png --doc_orientation_classify_model_dir your_orientation_classify_model_path

# 默认使用 PP-LCNet_x1_0_doc_ori 模型作为默认文本检测模型，如果微调的不是该模型，通过 --text_detection_model_name 修改模型名称
paddleocr seal_recognition -i ./seal_text_det.png --doc_orientation_classify_model_name PP-LCNet_x1_0_doc_ori --doc_orientation_classify_model_dir your_orientation_classify_model_path
```

脚本方式：

```python

from paddleocr import SealRecognition

# 通过 doc_orientation_classify_model_dir 指定本地模型路径
pipeline = SealRecognition(doc_orientation_classify_model_dir ="./your_orientation_classify_model_path")

# 默认使用 PP-LCNet_x1_0_doc_ori 模型作为默认文本检测模型，如果微调的不是该模型，通过 doc_orientation_classify_model_name 修改模型名称
# pipeline = SealRecognition(doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori", doc_orientation_classify_model_dir="./your_orientation_classify_model_path")

```


#### 4.2.2 通过配置文件指定本地模型路径


1. 获取产线配置文件

可调用 PaddleOCR 中 通用OCR 产线对象的 `export_paddlex_config_to_yaml` 方法，将当前产线配置导出为 YAML 文件：

```Python
from paddleocr import SealRecognition

pipeline = SealRecognition()
pipeline.export_paddlex_config_to_yaml("SealRecognition.yaml")
```

2. 修改配置文件

在得到默认的产线配置文件后，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可。例如

```yaml
......
SubPipelines:
  DocPreprocessor:
    SubModules:
      DocOrientationClassify:
        model_dir: null  # 替换为微调后的文档方向分类模型权重路径
        model_name: PP-LCNet_x1_0_doc_ori # 如果微调的模型名称与默认模型名称不同，请一并修改此处
        module_name: doc_text_orientation
      DocUnwarping:
        model_dir: null  # 替换为微调后的文档矫正模型权重路径
        model_name: UVDoc # 如果微调的模型名称与默认模型名称不同，请一并修改此处
        module_name: image_unwarping
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: true
    use_doc_unwarping: true
......
```

在产线配置文件中，不仅包含 SealRecognition CLI 和 Python API 支持的参数，还可进行更多高级配置，具体信息可在 [PaddleX模型产线使用概览](https://paddlepaddle.github.io/PaddleX/3.0/pipeline_usage/pipeline_develop_guide.html) 中找到对应的产线使用教程，参考其中的详细说明，根据需求调整各项配置。

3. 在 CLI 中加载产线配置文件

在修改完成配置文件后，通过命令行的 --paddlex_config 参数指定修改后的产线配置文件的路径，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```bash
paddleocr seal_recognition --paddlex_config SealRecognition.yaml ...
```

4. 在 Python API 中加载产线配置文件

初始化产线对象时，可通过 paddlex_config 参数传入 PaddleX 产线配置文件路径或配置字典，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```python
from paddleocr import SealRecognition

pipeline = SealRecognition(paddlex_config="SealRecognition.yaml")
```
