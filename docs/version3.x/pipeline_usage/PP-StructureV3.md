---
comments: true
---

# PP-StructureV3 产线使用教程

## 1. PP-StructureV3 产线介绍

版面解析是一种从文档图像中提取结构化信息的技术，主要用于将复杂的文档版面转换为机器可读的数据格式。这项技术在文档管理、信息提取和数据数字化等领域具有广泛的应用。版面解析通过结合光学字符识别（OCR）、图像处理和机器学习算法，能够识别和提取文档中的文本块、标题、段落、图片、表格以及其他版面元素。此过程通常包括版面分析、元素分析和数据格式化三个主要步骤，最终生成结构化的文档数据，提升数据处理的效率和准确性。<b>PP-StructureV3 产线在通用版面解析v1产线的基础上，强化了版面区域检测、表格识别、公式识别的能力，增加了图表理解能力和多栏阅读顺序的恢复能力、结果转换 Markdown 文件的能力，在多种文档数据中，表现优异，可以处理较复杂的文档数据。</b>本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<b>PP-StructureV3 产线中包含以下7个模块或子产线。每个模块或子产线均可独立进行训练和推理，并包含多个模型。有关详细信息，请点击相应链接以查看文档。</b>

- [版面区域检测模块](../module_usage/layout_detection.md)
- [通用OCR子产线](./OCR.md)
- [文档图像预处理子产线](./doc_preprocessor.md) （可选）
- [表格识别子产线](./table_recognition_v2.md) （可选）
- [印章文本识别子产线](./seal_recognition.md) （可选）
- [公式识别子产线](./formula_recognition.md) （可选）
- [图表解析模块](../module_usage/chart_parsing.md) （可选）

在本产线中，您可以根据下方的基准测试数据选择使用的模型。

> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

<details>
<summary><b>文档图像方向分类模块：</b></summary>
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
<summary><b>文本图像矫正模块：</b></summary>
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
<summary><b>版面区域检测模块：</b></summary>

* <b>版面检测模型，包含20个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、表格、图和表标题（图标题、表格标题和图表标题）、印章、图表、侧栏文本和参考文献内容</b>
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
<td>PP-DocLayout_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">训练模型</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位模型</td>
</tr>
<tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1300 张文档类型图片。</b>

* <b>文档图像版面子模块检测，包含1个 版面区域 类别，能检测多栏的报纸、杂志的每个子文章的文本区域：</b>
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
<td>PP-DocBlockLayout</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams">训练模型</a></td>
<td>95.9</td>
<td>34.60 / 28.54</td>
<td>506.43 / 256.83</td>
<td>123.92</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的文档图像版面子模块检测模型</td>
</tr>
<tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的版面子区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1000 张文档类型图片。</b>

* <b>版面检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>
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
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、报纸、研报和试卷等 500 张文档类型图片。</b>

> ❗ 以上列出的是版面检测模块重点支持的<b>5个核心模型</b>，该模块总共支持<b>13个全量模型</b>，包含多个预定义了不同类别的模型，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

* <b>表格版面检测模型</b>
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
<td>PicoDet_layout_1x_table</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>97.5</td>
<td>9.57 / 6.63</td>
<td>27.66 / 16.75</td>
<td>7.4</td>
<td>基于PicoDet-1x在自建数据集训练的高效率版面区域定位模型，可定位表格这1类区域</td>
</tr>
</tbody></table>

* <b>3类版面检测模型，包含表格、图像、印章</b>
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
<td>PicoDet-S_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>88.2</td>
<td>8.43 / 3.44</td>
<td>17.60 / 6.51</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody></table>

* <b>5类英文文档区域检测模型，包含文字、标题、表格、图片以及列表</b>
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
<td>PicoDet_layout_1x</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>97.8</td>
<td>9.62 / 6.75</td>
<td>26.96 / 12.77</td>
<td>7.4</td>
<td>基于PicoDet-1x在PubLayNet数据集训练的高效率英文文档版面区域定位模型</td>
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
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>8.80 / 3.62</td>
<td>17.51 / 6.35</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody>
</table>
</details>
</details>

<details>
<summary><b>表格结构识别模块：</b></summary>
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
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">85.92 / 85.92</td>
<td rowspan="2">- / 501.66</td>
<td rowspan="2">351</td>
<td rowspan="2">SLANeXt 系列是百度飞桨视觉团队自研的新一代表格结构识别模型。相较于 SLANet 和 SLANet_plus，SLANeXt 专注于对表格结构进行识别，并且对有线表格(wired)和无线表格(wireless)的识别分别训练了专用的权重，对各类型表格的识别能力都得到了明显提高，特别是对有线表格的识别能力得到了大幅提升。</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>表格分类模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">训练模型</a></td>
<td>94.2</td>
<td>2.62 / 0.60</td>
<td>3.17 / 1.14</td>
<td>6.6</td>
</tr>
</table>

<p><b>表格单元格检测模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">33.47 / 27.02</td>
<td rowspan="2">402.55 / 256.56</td>
<td rowspan="2">124</td>
<td rowspan="2">RT-DETR 是第一个实时的端到端目标检测模型。百度飞桨视觉团队基于 RT-DETR-L 作为基础模型，在自建表格单元格检测数据集上完成预训练，实现了对有线表格、无线表格均有较好性能的表格单元格检测。
</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
</details>

<details>
<summary><b>文本检测模块：</b></summary>
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
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
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
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
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
<summary><b>文本行方向分类模块（可选）：</b></summary>
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
<summary><b>公式识别模块：</b></summary>

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
<summary><b>印章文本检测模块：</b></summary>
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

<details>
<summary><b>图表解析模块：</b></summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>模型参数规模（B）</th>
<th>模型存储大小（GB）</th>
<th>模型分数 </th>
<th>介绍</th>
</tr>
<tr>
<td>PP-Chart2Table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.tar">推理模型</a></td>
<td>0.58</td>
<td>1.4</td>
<th>75.98</th>
<td>PP-Chart2Table是飞桨团队自研的一款专注于图表解析的多模态模型，在中英文图表解析任务中展现出卓越性能。团队采用精心设计的数据生成策略，构建了近70万条高质量的图表解析多模态数据集，全面覆盖饼图、柱状图、堆叠面积图等常见图表类型及各类应用场景。同时设计了二阶段训练方法，结合大模型蒸馏实现对海量无标注OOD数据的充分利用。在内部业务的中英文场景测试中，PP-Chart2Table不仅达到同参数量级模型中的SOTA水平，更在关键场景中实现了与7B参数量级VLM模型相当的精度。</td>
</tr>
</table>
</details>

<details>
<summary><b>测试环境说明：</b></summary>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
            <li><strong>测试数据集：
             </strong>
                <ul>
                  <li>文档图像方向分类模型：自建的内部数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li> 文本图像矫正模型：<a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>。</li>
                  <li>版面区域检测模型：PaddleOCR 自建的版面区域分析数据集，包含中英文论文、杂志和研报等常见的 1w 张文档类型图片。</li>
                  <li>表格结构识别模型：PaddleX 内部自建英文表格识别数据集。 </li>
                  <li>文本检测模型：PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。</li>
                  <li> 中文识别模型： PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。</li>
                  <li>ch_SVTRv2_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜评估集。</li>
                  <li> ch_RepSVTR_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜评估集。</li>
                  <li>英文识别模型：自建的内部英文数据集。</li>
                  <li> 多语言识别模型：自建的内部多语种数据集。</li>
                  <li>文本行方向分类模型：自建的内部数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li> 印章文本检测模型：自建的内部数据集，包含500张圆形印章图像。</li>
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
<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>

## 2. 快速开始

在本地使用 PP-StructureV3 产线前，请确保您已经按照[安装教程](../installation.md)完成了wheel包安装。安装完成后，可以在本地使用命令行体验或 Python 集成。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `doc-parser`。

**请注意，如果在执行过程中遇到程序失去响应、程序异常退出、内存资源耗尽、推理速度极慢等问题，请尝试参考文档调整配置，例如关闭不需要使用的功能或使用更轻量的模型。**

### 2.1 命令行方式体验

一行命令即可快速体验 PP-StructureV3 产线效果：

```bash
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png

# 通过 --use_doc_orientation_classify 指定是否使用文档方向分类模型
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_doc_orientation_classify True

# 通过 --use_doc_unwarping 指定是否使用文本图像矫正模块
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_doc_unwarping True

# 通过 --use_textline_orientation 指定是否使用文本行方向分类模型
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_textline_orientation False

# 通过 --device 指定模型推理时使用 GPU
paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --device gpu
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
<td>待预测数据，必填。
如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)。
</td>
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
<td><code>layout_detection_model_name</code></td>
<td>版面区域检测的模型名称。如果不设置，将会使用产线默认模型。</td>
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
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。<code>0-1</code> 之间的任意浮点数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>0.5</code>。
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
<td>版面区域检测模型检测框的扩张系数。
任意大于 <code>0</code>  浮点数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>1.0</code>。
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
<td><code>chart_recognition_model_name</code></td>
<td>图表解析的模型名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>图表解析模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>图表解析模型的batch size。如果不设置，将默认设置batch size为<code>1</code>。</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>文档图像版面子模块检测的模型名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>文档图像版面子模块检测模型的目录路径。如果不设置，将会下载官方模型。</td>
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
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的图像边长限制。
大于 <code>0</code> 的任意整数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>960</code>。
</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的图像边长限制类型。支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code>。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>max</code>。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>0.3</code>。
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
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>2.0</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>文本行方向模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>文本行方向模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>文本行方向模型的batch size。如果不设置，将默认设置batch size为<code>1</code>。</td>
<td><code>int</code></td>
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
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
大于 <code>0</code> 的任意浮点数。如果不设置，将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>表格分类模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>表格分类模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>有线表格结构识别模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>有线表格结构识别模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>无线表格结构识别模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>无线表格结构识别模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>有线表格单元格检测模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>有线表格单元格检测模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>无线表格单元格检测模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>无线表格单元格检测模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_name</code></td>
<td>表格方向分类模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_dir</code></td>
<td>表格方向分类模型的目录路径。如果不设置，将会下载官方模型。</td>
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
<td><code>seal_det_limit_side_len</code></td>
<td>印章文本检测的图像边长限制。
大于 <code>0</code> 的任意整数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>736</code>。
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
<td>检测像素阈值。输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>0.2</code>。
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>0.6</code>。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>0.5</code>。
</td>
<td><code>float</code></td>
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
<td><code>seal_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
大于 <code>0</code> 的任意浮点数
。如果不设置，将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>公式识别模型的名称。如果不设置，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>公式识别模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>公式识别模型的batch size。如果不设置，将默认设置batch size为<code>1</code>。</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向分类模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否加载并使用印章文本识别子产线。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
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
<td><code>use_formula_recognition</code></td>
<td>是否加载并使用公式识别子产线。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否加载并使用图表解析模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>是否加载并使用文档区域检测模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
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

运行结果会被打印到终端上，默认配置的 PP-StructureV3 产线的运行结果如下：

<details><summary> 👉点击展开</summary>
<pre>
<code>
{'res': {'input_path': 'pp_structure_v3_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9853514432907104, 'coordinate': [770.9531, 776.6814, 1122.6057, 1058.7322]}, {'cls_id': 1, 'label': 'image', 'score': 0.9848673939704895, 'coordinate': [775.7434, 202.27979, 1502.8113, 686.02136]}, {'cls_id': 2, 'label': 'text', 'score': 0.983731746673584, 'coordinate': [1152.3197, 1113.3275, 1503.3029, 1346.586]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832221865653992, 'coordinate': [1152.5602, 801.431, 1503.8436, 986.3563]}, {'cls_id': 2, 'label': 'text', 'score': 0.9829439520835876, 'coordinate': [9.549545, 849.5713, 359.1173, 1058.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9811657667160034, 'coordinate': [389.58298, 1137.2659, 740.66235, 1346.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9775941371917725, 'coordinate': [9.1302185, 201.85, 359.0409, 339.05692]}, {'cls_id': 2, 'label': 'text', 'score': 0.9750366806983948, 'coordinate': [389.71454, 752.96924, 740.544, 889.92456]}, {'cls_id': 2, 'label': 'text', 'score': 0.9738152027130127, 'coordinate': [389.94565, 298.55988, 740.5585, 435.5124]}, {'cls_id': 2, 'label': 'text', 'score': 0.9737328290939331, 'coordinate': [771.50256, 1065.4697, 1122.2582, 1178.7324]}, {'cls_id': 2, 'label': 'text', 'score': 0.9728517532348633, 'coordinate': [1152.5154, 993.3312, 1503.2349, 1106.327]}, {'cls_id': 2, 'label': 'text', 'score': 0.9725610017776489, 'coordinate': [9.372787, 1185.823, 359.31738, 1298.7227]}, {'cls_id': 2, 'label': 'text', 'score': 0.9724331498146057, 'coordinate': [389.62848, 610.7389, 740.83234, 746.2377]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720287322998047, 'coordinate': [389.29898, 897.0936, 741.41516, 1034.6616]}, {'cls_id': 2, 'label': 'text', 'score': 0.9713053703308105, 'coordinate': [10.323685, 1065.4663, 359.6786, 1178.8872]}, {'cls_id': 2, 'label': 'text', 'score': 0.9689728021621704, 'coordinate': [9.336395, 537.6609, 359.2901, 652.1881]}, {'cls_id': 2, 'label': 'text', 'score': 0.9684857130050659, 'coordinate': [10.7608185, 345.95068, 358.93616, 434.64087]}, {'cls_id': 2, 'label': 'text', 'score': 0.9681928753852844, 'coordinate': [9.674866, 658.89075, 359.56528, 770.4319]}, {'cls_id': 2, 'label': 'text', 'score': 0.9634978175163269, 'coordinate': [770.9464, 1281.1785, 1122.6522, 1346.7156]}, {'cls_id': 2, 'label': 'text', 'score': 0.96304851770401, 'coordinate': [390.0113, 201.28055, 740.1684, 291.53073]}, {'cls_id': 2, 'label': 'text', 'score': 0.962053120136261, 'coordinate': [391.21393, 1040.952, 740.5046, 1130.32]}, {'cls_id': 2, 'label': 'text', 'score': 0.9565253853797913, 'coordinate': [10.113251, 777.1482, 359.439, 842.437]}, {'cls_id': 2, 'label': 'text', 'score': 0.9497362375259399, 'coordinate': [390.31357, 537.86285, 740.47595, 603.9285]}, {'cls_id': 2, 'label': 'text', 'score': 0.9371236562728882, 'coordinate': [10.2034, 1305.9753, 359.5958, 1346.7295]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9338151216506958, 'coordinate': [791.6062, 1200.8479, 1103.3257, 1259.9324]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9326773285865784, 'coordinate': [408.0737, 457.37024, 718.9509, 516.63464]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9274250864982605, 'coordinate': [29.448685, 456.6762, 340.99194, 515.6999]}, {'cls_id': 2, 'label': 'text', 'score': 0.8742568492889404, 'coordinate': [1154.7095, 777.3624, 1330.3086, 794.5853]}, {'cls_id': 2, 'label': 'text', 'score': 0.8442489504814148, 'coordinate': [586.49316, 160.15454, 927.468, 179.64203]}, {'cls_id': 11, 'label': 'doc_title', 'score': 0.8332607746124268, 'coordinate': [133.80017, 37.41908, 1380.8601, 124.1429]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6770150661468506, 'coordinate': [812.1718, 705.1199, 1484.6973, 747.1692]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[133,  35],
        ...,
        [133, 131]],

       ...,

       [[ 13, 754],
        ...,
        [ 13, 777]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者', '沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等,曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院")举办"喜迎新年"中国', '黄鸣飞表示,随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设,预计今年上半年竣工,建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '多年来,厄立特里亚广大赴华留学生和', '培训人员积极投身国家建设,成为助力该国', '发展的人才和厄中友好的见证者和推动者。', '在厄立特里亚全国妇女联盟工作的约翰', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '中华女子学院攻读硕士学位,研究方向是女', '性领导力与社会发展。其间，她实地走访中国', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '手资料。', '中国驻厄立特里亚大使馆供图', '“这是中文歌曲初级班，共有32人。学', '“不管远近都是客人，请不用客气;相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '好了在一起,我们欢迎你"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。"尤斯拉告诉记者。', '年联谊活动上,四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器,罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和""禅”“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明,很早以前我们就通过海上丝绸之路进行', '习中文,在2017年第十届"汉语桥"世界中学生', '是其中一名演唱者,她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名,并和', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛,获得', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲,每周末两个课时。中国', '还是在中国留学的厄立特里亚学子,两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深,我希望我的学生们能够通过中', '民携手努力,必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐,你想去中国吗?"“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往,搭建友谊桥梁。"', '能歌善舞的姐妹,姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验,有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说：“这些年来,怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '“共同向世界展示非', '印象深刻。“中国博物馆不仅有许多保存完好', '和中国文化的热爱,我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说：“学', '的文物,还充分运用先进科技手段进行展示，', '励,一起学习。我们的中文一天比一天好,还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰,希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明,始终相', '去。学好中文,我们的未来不是梦!"', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发,沿着蜿蜒曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍,这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作,共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时,记者来到位于厄立特里亚港口城市马萨', '烂文明。”', '谈起在中国求学的经历,约翰娜记忆犹', '新：“中国的发展在当今世界是独一无二的。', '沿着中国特色社会主义道路坚定前行，中国', '创造了发展奇迹,这一切都离不开中国共产党', '的领导。中国的发展经验值得许多国家学习', '借鉴，”', '正在西南大学学习的厄立特里亚博士生', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '年前，在北京师范大学获得硕士学位后，穆卢', '盖塔在社交媒体上写下这样一段话：“这是我', '人生的重要一步，自此我拥有了一双坚固的', '鞋子.赋予我穿越荆棘的力量。”', '“鲜花曾告诉我你怎样走过，大地知道你', '心中的每一个角落"厄立特里亚阿斯马拉', '大学综合楼二层，一阵优美的歌声在走廊里回', '响。循着熟悉的旋律轻轻推开一间教室的门，', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '这是厄特孔院阿斯马拉大学教学点的一', '节中文歌曲课。为了让学生们更好地理解歌', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '字翻译和解释歌词。随着伴奏声响起，学生们', '边唱边随着节拍摇动身体，现场气氛热烈。'], 'rec_scores': array([0.99972075, ..., 0.96241361]), 'rec_polys': array([[[133,  35],
        ...,
        [133, 131]],

       ...,

       [[ 13, 754],
        ...,
        [ 13, 777]]], dtype=int16), 'rec_boxes': array([[133, ..., 131],
       ...,
       [ 13, ..., 777]], dtype=int16)}}}
</code></pre></details>

运行结果参数说明可以参考[2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

<b>注：</b>由于产线的默认模型较大，推理速度可能较慢，您可以参考第一节的模型列表，替换推理速度更快的模型。

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
# pipeline = PPStructureV3(lang="en") # 将 lang 参数设置为使用英文文本识别模型。对于其他支持的语言，请参阅第5节：附录部分。默认配置为中英文模型。
# pipeline = PPStructureV3(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# pipeline = PPStructureV3(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# pipeline = PPStructureV3(use_textline_orientation=True) # 通过 use_textline_orientation 指定是否使用文本行方向分类模型
# pipeline = PPStructureV3(device="gpu") # 通过 device 指定模型推理时使用 GPU
output = pipeline.predict("./pp_structure_v3_demo.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
```

如果是 PDF 文件，会将 PDF 的每一页单独处理，每一页的 Markdown 文件也会对应单独的结果。如果希望整个 PDF 文件转换为 Markdown 文件，建议使用以下的方式运行：

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

**注：**

- PP-StructureV3 产线使用的默认文本识别模型为 **中英文识别模型**，对于纯英文的识别能力有限，对于全英文场景，您可以设置`text_recognition_model_name`参数为 `en_PP-OCRv4_mobile_rec` 等英文识别模型以取得更好的识别效果。对应其他语言场景，也可以参考前文的模型列表，选择对应的语言识别模型进行替换。

- 在示例代码中，`use_doc_orientation_classify`、`use_doc_unwarping`、`use_textline_orientation` 参数默认均设置为 `False`，分别表示关闭文档方向分类、文本图像矫正、文本行方向分类功能，如果需要使用这些功能，可以手动设置为 `True`。

在上述 Python 脚本中，执行了如下几个步骤：

<details><summary>（1）实例化产线对象，具体参数说明如下：</summary>

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
<td>版面区域检测的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
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
<li><b>dict</b>，dict的key为<b>int</b>类型，代表<code>cls_id</code>, value为<b>tuple</b>类型，如<code>{0: (1.1, 2.0)}</code>，表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍；</li>
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
<li><b>dict</b>： dict的key为<b>int</b>类型，代表<code>cls_id</code>，value为<b>str</b>类型，如<code>{0: "large", 2: "small"}</code>，表示对第0类别检测框使用large模式，对第2类别检测框使用small模式；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为 <code>large</code>。</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td>图表解析的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>图表解析模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>图表解析模型的batch size。如果设置为<code>None</code>，将默认设置batch size为<code>1</code>。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>文档图像版面子模块检测的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>文档图像版面子模块检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
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
<td>文本检测的图像边长限制类型。
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
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.3</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.6</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>2.0</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>文本行方向模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>文本行方向模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>文本行方向模型的batch size。如果设置为<code>None</code>，将默认设置batch size为<code>1</code>。</td>
<td><code>int|None</code></td>
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
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.0</code>，即不设阈值。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>表格分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>表格分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>有线表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>有线表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>无线表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>无线表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>有线表格单元格检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>有线表格单元格检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>无线表格单元格检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>无线表格单元格检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_name</code></td>
<td>表格方向分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_orientation_classify_model_dir</code></td>
<td>表格方向分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
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
<td><code>seal_det_limit_side_len</code></td>
<td>印章文本检测的图像边长限制。
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
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
<li><b>float</b>：大于 <code>0</code> 的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.2</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.6</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.5</code>。</li></li></ul>
</td>
<td><code>float|None</code></td>
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
<td><code>seal_rec_score_thresh</code></td>
<td>印章文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于<code>0</code>的任意浮点数；
<li><b>None</b>：如果设置为<code>None</code>，将默认使用产线初始化的该参数值 <code>0.0</code>，即不设阈值。</li></li></ul>
</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>公式识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>公式识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>公式识别模型的batch size。如果设置为<code>None</code>，将默认设置batch size为<code>1</code>。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向分类模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否加载并使用印章文本识别子产线。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
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
<td><code>use_formula_recognition</code></td>
<td>是否加载并使用公式识别子产线。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否加载并使用图表解析模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>是否加载并使用文档区域检测模块。如果设置为<code>None</code>，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
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
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>（2）调用 PP-StructureV3 产线对象的 <code>predict()</code> 方法进行推理预测，该方法会返回一个结果列表。另外，产线还提供了 <code>predict_iter()</code> 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 <code>predict_iter()</code> 返回的是一个 <code>generator</code>，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。以下是 <code>predict()</code> 方法的参数及其说明：</summary>

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
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>list</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]。</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否在推理时使用文档方向分类模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否在推理时使用文本图像矫正模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否在推理时使用文本行方向分类模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否在推理时使用印章文本识别子产线。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否在推理时使用表格识别子产线。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>是否在推理时使用公式识别子产线。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否在推理时使用图表解析模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>是否在推理时使用文档区域检测模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
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
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>是否启用有线表单元格检测结果直转HTML，启用则直接基于有线表单元格检测结果的几何关系构建HTML。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_wireless_table_cells_trans_to_html</code></td>
<td>是否启用无线表单元格检测结果直转HTML，启用则直接基于无线表单元格检测结果的几何关系构建HTML。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td>是否启用表格使用表格方向分类，启用时当图像中的表格存在90/180/270度旋转时，能够将方向校正并正确完成表格识别。</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td>是否启用单元格切分OCR，启用时会基于单元格预测结果对OCR检测结果进行切分和重识别，避免出现文字缺失情况。</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td>是否启用有线表端到端表格识别模式，启用则不使用单元格检测模型，只使用表格结构识别模型。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td>是否启用无线表端到端表格识别模式，启用则不使用单元格检测模型，只使用表格结构识别模型。</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</table>
</details>

<details><summary>（3）对预测结果进行处理：每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为<code>json</code>文件的操作:</summary>

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
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
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
<td>将中间各个模块的可视化图像保存在png格式的图像</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>将图像或者PDF文件中的每一页分别保存为markdown格式的文件。</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>将文件中的表格保存为html格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>将文件中的表格保存为xlsx格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>concatenate_markdown_pages()</code></td>
<td>将多页Markdown内容拼接为单一文档</td>
<td><code>markdown_list</code></td>
<td><code>list</code></td>
<td>包含每一页Markdown数据的列表。</td>
<td>返回处理后的Markdown文本和图像列表。</td>
</tr>
</table>

<ul>
   <li>调用<code>print()</code> 方法会将结果打印到终端，打印到终端的内容解释如下：</li>
      <ul>
        <li><code>input_path</code>: <code>(str)</code>  待预测图像或者PDF的输入路径</li>
        <li><code>page_index</code>: <code>(Union[int, None])</code> 如果输入是PDF文件，则表示当前是PDF的第几页，否则为  <code>None</code></li>
        <li><code>model_settings</code>: <code>(Dict[str, bool])</code>  配置产线所需的模型参数</li>
            <ul>
                <li><code>use_doc_preprocessor</code>: <code>(bool)</code>  控制是否启用文档预处理子产线</li>
                <li><code>use_seal_recognition</code>: <code>(bool)</code> 控制是否启用印章文本识别子产线</li>
                <li><code>use_table_recognition</code>: <code>(bool)</code> 控制是否启用表格识别子产线</li>
                <li><code>use_formula_recognition</code>: <code>(bool)</code> 控制是否启用公式识别子产线</li>
            </ul>
        </li>
        <li><code>doc_preprocessor_res</code>: <code>(Dict[str, Union[List[float], str]])</code> 文档预处理结果dict，仅当<code>use_doc_preprocessor=True</code>时存在</li>
            <ul>
                <li><code>input_path</code>: <code>(str)</code>  文档预处理子产线接受的图像路径，当输入为 <code>numpy.ndarray</code> 时，保存为<code> None</code> ，此处为<code> None</code> </li>
                <li><code>page_index</code>: <code>None</code> ，此处的输入为<code>numpy.ndarray</code> ，所以值为<code> None</code> </li>
                <li><code>model_settings</code>: <code>(Dict[str, bool])</code>  文档预处理子产线的模型配置参数</li>
                    <ul>
                        <li><code>use_doc_orientation_classify</code>: <code>(bool)</code>  控制是否启用文档图像方向分类子模块</li>
                        <li><code>use_doc_unwarping</code>: <code>(bool)</code>  控制是否启用文本图像扭曲矫正子模块</li>
                    </ul>
                <li><code>angle</code>: <code>(int)</code>  文档图像方向分类子模块的预测结果，启用时返回实际角度值</li>
            </ul>
        <li><code>parsing_res_list</code>: <code>(List[Dict])</code> 解析结果的列表，每个元素为一个字典，列表顺序为解析后的阅读顺序。</li>
            <ul>
                <li><code>block_bbox</code>: <code>(np.ndarray)</code> 版面区域的边界框。</li>
                <li><code>block_label</code>: <code>(str)</code> 版面区域的标签，例如<code>text</code>, <code>table</code>等。</li>
                <li><code>block_content</code>: <code>(str)</code> 内容为版面区域内的内容。</li>
                <li><code>block_id</code>: <code>(int)</code> 版面区域的索引，用于显示版面排序结果。</li>
                <li><code>block_order</code> <code>(int)</code> 版面区域的顺序，用于显示版面阅读顺序,对于非排序部分，默认值为 <code> None</code> 。</li>
            </ul>
            <li><code>overall_ocr_res</code>: <code>(Dict[str, Union[List[str], List[float], numpy.ndarray]])</code> 全局 OCR 结果的dict</li>
                <ul>
                    <li><code>input_path</code>: <code>(Union[str, None])</code> 图像OCR子产线接受的图像路径，当输入为<code>numpy.ndarray</code>时，保存为<code> None</code> </li>
                    <li><code>page_index</code>: <code> None</code> ，此处的输入为<code>numpy.ndarray</code>，所以值为<code> None</code> </li>
                    <li><code>model_settings</code>: <code>(Dict)</code> OCR子产线的模型配置参数</li>
                    <li><code>dt_polys</code>: <code>(List[numpy.ndarray])</code> 文本检测的多边形框列表。每个检测框由4个顶点坐标构成的numpy数组表示，数组shape为(4, 2)，数据类型为int16</li>
                    <li><code>dt_scores</code>: <code>(List[float])</code> 文本检测框的置信度列表</li>
                    <li><code>text_det_params</code>: <code>(Dict[str, Dict[str, int, float]])</code> 文本检测模块的配置参数</li>
                        <ul>
                            <li><code>limit_side_len</code>: <code>(int)</code> 图像预处理时的边长限制值</li>
                            <li><code>limit_type</code>: <code>(str)</code> 边长限制的处理方式</li>
                            <li><code>thresh</code>: <code>(float)</code> 文本像素分类的置信度阈值</li>
                            <li><code>box_thresh</code>: <code>(float)</code> 文本检测框的置信度阈值</li>
                            <li><code>unclip_ratio</code>: <code>(float)</code> 文本检测框的膨胀系数</li>
                            <li><code>text_type</code>: <code>(str)</code> 文本检测的类型，当前固定为"general"</li>
                        </ul>
                    <li><code>text_type</code>: <code>(str)</code> 文本检测的类型，当前固定为"general"</li>
                    <li><code>textline_orientation_angles</code>: <code>(List[int])</code> 文本行方向分类的预测结果。启用时返回实际角度值（如[0,0,1]</li>
                    <li><code>text_rec_score_thresh</code>: <code>(float)</code> 文本识别结果的过滤阈值</li>
                    <li><code>rec_texts</code>: <code>(List[str])</code> 文本识别结果列表，仅包含置信度超过<code>text_rec_score_thresh</code>的文本</li>
                    <li><code>rec_scores</code>: <code>(List[float])</code> 文本识别的置信度列表，已按<code>text_rec_score_thresh</code>过滤</li>
                    <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> 经过置信度过滤的文本检测框列表，格式同<code>dt_polys</code></li>
                </ul>
            <li><code>formula_res_list</code>: <code>(List[Dict[str, Union[numpy.ndarray, List[float], str]]])</code> 公式识别结果列表，每个元素为一个dict</li>
                <ul>
                    <li><code>rec_formula</code>: <code>(str)</code> 公式识别结果</li>
                    <li><code>rec_polys</code>: <code>(numpy.ndarray)</code> 公式检测框，shape为(4, 2)，dtype为int16</li>
                    <li><code>formula_region_id</code>: <code>(int)</code> 公式所在的区域编号</li>
                </ul>
            <li><code>seal_res_list</code>: <code>(List[Dict[str, Union[numpy.ndarray, List[float], str]]])</code> 印章文本识别结果列表，每个元素为一个dict</li>
                <ul>
                    <li><code>input_path</code>: <code>(str)</code> 印章图像的输入路径</li>
                    <li><code>page_index</code>: <code> None</code> ，此处的输入为<code>numpy.ndarray</code>，所以值为<code> None</code> </li>
                    <li><code>model_settings</code>: <code>(Dict)</code> 印章文本识别子产线的模型配置参数</li>
                    <li><code>dt_polys</code>: <code>(List[numpy.ndarray])</code> 印章检测框列表，格式同<code>dt_polys</code></li>
                    <li><code>text_det_params</code>: <code>(Dict[str, Dict[str, int, float]])</code> 印章检测模块的配置参数, 具体参数含义同上</li>
                    <li><code>text_type</code>: <code>(str)</code> 印章检测的类型，当前固定为"seal"</li>
                    <li><code>text_rec_score_thresh</code>: <code>(float)</code> 印章文本识别结果的过滤阈值</li>
                    <li><code>rec_texts</code>: <code>(List[str])</code> 印章文本识别结果列表，仅包含置信度超过<code>text_rec_score_thresh</code>的文本</li>
                    <li><code>rec_scores</code>: <code>(List[float])</code> 印章文本识别的置信度列表，已按<code>dt_polys</code>过滤</li>
                    <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> 经过置信度过滤的印章检测框列表，格式同<code>dt_polys</code></li>
                    <li><code>rec_boxes</code>: <code>(numpy.ndarray)</code> 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形</li>
                </ul>
            <li><code>table_res_list</code>: <code>(List[Dict[str, Union[numpy.ndarray, List[float], str]]])</code> 表格识别结果列表，每个元素为一个dict</li>
                <ul>
                    <li><code>cell_box_list</code>: <code>(List[numpy.ndarray])</code> 表格单元格的边界框列表</li>
                    <li><code>pred_html</code>: <code>(str)</code> 表格的HTML格式字符串</li>
                    <li><code>table_ocr_pred</code>: <code>(dict)</code> 表格的OCR识别结果</li>
                    <ul>
                        <li><code>rec_polys</code>: <code>(List[numpy.ndarray])</code> 单元格的检测框列表</li>
                        <li><code>rec_texts</code>: <code>(List[str])</code> 单元格的识别结果</li>
                        <li><code>rec_scores</code>: <code>(List[float])</code>单元格的识别置信度</li>
                        <li><code>rec_boxes</code>: <code>(numpy.ndarray)</code> 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形</li>
                    </ul>
                </ul>
            </ul>
    </li>
    <li>调用<code>save_to_json()</code> 方法会将上述内容保存到指定的 <code>save_path</code> 中，如果指定为目录，则保存的路径为<code>save_path/{your_img_basename}_res.json</code>，如果指定为文件，则直接保存到该文件中。由于 json 文件不支持保存numpy数组，因此会将其中的 <code>numpy.array</code> 类型转换为列表形式。</li>
    <li>调用<code>save_to_img()</code> 方法会将可视化结果保存到指定的 <code>save_path</code> 中，如果指定为目录，则会将版面区域检测可视化图像、全局OCR可视化图像、版面阅读顺序可视化图像等内容保存，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)</li>
    <li>调用<code>save_to_markdown()</code> 方法会将转化后的 Markdown 文件保存到指定的 <code>save_path</code> 中，保存的文件路径为<code>save_path/{your_img_basename}.md</code>，如果输入是 PDF 文件，建议直接指定目录，否责多个 markdown 文件会被覆盖。
    调用 <code>concatenate_markdown_pages()</code> 方法将 <code>PP-StructureV3 pipeline</code> 输出的多页Markdown内容<code>markdown_list</code>合并为单个完整文档，并返回合并后的Markdown内容。</li>
</ul>
此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：
<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>json</code></td>
<td>获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"><code>markdown</code></td>
<td rowspan="3">获取格式为 <code>dict</code> 的 markdown 结果</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>
<ul>
    <li><code>json</code> 属性获取的预测结果为dict类型的数据，相关内容与调用 <code>save_to_json()</code> 方法保存的内容一致。</li>
    <li><code>img</code> 属性返回的预测结果是一个dict类型的数据。其中，键分别为 <code>layout_det_res</code>、<code>overall_ocr_res</code>、<code>text_paragraphs_ocr_res</code>、<code>formula_res_region1</code>、<code>table_cell_img</code> 和 <code>seal_res_region1</code>，对应的值是 <code>Image.Image</code> 对象：分别用于显示版面区域检测、OCR、OCR文本段落、公式、表格和印章结果的可视化图像。如果没有使用可选模块，则dict中只包含 <code>layout_det_res</code>。</li>
    <li><code>markdown</code> 属性返回的预测结果是一个dict类型的数据。其中，键分别为 <code>markdown_texts</code> 、 <code>markdown_images</code>和<code>page_continuation_flags</code>，对应的值分别是 markdown 文本，在 Markdown 中显示的图像（<code>Image.Image</code> 对象）和用于标识当前页面第一个元素是否为段开始以及最后一个元素是否为段结束的bool元组。</li>
</ul>
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
<li><b><code>infer</code></b></li>
</ul>
<p>进行版面解析。</p>
<p><code>POST /layout-parsing</code></p>
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
<td><code>integer</code>｜<code>null</code></td>
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
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_textline_orientation</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_seal_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_table_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_formula_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_chart_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useRegionDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_region_detection</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>object</code> | </code><code>null</code></td>
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
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_merge_bboxes_mode</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>text_det_limit_side_len</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>text_det_limit_type</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>text_det_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>text_det_box_thresh</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>text_det_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>text_rec_score_thresh</code> 参数相关说明。</td>
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
<tr>
<td><code>useWiredTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_wired_table_cells_trans_to_html</code> 参数相关说明。</td>
<td>No</td>
</tr>
<tr>
<td><code>useWirelessTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_wireless_table_cells_trans_to_html</code> 参数相关说明。</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableOrientationClassify</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_table_orientation_classify</code> 参数相关说明。</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrResultsWithTableCells</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_ocr_results_with_table_cells</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_e2e_wired_table_rec_model</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_e2e_wireless_table_rec_model</code> 参数相关说明。</td>
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
<td>版面解析结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
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
<td>产线对象的 <code>predict</code> 方法生成结果的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 和 <code>page_index</code> 字段。</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>Markdown结果。</td>
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
</table>
<p><code>markdown</code>为一个<code>object</code>，具有如下属性：</p>
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
<td><code>text</code></td>
<td><code>string</code></td>
<td>Markdown文本。</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>Markdown图片相对路径和Base64编码图像的键值对。</td>
</tr>
<tr>
<td><code>isStart</code></td>
<td><code>boolean</code></td>
<td>当前页面第一个元素是否为段开始。</td>
</tr>
<tr>
<td><code>isEnd</code></td>
<td><code>boolean</code></td>
<td>当前页面最后一个元素是否为段结束。</td>
</tr>
</tbody>
</table></details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests
import pathlib

API_URL = "http://localhost:8080/layout-parsing" # 服务URL

image_path = "./demo.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64编码的文件内容或者文件URL
    "fileType": 1, # 文件类型，1表示图像文件
}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
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


## 4. 二次开发
如果 PP-StructureV3 产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升 PP-StructureV3 产线的在您的场景中的识别效果。

### 4.1 模型微调
由于 PP-StructureV3 产线包含若干模块，模型产线的效果不及预期可能来自于其中任何一个模块。您可以对提取效果差的 case 进行分析，通过可视化图像，确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。


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
<td>表格结构识别模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/table_structure_recognition.html#_5">链接</a></td>
</tr>
<tr>
<td>公式识别不准</td>
<td>公式识别模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/version3.x/module_usage/formula_recognition.html#_5">链接</a></td>
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

可调用 PaddleOCR 中 PPStructureV3 产线对象的 `export_paddlex_config_to_yaml` 方法，将当前产线配置导出为 YAML 文件：

```Python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
pipeline.export_paddlex_config_to_yaml("PP-StructureV3.yaml")
```

2. 修改配置文件

在得到默认的产线配置文件后，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可。例如

```yaml
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout_plus-L
    model_dir: null # 替换为微调后的版面区域检测模型权重路径
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
        model_dir: null # 替换为微调后的文本测模型权重路径
        limit_side_len: 960
        limit_type: max
        max_side_limit: 4000
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 1.5

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv5_server_rec
        model_dir: null # 替换为微调后的文本识别模型权重路径
        batch_size: 1
        score_thresh: 0
......
```

在产线配置文件中，不仅包含 PaddleOCR CLI 和 Python API 支持的参数，还可进行更多高级配置，具体信息可在 [PaddleX模型产线使用概览](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/pipeline_develop_guide.html) 中找到对应的产线使用教程，参考其中的详细说明，根据需求调整各项配置。

3. 在 CLI 中加载产线配置文件

在修改完成配置文件后，通过命令行的 `--paddlex_config` 参数指定修改后的产线配置文件的路径，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```bash
paddleocr pp_structurev3 --paddlex_config PP-StructureV3.yaml ...
```

4. 在 Python API 中加载产线配置文件

初始化产线对象时，可通过 `paddlex_config` 参数传入 PaddleX 产线配置文件路径或配置dict，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
```

## 5. 附录

<details><summary><b>支持语言</b></summary>

<table border="1" cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th><code>lang</code></th>
      <th>语言名称</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>abq</code></td><td>阿布哈兹文</td></tr>
    <tr><td><code>af</code></td><td>南非荷兰文</td></tr>
    <tr><td><code>ang</code></td><td>古英文</td></tr>
    <tr><td><code>ar</code></td><td>阿拉伯文</td></tr>
    <tr><td><code>ava</code></td><td>阿瓦尔文</td></tr>
    <tr><td><code>az</code></td><td>阿塞拜疆文</td></tr>
    <tr><td><code>be</code></td><td>白俄罗斯文</td></tr>
    <tr><td><code>bg</code></td><td>保加利亚文</td></tr>
    <tr><td><code>bgc</code></td><td>哈里亚纳文</td></tr>
    <tr><td><code>bh</code></td><td>比哈尔文</td></tr>
    <tr><td><code>bho</code></td><td>博杰普尔文</td></tr>
    <tr><td><code>bs</code></td><td>波斯尼亚文</td></tr>
    <tr><td><code>ch</code></td><td>简体中文</td></tr>
    <tr><td><code>che</code></td><td>车臣文</td></tr>
    <tr><td><code>chinese_cht</code></td><td>繁体中文</td></tr>
    <tr><td><code>cs</code></td><td>捷克文</td></tr>
    <tr><td><code>cy</code></td><td>威尔士文</td></tr>
    <tr><td><code>da</code></td><td>丹麦文</td></tr>
    <tr><td><code>dar</code></td><td>达尔格瓦文</td></tr>
    <tr><td><code>de</code> or <code>german</code></td><td>德文</td></tr>
    <tr><td><code>en</code></td><td>英文</td></tr>
    <tr><td><code>es</code></td><td>西班牙文</td></tr>
    <tr><td><code>et</code></td><td>爱沙尼亚文</td></tr>
    <tr><td><code>fa</code></td><td>波斯文</td></tr>
    <tr><td><code>fr</code> or <code>french</code></td><td>法文</td></tr>
    <tr><td><code>ga</code></td><td>爱尔兰文</td></tr>
    <tr><td><code>gom</code></td><td>孔卡尼文</td></tr>
    <tr><td><code>hi</code></td><td>印地文</td></tr>
    <tr><td><code>hr</code></td><td>克罗地亚文</td></tr>
    <tr><td><code>hu</code></td><td>匈牙利文</td></tr>
    <tr><td><code>id</code></td><td>印尼文</td></tr>
    <tr><td><code>inh</code></td><td>印古什文</td></tr>
    <tr><td><code>is</code></td><td>冰岛文</td></tr>
    <tr><td><code>it</code></td><td>意大利文</td></tr>
    <tr><td><code>japan</code></td><td>日文</td></tr>
    <tr><td><code>ka</code></td><td>格鲁吉亚文</td></tr>
    <tr><td><code>kbd</code></td><td>卡巴尔达文</td></tr>
    <tr><td><code>korean</code></td><td>韩文</td></tr>
    <tr><td><code>ku</code></td><td>库尔德文</td></tr>
    <tr><td><code>la</code></td><td>拉丁文</td></tr>
    <tr><td><code>lbe</code></td><td>拉克文</td></tr>
    <tr><td><code>lez</code></td><td>列兹金文</td></tr>
    <tr><td><code>lt</code></td><td>立陶宛文</td></tr>
    <tr><td><code>lv</code></td><td>拉脱维亚文</td></tr>
    <tr><td><code>mah</code></td><td>马加希文</td></tr>
    <tr><td><code>mai</code></td><td>迈蒂利文</td></tr>
    <tr><td><code>mi</code></td><td>毛利文</td></tr>
    <tr><td><code>mn</code></td><td>蒙古文</td></tr>
    <tr><td><code>mr</code></td><td>马拉地文</td></tr>
    <tr><td><code>ms</code></td><td>马来文</td></tr>
    <tr><td><code>mt</code></td><td>马耳他文</td></tr>
    <tr><td><code>ne</code></td><td>尼泊尔文</td></tr>
    <tr><td><code>new</code></td><td>尼瓦尔文</td></tr>
    <tr><td><code>nl</code></td><td>荷兰文</td></tr>
    <tr><td><code>no</code></td><td>挪威文</td></tr>
    <tr><td><code>oc</code></td><td>奥克文</td></tr>
    <tr><td><code>pi</code></td><td>巴利文</td></tr>
    <tr><td><code>pl</code></td><td>波兰文</td></tr>
    <tr><td><code>pt</code></td><td>葡萄牙文</td></tr>
    <tr><td><code>ro</code></td><td>罗马尼亚文</td></tr>
    <tr><td><code>rs_cyrillic</code></td><td>塞尔维亚语西里尔字母</td></tr>
    <tr><td><code>rs_latin</code></td><td>塞尔维亚语拉丁字母</td></tr>
    <tr><td><code>ru</code></td><td>俄文</td></tr>
    <tr><td><code>sa</code></td><td>梵文</td></tr>
    <tr><td><code>sck</code></td><td>萨达里文</td></tr>
    <tr><td><code>sk</code></td><td>斯洛伐克文</td></tr>
    <tr><td><code>sl</code></td><td>斯洛文尼亚文</td></tr>
    <tr><td><code>sq</code></td><td>阿尔巴尼亚文</td></tr>
    <tr><td><code>sv</code></td><td>瑞典文</td></tr>
    <tr><td><code>sw</code></td><td>斯瓦希里文</td></tr>
    <tr><td><code>tab</code></td><td>塔巴萨兰文</td></tr>
    <tr><td><code>ta</code></td><td>泰米尔文</td></tr>
    <tr><td><code>te</code></td><td>泰卢固文</td></tr>
    <tr><td><code>tl</code></td><td>塔加洛文</td></tr>
    <tr><td><code>tr</code></td><td>土耳其文</td></tr>
    <tr><td><code>ug</code></td><td>维吾尔文</td></tr>
    <tr><td><code>uk</code></td><td>乌克兰文</td></tr>
    <tr><td><code>ur</code></td><td>乌尔都文</td></tr>
    <tr><td><code>uz</code></td><td>乌兹别克文</td></tr>
    <tr><td><code>vi</code></td><td>越南文</td></tr>
  </tbody>
</table>

</details>

<details><summary><b>OCR模型版本与支持语言的对应关系</b></summary>

<table border="1" cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th><code>ocr_version</code></th>
      <th>Supported <code>lang</code></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>PP-OCRv5</code></td>
      <td><code>ch</code>, <code>en</code>, <code>fr</code>, <code>de</code>, <code>japan</code>, <code>korean</code>, <code>chinese_cht</code>, <code>af</code>, <code>it</code>, <code>es</code>, <code>bs</code>, <code>pt</code>, <code>cs</code>, <code>cy</code>, <code>da</code>, <code>et</code>, <code>ga</code>, <code>hr</code>, <code>hu</code>, <code>rslatin</code>, <code>id</code>, <code>oc</code>, <code>is</code>, <code>lt</code>, <code>mi</code>, <code>ms</code>, <code>nl</code>, <code>no</code>, <code>pl</code>, <code>sk</code>, <code>sl</code>, <code>sq</code>, <code>sv</code>, <code>sw</code>, <code>tl</code>, <code>tr</code>, <code>uz</code>, <code>la</code>, <code>ru</code>, <code>be</code>, <code>uk</code></td>
    </tr>
    <tr>
      <td><code>PP-OCRv4</code></td>
      <td><code>ch</code>, <code>en</code></td>
    </tr>
    <tr>
      <td><code>PP-OCRv3</code></td>
      <td>
        <code>abq</code>, <code>af</code>, <code>ady</code>, <code>ang</code>, <code>ar</code>, <code>ava</code>, <code>az</code>, <code>be</code>,
        <code>bg</code>, <code>bgc</code>, <code>bh</code>, <code>bho</code>, <code>bs</code>, <code>ch</code>, <code>che</code>,
        <code>chinese_cht</code>, <code>cs</code>, <code>cy</code>, <code>da</code>, <code>dar</code>, <code>de</code>, <code>german</code>,
        <code>en</code>, <code>es</code>, <code>et</code>, <code>fa</code>, <code>fr</code>, <code>french</code>, <code>ga</code>, <code>gom</code>,
        <code>hi</code>, <code>hr</code>, <code>hu</code>, <code>id</code>, <code>inh</code>, <code>is</code>, <code>it</code>, <code>japan</code>,
        <code>ka</code>, <code>kbd</code>, <code>korean</code>, <code>ku</code>, <code>la</code>, <code>lbe</code>, <code>lez</code>, <code>lt</code>,
        <code>lv</code>, <code>mah</code>, <code>mai</code>, <code>mi</code>, <code>mn</code>, <code>mr</code>, <code>ms</code>, <code>mt</code>,
        <code>ne</code>, <code>new</code>, <code>nl</code>, <code>no</code>, <code>oc</code>, <code>pi</code>, <code>pl</code>, <code>pt</code>,
        <code>ro</code>, <code>rs_cyrillic</code>, <code>rs_latin</code>, <code>ru</code>, <code>sa</code>, <code>sck</code>, <code>sk</code>,
        <code>sl</code>, <code>sq</code>, <code>sv</code>, <code>sw</code>, <code>ta</code>, <code>tab</code>, <code>te</code>, <code>tl</code>,
        <code>tr</code>, <code>ug</code>, <code>uk</code>, <code>ur</code>, <code>uz</code>, <code>vi</code>
      </td>
    </tr>
  </tbody>
</table>

</details>
