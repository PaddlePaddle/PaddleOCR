---
comments: true
---

# PP-StructureV3 产线使用教程

## 1. PP-StructureV3 产线介绍

版面解析是一种从文档图像中提取结构化信息的技术，主要用于将复杂的文档版面转换为机器可读的数据格式。这项技术在文档管理、信息提取和数据数字化等领域具有广泛的应用。版面解析通过结合光学字符识别（OCR）、图像处理和机器学习算法，能够识别和提取文档中的文本块、标题、段落、图片、表格以及其他版面元素。此过程通常包括版面分析、元素分析和数据格式化三个主要步骤，最终生成结构化的文档数据，提升数据处理的效率和准确性。<b>PP-StructureV3 产线在通用版面解析v1产线的基础上，强化了版面区域检测、表格识别、公式识别的能力，增加了图表理解能力和多栏阅读顺序的恢复能力、结果转换 Markdown 文件的能力，在多种文档数据中，表现优异，可以处理较复杂的文档数据。</b>本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<b>PP-StructureV3 产线中包含以下6个模块。每个模块均可独立进行训练和推理，并包含多个模型。有关详细信息，请点击相应模块以查看文档。</b>

- [版面区域检测模块](../module_usage/layout_detection.md)
- [通用OCR子产线](./OCR.md)
- [文档图像预处理子产线](./doc_preprocessor.md) （可选）
- [表格识别子产线](./table_recognition_v2.md) （可选）
- [印章识别子产线](./seal_recognition.md) （可选）
- [公式识别子产线](./formula_recognition.md) （可选）

在本产线中，您可以根据下方的基准测试数据选择使用的模型。

<details>
<summary><b>文档图像方向分类模块：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
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
<summary><b>文本图像矫正模块：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>CER </th>
<th>模型存储大小（M）</th>
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
<summary><b>版面区域检测模块：</b></summary>

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

<b>注：以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1300 张文档类型图片。</b>

* <b>文档图像版面子模块检测，包含1个 版面区域 类别，能检测多栏的报纸、杂志的每个子文章的文本区域：</b>
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
<td>PP-DocBlockLayout</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams">训练模型</a></td>
<td>95.9</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>123.92 M</td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
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

* <b>5类英文文档区域检测模型，包含文字、标题、表格、图片以及列表</b>
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
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
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
<summary><b>表格结构识别模块：</b></summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>精度（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
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
<th>模型存储大小 (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">训练模型</a></td>
<td>94.2</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>6.6M</td>
</tr>
</table>

<p><b>表格单元格检测模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
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
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">训练模型</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>371.65 / 371.65</td>
<td>84.3</td>
<td>PP-OCRv5 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>79.0</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv5 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>69.2</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>63.8</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
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
<summary><b>文本行方向分类模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th>
<th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">训练模型</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
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
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">训练模型</a></td>
<td>85.91</td>
<td>43.50</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.53 G</td>
<td>UniMERNet是由上海AI Lab研发的一款公式识别模型。该模型采用Donut Swin作为编码器，MBartDecoder作为解码器，并通过在包含简单公式、复杂公式、扫描捕捉公式和手写公式在内的一百万数据集上进行训练，大幅提升了模型对真实场景公式的识别准确率</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">训练模型</a></td>
<td>87.00</td>
<td>45.71</td>
<td>202.25/-</td>
<td>-/-</td>
<td>224 M</td>
<td rowspan="2">PP-FormulaNet 是由百度飞桨视觉团队开发的一款先进的公式识别模型，支持5万个常见LateX源码词汇的识别。PP-FormulaNet-S 版本采用了 PP-HGNetV2-B4 作为其骨干网络，通过并行掩码和模型蒸馏等技术，大幅提升了模型的推理速度，同时保持了较高的识别精度，适用于简单印刷公式、跨行简单印刷公式等场景。而 PP-FormulaNet-L 版本则基于 Vary_VIT_B 作为骨干网络，并在大规模公式数据集上进行了深入训练，在复杂公式的识别方面，相较于PP-FormulaNet-S表现出显著的提升，适用于简单印刷公式、复杂印刷公式、手写公式等场景。 </td>

</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">训练模型</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>695 M</td>
<tr>
<td>PP-FormulaNet_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">训练模型</a></td>
<td>88.71</td>
<td>53.32</td>
<td>191.69/-</td>
<td>-/-</td>
<td>248 M</td>
<td rowspan="3">PP-FormulaNet_plus 是百度飞桨视觉团队在 PP-FormulaNet 的基础上开发的增强版公式识别模型。与原版相比，PP-FormulaNet_plus 在训练中使用了更为丰富的公式数据集，包括中文学位论文、专业书籍、教材试卷以及数学期刊等多种来源。这一扩展显著提升了模型的识别能力。

其中，PP-FormulaNet_plus-M 和 PP-FormulaNet_plus-L 模型新增了对中文公式的支持，并将公式的最大预测 token 数从 1024 扩大至 2560，大幅提升了对复杂公式的识别性能。同时，PP-FormulaNet_plus-S 模型则专注于增强英文公式的识别能力。通过这些改进，PP-FormulaNet_plus 系列模型在处理复杂多样的公式识别任务时表现更加出色。 </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">训练模型</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1301.56/-</td>
<td>-/-</td>
<td>592 M</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">训练模型</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1745.25/-</td>
<td>-/-</td>
<td>698 M</td>
</tr>

<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1244.61/-</td>
<td>-/-</td>
<td>99 M</td>
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
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
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
<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>

## 2. 快速开始

在本地使用 PP-StructureV3 产线前，请确保您已经按照[安装教程](../installation.md)完成了wheel包安装。安装完成后，可以在本地使用命令行体验或 Python 集成。

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
<td>待预测数据，支持多种输入类型，必填。
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
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
<td><code>layout_detection_model_name</code></td>
<td>版面区域检测的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。
<ul>
<li><b>float</b>：<code>0-1</code> 之间的任意浮点数；</li>
<li><b>dict</b>： <code>{0:0.1}</code> key为类别ID，value为该类别的阈值；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>0.5</code>；</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面区域检测模型是否使用NMS后处理。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数。
<ul>
<li><b>float</b>：任意大于 <code>0</code>  浮点数；</li>
<li><b>Tuple[float,float]</b>：在横纵两个方向各自的扩张系数；</li>
<li><b>字典</b>, 字典的key为<b>int</b>类型，代表<code>cls_id</code>, value为<b>tuple</b>类型，如<code>{0: (1.1, 2.0)}</code>, 表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>1.0</code>；</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测的重叠框过滤方式。
<ul>
<li><b>str</b>：<code>large</code>，<code>small</code>, <code>union</code>，分别表示重叠框过滤时选择保留大框，小框还是同时保留</li>
<li><b>dict</b>, 字典的key为<b>int</b>类型，代表<code>cls_id</code>, value为<b>str</b>类型, 如<code>{0: "large", 2: "small"}</code>, 表示对第0类别检测框使用large模式，对第2类别检测框使用small模式</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>large</code>；</li>
</ul>
</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td>图表解析的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>图表解析模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>图表解析模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>文档图像版面子模块检测的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>文档图像版面子模块检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>文本检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>文本检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的最大边长度限制。
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>960</code>；</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>max</code>。</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.3</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>2.0</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>文本行方向模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>文本行方向模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>文本行方向模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>表格分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>表格分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>有线表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>有线表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>无线表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>无线表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>有线表格单元格检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>有线表格单元格检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>无线表格单元格检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>无线表格单元格检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>印章文本检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>印章文本检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
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
<td><code>seal_text_recognition_model_name</code></td>
<td>印章文本识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>印章文本识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>印章文本识别模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
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
<td><code>formula_recognition_model_name</code></td>
<td>公式识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>公式识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>公式识别模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载文档方向分类模块。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载文本图像矫正模块。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否加载印章识别子产线。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否加载表格识别子产线。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>是否加载公式识别子产线。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否加载图表解析模型。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>是否加载文档图像版面子模块检测模型。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
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

运行结果会被打印到终端上，默认配置的 PP-StructureV3 产线的运行结果如下：

<details><summary> 👉点击展开</summary>
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

运行结果参数说明可以参考[2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

<b>注：</b>由于产线的默认模型较大，推理速度可能较慢，您可以参考第一节的模型列表，替换推理速度更快的模型。

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
# ocr = PPStructureV3(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# ocr = PPStructureV3(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# ocr = PPStructureV3(use_textline_orientation=True) # 通过 use_textline_orientation 指定是否使用文本行方向分类模型
# ocr = PPStructureV3(device="gpu") # 通过 device 指定模型推理时使用 GPU
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

**注：**

- PP-StructureV3 产线使用的默认文本识别模型为 **中英文识别模型**，对于纯英文的识别能力有限，对于全英文场景，您可以设置`text_recognition_model_name`参数为 `en_PP-OCRv4_mobile_rec` 等英文识别模型以取得更好的识别效果。对应其他语言场景，也可以参考前文的模型列表，选择对应的语言识别模型进行替换。

- 在示例代码中，`use_doc_orientation_classify`、`use_doc_unwarping`、`use_textline_orientation` 参数默认均设置为 `False`，分别表示关闭文档方向分类、文本图像矫正、文本行方向分类功能，如果需要使用这些功能，可以手动设置为 `True`。

在上述 Python 脚本中，执行了如下几个步骤：

<details><summary>（1）实例化 <code>PPStructureV3</code> 实例化产线对象，具体参数说明如下：</summary>

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
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。
<ul>
<li><b>float</b>：<code>0-1</code> 之间的任意浮点数；</li>
<li><b>dict</b>： <code>{0:0.1}</code> key为类别ID，value为该类别的阈值；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>0.5</code>；</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面区域检测模型是否使用NMS后处理。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数。
<ul>
<li><b>float</b>：任意大于 <code>0</code>  浮点数；</li>
<li><b>Tuple[float,float]</b>：在横纵两个方向各自的扩张系数；</li>
<li><b>字典</b>, 字典的key为<b>int</b>类型，代表<code>cls_id</code>, value为<b>tuple</b>类型，如<code>{0: (1.1, 2.0)}</code>, 表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>1.0</code>；</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测的重叠框过滤方式。
<ul>
<li><b>str</b>：<code>large</code>，<code>small</code>, <code>union</code>，分别表示重叠框过滤时选择保留大框，小框还是同时保留</li>
<li><b>dict</b>, 字典的key为<b>int</b>类型，代表<code>cls_id</code>, value为<b>str</b>类型, 如<code>{0: "large", 2: "small"}</code>, 表示对第0类别检测框使用large模式，对第2类别检测框使用small模式</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>large</code>；</li>
</ul>
</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_name</code></td>
<td>图表解析的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_model_dir</code></td>
<td>图表解析模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chart_recognition_batch_size</code></td>
<td>图表解析模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_name</code></td>
<td>文档图像版面子模块检测的模型名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>region_detection_model_dir</code></td>
<td>文档图像版面子模块检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_name</code></td>
<td>文本检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>文本检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的最大边长度限制。
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>960</code>；</li>
</ul>
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>max</code>。</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.3</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>2.0</code></li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>文本行方向模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>文本行方向模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>文本行方向模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_name</code></td>
<td>表格分类模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_classification_model_dir</code></td>
<td>表格分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_name</code></td>
<td>有线表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_structure_recognition_model_dir</code></td>
<td>有线表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_name</code></td>
<td>无线表格结构识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_structure_recognition_model_dir</code></td>
<td>无线表格结构识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_name</code></td>
<td>有线表格单元格检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wired_table_cells_detection_model_dir</code></td>
<td>有线表格单元格检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_name</code></td>
<td>无线表格单元格检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>wireless_table_cells_detection_model_dir</code></td>
<td>无线表格单元格检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_name</code></td>
<td>印章文本检测模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_detection_model_dir</code></td>
<td>印章文本检测模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
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
<td><code>seal_text_recognition_model_name</code></td>
<td>印章文本识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_model_dir</code></td>
<td>印章文本识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_text_recognition_batch_size</code></td>
<td>印章文本识别模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>印章文本识别阈值，得分大于该阈值的文本结果会被保留。
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul>
</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>公式识别模型的名称。如果设置为<code>None</code>，将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>公式识别模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>公式识别模型的批处理大小。如果设置为<code>None</code>，将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载文档方向分类模块。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载文本图像矫正模块。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否加载印章识别子产线。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否加载表格识别子产线。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>是否加载公式识别子产线。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否加载图表解析模型。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>是否加载文档图像版面子模块检测模型。如果设置为<code>None</code>，将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
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
<td><code>use_textline_orientation</code></td>
<td>是否在推理时使用文本行方向分类模块。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否在推理时使用印章识别子产线。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否在推理时使用表格识别子产线。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>是否在推理时使用公式识别子产线。</td>
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
<td><code>float|Tuple[float,float]|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>与实例化时的参数相同。</td>
<td><code>str|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>与实例化时的参数相同。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>与实例化时的参数相同。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>与实例化时的参数相同。</td>
<td><code>float</code></td>
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
<td>将中间各个模块的可视化图像保存在png格式的图像</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>将图像或者PDF文件中的每一页分别保存为markdown格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>将文件中的表格保存为html格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>将文件中的表格保存为xlsx格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>concatenate_markdown_pages()</code></td>
<td>将多页Markdown内容拼接为单一文档</td>
<td><code>markdown_list</code></td>
<td><code>list</code></td>
<td>包含每一页Markdown数据的列表</td>
<td>返回处理后的Markdown文本和图像列表</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：
    - `input_path`: `(str)` 待预测图像或者PDF的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_seal_recognition`: `(bool)` 控制是否启用印章识别子产线
        - `use_table_recognition`: `(bool)` 控制是否启用表格识别子产线
        - `use_formula_recognition`: `(bool)` 控制是否启用公式识别子产线

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` 文档预处理结果字典，仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(str)` 文档预处理子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`，此处为`None`
        - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
        - `model_settings`: `(Dict[str, bool])` 文档预处理子产线的模型配置参数
          - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档图像方向分类子模块
          - `use_doc_unwarping`: `(bool)` 控制是否启用文本图像扭曲矫正子模块
        - `angle`: `(int)` 文档图像方向分类子模块的预测结果，启用时返回实际角度值

    - `parsing_res_list`: `(List[Dict])` 解析结果的列表，每个元素为一个字典，列表顺序为解析后的阅读顺序。
        - `block_bbox`: `(np.ndarray)` 版面区域的边界框。
        - `block_label`: `(str)` 版面区域的标签，例如`text`, `table`等。
        - `block_content`: `(str)` 内容为版面区域内的内容。
        - `seg_start_flag`: `(bool)` 标识该版面区域是否是段落的开始。
        - `seg_end_flag`: `(bool)` 标识该版面区域是否是段落的结束。
        - `sub_label`: `(str)` 版面区域的子标签，例如`text`的子标签可能为`title_text`。
        - `sub_index`: `(int)` 版面区域的子索引，用于恢复Markdown。
        - `index`: `(int)` 版面区域的索引，用于显示版面排序结果。





    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` 全局 OCR 结果的字典
      - `input_path`: `(Union[str, None])` 图像OCR子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
      - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
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

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 公式识别结果列表，每个元素为一个字典
        - `rec_formula`: `(str)` 公式识别结果
        - `rec_polys`: `(numpy.ndarray)` 公式检测框，shape为(4, 2)，dtype为int16
        - `formula_region_id`: `(int)` 公式所在的区域编号

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 印章识别结果列表，每个元素为一个字典
        - `input_path`: `(str)` 印章图像的输入路径
        - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
        - `model_settings`: `(Dict)` 印章识别子产线的模型配置参数
        - `dt_polys`: `(List[numpy.ndarray])` 印章检测框列表，格式同`dt_polys`
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 印章检测模块的配置参数, 具体参数含义同上
        - `text_type`: `(str)` 印章检测的类型，当前固定为"seal"
        - `text_rec_score_thresh`: `(float)` 印章识别结果的过滤阈值
        - `rec_texts`: `(List[str])` 印章识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本
        - `rec_scores`: `(List[float])` 印章识别的置信度列表，已按`text_rec_score_thresh`过滤
        - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的印章检测框列表，格式同`dt_polys`
        - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 表格识别结果列表，每个元素为一个字典
        - `cell_box_list`: `(List[numpy.ndarray])` 表格单元格的边界框列表
        - `pred_html`: `(str)` 表格的HTML格式字符串
        - `table_ocr_pred`: `(dict)` 表格的OCR识别结果
            - `rec_polys`: `(List[numpy.ndarray])` 单元格的检测框列表
            - `rec_texts`: `(List[str])` 单元格的识别结果
            - `rec_scores`: `(List[float])` 单元格的识别置信度
            - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形

- 调用`save_to_json()` 方法会将上述内容保存到指定的 `save_path` 中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于 json 文件不支持保存numpy数组，因此会将其中的 `numpy.array` 类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的 `save_path` 中，如果指定为目录，则会将版面区域检测可视化图像、全局OCR可视化图像、版面阅读顺序可视化图像等内容保存，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)
- 调用`save_to_markdown()` 方法会将转化后的 Markdown 文件保存到指定的 `save_path` 中，保存的文件路径为`save_path/{your_img_basename}.md`，如果输入是 PDF 文件，建议直接指定目录，否责多个 markdown 文件会被覆盖。
- 调用 `concatenate_markdown_pages()` 方法将 `PP-StructureV3 pipeline` 输出的多页Markdown内容`markdown_list`合并为单个完整文档，并返回合并后的Markdown内容。

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

- `json` 属性获取的预测结果为字典类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `layout_det_res`、`overall_ocr_res`、`text_paragraphs_ocr_res`、`formula_res_region1`、`table_cell_img` 和 `seal_res_region1`，对应的值是 `Image.Image` 对象：分别用于显示版面区域检测、OCR、OCR文本段落、公式、表格和印章结果的可视化图像。如果没有使用可选模块，则字典中只包含 `layout_det_res`。
- `markdown` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `markdown_texts` 、 `markdown_images`和`page_continuation_flags`，对应的值分别是 markdown 文本，在 Markdown 中显示的图像（`Image.Image` 对象）和用于标识当前页面第一个元素是否为段开始以及最后一个元素是否为段结束的bool元组。

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
<td><code>useTableCellsOcrResults</code></td>
<td><code>boolean</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_table_cells_ocr_results</code> 参数相关说明。</td>
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
<td>markdown结果。</td>
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
<td>Markdown图片相对路径和base64编码图像的键值对。</td>
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

在产线配置文件中，不仅包含 PaddleOCR CLI 和 Python API 支持的参数，还可进行更多高级配置，具体信息可在 [PaddleX模型产线使用概览](https://paddlepaddle.github.io/PaddleX/3.0/pipeline_usage/pipeline_develop_guide.html) 中找到对应的产线使用教程，参考其中的详细说明，根据需求调整各项配置。

3. 在 CLI 中加载产线配置文件

在修改完成配置文件后，通过命令行的 --paddlex_config 参数指定修改后的产线配置文件的路径，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```bash
paddleocr pp_structurev3 --paddlex_config PP-StructureV3.yaml ...
```

4. 在 Python API 中加载产线配置文件

初始化产线对象时，可通过 paddlex_config 参数传入 PaddleX 产线配置文件路径或配置字典，PaddleOCR 会读取其中的内容作为产线配置。示例如下：

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
```
