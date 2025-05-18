---
comments: true
---

# 公式识别产线使用教程

## 1. 公式识别产线介绍

公式识别是一种自动从文档或图像中识别和提取LaTeX公式内容及其结构的技术，广泛应用于数学、物理、计算机科学等领域的文档编辑和数据分析。通过使用计算机视觉和机器学习算法，公式识别能够将复杂的数学公式信息转换为可编辑的LaTeX格式，方便用户进一步处理和分析数据。

公式识别产线用于解决公式识别任务，提取图片中的公式信息以LaTeX源码形式输出，本产线是一个集成了百度飞桨视觉团队自研的先进公式识别模型PP-FormulaNet 和业界知名公式识别模型 UniMERNet的端到端公式识别系统，支持简单印刷公式、复杂印刷公式、手写公式的识别，并在此基础上，增加了对图像的方向矫正和扭曲矫正功能。基于本产线，可实现公式内容精准预测，使用场景覆盖教育、科研、金融、制造等各个领域。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/03.png" style="width: 70%"/>

<b>公式识别产线中包含以下4个模块。每个模块均可独立进行训练和推理，并包含多个模型。有关详细信息，请点击相应模块以查看文档。</b>

- [公式识别模块](../module_usage/formula_recognition.md)
- [版面区域检测模块](../module_usage/layout_detection.md)（可选）
- [文档图像方向分类模块](../module_usage/doc_img_orientation_classification.md) （可选）
- [文本图像矫正模块](../module_usage/text_image_unwarping.md) （可选）

在本产线中，您可以根据下方的基准测试数据选择使用的模型。

<details>
<summary><b>文档图像方向分类模块（可选）：</b></summary>
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
<summary><b>文本图像矫正模块（可选）：</b></summary>
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
<summary><b>版面区域检测模块（可选）：</b></summary>
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

>❗ 以上列出的是版面区域检测模块重点支持的<b>3个核心模型</b>，该模块总共支持<b>6个全量模型</b>，包含多个预定义了不同类别的模型，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

* <b>17类版面区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody>
</table>


* <b>23类版面区域检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>
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
</details>
</details>

<details>
<summary><b>公式识别模块：</b></summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Avg-BLEU(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">训练模型</a></td>
<td>86.13</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.4 G</td>
<td>UniMERNet是由上海AI Lab研发的一款公式识别模型。该模型采用Donut Swin作为编码器，MBartDecoder作为解码器，并通过在包含简单公式、复杂公式、扫描捕捉公式和手写公式在内的一百万数据集上进行训练，大幅提升了模型对真实场景公式的识别准确率</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">训练模型</a></td>
<td>87.12</td>
<td>202.25/-</td>
<td>-/-</td>
<td>167.9 M</td>
<td rowspan="2">PP-FormulaNet 是由百度飞桨视觉团队开发的一款先进的公式识别模型，支持5万个常见LateX源码词汇的识别。PP-FormulaNet-S 版本采用了 PP-HGNetV2-B4 作为其骨干网络，通过并行掩码和模型蒸馏等技术，大幅提升了模型的推理速度，同时保持了较高的识别精度，适用于简单印刷公式、跨行简单印刷公式等场景。而 PP-FormulaNet-L 版本则基于 Vary_VIT_B 作为骨干网络，并在大规模公式数据集上进行了深入训练，在复杂公式的识别方面，相较于PP-FormulaNet-S表现出显著的提升，适用于简单印刷公式、复杂印刷公式、手写公式等场景。 </td>

</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">训练模型</a></td>
<td>92.13</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>535.2 M</td>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
<td>71.63</td>
<td>-/-</td>
<td>-/-</td>
<td>89.7 M</td>
<td>LaTeX-OCR是一种基于自回归大模型的公式识别算法，通过采用 Hybrid ViT 作为骨干网络，transformer作为解码器，显著提升了公式识别的准确性。</td>
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
                  <li>文档图像方向分类模型：PaddleX自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li> 文本图像矫正模型：<a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>。</li>
                  <li>版面区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志、合同、书本、试卷和研报等常见的 500 张文档类型图片。</li>
                  <li>17类区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。</li>
                  <li>公式识别模型：PaddleX 内部自建公式识别测试集。</li>
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

在本地使用通用版面解析v3产线前，请确保您已经按照[安装教程](../ppocr/installation.md)完成了wheel包安装。安装完成后，可以在本地使用命令行体验或 Python 集成。

### 2.1 命令行方式体验

一行命令即可快速体验 formula_recognition 产线效果：

```bash
paddleocr formula_recognition_pipeline -i https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png

# 通过 --use_doc_orientation_classify 指定是否使用文档方向分类模型
paddleocr formula_recognition_pipeline -i ./general_formula_recognition_001.png --use_doc_orientation_classify True

# 通过 --use_doc_unwarping 指定是否使用文本图像矫正模块
paddleocr formula_recognition_pipeline -i ./general_formula_recognition_001.png --use_doc_unwarping True

# 通过 --device 指定模型推理时使用 GPU
paddleocr formula_recognition_pipeline -i ./general_formula_recognition_001.png --device gpu
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
<td><code>doc_orientation_classify_batch_size</code></td>
<td>文档方向分类模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
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
<td><code>doc_unwarping_batch_size</code></td>
<td>文本图像矫正模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>版面区域检测模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面区域检测的阈值，用于过滤掉低置信度预测结果的阈值。
<ul>
<li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
<li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面区域检测是否使用NMS后处理，过滤重叠框。如果设置为<code>None</code>, 将会使用官方模型配置。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测中检测框的边长缩放倍数。
<ul>
<li><b>float</b>, 大于0的浮点数，如 1.1 , 表示将模型输出的检测框中心不变，宽和高都扩张1.1倍</li>
<li><b>列表</b>, 如 [1.2, 1.5] , 表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测中模型输出的检测框的合并处理模式。
<ul>
<li><b>large</b>, 设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>small</b>, 设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>union</b>, 不进行框的过滤处理，内外框都保留</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_batch_size</code></td>
<td>版面区域检测模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
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
<td>是否加载版面区域检测模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>公式识别模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>公式识别模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>公式识别模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
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
<td><code>None</code></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径。如果设置为<code>None</code>, 推理结果将不会保存到本地。</td>
<td><code>str</code></td>
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
</tbody>
</table>
</details>
<br />

运行结果会被打印到终端上，默认配置的 formula_recognition 产线的运行结果如下：

```bash
{'res': {'input_path': '/root/.paddlex/predict_input/general_formula_recognition_001.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 0}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.985754132270813, 'coordinate': [89.7193, 1086.395, 658.7713, 1553.3423]}, {'cls_id': 2, 'label': 'text', 'score': 0.9843020439147949, 'coordinate': [92.333496, 128.53197, 665.28827, 396.77502]}, {'cls_id': 2, 'label': 'text', 'score': 0.9766702651977539, 'coordinate': [698.1615, 590.9516, 1293.0541, 747.957]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720445275306702, 'coordinate': [697.04236, 752.37866, 1289.7733, 883.6316]}, {'cls_id': 2, 'label': 'text', 'score': 0.969851016998291, 'coordinate': [92.62311, 799.51917, 660.5987, 901.7046]}, {'cls_id': 2, 'label': 'text', 'score': 0.968906819820404, 'coordinate': [703.46436, 81.138016, 1304.8857, 187.78355]}, {'cls_id': 2, 'label': 'text', 'score': 0.9686803221702576, 'coordinate': [691.15967, 1513.7944, 1283.3694, 1639.1626]}, {'cls_id': 2, 'label': 'text', 'score': 0.9676252007484436, 'coordinate': [700.59705, 287.55557, 1299.9479, 391.25064]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9654340147972107, 'coordinate': [727.97864, 441.2702, 1221.3971, 570.22736]}, {'cls_id': 2, 'label': 'text', 'score': 0.962298572063446, 'coordinate': [696.6504, 958.38, 1288.2308, 1033.8015]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9593353867530823, 'coordinate': [155.30962, 924.0272, 598.61615, 1036.6716]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9584437012672424, 'coordinate': [811.00867, 1058.013, 1176.5062, 1118.1985]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9583907127380371, 'coordinate': [776.84436, 208.44116, 1224.5082, 267.0984]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9574459195137024, 'coordinate': [756.9298, 1211.8248, 1190.2643, 1267.3693]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9567654728889465, 'coordinate': [723.22705, 1332.768, 1254.1936, 1469.2213]}, {'cls_id': 2, 'label': 'text', 'score': 0.9535155296325684, 'coordinate': [87.32236, 1557.9272, 656.71436, 1632.439]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9531306028366089, 'coordinate': [116.53526, 714.33014, 613.72314, 773.89496]}, {'cls_id': 2, 'label': 'text', 'score': 0.9499222040176392, 'coordinate': [95.88785, 479.01178, 663.25146, 536.5941]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9488246440887451, 'coordinate': [165.14, 558.2904, 597.4905, 613.77295]}, {'cls_id': 2, 'label': 'text', 'score': 0.9445527791976929, 'coordinate': [96.62344, 639.164, 662.406, 693.54376]}, {'cls_id': 2, 'label': 'text', 'score': 0.9438745975494385, 'coordinate': [695.2748, 1138.9849, 1286.6161, 1188.8252]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9253508448600769, 'coordinate': [195.446, 425.10272, 567.61505, 452.4903]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9198653101921082, 'coordinate': [853.006, 908.8241, 1132.3086, 933.7346]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9072502255439758, 'coordinate': [165.8695, 129.74162, 512.8529, 156.56209]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9058157205581665, 'coordinate': [1246.4065, 1078.4541, 1287.0457, 1104.9424]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9032580256462097, 'coordinate': [1247.0944, 1229.402, 1287.0751, 1255.4117]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9020437598228455, 'coordinate': [1247.0134, 908.5498, 1288.0088, 934.7777]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9014809131622314, 'coordinate': [1252.9968, 492.4516, 1294.864, 518.0589]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9007395505905151, 'coordinate': [95.99875, 236.36539, 296.8511, 266.53656]}, {'cls_id': 2, 'label': 'text', 'score': 0.899124801158905, 'coordinate': [725.15186, 395.68433, 1263.7633, 423.32642]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8968735337257385, 'coordinate': [1241.6221, 1473.1089, 1283.1995, 1498.9155]}, {'cls_id': 2, 'label': 'text', 'score': 0.891890823841095, 'coordinate': [696.93774, 1286.4127, 1083.1498, 1310.9156]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8909994959831238, 'coordinate': [1270.2683, 219.9446, 1300.0896, 246.45982]}, {'cls_id': 2, 'label': 'text', 'score': 0.8866966366767883, 'coordinate': [94.425, 1058.3392, 441.26416, 1082.0751]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8848764300346375, 'coordinate': [94.911865, 1319.9253, 263.36142, 1345.916]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8829801678657532, 'coordinate': [634.6257, 427.7292, 661.91315, 453.2417]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8760316967964172, 'coordinate': [630.8843, 939.32605, 658.2267, 965.29675]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8740068078041077, 'coordinate': [634.3623, 576.0576, 660.5403, 601.4944]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8715350031852722, 'coordinate': [633.26575, 730.2727, 660.3563, 755.6184]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8699929714202881, 'coordinate': [630.99963, 1001.1361, 657.9286, 1025.9573]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8555153012275696, 'coordinate': [1089.9109, 1598.5446, 1277.5623, 1622.1991]}, {'cls_id': 7, 'label': 'formula', 'score': 0.833438515663147, 'coordinate': [694.6742, 1611.7349, 861.1708, 1635.6787]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7923287749290466, 'coordinate': [365.27258, 268.35327, 515.08936, 296.99475]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7592059969902039, 'coordinate': [917.18024, 1618.9021, 1009.52045, 1640.4705]}, {'cls_id': 3, 'label': 'number', 'score': 0.7468197345733643, 'coordinate': [1297.7268, 5.963439, 1310.3665, 26.294968]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7404024004936218, 'coordinate': [538.54333, 479.8123, 662.3668, 508.62253]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7284969091415405, 'coordinate': [99.5916, 508.4211, 253.29228, 535.67163]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7198922634124756, 'coordinate': [1116.627, 1572.7815, 1191.6616, 1594.5166]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7081565260887146, 'coordinate': [244.82803, 162.53033, 313.66757, 187.39536]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5778979659080505, 'coordinate': [255.87213, 323.67505, 326.8396, 349.7248]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5756691098213196, 'coordinate': [695.4659, 1561.6521, 900.0931, 1585.8818]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5596396923065186, 'coordinate': [175.38367, 350.68616, 242.63516, 376.44427]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5586691498756409, 'coordinate': [787.9116, 349.50732, 812.71045, 370.09338]}, {'cls_id': 7, 'label': 'formula', 'score': 0.546517550945282, 'coordinate': [1262.5737, 314.87128, 1296.2644, 338.0655]}, {'cls_id': 7, 'label': 'formula', 'score': 0.541178822517395, 'coordinate': [774.1763, 595.4717, 801.0121, 618.29297]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5409464836120605, 'coordinate': [848.8351, 619.38025, 959.8961, 646.0126]}]}, 'formula_res_list': [{'rec_formula': '\\small\\begin{aligned}{\\psi_{0}(M)-\\psi(M,z)=}&{{}\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{ln}(10)}\\times}\\\\ {}&{{}\\int_{0}^{z}d z^{\\prime}\\frac{d t}{d z^{\\prime}}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}},}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([727.97864, 441.2702, 1221.3971, 570.22736],)}, {'rec_formula': '\\begin{aligned}{\\rho_{\\mathrm{BH}}}&{{}=\\int d M\\psi(M)M}\\\\ {}&{{}=\\frac{1-\\epsilon_{r}}{\\epsilon_{r}c^{2}}\\int_{0}^{\\infty}d z\\frac{d t}{d z}\\int d\\log_{10}L\\phi(L,z)L,}\\\\ \\end{aligned}', 'formula_region_id': 2, 'dt_polys': ([155.30962, 924.0272, 598.61615, 1036.6716],)}, {'rec_formula': '{\\frac{d n}{d\\sigma}}d\\sigma=\\psi_{*}\\left({\\frac{\\sigma}{\\sigma_{*}}}\\right)^{\\alpha}{\\frac{e^{-(\\sigma/\\sigma_{*})^{\\beta}}}{\\Gamma(\\alpha/\\beta)}}\\beta{\\frac{d\\sigma}{\\sigma}}.', 'formula_region_id': 3, 'dt_polys': ([811.00867, 1058.013, 1176.5062, 1118.1985],)}, {'rec_formula': '\\phi(L)\\equiv\\frac{d n}{d\\log_{10}L}=\\frac{\\phi_{*}}{(L/L_{*})^{\\gamma_{1}}+(L/L_{*})^{\\gamma_{2}}}.', 'formula_region_id': 4, 'dt_polys': ([776.84436, 208.44116, 1224.5082, 267.0984],)}, {'rec_formula': '\\psi_{0}(M)=\\int d\\sigma\\frac{p(\\log_{10}M|\\log_{10}\\sigma)}{M\\log(10)}\\frac{d n}{d\\sigma}(\\sigma),', 'formula_region_id': 5, 'dt_polys': ([756.9298, 1211.8248, 1190.2643, 1267.3693],)}, {'rec_formula': '\\small\\begin{aligned}{p(\\operatorname{log}_{10}}&{{}M|\\operatorname{log}_{10}\\sigma)=\\frac{1}{\\sqrt{2\\pi}\\epsilon_{0}}}\\\\ {}&{{}\\times\\operatorname{exp}\\left[-\\frac{1}{2}\\left(\\frac{\\operatorname{log}_{10}M-a_{\\bullet}-b_{\\bullet}\\operatorname{log}_{10}\\sigma}{\\epsilon_{0}}\\right)^{2}\\right].}\\\\ \\end{aligned}', 'formula_region_id': 6, 'dt_polys': ([723.22705, 1332.768, 1254.1936, 1469.2213],)}, {'rec_formula': '\\frac{\\partial\\psi}{\\partial t}(M,t)+\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\mathrm{E}}^{2}\\ln(10)}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\mathrm{v}}}=0,', 'formula_region_id': 7, 'dt_polys': ([116.53526, 714.33014, 613.72314, 773.89496],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle\\psi(M,t)=\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}c^{2}\\operatorname{ln}(10)}\\phi(L,t)\\frac{d L}{d M}.', 'formula_region_id': 8, 'dt_polys': ([165.14, 558.2904, 597.4905, 613.77295],)}, {'rec_formula': '\\small\\begin{aligned}{\\phi(L,t)d\\operatorname{log}_{10}L=\\delta(M,t)\\psi(M,t)d M.}\\\\ \\end{aligned}', 'formula_region_id': 9, 'dt_polys': ([195.446, 425.10272, 567.61505, 452.4903],)}, {'rec_formula': '\\log_{10}M=a_{\\bullet}+b_{\\bullet}\\log_{10}X.', 'formula_region_id': 10, 'dt_polys': ([853.006, 908.8241, 1132.3086, 933.7346],)}, {'rec_formula': 't_{E}\\,=\\,\\sigma_{T}c/4\\pi G m_{v}\\,=\\,4.5\\times10^{8}\\mathrm{yr}', 'formula_region_id': 11, 'dt_polys': ([165.8695, 129.74162, 512.8529, 156.56209],)}, {'rec_formula': '\\dot{M}\\:=\\:(1\\:-\\:\\epsilon_{r})\\dot{M}_{\\mathrm{acc}}', 'formula_region_id': 12, 'dt_polys': ([95.99875, 236.36539, 296.8511, 266.53656],)}, {'rec_formula': 'M_{*}=L_{*}t_{E}/\\breve{\\lambda}c^{2}', 'formula_region_id': 13, 'dt_polys': ([94.911865, 1319.9253, 263.36142, 1345.916],)}, {'rec_formula': 'a_{\\bullet}\\,=\\,8.32\\pm0.05', 'formula_region_id': 14, 'dt_polys': ([1089.9109, 1598.5446, 1277.5623, 1622.1991],)}, {'rec_formula': 'b_{\\bullet}=5.64\\,\\dot{\\pm\\,0.32}', 'formula_region_id': 15, 'dt_polys': ([694.6742, 1611.7349, 861.1708, 1635.6787],)}, {'rec_formula': '\\phi(L,t)d\\operatorname{log}_{10}L', 'formula_region_id': 16, 'dt_polys': ([365.27258, 268.35327, 515.08936, 296.99475],)}, {'rec_formula': '\\epsilon_{0}=0.38', 'formula_region_id': 17, 'dt_polys': ([917.18024, 1618.9021, 1009.52045, 1640.4705],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle=', 'formula_region_id': 18, 'dt_polys': ([538.54333, 479.8123, 662.3668, 508.62253],)}, {'rec_formula': '\\delta(M,t)\\dot{M}(M,t)', 'formula_region_id': 19, 'dt_polys': ([99.5916, 508.4211, 253.29228, 535.67163],)}, {'rec_formula': 'M\\mathrm{~-~}\\sigma', 'formula_region_id': 20, 'dt_polys': ([1116.627, 1572.7815, 1191.6616, 1594.5166],)}, {'rec_formula': '\\epsilon_{r}\\dot{M}_{\\mathrm{acc}}', 'formula_region_id': 21, 'dt_polys': ([244.82803, 162.53033, 313.66757, 187.39536],)}, {'rec_formula': '\\delta(M,t)', 'formula_region_id': 22, 'dt_polys': ([255.87213, 323.67505, 326.8396, 349.7248],)}, {'rec_formula': 'X\\:=\\:\\sigma/200\\mathrm{km}\\:\\:\\mathrm{s}^{-1}', 'formula_region_id': 23, 'dt_polys': ([695.4659, 1561.6521, 900.0931, 1585.8818],)}, {'rec_formula': '\\phi(L,t)', 'formula_region_id': 24, 'dt_polys': ([175.38367, 350.68616, 242.63516, 376.44427],)}, {'rec_formula': '\\gamma_{2}', 'formula_region_id': 25, 'dt_polys': ([787.9116, 349.50732, 812.71045, 370.09338],)}, {'rec_formula': 'L_{*}.', 'formula_region_id': 26, 'dt_polys': ([1262.5737, 314.87128, 1296.2644, 338.0655],)}, {'rec_formula': '\\psi_{0}', 'formula_region_id': 27, 'dt_polys': ([774.1763, 595.4717, 801.0121, 618.29297],)}, {'rec_formula': 'z,\\ \\psi(M,z)', 'formula_region_id': 28, 'dt_polys': ([848.8351, 619.38025, 959.8961, 646.0126],)}]}}
```

运行结果参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，其中公式识别的可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04.png" style="width: 70%"/>

<b> 如果您需要对公式识别产线进行可视化，需要运行如下命令来对LaTeX渲染环境进行安装。目前公式识别产线可视化只支持Ubuntu环境，其他环境暂不支持。对于复杂公式，LaTeX 结果可能包含部分高级的表示，Markdown等环境中未必可以成功显示：</b>

```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-latex-extra -y
```

<b>备注</b>： 由于公式识别可视化过程中需要对每张公式图片进行渲染，因此耗时较长，请您耐心等待。

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddleocr import FormulaRecognitionPipeline

pipeline = FormulaRecognitionPipeline()
# ocr = FormulaRecognitionPipeline(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# ocr = FormulaRecognitionPipeline(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# ocr = FormulaRecognitionPipeline(device="gpu") # 通过 device 指定模型推理时使用 GPU
output = pipeline.predict("./general_formula_recognition_001.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `FormulaRecognitionPipeline()` 实例化公式识别产线对象，具体参数说明如下：

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
<td><code>doc_orientation_classify_batch_size</code></td>
<td>文档方向分类模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
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
<td><code>doc_unwarping_batch_size</code></td>
<td>文本图像矫正模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>版面区域检测模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面区域检测的阈值，用于过滤掉低置信度预测结果的阈值。
<ul>
<li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
<li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面区域检测是否使用NMS后处理，过滤重叠框。如果设置为<code>None</code>, 将会使用官方模型配置。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测中检测框的边长缩放倍数。
<ul>
<li><b>float</b>, 大于0的浮点数，如 1.1 , 表示将模型输出的检测框中心不变，宽和高都扩张1.1倍</li>
<li><b>列表</b>, 如 [1.2, 1.5] , 表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测中模型输出的检测框的合并处理模式。
<ul>
<li><b>large</b>, 设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>small</b>, 设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>union</b>, 不进行框的过滤处理，内外框都保留</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_batch_size</code></td>
<td>版面区域检测模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
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
<td>是否加载版面区域检测模块。如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>公式识别模型的名称。如果设置为<code>None</code>, 将会使用产线默认模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>公式识别模型的目录路径。如果设置为<code>None</code>, 将会下载官方模型。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>公式识别模型的批处理大小。如果设置为 <code>None</code>, 将默认设置批处理大小为<code>1</code>。</td>
<td><code>int</code></td>
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
</tbody>
</table>

（2）调用 公式识别产线对象的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。

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
<tbody>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td><code>None</code></td>
<tr>
<td><code>device</code></td>
<td>与实例化时的参数相同。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否在推理时使用文档区域检测模块。</td>
<td><code>bool</code></td>
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
<td><code>float|list</code></td>
<td><code>None</code></td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>与实例化时的参数相同。</td>
<td><code>string</code></td>
<td><code>None</code></td>
</tr>
</tr></tr></tbody>
</table>

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
        - `use_layout_detection`: `(bool)` 控制是否启用版面区域检测模块

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` 文档预处理子产线的输出结果。仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(Union[str, None])` 图像预处理子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `model_settings`: `(Dict)` 预处理子产线的模型配置参数
            - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档方向分类
            - `use_doc_unwarping`: `(bool)` 控制是否启用文本图像矫正
        - `angle`: `(int)` 文档方向分类的预测结果。启用时取值为[0,1,2,3]，分别对应[0°,90°,180°,270°]；未启用时为-1
    - `layout_det_res`: `(Dict[str, List[Dict]])` 版面区域检测模块的输出结果。仅当`use_layout_detection=True`时存在
        - `input_path`: `(Union[str, None])` 版面区域检测模块接收的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `boxes`: `(List[Dict[int, str, float, List[float]]])` 版面区域检测预测结果列表
            - `cls_id`: `(int)` 版面区域检测预测的类别id
            - `label`: `(str)` 版面区域检测预测的类别
            - `score`: `(float)` 版面区域检测预测的类别置信度分数
            - `coordinate`: `(List[float])` 版面区域检测预测的边界框坐标，格式为[x_min, y_min, x_max, y_max]，其中(x_min, y_min)为左上角坐标，(x_max, y_max) 为右上角坐标
    - `formula_res_list`:  `(List[Dict[str, int, List[float]]])` 公式识别的预测结果列表
        - `rec_formula`: `(str)` 公式识别预测的LaTeX源码
        - `formula_region_id`: `(int)` 公式识别预测的id编号
        - `dt_polys`:  `(List[float])` 公式识别预测的边界框坐标，格式为[x_min, y_min, x_max, y_max]，其中(x_min, y_min)为左上角坐标，(x_max, y_max) 为右上角坐标

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_formula_res_img.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

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
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `preprocessed_img`、 `layout_det_res`和 `formula_res_img`，对应的值是三个 `Image.Image` 对象：第一个用于展示图像预处理的可视化图像，第二个用于展示版面区域检测的可视化图像，第三个用于展示公式识别的可视化图像。如果没有使用图像预处理子模块，则字典中不包含 `preprocessed_img`；如果没有使用版面区域检测子模块，则字典中不包含`layout_det_res`。

## 3. 开发集成/部署

如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleOCR 也提供了其他两种部署方式，详细说明如下：

🚀 高性能推理：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleOCR 提供高性能推理功能，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[高性能推理指南](../deployment/high_performance_inference.md)。

☁️ 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。详细的产线服务化部署流程请参考[服务化部署指南](../deployment/serving.md)。

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
<p>获取图像公式识别结果。</p>
<p><code>POST /formula-recognition</code></p>
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
<td><code>formulaRecResults</code></td>
<td><code>object</code></td>
<td>公式识别结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>formulaRecResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>inputImage</code> | <code>null</code></td>
<td><code>string</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/formula-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["formulaRecResults"]):
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

如果公式识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升公式识别产线的在您的场景中的识别效果。

......
