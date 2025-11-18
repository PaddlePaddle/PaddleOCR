---

comments: true

---


# 版面区域检测模块使用教程 


## 一、概述


版面区域检测任务的核心是对输入的文档图像进行内容解析和区域划分。通过识别图像中的不同元素（如文字、图表、图像、公式、段落、摘要、参考文献等），将其归类为预定义的类别，并确定这些区域在文档中的位置。


## 二、支持模型列表

> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

### 📊📊 版面检测模型（20类）
**评估说明：** 以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1300 张文档类型图片。

#### PP-DocLayout_plus-L
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 126.01 MB  
**模型介绍：**  
基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷极速pdf、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 83.2 | 53.03 | 634.62 |
| **高性能模式** | - | 17.23 | 极速pdf378.32 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-DocLayout_plus-L  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37910  )

---

### 📊📊 文档图像版面子模块检测（1类）
**评估说明：** 以上精度指标的评估集是自建的版面子区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1000 张文档类型图片。

#### PP-DocBlockLayout
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 123.92 MB  
**模型介绍：**  
基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的文档图像版面子模块检测模型。

**性能指标：**
| 指标名称 | mAP(0.极速pdf5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 95.9 | 34.60 | 506.43 |
| **高性能模式** | - | 28.54 | 256.83 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/极速pdfPP-D极速pdfocBlockLayout_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocBlockLayout  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-DocBlockLayout  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37911  )


---

### 📊📊 版面检测模型（23类）
**评估说明：** 以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、报纸、研报和试卷等 500 张文档类型图片。

#### PP-DocLayout-L
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 123.76 MB  
**模型介绍：**  
基于RT-DETR-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的高精度版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 90.4 | 33.59 | 503.01 |
| **高性能模式** | - | 33.59 | 251.08 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayout-L  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-DocLayout-L  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37912  )

---

#### PP-DocLayout-M
**模型类型：** 推理模型/训练模型 | **模型存储大小：极速pdf** 22.578 MB  
**模型介绍：**  
基于PicoDet-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的精度效率平衡的版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :极速pdf--- | :--- | :--- |
| **常规模式** | 75.2 | 13.03 | 43.39 |
| **高性能模式** | - | 4.72 | 24.44 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayout-M  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-DocLayout-M  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37913  )

---

#### PP-DocLayout-S
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 4.834 MB  
**模型介绍：**  
基于PicoDet-S在中英文论文、杂志、合同、书本、试卷和研报等场景上自建数据集训练的高效率版面区域定位极速pdf模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 70.9 | 11.54 | 18.53 |
| **高性能模式** | - | 3.86 | 6.29 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3极速pdf.0.0/PP-DocLayout-S_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos极速pdf.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayout-S  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-DocLayout-S  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37914  )


---

> ❗ 以上列出的是版面检测模块重点支持的<b>5个核心模型</b>，该模块总共支持<b>13个全量模型</b>，包含多个预定义了不同类别的模型，完整的模型列表如下：

<details>
<summary> 👉模型列表详情</summary>

### 📊📊 表格版面检测模型（1类）
**评估说明：** 适用于表格区域的检测任务

#### PicoDet_layout_1x_table
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 7.4 MB  
**模型介绍：**  
基于PicoDet-1极速pdfx在自建数据集训练的高效率版面区域定位模型，可定位表格这1类区域。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 97.5 | 9.57 | 27.66 |
| **高性能模式** | - | 6.63 | 16.75 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PicoDet_layout_1x_table  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PicoDet_layout_1x_table  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37915  )


---

### 📊📊 3类版面检测模型
**评估说明：** 包含表格、图像、印章三类区域检测

#### PicoDet-S_layout_3cls
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 4.8 MB  
**模型介绍：**  
基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 88.2 | 8.43 | 极速pdf17.60 |
| **高性能模式** | - | 3.44 | 6.51 |

**下载极速pdf链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar) | [训练模型](https://paddle-model-ecology.b极速pdfj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PicoDet-S_layout_3cls  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PicoDet-S_layout_3cls  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37916  )


---

#### PicoDet-L_layout_3cls
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 22.6 MB  
**模型介绍：**  
基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 89.0 | 12.80 | 45.04 |
| **高性能模式** | - | 9.57 | 23.86 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PicoDet-L_layout_3cls  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PicoDet-L_layout_3cls  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37917  )


---

#### RT-DETR-H_layout_3cls
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 470.1 MB  
**模型介绍：**  
基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 95.8 | 114.80 | 924.38 |
| **高性能模式** | - | 25.65 | 924.38 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.极速pdf0.0/RT-DETR-H_layout_3cls_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/RT-DETR-H_layout_3cls  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/RT-DETR-H_layout_3cls  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37918  )


---

### 📊📊 5类英文文档区域检测模型
**评估说明：** 包含文字、标题、表格、图片以及列表五类区域

#### PicoDet_layout_1x
**模型类型：** 推理模型/训练模型 | **极速pdf模型存储大小：** 7.4 MB  
**模型介绍：**  
基于PicoDet-1x在PubLayNet数据集训练的高效率英文文档版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms极速pdf) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式极速pdf** | 97.8 | 9.62 | 26.极速pdf96 |
| **高性能模式** | - | 6.75 | 12.77 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1极速pdfx_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PicoDet_layout_1x  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PicoDet_layout_1x  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37919  )




---

### 📊📊 17类区域检测模型
**评估说明：** 包含17个版面常见类别

#### PicoDet-S_layout_17cls
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 4.8 MB  
**模型介绍：**  
基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 87.4 | 8.80 | 17.51 |
| **高性能模式** | - | 3.62 | 6.35 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/padd极速pdflex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PicoDet-S_layout_17cls  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PicoDet-S_layout_17cls  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37920  )

---

#### PicoDet-L_layout_17cls
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 22.6 MB  
**模型介绍：**  
基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 89.0 | 12.60 | 43.70 |
| **高性能模式** | - | 10.27 | 24.42 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/PicoDet-L_layout_17cls  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PicoDet-L_layout_17cls  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37921  )


---

#### RT-DETR-H_layout_17cls
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 470.2 MB  
**模型介绍：**  
基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型。

**性能指标：**
| 指标名称 | mAP(0.5)(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 98.3 | 115.29 | 964.75 |
| **高性能模式** | - | 101.18 | 964.75 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls极速pdf_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams)

[Hugging Face](https://huggingface.co/PaddlePaddle/RT-DETR-H_layout_17cls  )

[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/RT-DETR-H_layout_17cls  )

[AI Studio](https://aistudio.baidu.com/modelsdetail/37922  )

</details>




## 三、快速开始

> ❗ 在快速开始前，请先安装 PaddleOCR 的 wheel 包，详细请参考 [安装教程](../installation.md)。

使用一行命令即可快速体验：

```bash
paddleocr layout_detection -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg
```

<b>注：</b>PaddleOCR 官方模型默认从 HuggingFace 获取，如运行环境访问 HuggingFace 不便，可通过环境变量修改模型源为 BOS：`PADDLE_PDX_MODEL_SOURCE="BOS"`，未来将支持更多主流模型源；

您也可以将版面区域检测模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg)到本地。

```python
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': {'input_path': 'layout.jpg', 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9870226979255676, 'coordinate': [34.101906, 349.85275, 358.59213, 611.0772]}, {'cls_id': 2, 'label': 'text', 'score': 0.9866003394126892, 'coordinate': [34.500324, 647.1585, 358.29367, 848.66797]}, {'cls_id': 2, 'label': 'text', 'score': 0.9846674203872681, 'coordinate': [385.71445, 497.40973, 711.2261, 697.84265]}, {'cls_id': 8, 'label': 'table', 'score': 0.984126091003418, 'coordinate': [73.76879, 105.94899, 321.95303, 298.84888]}, {'cls_id': 8, 'label': 'table', 'score': 0.9834211468696594, 'coordinate': [436.95642, 105.81531, 662.7168, 313.48462]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832247495651245, 'coordinate': [385.62787, 346.2288, 710.10095, 458.77127]}, {'cls_id': 2, 'label': 'text', 'score': 0.9816061854362488, 'coordinate': [385.7802, 735.1931, 710.56134, 849.9764]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9577341079711914, 'coordinate': [34.421448, 20.055151, 358.71283, 76.53663]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9505634307861328, 'coordinate': [385.72278, 20.053688, 711.29333, 74.92744]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9001723527908325, 'coordinate': [386.46344, 477.03488, 699.4023, 490.07474]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8845751285552979, 'coordinate': [35.413048, 627.73596, 185.58383, 640.52264]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8837394118309021, 'coordinate': [387.17603, 716.3423, 524.7841, 729.258]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8508939743041992, 'coordinate': [35.50064, 331.18445, 141.6444, 344.81097]}]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `boxes`：预测的目标框信息，一个字典列表。每个字典代表一个检出的目标，包含以下信息：
  - `cls_id`：类别ID，一个整数
  - `label`：类别标签，一个字符串
  - `score`：目标框置信度，一个浮点数
  - `coordinate`：目标框坐标，一个浮点数列表，格式为<code>[xmin, ymin, xmax, ymax]</code>

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/layout_det/layout_res_plus.jpg"/>


相关方法、参数等说明如下：

* `LayoutDetection`实例化目标检测模型（此处以`PP-DocLayout_plus-L`为例），具体说明如下：

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
<td><code>model_name</code></td>
<td>模型名称。如果设置为<code>None</code>，则使用<code>PP-DocLayout-L</code></td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。<br/>
<b>例如：</b><code>"cpu"</code>、<code>"gpu"</code>、<code>"npu"</code>、<code>"gpu:0"</code>、<code>"gpu:0,1"</code>。<br/>
如指定多个设备，将进行并行推理。<br/>
默认情况下，优先使用 GPU 0；若不可用则使用 CPU。
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
<td>当使用 Paddle Inference 的 TensorRT 子图引擎时设置的计算精度。<br/><b>可选项：</b><code>"fp32"</code>、<code>"fp16"</code>。</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。<br/>
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
<td>在 CPU 上推理时使用的线程数量。</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>输入图像大小。
<ul>
<li><b>int</b>：如<code>640</code>，表示将输入图像resize到640x640大小。</li>
<li><b>list</b>：如<code>[640, 512]</code>，表示将输入图像resize到宽为640、高为512。</li>
</ul>
</td>
<td><code>int|list|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值。
<ul>
<li><b>float</b>：如<code>0.2</code>，表示过滤掉所有阈值小于0.2的目标框。</li>
<li><b>dict</b>：字典的键为<code>int</code>类型，代表类别ID；值为<code>float</code>类型阈值。如<code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对ID为0的类别应用阈值0.45、ID为1的类别应用阈值0.48、ID为7的类别应用阈值0.4。</li>
<li><b>None</b>：使用模型默认的配置。</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>是否使用NMS后处理，过滤重叠框。
<ul>
<li><b>bool</b>表示使用/不使用NMS进行检测框的后处理过滤重叠框。</li>
<li><b>None</b>使用模型默认的配置。</li>
</ul>
</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>检测框的边长缩放倍数。</b>
<ul>
<li><b>float</b>：大于0的浮点数，如<code>1.1</code>，表示将模型输出的检测框中心不变，宽和高都扩张1.1倍。</li>
<li><b>list</b>：如<code>[1.2, 1.5]</code>，表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍。</li>
<li><b>dict</b>：字典的键为<code>int</code>类型，代表类别ID；值为<code>tuple</code>类型，如<code>{0: (1.1, 2.0)}</code>，表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍。</li>
<li><b>None</b>：使用模型默认的配置。</li>
</ul>
</td>
<td><code>float|list|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>模型输出的检测框的合并处理模式。
<ul>
<li><b>"large"</b>：设置为<code>"large"</code>，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>"small"</b>：设置为<code>"small"</code>，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>"union"</b>：不进行框的过滤处理，内外框都保留。</li>
<li><b>dict</b>：字典的键为<code>int</code>类型，代表类别ID；值为<code>str</code>类型, 如<code>{0: "large", 2: "small"}</code>, 表示对第0类别检测框使用<code>large</code>模式，对第2类别检测框使用<code>small</code>。</li>
<li><b>None</b>：使用模型默认的配置。</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td>None</td>
</tr>
</tbody>
</table>


* 调用目标检测模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input`、`batch_size`和`threshold`，具体说明如下：
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
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>list</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小，可设置为任意正整数。</td>
<td><code>int</code></td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</b>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。
<td><code>float|list|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。
</td>
<td><code>str|dict|None</code></td>
<td>None</td>
</tr>
</tbody>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

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
<td rowspan="1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">获取格式为<code>dict</code>的可视化图像</td>
</tr>
</table>


## 四、二次开发

由于 PaddleOCR 并不直接提供版面区域检测模块的训练，因此，如果需要训练版面区域测模型，可以参考 [PaddleX 版面区域检测模块二次开发](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html#_5)部分进行训练。训练后的模型可以无缝集成到 PaddleOCR 的 API 中进行推理。

## 五、FAQ
