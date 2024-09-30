# PaddleX模型列表（昆仑 XPU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## 图像分类模块
|模型名称|Top1 Acc（%）|模型存储大小（M)|
|-|-|-|
|MobileNetV3_large_x0_5|69.2|9.6 M|
|MobileNetV3_large_x0_35|64.3|7.5 M|
|MobileNetV3_large_x0_75|73.1|14.0 M|
|MobileNetV3_large_x1_0|75.3|19.5 M|
|MobileNetV3_large_x1_25|76.4|26.5 M|
|MobileNetV3_small_x0_5|59.2|6.8 M|
|MobileNetV3_small_x0_35|53.0|6.0 M|
|MobileNetV3_small_x0_75|66.0|8.5 M|
|MobileNetV3_small_x1_0|68.2|10.5 M|
|MobileNetV3_small_x1_25|70.7|13.0 M|
|PP-HGNet_small|81.51|86.5 M|
|PP-LCNet_x0_5|63.14|6.7 M|
|PP-LCNet_x0_25|51.86|5.5 M|
|PP-LCNet_x0_35|58.09|5.9 M|
|PP-LCNet_x0_75|68.18|8.4 M|
|PP-LCNet_x1_0|71.32|10.5 M|
|PP-LCNet_x1_5|73.71|16.0 M|
|PP-LCNet_x2_0|75.18|23.2 M|
|PP-LCNet_x2_5|76.60|32.1 M|
|ResNet18|71.0|41.5 M|
|ResNet34|74.6|77.3 M|
|ResNet50|76.5|90.8 M|
|ResNet101|77.6|158.7 M|
|ResNet152|78.3|214.2 M|

**注：以上精度指标为**[ImageNet-1k](https://www.image-net.org/index.php)**验证集 Top1 Acc。**

## 目标检测模块
|模型名称|mAP（%）|模型存储大小（M)|
|-|-|-|
|PicoDet-L|42.6|20.9 M|
|PicoDet-S|29.1|4.4 M |
|PP-YOLOE_plus-L|52.9|185.3 M|
|PP-YOLOE_plus-M|49.8|83.2 M|
|PP-YOLOE_plus-S|43.7|28.3 M|
|PP-YOLOE_plus-X|54.7|349.4 M|

**注：以上精度指标为**[COCO2017](https://cocodataset.org/#home)**验证集 mAP(0.5:0.95)。**

## 语义分割模块
|模型名称|mloU（%）|模型存储大小（M)|
|-|-|-|
|PP-LiteSeg-T|73.10|28.5 M|

**注：以上精度指标为**[Cityscapes](https://www.cityscapes-dataset.com/)**数据集 mloU。**

## 文本检测模块
|模型名称|检测Hmean（%）|模型存储大小（M)|
|-|-|-|
|PP-OCRv4_mobile_det|77.79|4.2 M|
|PP-OCRv4_server_det ）|82.69|100.1M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。**

## 文本识别模块
|模型名称|识别Avg Accuracy(%)|模型存储大小（M)|
|-|-|-|
|PP-OCRv4_mobile_rec|78.20|10.6 M|
|PP-OCRv4_server_rec|79.20|71.2 M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。**

## 版面区域分析模块
|模型名称|mAP（%）|模型存储大小（M)|
|-|-|-|
|PicoDet_layout_1x|86.8|7.4M |

**注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含 1w 张图片。**

## 时序预测模块
|模型名称|mse|mae|模型存储大小（M)|
|-|-|-|-|
|DLinear|0.382|0.394|72K|
|NLinear|0.386|0.392|40K |
|RLinear|0.384|0.392|40K|

**注：以上精度指标测量自**[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)**数据集 ****（在测试集test.csv上的评测结果）****。**