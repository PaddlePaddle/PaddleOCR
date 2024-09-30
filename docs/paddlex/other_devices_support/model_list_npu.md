# PaddleX模型列表（昇腾 NPU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## 图像分类模块
|模型名称|Top1 Acc（%）|模型存储大小（M)|
|-|-|-|
|CLIP_vit_base_patch16_224|85.36|306.5 M|
|CLIP_vit_large_patch14_224|88.1|1.04 G|
|ConvNeXt_base_224|83.84|313.9 M|
|ConvNeXt_base_384|84.90|313.9 M|
|ConvNeXt_large_224|84.26|700.7 M|
|ConvNeXt_large_384|85.27|700.7 M|
|ConvNeXt_small|83.13|178.0 M|
|ConvNeXt_tiny|82.03|101.4 M|
|MobileNetV1_x0_75|68.8|9.3 M|
|MobileNetV1_x1_0|71.0|15.2 M|
|MobileNetV2_x0_5|65.0|7.1 M|
|MobileNetV2_x0_25|53.2|5.5 M|
|MobileNetV2_x1_0|72.2|12.6 M|
|MobileNetV2_x1_5|74.1|25.0 M|
|MobileNetV2_x2_0|75.2|41.2 M|
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
|PP-HGNet_base|85.0|249.4 M|
|PP-HGNet_small|81.51|86.5 M|
|PP-HGNet_tiny|79.83|52.4 M|
|PP-HGNetV2-B0|77.77|21.4 M|
|PP-HGNetV2-B1|79.18|22.6 M|
|PP-HGNetV2-B2|81.74|39.9 M|
|PP-HGNetV2-B3|82.98|57.9 M|
|PP-HGNetV2-B4|83.57|70.4 M|
|PP-HGNetV2-B5|84.75|140.8 M|
|PP-HGNetV2-B6|86.30|268.4 M|
|PP-LCNet_x0_5|63.14|6.7 M|
|PP-LCNet_x0_25|51.86|5.5 M|
|PP-LCNet_x0_35|58.09|5.9 M|
|PP-LCNet_x0_75|68.18|8.4 M|
|PP-LCNet_x1_0|71.32|10.5 M|
|PP-LCNet_x1_5|73.71|16.0 M|
|PP-LCNet_x2_0|75.18|23.2 M|
|PP-LCNet_x2_5|76.60|32.1 M|
|PP-LCNetV2_base|77.05|23.7 M|
|ResNet18_vd|72.3|41.5 M|
|ResNet18|71.0|41.5 M|
|ResNet34_vd|76.0|77.3 M|
|ResNet34|74.6|77.3 M|
|ResNet50_vd|79.1|90.8 M|
|ResNet50|76.5|90.8 M|
|ResNet101_vd|80.2|158.4 M|
|ResNet101|77.6|158.7 M|
|ResNet152_vd|80.6|214.3 M|
|ResNet152|78.3|214.2 M|
|ResNet200_vd|80.9|266.0 M|
|SwinTransformer_base_patch4_window7_224|83.37|310.5 M|
|SwinTransformer_small_patch4_window7_224|83.21|175.6 M|
|SwinTransformer_tiny_patch4_window7_224|81.10|100.1 M|

**注：以上精度指标为**[ImageNet-1k](https://www.image-net.org/index.php)**验证集 Top1 Acc。**

## 目标检测模块
|模型名称|mAP（%）|模型存储大小（M)|
|-|-|-|
|CenterNet-DLA-34|37.6|75.4 M|
|CenterNet-ResNet50|38.9|319.7 M|
|DETR-R50|42.3|159.3 M|
|FasterRCNN-ResNet34-FPN|37.8|137.5 M|
|FasterRCNN-ResNet50-FPN|38.4|148.1 M|
|FasterRCNN-ResNet50-vd-FPN|39.5|148.1 M|
|FasterRCNN-ResNet50-vd-SSLDv2-FPN|41.4|148.1 M|
|FasterRCNN-ResNet101-FPN|41.4|216.3 M|
|FCOS-ResNet50|39.6|124.2 M|
|PicoDet-L|42.6|20.9 M|
|PicoDet-M|37.5|16.8 M|
|PicoDet-S|29.1|4.4 M |
|PicoDet-XS|26.2|5.7M |
|PP-YOLOE_plus-L|52.9|185.3 M|
|PP-YOLOE_plus-M|49.8|83.2 M|
|PP-YOLOE_plus-S|43.7|28.3 M|
|PP-YOLOE_plus-X|54.7|349.4 M|
|RT-DETR-H|56.3|435.8 M|
|RT-DETR-L|53.0|113.7 M|
|RT-DETR-R18|46.5|70.7 M|
|RT-DETR-R50|53.1|149.1 M|
|RT-DETR-X|54.8|232.9 M|
|YOLOv3-DarkNet53|39.1|219.7 M|
|YOLOv3-MobileNetV3|31.4|83.8 M|
|YOLOv3-ResNet50_vd_DCN|40.6|163.0 M|

**注：以上精度指标为**[COCO2017](https://cocodataset.org/#home)**验证集 mAP(0.5:0.95)。**

## 语义分割模块
|模型名称|mloU（%）|模型存储大小（M)|
|-|-|-|
|Deeplabv3_Plus-R50 |80.36|94.9 M|
|Deeplabv3_Plus-R101|81.10|162.5 M|
|Deeplabv3-R50|79.90|138.3 M|
|Deeplabv3-R101|80.85|205.9 M|
|OCRNet_HRNet-W48|82.15|249.8 M|
|PP-LiteSeg-T|73.10|28.5 M|

**注：以上精度指标为**[Cityscapes](https://www.cityscapes-dataset.com/)**数据集 mloU。**

## 实例分割模块
|模型名称|Mask AP|模型存储大小（M)|
|-|-|-|
|Mask-RT-DETR-H|50.6|449.9|
|Mask-RT-DETR-L|45.7|113.6|
|Mask-RT-DETR-M|42.7|66.6 M|
|Cascade-MaskRCNN-ResNet50-FPN|36.3|254.8|
|Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN|39.1|254.7|
|PP-YOLOE_seg-S|32.5|31.5 M|

**注：以上精度指标为**[COCO2017](https://cocodataset.org/#home)**验证集 Mask AP(0.5:0.95)。**

## 文本检测模块
|模型名称|检测Hmean（%）|模型存储大小（M)|
|-|-|-|
|PP-OCRv4_mobile_det |77.79|4.2 M|
|PP-OCRv4_server_det |82.69|100.1M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。**

## 文本识别模块
|模型名称|识别Avg Accuracy(%)|模型存储大小（M)|
|-|-|-|
|PP-OCRv4_mobile_rec |78.20|10.6 M|
|PP-OCRv4_server_rec |79.20|71.2 M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。**

|模型名称|识别Avg Accuracy(%)|模型存储大小（M)|
|-|-|-|
|ch_SVTRv2_rec|68.81|73.9 M|

**注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)A榜。**

|模型名称|识别Avg Accuracy(%)|模型存储大小（M)|
|-|-|-|
|ch_RepSVTR_rec|65.07|22.1 M|

**注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)B榜。**

## 表格结构识别模块
|模型名称|精度（%）|模型存储大小（M)|
|-|-|-|
|SLANet|76.31|6.9 M |

**注：以上精度指标测量自PubtabNet英文表格识别数据集。**

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
|Nonstationary|0.600|0.515|55.5 M|
|PatchTST|0.385|0.397|2.0M |
|RLinear|0.384|0.392|40K|
|TiDE|0.405|0.412|31.7M|
|TimesNet|0.417|0.431|4.9M|

**注：以上精度指标测量自**[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)**数据集 ****（在测试集test.csv上的评测结果）****。**

## 时序异常检测模块
|模型名称|precison|recall|f1_score|模型存储大小（M)|
|-|-|-|-|-|
|AutoEncoder_ad|99.36|84.36|91.25|52K |
|DLinear_ad|98.98|93.96|96.41|112K|
|Nonstationary_ad|98.55|88.95|93.51|1.8M |
|PatchTST_ad|98.78|90.70|94.57|320K |
|TimesNet_ad|98.37|94.80|96.56|1.3M |

**注：以上精度指标测量自**[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)**数据集。**

## 时序分类模块
|模型名称|acc(%)|模型存储大小（M)|
|-|-|-|
|TimesNet_cls|87.5|792K|

**注：以上精度指标测量自UWaveGestureLibrary：[训练](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TRAIN.csv)、[评测](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv)数据集。**
