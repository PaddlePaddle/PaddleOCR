# PaddleX模型列表（海光 DCU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## 图像分类模块
|模型名称|Top1 Acc（%）|模型存储大小（M)|
|-|-|-|
|ResNet18|71.0|41.5 M|
|ResNet34|74.6|77.3 M|
|ResNet50|76.5|90.8 M|
|ResNet101|77.6|158.7 M|
|ResNet152|78.3|214.2 M|

**注：以上精度指标为**[ImageNet-1k](https://www.image-net.org/index.php)**验证集 Top1 Acc。**

## 语义分割模块
|模型名称|mloU（%）|模型存储大小（M)|
|-|-|-|
|Deeplabv3_Plus-R50 |80.36|94.9 M|
|Deeplabv3_Plus-R101|81.10|162.5 M|

**注：以上精度指标为**[Cityscapes](https://www.cityscapes-dataset.com/)**数据集 mloU。**