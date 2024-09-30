# 文本图像矫正模块开发教程

## 一、概述
文本图像矫正的主要目的是针对图像进行几何变换，以纠正图像中的文档扭曲、倾斜、透视变形等问题，以供后续的文本识别模块进行更准确的识别。

## 二、支持模型列表

<details>
   <summary> 👉模型列表详情</summary>
|模型|MS-SSIM （%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|介绍|
|-|-|-|-|-|-|
|UVDoc|54.40|-|-|31|高精度文本图像矫正模型|


**模型的精度指标测量自**[DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html)。**GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。**

</details>

## 三、快速集成
在快速集成前，首先需要安装PaddleX的wheel包，wheel的安装方式请参考 [PaddleX本地安装教程](../installation/installation.md)。完成wheel包的安装后，几行代码即可完成文本检测模块的推理，可以任意切换该模块下的模型，您也可以将文本检测的模块中的模型推理集成到您的项目中。

```
from paddlex import create_model
model = create_model("UVDoc")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../instructions/model_python_API.md)。

## 四、二次开发
当前模块暂时不支持微调训练，仅支持推理集成。关于该模块的微调训练，计划在未来支持。
