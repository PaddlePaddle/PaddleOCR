# SPTS

- [1. 算法简介](#1)
- [2. 环境配置](#2)
- [3. 模型训练、评估、预测](#3)
    - [3.1 训练](#3-1)
    - [3.2 评估](#3-2)
    - [3.3 预测](#3-3)
- [4. 推理部署](#4)
    - [4.1 Python推理](#4-1)
    - [4.2 C++推理](#4-2)
    - [4.3 Serving服务化部署](#4-3)
    - [4.4 更多推理部署](#4-4)
- [5. FAQ](#5)

<a name="1"></a>
## 1. 算法简介

论文信息：
> [SPTS: Single-Point Text Spotting](https://arxiv.org/abs/2112.07917)
> Dezhi Peng, Xinyu Wang, Yuliang Liu, Jiaxin Zhang, Mingxin Huang, Songxuan Lai, Shenggao Zhu, Jing Li, Dahua Lin, Chunhua Shen, Xiang Bai, Lianwen Jin
> ACM MM 2022 Oral

在icdar2015, SCUT-CTW1500, Total-Text文本检测公开数据集上，算法复现效果如下：

|数据集|模型|配置文件|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- | --- |
|icdar2015|SPTS|[configs/REC/rec_spts.yml](../../configs/rec/rec_spts.yml)|77.52%|46.32%|57.99%|[训练模型]()|
|SCUT-CTW1500|SPTS|[configs/REC/rec_spts.yml](../../configs/rec/rec_spts.yml)|84.68%|77.75%|81.07%|[训练模型]()|
|Total-Text|SPTS|[configs/REC/rec_spts.yml](../../configs/rec/rec_spts.yml)|86.09%|67.66%|75.77%|[训练模型]()|


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

SPTS模型使用文本检测公开数据集训练得到，数据集下载可参考[CurvedSynText150k(part1)](https://universityofadelaide.app.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p), [CurvedSynText150k(part2)](https://universityofadelaide.app.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs), [MLT](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m), [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector), [Icdar2013](https://rrc.cvc.uab.es/?ch=2), [Icdar2015](https://rrc.cvc.uab.es/?ch=4), [Total-Text-Dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset)。

请参考[文本检测训练教程](./detection.md)。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将CT文本检测训练过程中保存的模型，转换成inference model。以基于Resnet18_vd骨干网络，在Total-Text英文数据集训练的模型为例（ [模型下载地址]() )，可以使用如下命令进行转换：

```shell
python3 tools/export_model.py -c configs/rec/rec_spts.yml -o Global.pretrained_model=./output/pretrain/best_accuracy  Global.save_inference_dir=./output/inference/rec_spts
```

CT文本检测模型推理，可以执行如下命令：

```shell
python3 tools/infer/predict_det.py --image_dir="./output/inference/img_78.jpg" --rec_model_dir="./inference/SPTS/" --rec_algorithm="SPTS"
```
可视化文本检测结果默认保存到`./output/inference`文件夹里面。结果示例如下：

![]()


<a name="4-2"></a>
### 4.2 C++推理

暂不支持

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂不支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂不支持

<a name="5"></a>
## 5. FAQ


## 引用

```bibtex
@inproceedings{
    title={SPTS: Single-Point Text Spotting},
    author={Dezhi Peng, Xinyu Wang, Yuliang Liu, Jiaxin Zhang, Mingxin Huang, Songxuan Lai, Shenggao Zhu, Jing Li, Dahua Lin, Chunhua Shen, Xiang Bai, Lianwen Jin},
    booktitle={ACM MM 2022 Oral},
    year={2022}
}
```