---
typora-copy-images-to: images
comments: true
---

# FCENet

## 1. 算法简介

论文信息：
> [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)
> Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang
> CVPR, 2021

在CTW1500文本检测公开数据集上，算法复现效果如下：

| 模型  |骨干网络|配置文件|precision|recall|Hmean|下载链接|
|-----| --- | --- | --- | --- | --- | --- |
| FCE | ResNet50_dcn | [configs/det/det_r50_vd_dcn_fce_ctw.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_vd_dcn_fce_ctw.yml)| 88.39%|82.18%|85.27%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

上述FCE模型使用CTW1500文本检测公开数据集训练得到，数据集下载可参考 [ocr_datasets](../../datasets/ocr_datasets.md)。

数据下载完成后，请参考[文本检测训练教程](../../ppocr/model_train/detection.md)进行训练。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。

## 4. 推理部署

### 4.1 Python推理

首先将FCE文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd_dcn骨干网络，在CTW1500英文数据集训练的模型为例（ [模型下载地址](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar) )，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/det/det_r50_vd_dcn_fce_ctw.yml -o Global.pretrained_model=./det_r50_dcn_fce_ctw_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_fce
```

FCE文本检测模型推理，执行非弯曲文本检测，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_fce/" --det_algorithm="FCE" --det_fce_box_type=quad
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![img](./images/det_res_img_10_fce.jpg)

如果想执行弯曲文本检测，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img623.jpg" --det_model_dir="./inference/det_fce/" --det_algorithm="FCE" --det_fce_box_type=poly
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![img](./images/det_res_img623_fce.jpg)

**注意**：由于CTW1500数据集只有1000张训练图像，且主要针对英文场景，所以上述模型对中文文本图像检测效果会比较差。

### 4.2 C++推理

由于后处理暂未使用CPP编写，FCE文本检测模型暂不支持CPP推理。

### 4.3 Serving服务化部署

暂未支持

### 4.4 更多推理部署

暂未支持

## 5. FAQ

## 引用

```bibtex
@InProceedings{zhu2021fourier,
  title={Fourier Contour Embedding for Arbitrary-Shaped Text Detection},
  author={Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang},
  year={2021},
  booktitle = {CVPR}
}
```
