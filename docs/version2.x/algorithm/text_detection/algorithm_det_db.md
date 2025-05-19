---
typora-copy-images-to: images
comments: true
---

# DB与DB++

## 1. 算法简介

论文信息：
> [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
> Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang
> AAAI, 2020

> [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)
> Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang
> TPAMI, 2022

在ICDAR2015文本检测公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- | --- |
|DB|ResNet50_vd|[configs/det/det_r50_vd_db.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_vd_db.yml)|86.41%|78.72%|82.38%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|[configs/det/det_mv3_db.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_mv3_db.yml)|77.29%|73.08%|75.12%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|DB++|ResNet50|[configs/det/det_r50_db++_icdar15.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_db++_icdar15.yml)|90.89%|82.66%|86.58%|[合成数据预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_icdar15_train.tar)|

在TD_TR文本检测公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- | --- |
|DB++|ResNet50|[configs/det/det_r50_db++_td_tr.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_db++_td_tr.yml)|92.92%|86.48%|89.58%|[合成数据预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_td_tr_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

请参考[文本检测训练教程](../../ppocr/model_train/detection.md)。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。

## 4. 推理部署

### 4.1 Python推理

首先将DB文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例（ [模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar) )，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/det/det_r50_vd_db.yml -o Global.pretrained_model=./det_r50_vd_db_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_db
```

DB文本检测模型推理，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_db/" --det_algorithm="DB"
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为`det_res`。结果示例如下：

![img](./images/det_res_img_10_db.jpg)

**注意**：由于ICDAR2015数据集只有1000张训练图像，且主要针对英文场景，所以上述模型对中文文本图像检测效果会比较差。

### 4.2 C++推理

准备好推理模型后，参考[cpp infer](../../ppocr/infer_deploy/cpp_infer.md)教程进行操作即可。

### 4.3 Serving服务化部署

准备好推理模型后，参考[pdserving](../../ppocr/infer_deploy/paddle_server.md)教程进行Serving服务化部署，包括Python Serving和C++ Serving两种模式。

### 4.4 更多推理部署

DB模型还支持以下推理部署方式：

- Paddle2ONNX推理：准备好推理模型后，参考[paddle2onnx](../../ppocr/infer_deploy/paddle2onnx.md)教程操作。

## 5. FAQ

## 引用

```bibtex
@inproceedings{liao2020real,
  title={Real-time scene text detection with differentiable binarization},
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11474--11481},
  year={2020}
}

@article{liao2022real,
  title={Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion},
  author={Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
