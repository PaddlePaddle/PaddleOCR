> 运行示例前请先安装1.2.0或更高版本PaddleSlim

# 模型量化压缩教程

## 概述

该示例使用PaddleSlim提供的[量化压缩API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)对检测模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [OCR模型的常规训练方法](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)
- [PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)

## 安装PaddleSlim
可按照[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)中的步骤安装PaddleSlim。



## 量化训练

进入PaddleOCR根目录，通过以下命令对模型进行量化：

```bash
python deploy/slim/quantization/quant.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=det_mv3_db/best_accuracy Global.save_model_dir=./output/quant_model
```



## 评估并导出

在得到量化训练保存的模型后，我们可以将其导出为inference_model，用于预测部署：

```bash
python deploy/slim/quantization/export_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_model_dir=./output/quant_model
```
