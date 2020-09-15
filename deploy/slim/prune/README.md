> 运行示例前请先安装develop版本PaddleSlim

# 模型裁剪压缩教程

## 概述

该示例使用PaddleSlim提供的[裁剪压缩API](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)对OCR模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [OCR模型的常规训练方法](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)
- [PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)

## 安装PaddleSlim
可按照[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)中的步骤安装PaddleSlim。



## 敏感度分析训练

进入PaddleOCR根目录，通过以下命令对模型进行敏感度分析：

```bash
python deploy/slim/prune/sensitivity_anal.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1
```

## 裁剪模型与fine-tune

```bash
python deploy/slim/prune/pruning_and_finetune.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1
```



## 评估并导出

在得到裁剪训练保存的模型后，我们可以将其导出为inference_model，用于预测部署：

```bash
python deploy/slim/prune/export_prune_model.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./output/det_db/best_accuracy Global.test_batch_size_per_card=1 Global.save_inference_dir=inference_model
```
