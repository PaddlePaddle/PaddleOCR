
# 基于预测引擎推理

inference 模型（fluid.io.save_inference_model保存的模型）
一般是模型训练完成后保存的固化模型，多用于预测部署。
训练过程中保存的模型是checkpoints模型，保存的是模型的参数，多用于恢复训练等。
与checkpoints模型相比，inference 模型会额外保存模型的结构信息，在预测部署、加速推理上性能优越，灵活方便，适合与实际系统集成。更详细的介绍请参考文档[分类预测框架](https://paddleclas.readthedocs.io/zh_CN/latest/extension/paddle_inference.html). 接下来将依次介绍文本检测、文本识别以及两者串联基于预测引擎推理。与此同时也会介绍checkpoints转换成inference model的实现。


## 文本检测模型推理

将文本检测模型训练过程中保存的模型，转换成inference model，可以使用如下命令：

```
python tools/export_model.py -c configs/det/det_db_mv3.yml -o Global.checkpoints="./output/best_accuracy" \
        Global.save_inference_dir="./inference/det/"
```

推理模型保存在$./inference/det/model$, $./inference/det/params$

使用保存的inference model实现在单张图像上的预测：

```
python  tools/infer/predict_det.py --image_dir="/demo.jpg" --det_model_dir="./inference/det/"
```


## 文本识别模型推理

将文本识别模型训练过程中保存的模型，转换成inference model，可以使用如下命令：

```
python tools/export_model.py -c configs/rec/rec_chinese_lite_train.yml -o Global.checkpoints="./output/best_accuracy" \
        Global.save_inference_dir="./inference/rec/"
```

推理模型保存在$./inference/rec/model$, $./inference/rec/params$

使用保存的inference model实现在单张图像上的预测：

```
python  tools/infer/predict_rec.py --image_dir="/demo.jpg" --rec_model_dir="./inference/rec/"
```

## 文本检测、识别串联推理

实现文本检测、识别串联推理，预测$image_dir$指定的单张图像：
```
python tools/infer/predict_eval.py --image_dir="/Demo.jpg" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"
```

实现文本检测、识别串联推理，预测$image_dir$指指定文件夹下的所有图像：

```
python tools/infer/predict_eval.py --image_dir="/test_imgs/" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"
```
