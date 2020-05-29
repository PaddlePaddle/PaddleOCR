# 如何生产自定义超轻量模型？

生产自定义的超轻量模型可分为三步：训练文本检测模型、训练文本识别模型、模型串联预测。

## step1：训练文本检测模型

PaddleOCR提供了EAST、DB两种文本检测算法，均支持MobileNetV3、ResNet50_vd两种骨干网络，根据需要选择相应的配置文件，启动训练。例如，训练使用MobileNetV3作为骨干网络的DB检测模型（即超轻量模型使用的配置）：
```
python3 tools/train.py -c configs/det/det_mv3_db.yml
```
更详细的数据准备和训练教程参考文档教程中[文本检测模型训练/评估/预测](./detection.md)。

## step2：训练文本检测模型

PaddleOCR提供了CRNN、Rosetta、STAR-Net、RARE四种文本识别算法，均支持MobileNetV3、ResNet34_vd两种骨干网络，根据需要选择相应的配置文件，启动训练。例如，训练使用MobileNetV3作为骨干网络的CRNN识别模型（即超轻量模型使用的配置）：
```
python3 tools/train.py -c configs/rec/rec_chinese_lite_train.yml
```
更详细的数据准备和训练教程参考文档教程中[文本识别模型训练/评估/预测](./recognition.md)。

## step3：模型串联预测

PaddleOCR提供了检测和识别模型的串联工具，可以将训练好的任一检测模型和任一识别模型串联成两阶段的文本识别系统。输入图像经过文本检测、检测框矫正、文本识别、得分过滤四个主要阶段输出文本位置和识别结果，同时可选择对结果进行可视化。

在执行预测时，需要通过参数image_dir指定单张图像或者图像集合的路径、参数det_model_dir指定检测inference模型的路径和参数rec_model_dir指定识别inference模型的路径。可视化识别结果默认保存到 ./inference_results 文件夹里面。

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"
```
更多的文本检测、识别串联推理使用方式请参考文档教程中的[基于预测引擎推理](./inference.md)。
