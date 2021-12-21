# 如何训练自定义超轻量模型？

可分为通过下面三步训练自定义超轻量模型：文本检测模型训练、文本识别模型训练、模型预测。

## step1：文本检测模型训练

PaddleOCR提供了EAST、DB两种文本检测算法，均支持MobileNetV3、ResNet50_vd两种骨干网络，选择相应的配置文件开始训练。例如，使用MobileNetV3作为骨干网络DB检测模型来训练（即超轻量模型使用的配置）：
```
python3 tools/train.py -c configs/det/det_mv3_db.yml 2>&1 | tee det_db.log
```
详细教程见[文本检测模型训练/评估/预测](./detection.md)。

## step2：文本识别模型训练

PaddleOCR提供了CRNN、Rosetta、STAR-Net、RARE四种文本识别算法，均支持MobileNetV3、ResNet34_vd两种骨干网络，选择相应的配置文件开始训练。例如，使用MobileNetV3作为骨干网络CRNN识别模型来训练（即超轻量模型使用的配置）：
```
python3 tools/train.py -c configs/rec/rec_chinese_lite_train.yml 2>&1 | tee rec_ch_lite.log
```
详细教程见[文本识别模型训练/评估/预测](./recognition.md)。

## step3：模型预测

PaddleOCR提供了一个使用文本检测模型和文本识别模型进行预测文本识别系统。输入图像经过文本检测、检测框矫正、文本识别、置信度过滤处理，输出文本位置和文本内容，同时提供识别结果可视化。

参数：

参数image_dir为单张图像或者图像集合的路径；
参数det_model_dir为检测inference模型的路径；
参数rec_model_dir为识别inference模型的路径。

结果：

可视化识别结果默认保存到 ./inference_results 文件夹里面。

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"
```
详细教程见[基于预测引擎推理](./inference.md)。
