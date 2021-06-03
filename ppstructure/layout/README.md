# Python端预测部署

Python预测可以使用`tools/infer.py`，此种方式依赖PaddleDetection源码；也可以使用本篇教程预测方式，先将模型导出，使用一个独立的文件进行预测。


本篇教程使用AnalysisPredictor对[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/EXPORT_MODEL.md)进行高性能预测。

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法, 预测引擎使用了AnalysisPredictor，专门针对推理进行了优化，是基于[C++预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)的Python接口，该引擎可以对模型进行多项图优化，减少不必要的内存拷贝。如果用户在部署已训练模型的过程中对性能有较高的要求，我们提供了独立于PaddleDetection的预测脚本，方便用户直接集成部署。


主要包含两个步骤：

- 导出预测模型
- 基于Python的预测

## 1. 导出预测模型

PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/EXPORT_MODEL.md)

导出后目录下，包括`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`四个文件。

## 2. 基于python的预测

### 2.1 安装依赖
  - `PaddlePaddle`的安装:
    请点击[官方安装文档](https://paddlepaddle.org.cn/install/quick) 选择适合的方式，版本为2.0rc1以上即可
  - 切换到`PaddleDetection`代码库根目录，执行`pip install -r requirements.txt`安装其它依赖

### 2.2 执行预测程序
在终端输入以下命令进行预测：

```bash
python deploy/python/infer.py --model_dir=/path/to/models --image_file=/path/to/image
--use_gpu=(False/True)
```

参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| --model_dir | Yes|上述导出的模型路径 |
| --image_file | Option |需要预测的图片 |
| --video_file | Option |需要预测的视频 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按`q`退出输出预测结果到：output/output.mp4|
| --use_gpu |No|是否GPU，默认为False|
| --run_mode |No|使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16/trt_int8）|
| --threshold |No|预测得分的阈值，默认为0.5|
| --output_dir |No|可视化结果保存的根目录，默认为output/|
| --run_benchmark |No|是否运行benchmark，同时需指定--image_file|

说明：

- run_mode：fluid代表使用AnalysisPredictor，精度float32来推理，其他参数指用AnalysisPredictor，TensorRT不同精度来推理。
- PaddlePaddle默认的GPU安装包(<=1.7)，不支持基于TensorRT进行预测，如果想基于TensorRT加速预测，需要自行编译，详细可参考[预测库编译教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/paddle_tensorrt_infer.html)。

## 3. 部署性能对比测试
对比AnalysisPredictor相对Executor的推理速度

### 3.1 测试环境:

- CUDA 9.0
- CUDNN 7.5
- PaddlePaddle 1.71
- GPU: Tesla P40

### 3.2 测试方式:

- Batch Size=1
- 去掉前100轮warmup时间，测试100轮的平均时间，单位ms/image，只计算模型运行时间，不包括数据的处理和拷贝。


### 3.3 测试结果

|模型 | AnalysisPredictor | Executor | 输入|
|---|----|---|---|
| YOLOv3-MobileNetv1 | 15.20 | 19.54 |  608*608
| faster_rcnn_r50_fpn_1x | 50.05 | 69.58 |800*1088
| faster_rcnn_r50_1x | 326.11 | 347.22 | 800*1067
| mask_rcnn_r50_fpn_1x | 67.49 | 91.02 | 800*1088
| mask_rcnn_r50_1x | 326.11 | 350.94 | 800*1067
