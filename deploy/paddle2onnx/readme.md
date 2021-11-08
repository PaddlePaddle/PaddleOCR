# paddle2onnx 模型转化与预测

本章节介绍 PaddleOCR 模型如何转化为 ONNX 模型，并基于 ONNX 引擎预测。

## 1. 环境准备

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境

###  Paddle2ONNX
paddle2onnx 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换。
更多细节可参考 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

- 安装 Paddle2ONNX
```
python3.7 -m pip install paddle2onnx
```

- 安装 ONNX
```
# 建议安装 1.4.0 版本，可根据环境更换版本号
python3.7 -m pip install onnxruntime==1.4.0
```
- Paddle 模型下载

有两种方式获取Paddle静态图模型：在 [model_list](../../doc/doc_ch/models_list.md) 中下载PaddleOCR提供的预测模型；
参考[模型导出说明](../../doc/doc_ch/inference.md#训练模型转inference模型)把训练好的权重转为 inference_model。

以 ppocr 检测模型为例：

```
wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && cd ..
```

## 2. 模型转换

使用 Paddle2ONNX 将Paddle静态图模型转换为ONNX模型格式：

```
paddle2onnx --model_dir=./inference/ch_ppocr_mobile_v2.0_det_infer/ \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--save_file=./inference/det_mobile_onnx/model.onnx \
--opset_version=10 \
--enable_onnx_checker=True
```

执行完毕后，ONNX模型会被保存在 `./inference/det_mobile_onnx/` 路径下

## 3. onnx 预测

以检测模型为例，使用onnx预测可执行如下命令：

```
python3.7 ../../tools/infer/predict_det.py --use_gpu=False --use_onnx=True \
--det_model_dir=./inference/det_mobile_onnx/model.onnx \
--image_dir=../../doc/imgs/1.jpg
```

执行命令后在终端会打印出预测的检测框坐标，并在 `./inference_results/` 下保存可视化结果。

```
root INFO: 1.jpg  [[[291, 295], [334, 292], [348, 844], [305, 847]], [[344, 296], [379, 294], [387, 669], [353, 671]]]
The predict time of ../../doc/imgs/1.jpg: 0.06162881851196289
The visualized image saved in ./inference_results/det_res_1.jpg
```

* 注意：ONNX暂时不支持变长预测，因为需要将输入resize到固定输入，预测结果可能与直接使用Paddle预测有细微不同。
