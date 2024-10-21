---
typora-copy-images-to: images
comments: true
---

# Paddle2ONNX模型转化与预测

本章节介绍 PaddleOCR 模型如何转化为 ONNX 模型，并基于 ONNXRuntime 引擎预测。

## 1. 环境准备

需要准备 PaddleOCR、Paddle2ONNX 模型转化环境，和 ONNXRuntime 预测环境

### PaddleOCR

克隆PaddleOCR的仓库，使用 main 分支，并进行安装，由于 PaddleOCR 仓库比较大，git clone 速度比较慢，所以本教程已下载

```bash linenums="1"
git clone  -b main https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && python3 -m pip install -e .
```

### Paddle2ONNX

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~18，部分Paddle算子支持更低的ONNX Opset转换。
更多细节可参考 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

- 安装 Paddle2ONNX

  ```bash linenums="1"
  python3 -m pip install paddle2onnx
  ```

- 安装 ONNXRuntime

  ```bash linenums="1"
  python3 -m pip install onnxruntime
  ```

## 2. 模型转换

### Paddle 模型下载

有两种方式获取Paddle静态图模型：在 [model_list](../model_list.md) 中下载PaddleOCR提供的预测模型；

以 PP-OCRv3 中文检测、识别、分类模型为例：

```bash linenums="1"
wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
cd ./inference && tar xf ch_PP-OCRv3_rec_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..
```

### 模型转换

使用 Paddle2ONNX 将Paddle静态图模型转换为ONNX模型格式：

```bash linenums="1"
paddle2onnx --model_dir ./inference/ch_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/rec_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_ppocr_mobile_v2.0_cls_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/cls_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True
```

执行完毕后，ONNX 模型会被分别保存在 `./inference/det_onnx/`，`./inference/rec_onnx/`，`./inference/cls_onnx/`路径下

- 注意：对于OCR模型，转化过程中必须采用动态shape的形式，否则预测结果可能与直接使用Paddle预测有细微不同。
  另外，以下几个模型暂不支持转换为 ONNX 模型：
  NRTR、SAR、RARE、SRN

- 注意：[当前Paddle2ONNX版本(v1.2.3)](https://github.com/PaddlePaddle/Paddle2ONNX/releases/tag/v1.2.3)现已默认支持动态shape，即 `float32[p2o.DynamicDimension.0,3,p2o.DynamicDimension.1,p2o.DynamicDimension.2]`，选项 `--input_shape_dict` 已废弃。如果有shape调整需求可使用如下命令进行Paddle模型输入shape调整。

  ```bash linenums="1"
  python3 -m paddle2onnx.optimize --input_model inference/det_onnx/model.onnx \
    --output_model inference/det_onnx/model.onnx \
    --input_shape_dict "{'x': [-1,3,-1,-1]}"
  ```

如你对导出的 ONNX 模型有优化的需求，推荐使用 `onnxslim` 对模型进行优化:
```bash linenums="1"
pip install onnxslim
onnxslim model.onnx slim.onnx
```

## 3. 推理预测

以中文OCR模型为例，使用 ONNXRuntime 预测可执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_system.py --use_gpu=False --use_onnx=True \
--det_model_dir=./inference/det_onnx/model.onnx  \
--rec_model_dir=./inference/rec_onnx/model.onnx  \
--cls_model_dir=./inference/cls_onnx/model.onnx  \
--image_dir=./deploy/lite/imgs/lite_demo.png
```

以中文OCR模型为例，使用 Paddle Inference 预测可执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_system.py --use_gpu=False \
--cls_model_dir=./inference/ch_ppocr_mobile_v2.0_cls_infer \
--rec_model_dir=./inference/ch_PP-OCRv3_rec_infer \
--det_model_dir=./inference/ch_PP-OCRv3_det_infer \
--image_dir=./deploy/lite/imgs/lite_demo.png
```

执行命令后在终端会打印出预测的识别信息，并在 `./inference_results/` 下保存可视化结果。

ONNXRuntime 执行效果：

![](./images/lite_demo_onnx.png)

Paddle Inference 执行效果：

![](./images/lite_demo_paddle.png)

使用 ONNXRuntime 预测，终端输出：

```bash linenums="1"
[2022/02/22 17:48:27] root DEBUG: dt_boxes num : 38, elapse : 0.043187856674194336
[2022/02/22 17:48:27] root DEBUG: rec_res num  : 38, elapse : 0.592170000076294
[2022/02/22 17:48:27] root DEBUG: 0  Predict time of ./deploy/lite/imgs/lite_demo.png: 0.642s
[2022/02/22 17:48:27] root DEBUG: The, 0.984
[2022/02/22 17:48:27] root DEBUG: visualized, 0.882
[2022/02/22 17:48:27] root DEBUG: etect18片, 0.720
[2022/02/22 17:48:27] root DEBUG: image saved in./vis.jpg, 0.947
[2022/02/22 17:48:27] root DEBUG: 纯臻营养护发素0.993604, 0.996
[2022/02/22 17:48:27] root DEBUG: 产品信息/参数, 0.922
[2022/02/22 17:48:27] root DEBUG: 0.992728, 0.914
[2022/02/22 17:48:27] root DEBUG: （45元／每公斤，100公斤起订）, 0.926
[2022/02/22 17:48:27] root DEBUG: 0.97417, 0.977
[2022/02/22 17:48:27] root DEBUG: 每瓶22元，1000瓶起订）0.993976, 0.962
[2022/02/22 17:48:27] root DEBUG: 【品牌】：代加工方式/0EMODM, 0.945
[2022/02/22 17:48:27] root DEBUG: 0.985133, 0.980
[2022/02/22 17:48:27] root DEBUG: 【品名】：纯臻营养护发素, 0.921
[2022/02/22 17:48:27] root DEBUG: 0.995007, 0.883
[2022/02/22 17:48:27] root DEBUG: 【产品编号】：YM-X-30110.96899, 0.955
[2022/02/22 17:48:27] root DEBUG: 【净含量】：220ml, 0.943
[2022/02/22 17:48:27] root DEBUG: Q.996577, 0.932
[2022/02/22 17:48:27] root DEBUG: 【适用人群】：适合所有肤质, 0.913
[2022/02/22 17:48:27] root DEBUG: 0.995842, 0.969
[2022/02/22 17:48:27] root DEBUG: 【主要成分】：鲸蜡硬脂醇、燕麦B-葡聚, 0.883
[2022/02/22 17:48:27] root DEBUG: 0.961928, 0.964
[2022/02/22 17:48:27] root DEBUG: 10, 0.812
[2022/02/22 17:48:27] root DEBUG: 糖、椰油酰胺丙基甜菜碱、泛醒, 0.866
[2022/02/22 17:48:27] root DEBUG: 0.925898, 0.943
[2022/02/22 17:48:27] root DEBUG: （成品包材）, 0.974
[2022/02/22 17:48:27] root DEBUG: 0.972573, 0.961
[2022/02/22 17:48:27] root DEBUG: 【主要功能】：可紧致头发磷层，从而达到, 0.936
[2022/02/22 17:48:27] root DEBUG: 0.994448, 0.952
[2022/02/22 17:48:27] root DEBUG: 13, 0.998
[2022/02/22 17:48:27] root DEBUG: 即时持久改善头发光泽的效果，给干燥的头, 0.994
[2022/02/22 17:48:27] root DEBUG: 0.990198, 0.975
[2022/02/22 17:48:27] root DEBUG: 14, 0.977
[2022/02/22 17:48:27] root DEBUG: 发足够的滋养, 0.991
[2022/02/22 17:48:27] root DEBUG: 0.997668, 0.918
[2022/02/22 17:48:27] root DEBUG: 花费了0.457335秒, 0.901
[2022/02/22 17:48:27] root DEBUG: The visualized image saved in ./inference_results/lite_demo.png
[2022/02/22 17:48:27] root INFO: The predict total time is 0.7003889083862305
```

使用 Paddle Inference 预测，终端输出：

```bash linenums="1"
[2022/02/22 17:47:25] root DEBUG: dt_boxes num : 38, elapse : 0.11791276931762695
[2022/02/22 17:47:27] root DEBUG: rec_res num  : 38, elapse : 2.6206860542297363
[2022/02/22 17:47:27] root DEBUG: 0  Predict time of ./deploy/lite/imgs/lite_demo.png: 2.746s
[2022/02/22 17:47:27] root DEBUG: The, 0.984
[2022/02/22 17:47:27] root DEBUG: visualized, 0.882
[2022/02/22 17:47:27] root DEBUG: etect18片, 0.720
[2022/02/22 17:47:27] root DEBUG: image saved in./vis.jpg, 0.947
[2022/02/22 17:47:27] root DEBUG: 纯臻营养护发素0.993604, 0.996
[2022/02/22 17:47:27] root DEBUG: 产品信息/参数, 0.922
[2022/02/22 17:47:27] root DEBUG: 0.992728, 0.914
[2022/02/22 17:47:27] root DEBUG: （45元／每公斤，100公斤起订）, 0.926
[2022/02/22 17:47:27] root DEBUG: 0.97417, 0.977
[2022/02/22 17:47:27] root DEBUG: 每瓶22元，1000瓶起订）0.993976, 0.962
[2022/02/22 17:47:27] root DEBUG: 【品牌】：代加工方式/0EMODM, 0.945
[2022/02/22 17:47:27] root DEBUG: 0.985133, 0.980
[2022/02/22 17:47:27] root DEBUG: 【品名】：纯臻营养护发素, 0.921
[2022/02/22 17:47:27] root DEBUG: 0.995007, 0.883
[2022/02/22 17:47:27] root DEBUG: 【产品编号】：YM-X-30110.96899, 0.955
[2022/02/22 17:47:27] root DEBUG: 【净含量】：220ml, 0.943
[2022/02/22 17:47:27] root DEBUG: Q.996577, 0.932
[2022/02/22 17:47:27] root DEBUG: 【适用人群】：适合所有肤质, 0.913
[2022/02/22 17:47:27] root DEBUG: 0.995842, 0.969
[2022/02/22 17:47:27] root DEBUG: 【主要成分】：鲸蜡硬脂醇、燕麦B-葡聚, 0.883
[2022/02/22 17:47:27] root DEBUG: 0.961928, 0.964
[2022/02/22 17:47:27] root DEBUG: 10, 0.812
[2022/02/22 17:47:27] root DEBUG: 糖、椰油酰胺丙基甜菜碱、泛醒, 0.866
[2022/02/22 17:47:27] root DEBUG: 0.925898, 0.943
[2022/02/22 17:47:27] root DEBUG: （成品包材）, 0.974
[2022/02/22 17:47:27] root DEBUG: 0.972573, 0.961
[2022/02/22 17:47:27] root DEBUG: 【主要功能】：可紧致头发磷层，从而达到, 0.936
[2022/02/22 17:47:27] root DEBUG: 0.994448, 0.952
[2022/02/22 17:47:27] root DEBUG: 13, 0.998
[2022/02/22 17:47:27] root DEBUG: 即时持久改善头发光泽的效果，给干燥的头, 0.994
[2022/02/22 17:47:27] root DEBUG: 0.990198, 0.975
[2022/02/22 17:47:27] root DEBUG: 14, 0.977
[2022/02/22 17:47:27] root DEBUG: 发足够的滋养, 0.991
[2022/02/22 17:47:27] root DEBUG: 0.997668, 0.918
[2022/02/22 17:47:27] root DEBUG: 花费了0.457335秒, 0.901
[2022/02/22 17:47:27] root DEBUG: The visualized image saved in ./inference_results/lite_demo.png
[2022/02/22 17:47:27] root INFO: The predict total time is 2.8338775634765625
```
