[English](README.md) | 简体中文
# PaddleOCR CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PP-OCRv3在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例.
## 1. 说明
PaddleOCR支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署OCR模型.

## 2. 部署环境准备
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库.

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载所需模型.

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.0以上(x.x.x>=1.0.0)

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/ocr/PP-OCR/cpu-gpu/cpp

# 如果您希望从PaddleOCR下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到dygraph分支
git checkout dygraph
cd PaddleOCR/deploy/fastdeploy/cpu-gpu/cpp

# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz

# 编译部署示例
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PP-OCRv3文字检测模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar
# 下载文字方向分类器模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar
# 下载PP-OCRv3文字识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

# 下载预测图片与字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 运行部署示例
# 在CPU上使用Paddle Inference推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# 在CPU上使用OenVINO推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
# 在CPU上使用ONNX Runtime推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2
# 在CPU上使用Paddle Lite推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 3
# 在GPU上使用Paddle Inference推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 4
# 在GPU上使用Paddle TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 5
# 在GPU上使用ONNX Runtime推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 6
# 在GPU上使用Nvidia TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 7

# 同时, FastDeploy提供文字检测,文字分类,文字识别三个模型的单独推理,
# 有需要的用户, 请准备合适的图片, 同时根据自己的需求, 参考infer.cc来配置自定义硬件与推理后端.

# 在CPU上,单独使用文字检测模型部署
./infer_det ./ch_PP-OCRv3_det_infer ./12.jpg 0

# 在CPU上,单独使用文字方向分类模型部署
./infer_cls ./ch_ppocr_mobile_v2.0_cls_infer ./12.jpg 0

# 在CPU上,单独使用文字识别模型部署
./infer_rec ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
```

运行完成可视化结果如下图所示
<div  align="center">
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">
</div>

- 注意，以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考文档: [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)
- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 5. 部署示例选项说明
在我们使用`infer_demo`时, 输入了6个参数, 分别为文字检测模型, 文字分类模型, 文字识别模型, 预测图片, 字典文件与最后一位的数字选项.
现在下表将解释最后一位数字选项的含义.
|数字选项|含义|
|:---:|:---:|
|0| 在CPU上使用Paddle Inference推理 |
|1| 在CPU上使用OenVINO推理 |
|2| 在CPU上使用ONNX Runtime推理 |
|3| 在CPU上使用Paddle Lite推理 |
|4| 在GPU上使用Paddle Inference推理 |
|5| 在GPU上使用Paddle TensorRT推理 |
|6| 在GPU上使用ONNX Runtime推理 |
|7| 在GPU上使用Nvidia TensorRT推理 |

关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南

### 6.1 如何使用C++部署PP-OCRv2系列模型.
本目录下的`infer.cc`代码是以PP-OCRv3模型为例, 如果用户有使用PP-OCRv2的需求, 只需要按照下面所示的方式, 来创建PP-OCRv2并使用.

```cpp
// 此行为创建PP-OCRv3模型的代码
auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);
// 只需要将PPOCRv3改为PPOCRv2,即可创造PPOCRv2模型, 同时, 后续的接口均使用ppocr_v2来调用
auto ppocr_v2 = fastdeploy::pipeline::PPOCRv2(&det_model, &cls_model, &rec_model);

// 如果用户在部署PP-OCRv2时, 需要使用TensorRT推理, 还需要改动Rec模型的TensorRT的输入shape.
// 建议如下修改, 需要把 H 维度改为32, W 纬度按需修改.
rec_option.SetTrtInputShape("x", {1, 3, 32, 10}, {rec_batch_size, 3, 32, 320},
                                {rec_batch_size, 3, 32, 2304});
```
### 6.2 如何在PP-OCRv2/v3系列模型中, 关闭文字方向分类器的使用.

在PP-OCRv3/v2中, 文字方向分类器是可选的, 用户可以按照以下方式, 来决定自己是否使用方向分类器.
```cpp
// 使用 Cls 模型
auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

// 不使用 Cls 模型
auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);

// 当不使用Cls模型时, 请删掉或者注释掉相关代码
```

### 6.3 如何修改前后处理超参数.
在示例代码中, 我们展示出了修改前后处理超参数的接口,并设置为默认值,其中, FastDeploy提供的超参数的含义与文档[PaddleOCR推理模型参数解释](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/inference_args.md)是相同的. 如果用户想要进行更多定制化的开发, 请阅读[PP-OCR系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1ocr.html)

```cpp
// 设置检测模型的max_side_len
det_model.GetPreprocessor().SetMaxSideLen(960);
// 其他...
```

### 6.4 其他指南
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PP-OCRv3 Python部署](../python)
- [PP-OCRv3 C 部署](../c)
- [PP-OCRv3 C# 部署](../csharp)

## 7. 常见问题
- PaddleOCR能在FastDeploy支持的多种后端上推理,支持情况如下表所示, 如何切换后端, 详见文档[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

|硬件类型|支持的后端|
|:---:|:---:|
|X86 CPU| Paddle Inference, ONNX Runtime, OpenVINO |
|ARM CPU| Paddle Lite |
|飞腾 CPU| ONNX Runtime |
|NVIDIA GPU| Paddle Inference, ONNX Runtime, TensorRT |

- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)
