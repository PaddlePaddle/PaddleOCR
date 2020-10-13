
# 中文OCR模型快速使用

## 1.环境配置

请先参考[快速安装](./installation.md)配置PaddleOCR运行环境。

*注意：也可以通过 whl 包安装使用PaddleOCR，具体参考[Paddleocr Package使用说明](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/whl.md)。*

## 2.inference模型下载

* 移动端和服务器端的检测与识别模型如下，更多模型下载（包括多语言），可以参考[PP-OCR v1.1 系列模型下载](../doc_ch/models_list.md)

| 模型简介     | 模型名称     |推荐场景          | 检测模型 | 方向分类器 | 识别模型 |
| ------------ | --------------- | ----------------|---- | ---------- | -------- |
| 中英文超轻量OCR模型（8.1M） | ch_ppocr_mobile_v1.1_xx |移动端&服务器端|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_train.tar)|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_train.tar) |[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_pre.tar)      |
| 中英文通用OCR模型（155.1M）   |ch_ppocr_server_v1.1_xx|服务器端 |[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_train.tar)          |[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_train.tar)    |[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_pre.tar)  |


* windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下

复制上表中的检测和识别的`inference模型`下载地址，并解压

```
mkdir inference && cd inference
# 下载检测模型并解压
wget {url/of/detection/inference_model} && tar xf {name/of/detection/inference_model/package}
# 下载识别模型并解压
wget {url/of/recognition/inference_model} && tar xf {name/of/recognition/inference_model/package}
# 下载方向分类器模型并解压
wget {url/of/classification/inference_model} && tar xf {name/of/classification/inference_model/package}
cd ..
```

以超轻量级模型为例：

```
mkdir inference && cd inference
# 下载超轻量级中文OCR模型的检测模型并解压
wget https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar && tar xf ch_ppocr_mobile_v1.1_det_infer.tar
# 下载超轻量级中文OCR模型的识别模型并解压
wget https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar && tar xf ch_ppocr_mobile_v1.1_rec_infer.tar
# 下载超轻量级中文OCR模型的文本方向分类器模型并解压
wget https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar && tar xf ch_ppocr_mobile_v1.1_cls_infer.tar
cd ..
```

解压完毕后应有如下文件结构：

```
|-inference
    |-ch_ppocr_mobile_v1.1_det_infer
        |- model
        |- params
    |-ch_ppocr_mobile_v1.1_rec_infer
        |- model
        |- params
    |-ch_ppocr_mobile-v1.1_cls_infer
        |- model
        |- params
    ...
```

## 3.单张图像或者图像集合预测

以下代码实现了文本检测、识别串联推理，在执行预测时，需要通过参数image_dir指定单张图像或者图像集合的路径、参数`det_model_dir`指定检测inference模型的路径、参数`rec_model_dir`指定识别inference模型的路径、参数`use_angle_cls`指定是否使用方向分类器、参数`cls_model_dir`指定方向分类器inference模型的路径、参数`use_space_char`指定是否预测空格字符。可视化识别结果默认保存到`./inference_results`文件夹里面。

```bash

# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_angle_cls=True --use_space_char=True

# 预测image_dir指定的图像集合
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_angle_cls=True --use_space_char=True

# 如果想使用CPU进行预测，需设置use_gpu参数为False
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_angle_cls=True --use_space_char=True --use_gpu=False
```

- 通用中文OCR模型

请按照上述步骤下载相应的模型，并且更新相关的参数，示例如下：

```bash
# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_server_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_angle_cls=True --use_space_char=True
```

* 注意：
    - 如果希望使用不支持空格的识别模型，在预测的时候需要注意：请将代码更新到最新版本，并添加参数 `--use_space_char=False`。
    - 如果不希望使用方向分类器，在预测的时候需要注意：请将代码更新到最新版本，并添加参数 `--use_angle_cls=False`。


更多的文本检测、识别串联推理使用方式请参考文档教程中[基于Python预测引擎推理](./inference.md)。

此外，文档教程中也提供了中文OCR模型的其他预测部署方式：
- [基于C++预测引擎推理](../../deploy/cpp_infer/readme.md)
- [服务部署](../../deploy/pdserving/readme.md)
- [端侧部署](../../deploy/lite/readme.md)
