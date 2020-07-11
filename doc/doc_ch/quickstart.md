
# 中文OCR模型快速使用

## 1.环境配置

请先参考[快速安装](./installation.md)配置PaddleOCR运行环境。

## 2.inference模型下载

|模型名称|模型简介|检测模型地址|识别模型地址|支持空格的识别模型地址|
|-|-|-|-|-|
|chinese_db_crnn_mobile|超轻量级中文OCR模型|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|chinese_db_crnn_server|通用中文OCR模型|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)

*windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下*

复制上表中的检测和识别的`inference模型`下载地址，并解压

```
mkdir inference && cd inference
# 下载检测模型并解压
wget {url/of/detection/inference_model} && tar xf {name/of/detection/inference_model/package}
# 下载识别模型并解压
wget {url/of/recognition/inference_model} && tar xf {name/of/recognition/inference_model/package}
cd ..
```

以超轻量级模型为例：

```
mkdir inference && cd inference
# 下载超轻量级中文OCR模型的检测模型并解压
wget https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar && tar xf ch_det_mv3_db_infer.tar
# 下载超轻量级中文OCR模型的识别模型并解压
wget https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar && tar xf ch_rec_mv3_crnn_infer.tar
cd ..
```

解压完毕后应有如下文件结构：

```
|-inference
    |-ch_rec_mv3_crnn
        |- model
        |- params
    |-ch_det_mv3_db
        |- model
        |- params
    ...
```

## 3.单张图像或者图像集合预测

以下代码实现了文本检测、识别串联推理，在执行预测时，需要通过参数image_dir指定单张图像或者图像集合的路径、参数det_model_dir指定检测inference模型的路径和参数rec_model_dir指定识别inference模型的路径。可视化识别结果默认保存到 ./inference_results 文件夹里面。

```bash

# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/"

# 预测image_dir指定的图像集合
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/"

# 如果想使用CPU进行预测，需设置use_gpu参数为False
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/" --use_gpu=False
```

通用中文OCR模型的体验可以按照上述步骤下载相应的模型，并且更新相关的参数，示例如下：
```
# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_r50_vd_db/"  --rec_model_dir="./inference/ch_rec_r34_vd_crnn/"
```

带空格的通用中文OCR模型的体验可以按照上述步骤下载相应的模型，并且更新相关的参数，示例如下：

```
# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs_en/img_12.jpg" --det_model_dir="./inference/ch_det_r50_vd_db/"  --rec_model_dir="./inference/ch_rec_r34_vd_crnn_enhance/"
```

更多的文本检测、识别串联推理使用方式请参考文档教程中[基于Python预测引擎推理](./inference.md)。

此外，文档教程中也提供了中文OCR模型的其他预测部署方式：
- 基于C++预测引擎推理(comming soon)
- [服务部署](./doc/doc_ch/serving.md)
- 端测部署(comming soon)
