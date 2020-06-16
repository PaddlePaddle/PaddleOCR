[English](README_en.md) | 简体中文

## 简介
PaddleOCR旨在打造一套丰富、领先、且实用的OCR工具库，助力使用者训练出更好的模型，并应用落地。

**近期更新**
- 2020.6.8 添加[数据集](./doc/doc_ch/datasets.md)，并保持持续更新
- 2020.6.5 支持 `attetnion` 模型导出 `inference_model`
- 2020.6.5 支持单独预测识别时，输出结果得分
- 2020.5.30 提供超轻量级中文OCR在线体验
- 2020.5.30 模型预测、训练支持Windows系统
- [more](./doc/doc_ch/update.md)

## 特性
- 超轻量级中文OCR，总模型仅8.6M
    - 单模型支持中英文数字组合识别、竖排文本识别、长文本识别
    - 检测模型DB（4.1M）+识别模型CRNN（4.5M）
- 多种文本检测训练算法，EAST、DB
- 多种文本识别训练算法，Rosetta、CRNN、STAR-Net、RARE

### 支持的中文模型列表:

|模型名称|模型简介|检测模型地址|识别模型地址|
|-|-|-|-|
|chinese_db_crnn_mobile|超轻量级中文OCR模型|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) & [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) & [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|
|chinese_db_crnn_server|通用中文OCR模型|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) & [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) & [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|

超轻量级中文OCR在线体验地址：https://www.paddlepaddle.org.cn/hub/scene/ocr

**也可以按如下教程快速体验超轻量级中文OCR和通用中文OCR模型。**

## **超轻量级中文OCR以及通用中文OCR体验**

![](doc/imgs_results/11.jpg)

上图是超轻量级中文OCR模型效果展示，更多效果图请见文末[超轻量级中文OCR效果展示](#超轻量级中文OCR效果展示)和[通用中文OCR效果展示](#通用中文OCR效果展示)。

#### 1.环境配置

请先参考[快速安装](./doc/doc_ch/installation.md)配置PaddleOCR运行环境。

#### 2.inference模型下载

*windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下*


#### (1)超轻量级中文OCR模型下载
```
mkdir inference && cd inference
# 下载超轻量级中文OCR模型的检测模型并解压
wget https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar && tar xf ch_det_mv3_db_infer.tar
# 下载超轻量级中文OCR模型的识别模型并解压
wget https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar && tar xf ch_rec_mv3_crnn_infer.tar
cd ..
```
#### (2)通用中文OCR模型下载
```
mkdir inference && cd inference
# 下载通用中文OCR模型的检测模型并解压
wget https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar && tar xf ch_det_r50_vd_db_infer.tar
# 下载通用中文OCR模型的识别模型并解压
wget https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar && tar xf ch_rec_r34_vd_crnn_infer.tar
cd ..
```

#### 3.单张图像或者图像集合预测

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

更多的文本检测、识别串联推理使用方式请参考文档教程中[基于预测引擎推理](./doc/doc_ch/inference.md)。

## 文档教程
- [快速安装](./doc/doc_ch/installation.md)
- [文本检测模型训练/评估/预测](./doc/doc_ch/detection.md)
- [文本识别模型训练/评估/预测](./doc/doc_ch/recognition.md)
- [基于预测引擎推理](./doc/doc_ch/inference.md)
- [数据集](./doc/doc_ch/datasets.md)

## 文本检测算法

PaddleOCR开源的文本检测算法列表：
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))
- [ ]  SAST([paper](https://arxiv.org/abs/1908.05498))(百度自研, comming soon)

在ICDAR2015文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
|-|-|-|-|-|-|
|EAST|ResNet50_vd|88.18%|85.51%|86.82%|[下载链接](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar)|
|EAST|MobileNetV3|81.67%|79.83%|80.74%|[下载链接](https://paddleocr.bj.bcebos.com/det_mv3_east.tar)|
|DB|ResNet50_vd|83.79%|80.65%|82.19%|[下载链接](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar)|
|DB|MobileNetV3|75.92%|73.18%|74.53%|[下载链接](https://paddleocr.bj.bcebos.com/det_mv3_db.tar)|

使用[LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/datasets.md#1icdar2019-lsvt)街景数据集共3w张数据，训练中文检测模型的相关配置和预训练文件如下：
|模型|骨干网络|配置文件|预训练模型|
|-|-|-|-|
|超轻量中文模型|MobileNetV3|det_mv3_db.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|
|通用中文OCR模型|ResNet50_vd|det_r50_vd_db.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|

* 注： 上述DB模型的训练和评估，需设置后处理参数box_thresh=0.6，unclip_ratio=1.5，使用不同数据集、不同模型训练，可调整这两个参数进行优化

PaddleOCR文本检测算法的训练和使用请参考文档教程中[文本检测模型训练/评估/预测](./doc/doc_ch/detection.md)。

## 文本识别算法

PaddleOCR开源的文本识别算法列表：
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))
- [ ]  SRN([paper](https://arxiv.org/abs/2003.12294))(百度自研, comming soon)

参考[DTRB](https://arxiv.org/abs/1904.01906)文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.24%|rec_r34_vd_none_none_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_none_ctc.tar)|
|Rosetta|MobileNetV3|78.16%|rec_mv3_none_none_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_none_none_ctc.tar)|
|CRNN|Resnet34_vd|82.20%|rec_r34_vd_none_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_bilstm_ctc.tar)|
|CRNN|MobileNetV3|79.37%|rec_mv3_none_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar)|
|STAR-Net|Resnet34_vd|83.93%|rec_r34_vd_tps_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)|
|STAR-Net|MobileNetV3|81.56%|rec_mv3_tps_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_ctc.tar)|
|RARE|Resnet34_vd|84.90%|rec_r34_vd_tps_bilstm_attn|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_attn.tar)|
|RARE|MobileNetV3|83.32%|rec_mv3_tps_bilstm_attn|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_attn.tar)|

使用[LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/datasets.md#1icdar2019-lsvt)街景数据集根据真值将图crop出来30w数据，进行位置校准。此外基于LSVT语料生成500w合成数据训练中文模型，相关配置和预训练文件如下：
|模型|骨干网络|配置文件|预训练模型|
|-|-|-|-|
|超轻量中文模型|MobileNetV3|rec_chinese_lite_train.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|
|通用中文OCR模型|Resnet34_vd|rec_chinese_common_train.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|

PaddleOCR文本识别算法的训练和使用请参考文档教程中[文本识别模型训练/评估/预测](./doc/doc_ch/recognition.md)。

## 端到端OCR算法
- [ ]  [End2End-PSL](https://arxiv.org/abs/1909.07808)(百度自研, comming soon)

<a name="超轻量级中文OCR效果展示"></a>
## 超轻量级中文OCR效果展示
![](doc/imgs_results/1.jpg)
![](doc/imgs_results/7.jpg)
![](doc/imgs_results/12.jpg)
![](doc/imgs_results/4.jpg)
![](doc/imgs_results/6.jpg)
![](doc/imgs_results/9.jpg)
![](doc/imgs_results/16.png)
![](doc/imgs_results/22.jpg)

<a name="通用中文OCR效果展示"></a>
## 通用中文OCR效果展示
![](doc/imgs_results/chinese_db_crnn_server/11.jpg)
![](doc/imgs_results/chinese_db_crnn_server/2.jpg)
![](doc/imgs_results/chinese_db_crnn_server/8.jpg)

## FAQ
1. **转换attention识别模型时报错：KeyError: 'predict'**  
问题已解，请更新到最新代码。  

2. **关于推理速度**  
图片中的文字较多时，预测时间会增，可以使用--rec_batch_num设置更小预测batch num，默认值为30，可以改为10或其他数值。  

3. **服务部署与移动端部署**  
预计6月中下旬会先后发布基于Serving的服务部署方案和基于Paddle Lite的移动端部署方案，欢迎持续关注。  

4. **自研算法发布时间**  
自研算法SAST、SRN、End2End-PSL都将在6-7月陆续发布，敬请期待。  

[more](./doc/doc_ch/FAQ.md)

## 欢迎加入PaddleOCR技术交流群
加微信：paddlehelp，备注OCR，小助手拉你进群～

## 参考文献
```
1. EAST:
@inproceedings{zhou2017east,
  title={EAST: an efficient and accurate scene text detector},
  author={Zhou, Xinyu and Yao, Cong and Wen, He and Wang, Yuzhi and Zhou, Shuchang and He, Weiran and Liang, Jiajun},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={5551--5560},
  year={2017}
}

2. DB:
@article{liao2019real,
  title={Real-time Scene Text Detection with Differentiable Binarization},
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  journal={arXiv preprint arXiv:1911.08947},
  year={2019}
}

3. DTRB:
@inproceedings{baek2019wrong,
  title={What is wrong with scene text recognition model comparisons? dataset and model analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4715--4723},
  year={2019}
}

4. SAST:
@inproceedings{wang2019single,
  title={A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning},
  author={Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Huang, Zuming and En, Mengyi and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={1277--1285},
  year={2019}
}

5. SRN:
@article{yu2020towards,
  title={Towards Accurate Scene Text Recognition with Semantic Reasoning Networks},
  author={Yu, Deli and Li, Xuan and Zhang, Chengquan and Han, Junyu and Liu, Jingtuo and Ding, Errui},
  journal={arXiv preprint arXiv:2003.12294},
  year={2020}
}

6. end2end-psl:
@inproceedings{sun2019chinese,
  title={Chinese Street View Text: Large-scale Chinese Text Reading with Partially Supervised Learning},
  author={Sun, Yipeng and Liu, Jiaming and Liu, Wei and Han, Junyu and Ding, Errui and Liu, Jingtuo},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9086--9095},
  year={2019}
}
```

## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。

## 如何贡献代码
我们非常欢迎你为PaddleOCR贡献代码，也十分感谢你的反馈。

- 非常感谢 [Khanh Tran](https://github.com/xxxpsyduck) 贡献了英文文档。
