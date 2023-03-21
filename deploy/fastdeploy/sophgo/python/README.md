[English](README.md) | 简体中文
# PP-OCRv3 SOPHGO Python部署示例
本目录下提供`infer.py`快速完成 PP-OCRv3 在SOPHGO TPU上部署的示例。

## 1. 部署环境准备

在部署前，需自行编译基于算能硬件的FastDeploy python wheel包并安装，参考文档[算能硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)


## 2.运行部署示例

### 2.1 模型准备
将Paddle模型转换为SOPHGO bmodel模型, 转换步骤参考[文档](../README.md)  

### 2.2 开始部署
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/ocr/PP-OCR/sophgo/python

# 如果您希望从PaddleOCR下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到dygraph分支
git checkout dygraph
cd PaddleOCR/deploy/fastdeploy/sophgo/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

#下载字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 推理
python3 infer.py --det_model ocr_bmodel/ch_PP-OCRv3_det_1684x_f32.bmodel \
                 --cls_model ocr_bmodel/ch_ppocr_mobile_v2.0_cls_1684x_f32.bmodel \
                 --rec_model ocr_bmodel/ch_PP-OCRv3_rec_1684x_f32.bmodel \
                 --rec_label_file ../ppocr_keys_v1.txt \  
                 --image ../12.jpg

# 运行完成后返回结果如下所示
det boxes: [[42,413],[483,391],[484,428],[43,450]]rec text: 上海斯格威铂尔大酒店 rec score:0.952958 cls label: 0 cls score: 1.000000
det boxes: [[187,456],[399,448],[400,480],[188,488]]rec text: 打浦路15号 rec score:0.897335 cls label: 0 cls score: 1.000000
det boxes: [[23,507],[513,488],[515,529],[24,548]]rec text: 绿洲仕格维花园公寓 rec score:0.994589 cls label: 0 cls score: 1.000000
det boxes: [[74,553],[427,542],[428,571],[75,582]]rec text: 打浦路252935号 rec score:0.900663 cls label: 0 cls score: 1.000000

可视化结果保存在sophgo_result.jpg中
```

## 3. 其它文档
- [PP-OCRv3 C++部署](../cpp)
- [转换 PP-OCRv3 SOPHGO模型文档](../README.md)
- 如果用户想要调整前后处理超参数、单独使用文字检测识别模型、使用其他模型等，更多详细文档与说明请参考[PP-OCR系列在CPU/GPU上的部署](../../cpu-gpu/cpp/README.md)
