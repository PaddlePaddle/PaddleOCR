[English](README.md) | 简体中文
# PP-OCRv3 Ascend Python部署示例

本目录下提供`infer.py`, 供用户完成PP-OCRv3在华为昇腾AI处理器上的部署.

## 1. 部署环境准备
在部署前，需自行编译基于华为昇腾AI处理器的FastDeploy python wheel包并安装，参考文档，参考文档[华为昇腾AI处理器部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2.部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载所需模型.

## 3.运行部署示例
```
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/ocr/PP-OCR/ascend/python

# 如果您希望从PaddleOCR下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到dygraph分支
git checkout dygraph
cd PaddleOCR/deploy/fastdeploy/ascend/python

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

python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg
# NOTE:若用户需要连续地预测图片, 输入图片尺寸需要准备为统一尺寸, 例如 N 张, 尺寸为 A * B 的图片.
```

运行完成可视化结果如下图所示

<div  align="center">  
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">
</div>

## 4. 更多指南
- [PP-OCR系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/ocr.html)
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PP-OCRv3 C++部署](../cpp)
- 如果用户想要调整前后处理超参数、单独使用文字检测识别模型、使用其他模型等，更多详细文档与说明请参考[PP-OCR系列在CPU/GPU上的部署](../../cpu-gpu/python/README.md)

## 5. 常见问题
- [如何将视觉模型预测结果转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)
