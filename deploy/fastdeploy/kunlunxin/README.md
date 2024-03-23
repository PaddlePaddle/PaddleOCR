[English](README.md) | 简体中文

# PaddleOCR 在昆仑芯上部署方案-FastDeploy

## 1. 说明  
PaddleOCR支持利用FastDeploy在昆仑芯片上部署模型.

支持如下芯片的部署
- 昆仑 818-100（推理芯片）
- 昆仑 818-300（训练芯片）

支持如下芯片的设备
- K100/K200 昆仑 AI 加速卡
- R200 昆仑芯 AI 加速卡

## 2. 支持的PaddleOCR推理模型

下表中的推理模型为FastDeploy测试过的模型, 下载链接由PaddleOCR模型库提供,
更多的模型, 详见[PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md), 欢迎用户尝试.

| PaddleOCR版本 | 文本框检测 | 方向分类模型 | 文字识别 |字典文件| 说明 |
|:----|:----|:----|:----|:----|:--------|
| ch_PP-OCRv3[推荐] |[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv3系列原始超轻量模型，支持中英文、多语种文本检测 |
| en_PP-OCRv3[推荐] |[en_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [en_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) | [en_dict.txt](https://bj.bcebos.com/paddlehub/fastdeploy/en_dict.txt) | OCRv3系列原始超轻量模型，支持英文与数字识别，除检测模型和识别模型的训练数据与中文模型不同以外，无其他区别 |
| ch_PP-OCRv2 |[ch_PP-OCRv2_det](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv2_rec](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测 |
| ch_PP-OCRv2_mobile |[ch_ppocr_mobile_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测,比PPOCRv2更加轻量 |
| ch_PP-OCRv2_server |[ch_ppocr_server_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_server_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) |[ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2服务器系列模型, 支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|


## 3. 详细部署的部署示例  
- [Python部署](python)
- [C++部署](cpp)
