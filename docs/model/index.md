---
comments: true
hide:
    - toc
#     - navigation
---

## PP-OCR 系列模型列表（更新中）

| 模型简介 |  模型名称 | 推荐场景  | 检测模型| 方向分类器   | 识别模型|
| ----- | -------------- | --------------- | ------- | ------- | ---------- |
| 中英文超轻量 PP-OCRv4 模型（15.8M） | ch_PP-OCRv4_xx | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar)         | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar) |
| 中英文超轻量 PP-OCRv3 模型（16.2M） | ch_PP-OCRv3_xx | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| 英文超轻量 PP-OCRv3 模型（13.4M）   | en_PP-OCRv3_xx | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |

- 超轻量 OCR 系列更多模型下载（包括多语言），可以参考[PP-OCR 系列模型下载](../ppocr/model_list.md)，文档分析相关模型参考[PP-Structure 系列模型下载](../ppstructure/models_list.md)

### PaddleOCR 场景应用模型

| 行业 | 类别 | 亮点| 文档说明| 模型下载 |
| ---- | ---- | -------- | ---- | ----- |
| 制造 | 数码管识别   | 数码管数据合成、漏识别调优         | [光功率计数码管字符识别](../applications/光功率计数码管字符识别.md) | [下载链接](../applications/overview.md) |
| 金融 | 通用表单识别 | 多模态通用表单结构化提取           | [多模态表单识别](../applications/多模态表单识别.md)                                        | [下载链接](../applications/overview.md) |
| 交通 | 车牌识别     | 多角度图像处理、轻量模型、端侧部署 | [轻量级车牌识别](../applications/轻量级车牌识别.md)                                        | [下载链接](../applications/overview.md) |

- 更多制造、金融、交通行业的主要 OCR 垂类应用模型（如电表、液晶屏、高精度 SVTR 模型等），可参考[场景应用模型下载](../applications/overview.md)
