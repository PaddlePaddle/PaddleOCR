## OCR模型列表（V1.1，9月22日更新）

- [一、文本检测模型](#文本检测模型)
- [二、文本识别模型](#文本识别模型)
    - [1. 中文识别模型](#中文识别模型)
    - [2. 英文识别模型](#英文识别模型)
    - [3. 多语言识别模型](#多语言识别模型)
- [三、文本方向分类模型](#文本方向分类模型)

PaddleOCR提供的可下载模型包括`推理模型`、`训练模型`、`预训练模型`、`slim模型`，模型区别说明如下：

|模型类型|模型格式|简介|
|-|-|-|
|推理模型|model、params|用于python预测引擎推理，[详情](./inference.md)|
|训练模型、预训练模型|\*.pdmodel、\*.pdopt、\*.pdparams|训练过程中保存的checkpoints模型，保存的是模型的参数，多用于模型指标评估和恢复训练|
|slim模型|\*.nb|用于lite部署|


<a name="文本检测模型"></a>
### 一、文本检测模型
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
|-|-|-|-|-|
|ch_ppocr_mobile_slim_v1.1_det|slim裁剪版超轻量模型，支持中英文、多语种文本检测|[det_mv3_db_v1.1.yml](../../configs/det/det_mv3_db_v1.1.yml)|1.4M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/det/ch_ppocr_mobile_v1.1_det_prune_infer.tar) / [slim模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_det_prune_opt.nb)|
|ch_ppocr_mobile_v1.1_det|原始超轻量模型，支持中英文、多语种文本检测|[det_mv3_db_v1.1.yml](../../configs/det/det_mv3_db_v1.1.yml)|2.6M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_train.tar)|
|ch_ppocr_server_v1.1_det|通用模型，支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|[det_r18_vd_db_v1.1.yml](../../configs/det/det_r18_vd_db_v1.1.yml)|47.2M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_train.tar)|


<a name="文本识别模型"></a>
### 二、文本识别模型

<a name="中文识别模型"></a>
#### 1. 中文识别模型
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
|-|-|-|-|-|
|ch_ppocr_mobile_slim_v1.1_rec|slim裁剪量化版超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v1.1.yml](../../configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml)|1.6M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/rec/ch_ppocr_mobile_v1.1_rec_quant_infer.tar) / [slim模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_rec_quant_opt.nb) |
|ch_ppocr_mobile_v1.1_rec|原始超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v1.1.yml](../../configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml)|4.6M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_pre.tar) |
|ch_ppocr_server_v1.1_rec|通用模型，支持中英文、数字识别|[rec_chinese_common_train_v1.1.yml](../../configs/rec/ch_ppocr_v1.1/rec_chinese_common_train_v1.1.yml)|105M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_pre.tar) |

**说明：** `训练模型`是基于预训练模型在真实数据与竖排合成文本数据上finetune得到的模型，在真实应用场景中有着更好的表现，`预训练模型`则是直接基于全量真实数据与合成数据训练得到，更适合用于在自己的数据集上finetune。

<a name="英文识别模型"></a>
#### 2. 英文识别模型
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
|-|-|-|-|-|
|en_ppocr_mobile_slim_v1.1_rec|slim裁剪量化版超轻量模型，支持英文、数字识别|[rec_en_lite_train.yml](../../configs/rec/multi_languages/rec_en_lite_train.yml)|0.9M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/en/en_ppocr_mobile_v1.1_rec_quant_infer.tar) / [slim模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/en/en_ppocr_mobile_v1.1_rec_quant_opt.nb) |
|en_ppocr_mobile_v1.1_rec|原始超轻量模型，支持英文、数字识别|[rec_en_lite_train.yml](../../configs/rec/multi_languages/rec_en_lite_train.yml)|2.0M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/en/en_ppocr_mobile_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/en/en_ppocr_mobile_v1.1_rec_train.tar) |

<a name="多语言识别模型"></a>
#### 3. 多语言识别模型（更多语言持续更新中...）
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
|-|-|-|-|-|
| french_ppocr_mobile_v1.1_rec |法文识别|[rec_french_lite_train.yml](../../configs/rec/multi_languages/rec_french_lite_train.yml)|2.1M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/fr/french_ppocr_mobile_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/fr/french_ppocr_mobile_v1.1_rec_train.tar) |
| german_ppocr_mobile_v1.1_rec |德文识别|[rec_ger_lite_train.yml](../../configs/rec/multi_languages/rec_ger_lite_train.yml)|2.1M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/ge/german_ppocr_mobile_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/ge/german_ppocr_mobile_v1.1_rec_train.tar) |
| korean_ppocr_mobile_v1.1_rec |韩文识别|[rec_korean_lite_train.yml](../../configs/rec/multi_languages/rec_korean_lite_train.yml)|3.4M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/kr/korean_ppocr_mobile_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/kr/korean_ppocr_mobile_v1.1_rec_train.tar) |
| japan_ppocr_mobile_v1.1_rec |日文识别|[rec_japan_lite_train.yml](../../configs/rec/multi_languages/rec_japan_lite_train.yml)|3.7M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/jp/japan_ppocr_mobile_v1.1_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/jp/japan_ppocr_mobile_v1.1_rec_train.tar) |


<a name="文本方向分类模型"></a>
### 三、文本方向分类模型
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
|-|-|-|-|-|
|ch_ppocr_mobile_v1.1_cls_quant|slim量化版模型|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|0.5M|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_quant_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_quant_train.tar) / [slim模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_cls_quant_opt.nb) |
|ch_ppocr_mobile_v1.1_cls|原始模型|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|850kb|[推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_train.tar) |


## OCR模型列表（V1.0，7月16日更新）

|模型名称|模型简介|检测模型地址|识别模型地址|支持空格的识别模型地址|
|-|-|-|-|-|
|chinese_db_crnn_mobile|8.6M超轻量级中文OCR模型|[推理模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar) |[推理模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar) |[推理模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|chinese_db_crnn_server|通用中文OCR模型|[推理模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar) |[推理模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar) |[推理模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)
