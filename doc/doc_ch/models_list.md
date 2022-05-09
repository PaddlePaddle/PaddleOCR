# PP-OCR系列模型列表（V3，2022年4月28日更新）

> **说明**
> 1. V3版模型相比V2版模型，在模型精度上有进一步提升
> 2. 2.0+版模型和[1.1版模型](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/models_list.md) 的主要区别在于动态图训练vs.静态图训练，模型性能上无明显差距。
> 3. 本文档提供的是PPOCR自研模型列表，更多基于公开数据集的算法介绍与预训练模型可以参考：[算法概览文档](./algorithm_overview.md)。


- PP-OCR系列模型列表（V3，2022年4月28日更新）
  - [1. 文本检测模型](#1-文本检测模型)
    - [1.1 中文检测模型](#1.1)
    - [2.2 英文检测模型](#1.2)
    - [1.3 多语言检测模型](#1.3)
  - [2. 文本识别模型](#2-文本识别模型)
    - [2.1 中文识别模型](#21-中文识别模型)
    - [2.2 英文识别模型](#22-英文识别模型)
    - [2.3 多语言识别模型（更多语言持续更新中...）](#23-多语言识别模型更多语言持续更新中)
  - [3. 文本方向分类模型](#3-文本方向分类模型)
  - [4. Paddle-Lite 模型](#4-paddle-lite-模型)

PaddleOCR提供的可下载模型包括`推理模型`、`训练模型`、`预训练模型`、`nb模型`，模型区别说明如下：

|模型类型|模型格式|简介|
|--- | --- | --- |
|推理模型|inference.pdmodel、inference.pdiparams|用于预测引擎推理，[详情](./inference.md)|
|训练模型、预训练模型|\*.pdparams、\*.pdopt、\*.states |训练过程中保存的模型的参数、优化器状态和训练中间信息，多用于模型指标评估和恢复训练|
|nb模型|\*.nb|经过飞桨Paddle-Lite工具优化后的模型，适用于移动端/IoT端等端侧部署场景（需使用飞桨Paddle Lite部署）。|


各个模型的关系如下面的示意图所示。

![](../imgs/model_prod_flow_ch.png)


<a name="文本检测模型"></a>
## 1. 文本检测模型

<a name="1.1"></a>

### 1.1 中文检测模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv3_det_slim|【最新】slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测|[ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)| 1.1M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_distill_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.nb)|
|ch_PP-OCRv3_det| 【最新】原始超轻量模型，支持中英文、多语种文本检测 |[ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)| 3.8M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar)|
|ch_PP-OCRv2_det_slim| slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测|[ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)| 3M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar)|
|ch_PP-OCRv2_det| 原始超轻量模型，支持中英文、多语种文本检测|[ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)|3M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)|
|ch_ppocr_mobile_slim_v2.0_det|slim裁剪版超轻量模型，支持中英文、多语种文本检测|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)| 2.6M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar)|
|ch_ppocr_mobile_v2.0_det|原始超轻量模型，支持中英文、多语种文本检测|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)|3M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|通用模型，支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|[ch_det_res18_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml)|47M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|

<a name="1.2"></a>

### 1.2 英文检测模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|en_PP-OCRv3_det_slim |【最新】slim量化版超轻量模型，支持英文、数字检测 | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) | 1.1M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.nb) |
|en_PP-OCRv3_det |【最新】原始超轻量模型，支持英文、数字检测|[ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)| 3.8M | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) |

* 注：英文检测模型与中文检测模型结构完全相同，只有训练数据不同，在此仅提供相同的配置文件。

<a name="1.3"></a>

### 1.3 多语言检测模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
| ml_PP-OCRv3_det_slim |【最新】slim量化版超轻量模型，支持多语言检测 | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) | 1.1M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_distill_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.nb) |
| ml_PP-OCRv3_det |【最新】原始超轻量模型，支持多语言检测 | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)| 3.8M | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) |

* 注：多语言检测模型与中文检测模型结构完全相同，只有训练数据不同，在此仅提供相同的配置文件。


<a name="文本识别模型"></a>
## 2. 文本识别模型

<a name="中文识别模型"></a>

### 2.1 中文识别模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv3_rec_slim |【最新】slim量化版超轻量模型，支持中英文、数字识别|[ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)| 4.9M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.nb) |
|ch_PP-OCRv3_rec|【最新】原始超轻量模型，支持中英文、数字识别|[ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)| 12.4M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
|ch_PP-OCRv2_rec_slim| slim量化版超轻量模型，支持中英文、数字识别|[ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml)| 9M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
|ch_PP-OCRv2_rec| 原始超轻量模型，支持中英文、数字识别|[ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml)|8.5M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
|ch_ppocr_mobile_slim_v2.0_rec|slim裁剪量化版超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)| 6M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_train.tar) |
|ch_ppocr_mobile_v2.0_rec|原始超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)|5.2M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|通用模型，支持中英文、数字识别|[rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml)|94.8M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

**说明：** `训练模型`是基于预训练模型在真实数据与竖排合成文本数据上finetune得到的模型，在真实应用场景中有着更好的表现，`预训练模型`则是直接基于全量真实数据与合成数据训练得到，更适合用于在自己的数据集上finetune。

<a name="英文识别模型"></a>
### 2.2 英文识别模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|en_PP-OCRv3_rec_slim |【最新】slim量化版超轻量模型，支持英文、数字识别 | [en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml)| 3.2M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.nb) |
|en_PP-OCRv3_rec |【最新】原始超轻量模型，支持英文、数字识别|[en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml)| 9.6M | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
|en_number_mobile_slim_v2.0_rec|slim裁剪量化版超轻量模型，支持英文、数字识别|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)| 2.7M | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_train.tar) |
|en_number_mobile_v2.0_rec|原始超轻量模型，支持英文、数字识别|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)|2.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |


<a name="多语言识别模型"></a>
### 2.3 多语言识别模型（更多语言持续更新中...）

|模型名称|字典文件|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- |--- | --- |
| korean_PP-OCRv3_rec | ppocr/utils/dict/korean_dict.txt |韩文识别|[korean_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml)|11M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar) |
| japan_PP-OCRv3_rec | ppocr/utils/dict/japan_dict.txt |日文识别|[japan_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/japan_PP-OCRv3_rec.yml)|11M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar) |
| chinese_cht_PP-OCRv3_rec | ppocr/utils/dict/chinese_cht_dict.txt | 中文繁体识别|[chinese_cht_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/chinese_cht_PP-OCRv3_rec.yml)|12M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar) |
| te_PP-OCRv3_rec | ppocr/utils/dict/te_dict.txt | 泰卢固文识别|[te_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/te_PP-OCRv3_rec.yml)|9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_train.tar) |
| ka_PP-OCRv3_rec | ppocr/utils/dict/ka_dict.txt |卡纳达文识别|[ka_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/ka_PP-OCRv3_rec.yml)|9.9M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_train.tar) |
| ta_PP-OCRv3_rec | ppocr/utils/dict/ta_dict.txt |泰米尔文识别|[ta_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/ta_PP-OCRv3_rec.yml)|9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_train.tar) |
| latin_PP-OCRv3_rec |  ppocr/utils/dict/latin_dict.txt | 拉丁文识别 |  [latin_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/latin_PP-OCRv3_rec.yml) |9.7M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar) |
| arabic_PP-OCRv3_rec | ppocr/utils/dict/arabic_dict.txt | 阿拉伯字母 | [arabic_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/rec_arabic_lite_train.yml) |9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar) |
| cyrillic_PP-OCRv3_rec | ppocr/utils/dict/cyrillic_dict.txt | 斯拉夫字母 | [cyrillic_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/cyrillic_PP-OCRv3_rec.yml) |9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar) |
| devanagari_PP-OCRv3_rec | ppocr/utils/dict/devanagari_dict.txt |梵文字母 | [devanagari_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml) |9.9M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_train.tar) |

更多支持语种请参考: [多语言模型](./multi_languages.md)


<a name="文本方向分类模型"></a>
## 3. 文本方向分类模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|slim量化版模型，对检测到的文本行文字角度分类|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)| 2.1M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_infer_opt.nb) |
|ch_ppocr_mobile_v2.0_cls|原始分类器模型，对检测到的文本行文字角度分类|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|1.38M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |


<a name="Paddle-Lite模型"></a>
## 4. Paddle-Lite 模型

Paddle-Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，它可以对inference模型进一步优化，得到适用于移动端/IoT端等端侧部署场景的`nb模型`。一般建议基于量化模型进行转换，因为可以将模型以INT8形式进行存储与推理，从而进一步减小模型大小，提升模型速度。

本节主要列出PP-OCRv2以及更早版本的检测与识别nb模型，最新版本的nb模型可以直接从上面的模型列表中获得。


|模型版本|模型简介|模型大小|检测模型|文本方向分类模型|识别模型|Paddle-Lite版本|
|---|---|---|---|---|---|---|
|PP-OCRv2|蒸馏版超轻量中文OCR移动端模型|11M|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_det_infer_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_infer_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_rec_infer_opt.nb)|v2.10|
|PP-OCRv2(slim)|蒸馏版超轻量中文OCR移动端模型|4.6M|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_det_slim_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_rec_slim_opt.nb)|v2.10|
|PP-OCRv2|蒸馏版超轻量中文OCR移动端模型|11M|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer_opt.nb)|v2.9|
|PP-OCRv2(slim)|蒸馏版超轻量中文OCR移动端模型|4.9M|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_opt.nb)|v2.9|
|V2.0|ppocr_v2.0超轻量中文OCR移动端模型|7.8M|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_rec_opt.nb)|v2.9|
|V2.0(slim)|ppocr_v2.0超轻量中文OCR移动端模型|3.3M|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_slim_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_rec_slim_opt.nb)|v2.9|
