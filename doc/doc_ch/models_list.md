## OCR模型列表（V2.0，2020年12月12日更新）
**说明** ：2.0版模型和[1.1版模型](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/models_list.md)的主要区别在于动态图训练vs.静态图训练，模型性能上无明显差距。

- [一、文本检测模型](#文本检测模型)
- [二、文本识别模型](#文本识别模型)
    - [1. 中文识别模型](#中文识别模型)
    - [2. 英文识别模型](#英文识别模型)
    - [3. 多语言识别模型](#多语言识别模型)
- [三、文本方向分类模型](#文本方向分类模型)

PaddleOCR提供的可下载模型包括`推理模型`、`训练模型`、`预训练模型`、`slim模型`，模型区别说明如下：

|模型类型|模型格式|简介|
|--- | --- | --- |
|推理模型|inference.pdmodel、inference.pdiparams|用于python预测引擎推理，[详情](./inference.md)|
|训练模型、预训练模型|\*.pdparams、\*.pdopt、\*.states |训练过程中保存的模型的参数、优化器状态和训练中间信息，多用于模型指标评估和恢复训练|
|slim模型|\*.nb|用于lite部署|


<a name="文本检测模型"></a>
### 一、文本检测模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_det|slim裁剪版超轻量模型，支持中英文、多语种文本检测|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)| |推理模型 (coming soon) / slim模型 (coming soon)|
|ch_ppocr_mobile_v2.0_det|原始超轻量模型，支持中英文、多语种文本检测|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)|3M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|通用模型，支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|[ch_det_res18_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml)|47M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|


<a name="文本识别模型"></a>
### 二、文本识别模型

<a name="中文识别模型"></a>
#### 1. 中文识别模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_rec|slim裁剪量化版超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)| |推理模型 (coming soon) / slim模型 (coming soon) |
|ch_ppocr_mobile_v2.0_rec|原始超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)|3.71M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|通用模型，支持中英文、数字识别|[rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml)|94.8M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

**说明：** `训练模型`是基于预训练模型在真实数据与竖排合成文本数据上finetune得到的模型，在真实应用场景中有着更好的表现，`预训练模型`则是直接基于全量真实数据与合成数据训练得到，更适合用于在自己的数据集上finetune。

<a name="英文识别模型"></a>
#### 2. 英文识别模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|en_number_mobile_slim_v2.0_rec|slim裁剪量化版超轻量模型，支持英文、数字识别|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)| | 推理模型 (coming soon) / slim模型 (coming soon) |
|en_number_mobile_v2.0_rec|原始超轻量模型，支持英文、数字识别|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)|2.56M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |

<a name="多语言识别模型"></a>
#### 3. 多语言识别模型（更多语言持续更新中...）

**说明：** 新增的多语言模型的配置文件通过代码方式生成，以生成意大利语配置文件为例：

```bash
# 该代码需要在指定目录运行
cd PaddleOCR/configs/rec/multi_language/
# 通过-l或者--language参数设置需要生成的语种的配置文件，该命令会将默认参数写入配置文件
python3 generate_multi_language_configs.py -l it
```
您可以通过`--help`参数查看当前PaddleOCR支持生成哪些多语言的配置文件：

```bash
python3 generate_multi_language_configs.py --help
```
如果您不想使用默认的路径或者默认参数可以根据以下命令修改：

```bash
# -l或者--language字段是必须的
# --train修改训练集，--val修改验证集，--data_dir修改数据集目录，-o修改对应默认参数
python3 generate_multi_language_configs.py -l it \
--train {path/to/train_list} \
--val {path/to/val_list} \
--data_dir {path/to/data_dir} \
-o Global.use_gpu=False
```

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
| french_mobile_v2.0_rec |法文识别|[rec_french_lite_train.yml](../../configs/rec/multi_language/rec_french_lite_train.yml)|2.65M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec |德文识别|[rec_german_lite_train.yml](../../configs/rec/multi_language/rec_german_lite_train.yml)|2.65M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec |韩文识别|[rec_korean_lite_train.yml](../../configs/rec/multi_language/rec_korean_lite_train.yml)|3.9M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec |日文识别|[rec_japan_lite_train.yml](../../configs/rec/multi_language/rec_japan_lite_train.yml)|4.23M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| it_mobile_v2.0_rec |意大利文识别|rec_it_lite_train.yml|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_train.tar) |
| xi_mobile_v2.0_rec |西班牙文识别|rec_xi_lite_train.yml|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/xi_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/xi_mobile_v2.0_rec_train.tar) |
| pu_mobile_v2.0_rec |葡萄牙文识别|rec_pu_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pu_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pu_mobile_v2.0_rec_train.tar) |
| ru_mobile_v2.0_rec |俄罗斯文识别|rec_ru_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_train.tar) |
| ar_mobile_v2.0_rec |阿拉伯文识别|rec_ar_lite_train.yml|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_train.tar) |
| hi_mobile_v2.0_rec |印地文识别|rec_hi_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_train.tar) |
| chinese_cht_mobile_v2.0_rec |中文繁体识别|rec_chinese_cht_lite_train.yml|5.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_train.tar) |
| ug_mobile_v2.0_rec |维吾尔文识别|rec_ug_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_train.tar) |
| fa_mobile_v2.0_rec |波斯文识别|rec_fa_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_train.tar) |
| ur_mobile_v2.0_rec |乌尔都文识别|rec_ur_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_train.tar) |
| rs_mobile_v2.0_rec |塞尔维亚文（latin）识别|rec_rs_lite_train.yml|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_mobile_v2.0_rec_train.tar) |
| oc_mobile_v2.0_rec |欧西坦文识别|rec_oc_lite_train.yml|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_train.tar) |
| mr_mobile_v2.0_rec |马拉地文识别|rec_mr_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_train.tar) |
| ne_mobile_v2.0_rec |尼泊尔文识别|rec_ne_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_train.tar) |
| rsc_mobile_v2.0_rec |塞尔维亚文（cyrillic）识别|rec_rsc_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rsc_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rsc_mobile_v2.0_rec_train.tar) |
| bg_mobile_v2.0_rec |保加利亚文识别|rec_bg_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_train.tar) |
| uk_mobile_v2.0_rec |乌克兰文识别|rec_uk_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_train.tar) |
| be_mobile_v2.0_rec |白俄罗斯文识别|rec_be_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec |泰卢固文识别|rec_te_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| ka_mobile_v2.0_rec |卡纳达文识别|[rec_ka_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec |泰米尔文识别|rec_ta_lite_train.yml|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |

<a name="文本方向分类模型"></a>
### 三、文本方向分类模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|slim量化版模型|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)| |推理模型 (coming soon) / 训练模型 / slim模型 |
|ch_ppocr_mobile_v2.0_cls|原始模型|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|1.38M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |
