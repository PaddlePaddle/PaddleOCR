## OCR model list（V1.1, updated on 9.22）

- [1. Text Detection Model](#Detection)
- [2. Text Recognition Model](#Recognition)
    - [Chinese Recognition Model](#Chinese)
    - [English Recognition Model](#English)
    - [Multilingual Recognition Model](#Multilingual)
- [3. Text Angle Classification Model](#Angle)

The downloadable models provided by PaddleOCR include `inference model`, `trained model`, `pre-trained model` and `slim model`. The differences between the models are as follows:

|model type|model format|description|
|-|-|-|
|inference model|model、params|Used for reasoning based on Python prediction engine. [detail](./inference_en.md)|
|trained model / pre-trained model|\*.pdmodel、\*.pdopt、\*.pdparams|The checkpoints model saved in the training process, which stores the parameters of the model, mostly used for model evaluation and continuous training.|
|slim model|\*.nb|Generally used for Lite deployment|


<a name="Detection"></a>
### 1. Text Detection Model
|model name|description|config|model size|download|
|-|-|-|-|-|
|ch_ppocr_mobile_slim_v1.1_det|Slim pruned lightweight model, supporting Chinese, English, multilingual text detection|[det_mv3_db_v1.1.yml](../../configs/det/det_mv3_db_v1.1.yml)|1.4M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/det/ch_ppocr_mobile_v1.1_det_prune_infer.tar) / [slim model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_det_prune_opt.nb)|
|ch_ppocr_mobile_v1.1_det|Original lightweight model, supporting Chinese, English, multilingual text detection|[det_mv3_db_v1.1.yml](../../configs/det/det_mv3_db_v1.1.yml)|2.6M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_train.tar)|
|ch_ppocr_server_v1.1_det|General model, which is larger than the lightweight model, but achieved better performance|[det_r18_vd_db_v1.1.yml](../../configs/det/det_r18_vd_db_v1.1.yml)|47.2M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_train.tar)|


<a name="Recognition"></a>
### 2. Text Recognition Model

<a name="Chinese"></a>
#### Chinese Recognition Model
|model name|description|config|model size|download|
|-|-|-|-|-|
|ch_ppocr_mobile_slim_v1.1_rec|Slim pruned and quantized lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v1.1.yml](../../configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml)|1.6M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/rec/ch_ppocr_mobile_v1.1_rec_quant_infer.tar) / [slim model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_rec_quant_opt.nb) |
|ch_ppocr_mobile_v1.1_rec|Original lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v1.1.yml](../../configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml)|4.6M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_pre.tar) |
|ch_ppocr_server_v1.1_rec|General model, supporting Chinese, English and number recognition|[rec_chinese_common_train_v1.1.yml](../../configs/rec/ch_ppocr_v1.1/rec_chinese_common_train_v1.1.yml)|105M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_pre.tar) |

**Note:** The `trained model` is finetuned on the `pre-trained model` with real data and synthsized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthsized data, which is more suitable for finetune on your own dataset.

<a name="English"></a>
#### English Recognition Model
|model name|description|config|model size|download|
|-|-|-|-|-|
|en_ppocr_mobile_slim_v1.1_rec|Slim pruned and quantized lightweight model, supporting English and number recognition|[rec_en_lite_train.yml](../../configs/rec/multi_languages/rec_en_lite_train.yml)|0.9M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/en/en_ppocr_mobile_v1.1_rec_quant_infer.tar) / [slim model](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/en/en_ppocr_mobile_v1.1_rec_quant_opt.nb) |
|en_ppocr_mobile_v1.1_rec|Original lightweight model, supporting English and number recognition|[rec_en_lite_train.yml](../../configs/rec/multi_languages/rec_en_lite_train.yml)|2.0M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/en/en_ppocr_mobile_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/en/en_ppocr_mobile_v1.1_rec_train.tar) |

<a name="Multilingual"></a>
#### Multilingual Recognition Model（Updating...）
|model name|description|config|model size|download|
|-|-|-|-|-|
| french_ppocr_mobile_v1.1_rec |Lightweight model for French recognition|[rec_french_lite_train.yml](../../configs/rec/multi_languages/rec_french_lite_train.yml)|2.1M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/fr/french_ppocr_mobile_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/fr/french_ppocr_mobile_v1.1_rec_train.tar) |
| german_ppocr_mobile_v1.1_rec |German model for French recognition|[rec_ger_lite_train.yml](../../configs/rec/multi_languages/rec_ger_lite_train.yml)|2.1M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/ge/german_ppocr_mobile_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/ge/german_ppocr_mobile_v1.1_rec_train.tar) |
| korean_ppocr_mobile_v1.1_rec |Lightweight model for Korean recognition|[rec_korean_lite_train.yml](../../configs/rec/multi_languages/rec_korean_lite_train.yml)|3.4M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/kr/korean_ppocr_mobile_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/kr/korean_ppocr_mobile_v1.1_rec_train.tar) |
| japan_ppocr_mobile_v1.1_rec |Lightweight model for Japanese recognition|[rec_japan_lite_train.yml](../../configs/rec/multi_languages/rec_japan_lite_train.yml)|3.7M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/jp/japan_ppocr_mobile_v1.1_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/jp/japan_ppocr_mobile_v1.1_rec_train.tar) |


<a name="Angle"></a>
### 3. Text Angle Classification Model
|model name|description|config|model size|download|
|-|-|-|-|-|
|ch_ppocr_mobile_v1.1_cls_quant|Slim quantized model|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|0.5M|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_quant_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_quant_train.tar) / [slim model](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_cls_quant_opt.nb) |
|ch_ppocr_mobile_v1.1_cls|Original model|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|850kb|[inference model](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_train.tar) |


## OCR model list（V1.0, updated on 7.16）
|model name|description|detection model|recognition model|recognition model supporting space recognition|
|-|-|-|-|-|
|chinese_db_crnn_mobile|8.6M lightweight OCR model|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar) | [inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar) |[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|chinese_db_crnn_server|General OCR model|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar) | [inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar) |[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)
