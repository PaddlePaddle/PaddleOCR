# OCR Model List（V2.1, updated on 2021.9.6）
> **Note**
> 1. Compared with the model v2.0, the 2.1 version of the detection model has a improvement in accuracy, and the 2.1 version of the recognition model has optimizations in accuracy and speed with CPU.
> 2. Compared with [models 1.1](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/models_list_en.md), which are trained with static graph programming paradigm, models 2.0 are the dynamic graph trained version and achieve close performance.
> 3. All models in this tutorial are all ppocr-series models, for more introduction of algorithms and models based on public dataset, you can refer to [algorithm overview tutorial](./algorithm_overview_en.md).

- [1. Text Detection Model](#Detection)
- [2. Text Recognition Model](#Recognition)
    - [2.1 Chinese Recognition Model](#Chinese)
    - [2.2 English Recognition Model](#English)
    - [2.3 Multilingual Recognition Model](#Multilingual)
- [3. Text Angle Classification Model](#Angle)
- [4. Paddle-Lite Model](#Paddle-Lite)

The downloadable models provided by PaddleOCR include `inference model`, `trained model`, `pre-trained model` and `slim model`. The differences between the models are as follows:

|model type|model format|description|
|--- | --- | --- |
|inference model|inference.pdmodel、inference.pdiparams|Used for inference based on Paddle inference engine，[detail](./inference_en.md)|
|trained model, pre-trained model|\*.pdparams、\*.pdopt、\*.states |The checkpoints model saved in the training process, which stores the parameters of the model, mostly used for model evaluation and continuous training.|
|slim model|\*.nb| Model compressed by PaddleSlim (a model compression tool using PaddlePaddle), which is suitable for mobile-side deployment scenarios (Paddle-Lite is needed for slim model deployment). |

Relationship of the above models is as follows.

![](../imgs_en/model_prod_flow_en.png)

<a name="Detection"></a>
## 1. Text Detection Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv2_det_slim|[New] slim quantization with distillation lightweight model, supporting Chinese, English, multilingual text detection|[ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)| 3M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar)|
|ch_PP-OCRv2_det|[New] Original lightweight model, supporting Chinese, English, multilingual text detection|[ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)|3M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)|
|ch_ppocr_mobile_slim_v2.0_det|Slim pruned lightweight model, supporting Chinese, English, multilingual text detection|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)|2.6M |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar)|
|ch_ppocr_mobile_v2.0_det|Original lightweight model, supporting Chinese, English, multilingual text detection|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)|3M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|General model, which is larger than the lightweight model, but achieved better performance|[ch_det_res18_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml)|47M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|

<a name="Recognition"></a>
## 2. Text Recognition Model

<a name="Chinese"></a>
### 2.1 Chinese Recognition Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv2_rec_slim|[New] Slim qunatization with distillation lightweight model, supporting Chinese, English, multilingual text recognition|[ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml)| 9M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
|ch_PP-OCRv2_rec|[New] Original lightweight model, supporting Chinese, English, multilingual text recognition|[ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml)|8.5M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
|ch_ppocr_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)| 6M | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_train.tar) |
|ch_ppocr_mobile_v2.0_rec|Original lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)|5.2M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|General model, supporting Chinese, English and number recognition|[rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml)|94.8M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |


**Note:** The `trained model` is fine-tuned on the `pre-trained model` with real data and synthesized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthesized data, which is more suitable for fine-tune on your own dataset.

<a name="English"></a>
### 2.2 English Recognition Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|en_number_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting English and number recognition|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)| 2.7M | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_train.tar) |
|en_number_mobile_v2.0_rec|Original lightweight model, supporting English and number recognition|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)|2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |

<a name="Multilingual"></a>
### 2.3 Multilingual Recognition Model（Updating...）

|model name| dict file | description|config|model size|download|
| --- | --- | --- |--- | --- | --- |
| french_mobile_v2.0_rec | ppocr/utils/dict/french_dict.txt | Lightweight model for French recognition|[rec_french_lite_train.yml](../../configs/rec/multi_language/rec_french_lite_train.yml)|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec | ppocr/utils/dict/german_dict.txt | Lightweight model for German recognition|[rec_german_lite_train.yml](../../configs/rec/multi_language/rec_german_lite_train.yml)|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec | ppocr/utils/dict/korean_dict.txt | Lightweight model for Korean recognition|[rec_korean_lite_train.yml](../../configs/rec/multi_language/rec_korean_lite_train.yml)|3.9M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec | ppocr/utils/dict/japan_dict.txt | Lightweight model for Japanese recognition|[rec_japan_lite_train.yml](../../configs/rec/multi_language/rec_japan_lite_train.yml)|4.23M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| chinese_cht_mobile_v2.0_rec | ppocr/utils/dict/chinese_cht_dict.txt | Lightweight model for chinese cht recognition|rec_chinese_cht_lite_train.yml|5.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec | ppocr/utils/dict/te_dict.txt | Lightweight model for Telugu recognition|rec_te_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| ka_mobile_v2.0_rec | ppocr/utils/dict/ka_dict.txt | Lightweight model for Kannada recognition|rec_ka_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec | ppocr/utils/dict/ta_dict.txt | Lightweight model for Tamil recognition|rec_ta_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |
| latin_mobile_v2.0_rec | ppocr/utils/dict/latin_dict.txt | Lightweight model for latin recognition |  [rec_latin_lite_train.yml](../../configs/rec/multi_language/rec_latin_lite_train.yml) |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_train.tar) |
| arabic_mobile_v2.0_rec |  ppocr/utils/dict/arabic_dict.txt | Lightweight model for arabic recognition | [rec_arabic_lite_train.yml](../../configs/rec/multi_language/rec_arabic_lite_train.yml) |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_train.tar) |
| cyrillic_mobile_v2.0_rec | ppocr/utils/dict/cyrillic_dict.txt | Lightweight model for cyrillic recognition | [rec_cyrillic_lite_train.yml](../../configs/rec/multi_language/rec_cyrillic_lite_train.yml) |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_train.tar) |
| devanagari_mobile_v2.0_rec | ppocr/utils/dict/devanagari_dict.txt | Lightweight model for devanagari recognition | [rec_devanagari_lite_train.yml](../../configs/rec/multi_language/rec_devanagari_lite_train.yml) |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_train.tar) |

For more supported languages, please refer to : [Multi-language model](./multi_languages_en.md)


<a name="Angle"></a>
## 3. Text Angle Classification Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|Slim quantized model for text angle classification|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)| 2.1M | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_train.tar) |
|ch_ppocr_mobile_v2.0_cls|Original model for text angle classification|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|1.38M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

<a name="Paddle-Lite"></a>
## 4. Paddle-Lite Model
|Version|Introduction|Model size|Detection model|Text Direction model|Recognition model|Paddle-Lite branch|
|---|---|---|---|---|---|---|
|PP-OCRv2|extra-lightweight chinese OCR optimized model|11M|[download link](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer_opt.nb)|v2.9|
|PP-OCRv2(slim)|extra-lightweight chinese OCR optimized model|4.9M|[download link](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_opt.nb)|v2.9|
|V2.0|ppocr_v2.0 extra-lightweight chinese OCR optimized model|7.8M|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_rec_opt.nb)|v2.9|
|V2.0(slim)|ppovr_v2.0 extra-lightweight chinese OCR optimized model|3.3M|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_slim_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb)|[download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_rec_slim_opt.nb)|v2.9|
