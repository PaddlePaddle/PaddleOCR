# PP-OCR Series Model List (V4, Updated February 8, 2024)

> **Explanation**
> 1. The V4 models exhibit further accuracy improvements over the V3 models.
> 2. The V3 models have shown accuracy enhancements compared to the V2 models.
> 3. The primary distinction between the 2.0+ models and [the 1.1 models](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/models_list.md) is dynamic versus static graph training, with no significant differences in model performance.
> 4. This document provides a list of PPOCR's self-developed models. For introductions to more algorithms based on public datasets and pre-trained models, refer to: [the Algorithm Overview Document](./algorithm_overview.md).


- PP-OCR Series Model List (V4, Updated on August 1, 2023)
  - [1. Text Detection Models](#1-text-detection-models)
    - [1.1 Chinese Detection Models](#11-chinese-detection-models)
    - [1.2 English Detection Models](#12-english-detection-models)
    - [1.3 Multilingual Detection Models](#13-multilingual-detection-models)
  - [2. Text Recognition Models](#2-text-recognition-models)
    - [2.1 Chinese Recognition Models](#21-chinese-recognition-models)
    - [2.2 English Recognition Models](#22-english-recognition-models)
    - [2.3 Multilingual Recognition Models (More languages being updated...)](#23-multilingual-recognition-models-more-languages-being-updated)
  - [3. Text Direction Classification Models](#3-text-direction-classification-models)
  - [4. Paddle-Lite Models](#4-paddle-lite-models)

The downloadable models provided by PaddleOCR include `inference model`, `trained model`, `pre-trained model` and `nb model`. The differences between the models are as follows:

| Model Type | Model Format | Description |
|--- | --- | --- |
| Inference Models | inference.pdmodel, inference.pdiparams | For prediction engine inference, [details](./inference_ppocr.md) |
| Training Models, Pre-trained Models | \*.pdparams, \*.pdopt, \*.states | Parameters, optimizer states, and training info saved during training, mostly used for model evaluation and resuming training |
| nb Models | \*.nb | Optimized by PaddlePaddle Paddle-Lite tool, suitable for mobile/IoT end deployment scenarios (requires Paddle Lite deployment). |


Relationship of the above models is as follows.

![](../imgs_en/model_prod_flow_en.png)

<a name="Text-Detection-Model"></a>
## 1. Text Detection Model

<a name="1.1"></a>

### 1.1 Chinese Detection Model

| Model Name                | Description                                                | Config                             | Model Size | Download                                                                                       |
|---------------------------|------------------------------------------------------------------|---------------------------------------------------|----------------------|-----------------------------------------------------------------------------------------------------|
| ch_PP-OCRv4_det           | **[Latest]** Original ultra-lightweight model supporting Chinese, English, and multilingual text detection.   | [ch_PP-OCRv4_det_cml.yml](../../configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_cml.yml)                        | 4.70M                | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar) |
| ch_PP-OCRv4_server_det    | **[Latest]** Original high-precision model supporting Chinese, English, and multilingual text detection.   | [ch_PP-OCRv4_det_teacher.yml](../../configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml)              | 110M                 | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_train.tar) |
| ch_PP-OCRv3_det_slim      | Slim quantization + distillation version of the ultra-lightweight model supporting Chinese, English, and multilingual text detection. | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)                        | 1.1M                 | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_distill_train.tar) / [Nb Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.nb) |
| ch_PP-OCRv3_det           | Original ultra-lightweight model supporting Chinese, English, and multilingual text detection.               | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)                        | 3.80M                | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) |
| ch_PP-OCRv2_det_slim      | Slim quantization + distillation version of the ultra-lightweight model supporting Chinese, English, and multilingual text detection. | [ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)                        | 3.0M                 | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar) |
| ch_PP-OCRv2_det           | Original ultra-lightweight model supporting Chinese, English, and multilingual text detection.               | [ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)                        | 3.0M                 | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar) |
| ch_ppocr_mobile_slim_v2.0_det | Slim cropped version of the ultra-lightweight model supporting Chinese, English, and multilingual text detection. | [ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)                      | 2.60M                | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar) |
| ch_ppocr_mobile_v2.0_det | Original ultra-lightweight model supporting Chinese, English, and multilingual text detection.               | [ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)                      | 3.0M                 | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar) |
| ch_ppocr_server_v2.0_det | General model supporting Chinese, English, and multilingual text detection, larger than ultra-lightweight models but with better performance. | [ch_det_res18_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml)                | 47.0M                | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar) |


<a name="1.2"></a>

### 1.2 English Detection Model

| Model Name | Description | Config | Model Size | Download |
| --- | --- | --- | --- | --- |
| en_PP-OCRv3_det_slim | 【Latest】Slim quantized version of ultra-lightweight model, supporting English and digit detection | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) | 1.1M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar) / [nb Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.nb) |
| en_PP-OCRv3_det | 【Latest】Original ultra-lightweight model, supporting English and digit detection | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) | 3.8M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) |


<a name="1.3"></a>

### 1.3 Multilingual Detection Model

| Model Name | Description | Config | Model Size | Download |
| --- | --- | --- | --- | --- |
| ml_PP-OCRv3_det_slim | 【Latest】Slim quantized version of ultra-lightweight model, supporting multilingual text detection | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) | 1.1M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_distill_train.tar) / [nb Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.nb) |
| ml_PP-OCRv3_det | 【Latest】Original ultra-lightweight model, supporting multilingual text detection | [ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) | 3.8M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) |

* Note: English configuration file is same as Chinese except training data, here we only provide one configuration file.

<a name="Text_Recognition_Model"></a>
## 2. Text Recognition Model

<a name="Chinese_Recognition_Model"></a>

### 2.1 Chinese Recognition Model

| Model Name | Description | Config | Model Size | Download |
| --- | --- | --- | --- | --- |
| ch_PP-OCRv4_rec | 【Latest】Ultra-lightweight model supporting Chinese and English text recognition | [ch_PP-OCRv4_rec_distill.yml](../../configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_distill.yml) | 10M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar) |
| ch_PP-OCRv4_server_rec | 【Latest】High-precision model supporting Chinese and English text recognition | [ch_PP-OCRv4_rec_hgnet.yml](../../configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml) | 88M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_train.tar) |
| ch_PP-OCRv3_rec_slim | Slim quantized ultra-lightweight model supporting Chinese and English text recognition | [ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml) | 4.9M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_train.tar) / [nb Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.nb) |
| ch_PP-OCRv3_rec | Original ultra-lightweight model supporting Chinese and English text recognition | [ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml) | 12.4M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| ch_PP-OCRv2_rec_slim | Slim quantized ultra-lightweight model supporting Chinese and English text recognition | [ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml) | 9.0M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
| ch_PP-OCRv2_rec | Original ultra-lightweight model supporting Chinese and English text recognition | [ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml) | 8.50M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
| ch_ppocr_mobile_slim_v2.0_rec | Slim pruned quantized ultra-lightweight model supporting Chinese and English text recognition | [rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml) | 6.0M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_train.tar) |
| ch_ppocr_mobile_v2.0_rec | Original ultra-lightweight model supporting Chinese and English text recognition | [rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml) | 5.20M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [Pretrained Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
| ch_ppocr_server_v2.0_rec | Universal model supporting Chinese and English text recognition | [rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml) | 94.8M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [Pretrained Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

**Note:** The `trained model` is fine-tuned on the `pre-trained model` with real data and synthesized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthesized data, which is more suitable for fine-tune on your own dataset.

<a name="English_Recognition_Model"></a>

### 2.2 English Recognition Model

| Model Name | Description | Config | Model Size | Download |
| --- | --- | --- | --- | --- |
| en_PP-OCRv4_rec | **Latest**: Original ultra-lightweight model supporting English and number recognition | [en_PP-OCRv4_rec.yml](../../configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml) | 9.7M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar) |
| en_PP-OCRv3_rec_slim | Slim quantized ultra-lightweight model supporting English and number recognition | [en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml) | 3.2M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_train.tar) / [nb Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.nb) |
| en_PP-OCRv3_rec | Original ultra-lightweight model supporting English and number recognition | [en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml) | 9.6M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| en_number_mobile_slim_v2.0_rec | Slim pruned quantized ultra-lightweight model supporting English and number recognition | [rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml) | 2.7M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_train.tar) |
| en_number_mobile_v2.0_rec | Original ultra-lightweight model supporting English and number recognition | [rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml) | 2.6M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |


<a name="multilingual_recognition_model"></a>
### 2.3 Multilingual Recognition Model（Updating...）


| Model Name | Dict File | Description | Config | Model Size | Download |
| --- | --- | --- | --- | --- | --- |
| korean_PP-OCRv3_rec | ppocr/utils/dict/korean_dict.txt | Korean Recognition | [korean_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml) | 11.0M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar) |
| japan_PP-OCRv3_rec | ppocr/utils/dict/japan_dict.txt | Japanese Recognition | [japan_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/japan_PP-OCRv3_rec.yml) | 11.0M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar) |
| chinese_cht_PP-OCRv3_rec | ppocr/utils/dict/chinese_cht_dict.txt | Traditional Chinese Recognition | [chinese_cht_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/chinese_cht_PP-OCRv3_rec.yml) | 12.0M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar) |
| te_PP-OCRv3_rec | ppocr/utils/dict/te_dict.txt | Telugu Recognition | [te_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/te_PP-OCRv3_rec.yml) | 9.6M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_train.tar) |
| ka_PP-OCRv3_rec | ppocr/utils/dict/ka_dict.txt | Kannada Recognition | [ka_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/ka_PP-OCRv3_rec.yml) | 9.9M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_train.tar) |
| ta_PP-OCRv3_rec | ppocr/utils/dict/ta_dict.txt | Tamil Recognition | [ta_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/ta_PP-OCRv3_rec.yml) | 9.6M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_train.tar) |
| latin_PP-OCRv3_rec |  ppocr/utils/dict/latin_dict.txt | Latin Recognition |  [latin_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/latin_PP-OCRv3_rec.yml) | 9.7M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar) |
| arabic_PP-OCRv3_rec | ppocr/utils/dict/arabic_dict.txt | Arabic Recognition | [arabic_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/arabic_PP-OCRv3_rec.yml) | 9.6M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar) |
| cyrillic_PP-OCRv3_rec | ppocr/utils/dict/cyrillic_dict.txt | Cyrillic Recognition | [cyrillic_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/cyrillic_PP-OCRv3_rec.yml) | 9.6M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar) |
| devanagari_PP-OCRv3_rec | ppocr/utils/dict/devanagari_dict.txt | Devanagari Recognition | [devanagari_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml) | 9.9M | [Inference Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_train.tar) |

For a complete list of languages ​​and tutorials, please refer to : [Multi-language model](./multi_languages_en.md)


<a name="text_direction_classification_model"></a>
## 3. Text Angle Classification Model

| Model Name | Description | Config | Model Size | Download |
| --- | --- | --- | --- | --- |
| ch_ppocr_mobile_slim_v2.0_cls | Slim quantized model, classifying the text line angle of detected text | [cls_mv3.yml](../../configs/cls/cls_mv3.yml) | 2.1M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [NB Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_infer_opt.nb) |
| ch_ppocr_mobile_v2.0_cls | Original classifier model, classifying the text line angle of detected text | [cls_mv3.yml](../../configs/cls/cls_mv3.yml) | 1.38M | [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [Training Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |


<a name="Paddle-Lite Model"></a>
## 4. Paddle-Lite Model

Paddle Lite is an updated version of Paddle-Mobile, an open-open source deep learning framework designed to make it easy to perform inference on mobile, embeded, and IoT devices. It can further optimize the inference model and generate `nb model` used for edge devices. It's suggested to optimize the quantization model using Paddle-Lite because `INT8` format is used for the model storage and inference.

This chapter lists OCR nb models with PP-OCRv2 or earlier versions. You can access to the latest nb models from the above tables.


| Version | Introduction | Model Size | Detection Model | Text Direction  Model | Recognition Model | Paddle-Lite Branch |
| --- | --- | --- | --- | --- | --- | --- |
| PP-OCRv2 | Distilled ultra-lightweight Chinese OCR mobile model | 11.0M | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_det_infer_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_infer_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_rec_infer_opt.nb) | v2.10 |
| PP-OCRv2(slim) | Distilled ultra-lightweight Chinese OCR mobile model | 4.6M | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_det_slim_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_PP-OCRv2_rec_slim_opt.nb) | v2.10 |
| PP-OCRv2 | Distilled ultra-lightweight Chinese OCR mobile model | 11.0M | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer_opt.nb) | v2.9 |
| PP-OCRv2(slim) | Distilled ultra-lightweight Chinese OCR mobile model | 4.9M | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_opt.nb) | v2.9 |
| V2.0 | ppocr_v2.0 ultra-lightweight Chinese OCR mobile model | 7.8M | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_rec_opt.nb) | v2.9 |
| V2.0(slim) | ppocr_v2.0 ultra-lightweight Chinese OCR mobile model | 3.3M | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_det_slim_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_cls_slim_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/dygraph_v2.0/lite/ch_ppocr_mobile_v2.0_rec_slim_opt.nb) | v2.9 |

