## OCR model list（V2.0, updated on 2021.1.20）
**Note** : Compared with [models 1.1](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/models_list_en.md), which are trained with static graph programming paradigm, models 2.0 are the dynamic graph trained version and achieve close performance.

- [1. Text Detection Model](#Detection)
- [2. Text Recognition Model](#Recognition)
    - [Chinese Recognition Model](#Chinese)
    - [English Recognition Model](#English)
    - [Multilingual Recognition Model](#Multilingual)
- [3. Text Angle Classification Model](#Angle)

The downloadable models provided by PaddleOCR include `inference model`, `trained model`, `pre-trained model` and `slim model`. The differences between the models are as follows:

|model type|model format|description|
|--- | --- | --- |
|inference model|inference.pdmodel、inference.pdiparams|Used for reasoning based on Python prediction engine，[detail](./inference_en.md)|
|trained model, pre-trained model|\*.pdparams、\*.pdopt、\*.states |The checkpoints model saved in the training process, which stores the parameters of the model, mostly used for model evaluation and continuous training.|
|slim model|\*.nb|Generally used for Lite deployment|

<a name="Detection"></a>
### 1. Text Detection Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_det|Slim pruned lightweight model, supporting Chinese, English, multilingual text detection|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)| |inference model (coming soon) / slim model (coming soon)|
|ch_ppocr_mobile_v2.0_det|Original lightweight model, supporting Chinese, English, multilingual text detection|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)|3M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|General model, which is larger than the lightweight model, but achieved better performance|[ch_det_res18_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml)|47M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|

<a name="Recognition"></a>
### 2. Text Recognition Model

<a name="Chinese"></a>
#### Chinese Recognition Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)| |inference model (coming soon) / slim model (coming soon) |
|ch_ppocr_mobile_v2.0_rec|Original lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)|3.71M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|General model, supporting Chinese, English and number recognition|[rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml)|94.8M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |


**Note:** The `trained model` is finetuned on the `pre-trained model` with real data and synthsized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthsized data, which is more suitable for finetune on your own dataset.

<a name="English"></a>
#### English Recognition Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|en_number_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting English and number recognition|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)| |inference model (coming soon ) / slim model (coming soon) |
|en_number_mobile_v2.0_rec|Original lightweight model, supporting English and number recognition|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)|2.56M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |

<a name="Multilingual"></a>
#### Multilingual Recognition Model（Updating...）

**Note：** The configuration file of the new multi language model is generated by code. You can use the `--help` parameter to check which multi language are supported by current PaddleOCR.

```bash
# The code needs to run in the specified directory
cd {your/path/}PaddleOCR/configs/rec/multi_language/
python3 generate_multi_language_configs.py --help
```

Take the Italian configuration file as an example：
##### 1.Generate Italian configuration file to test the model provided
you can generate the default configuration file through the following command, and use the default language dictionary provided by paddleocr for prediction.
```bash
# The code needs to run in the specified directory
cd {your/path/}PaddleOCR/configs/rec/multi_language/
# Set the required language configuration file through -l or --language parameter
# This command will write the default parameter to the configuration file.
python3 generate_multi_language_configs.py -l it
```
##### 2. Generate Italian configuration file to train your own data
If you want to train your own model, you can prepare the training set file, verification set file, dictionary file and training data path. Here we assume that the Italian training set, verification set, dictionary and training data path are:
- Training set:{your/path/}PaddleOCR/train_data/train_list.txt
- Validation set: {your/path/}PaddleOCR/train_data/val_list.txt
- Use the default dictionary provided by paddleocr:{your/path/}PaddleOCR/ppocr/utils/dict/it_dict.txt
- Training data path:{your/path/}PaddleOCR/train_data
```bash
# The code needs to run in the specified directory
cd {your/path/}PaddleOCR/configs/rec/multi_language/
# The -l or --language parameter is required
# --train modify train_list path
# --val modify eval_list path 
# --data_dir modify data dir
# -o modify default parameters
# --dict Change the dictionary path. The example uses the default dictionary path, so that this parameter can be empty.
python3 generate_multi_language_configs.py -l it \
--train {path/to/train_list} \
--val {path/to/val_list} \
--data_dir {path/to/data_dir} \
-o Global.use_gpu=False
```
|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
| french_mobile_v2.0_rec |Lightweight model for French recognition|[rec_french_lite_train.yml](../../configs/rec/multi_language/rec_french_lite_train.yml)|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec |Lightweight model for French recognition|[rec_german_lite_train.yml](../../configs/rec/multi_language/rec_german_lite_train.yml)|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec |Lightweight model for Korean recognition|[rec_korean_lite_train.yml](../../configs/rec/multi_language/rec_korean_lite_train.yml)|3.9M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec |Lightweight model for Japanese recognition|[rec_japan_lite_train.yml](../../configs/rec/multi_language/rec_japan_lite_train.yml)|4.23M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| it_mobile_v2.0_rec |Lightweight model for Italian recognition|rec_it_lite_train.yml|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_train.tar) |
| xi_mobile_v2.0_rec |Lightweight model for Spanish recognition|rec_xi_lite_train.yml|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/xi_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/xi_mobile_v2.0_rec_train.tar) |
| pu_mobile_v2.0_rec |Lightweight model for Portuguese recognition|rec_pu_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pu_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pu_mobile_v2.0_rec_train.tar) |
| ru_mobile_v2.0_rec |Lightweight model for Russia recognition|rec_ru_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_train.tar) |
| ar_mobile_v2.0_rec |Lightweight model for Arabic recognition|rec_ar_lite_train.yml|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_train.tar) |
| hi_mobile_v2.0_rec |Lightweight model for Hindi recognition|rec_hi_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_train.tar) |
| chinese_cht_mobile_v2.0_rec |Lightweight model for chinese traditional recognition|rec_chinese_cht_lite_train.yml|5.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_train.tar) |
| ug_mobile_v2.0_rec |Lightweight model for Uyghur recognition|rec_ug_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_train.tar) |
| fa_mobile_v2.0_rec |Lightweight model for Persian recognition|rec_fa_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_train.tar) |
| ur_mobile_v2.0_rec |Lightweight model for Urdu recognition|rec_ur_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_train.tar) |
| rs_mobile_v2.0_rec |Lightweight model for Serbian(latin) recognition|rec_rs_lite_train.yml|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_mobile_v2.0_rec_train.tar) |
| oc_mobile_v2.0_rec |Lightweight model for Occitan recognition|rec_oc_lite_train.yml|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_train.tar) |
| mr_mobile_v2.0_rec |Lightweight model for Marathi recognition|rec_mr_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_train.tar) |
| ne_mobile_v2.0_rec |Lightweight model for Nepali recognition|rec_ne_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_train.tar) |
| rsc_mobile_v2.0_rec |Lightweight model for Serbian(cyrillic) recognition|rec_rsc_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rsc_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rsc_mobile_v2.0_rec_train.tar) |
| bg_mobile_v2.0_rec |Lightweight model for Bulgarian recognition|rec_bg_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_train.tar) |
| uk_mobile_v2.0_rec |Lightweight model for Ukranian recognition|rec_uk_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_train.tar) |
| be_mobile_v2.0_rec |Lightweight model for Belarusian recognition|rec_be_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec |Lightweight model for Telugu recognition|rec_te_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| ka_mobile_v2.0_rec |Lightweight model for Kannada recognition|rec_ka_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec |Lightweight model for Tamil recognition|rec_ta_lite_train.yml|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |


<a name="Angle"></a>
### 3. Text Angle Classification Model

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|Slim quantized model|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)| |inference model (coming soon) / trained model / slim model|
|ch_ppocr_mobile_v2.0_cls|Original model|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)|1.38M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

