# PP-Structure Model list

- [1. Layout Analysis](#1)
- [2. OCR and Table Recognition](#2)
    - [2.1 OCR](#21)
    - [2.2 Table Recognition](#22)
- [3. VQA](#3)
- [4. KIE](#4)


<a name="1"></a>
## 1. Layout Analysis

|model name| description                                                                                                                                             |download|label_map|
| --- |---------------------------------------------------------------------------------------------------------------------------------------------------------| --- | --- |
| ppyolov2_r50vd_dcn_365e_publaynet | The layout analysis model trained on the PubLayNet dataset, the model can recognition 5 types of areas such as **text, title, table, picture and list** | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) / [trained model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams) |{0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}|
| ppyolov2_r50vd_dcn_365e_tableBank_word | The layout analysis model trained on the TableBank Word dataset, the model can only detect tables                                                       | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar) | {0:"Table"}|
| ppyolov2_r50vd_dcn_365e_tableBank_latex | The layout analysis model trained on the TableBank Latex dataset, the model can only detect tables                                                      | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar) | {0:"Table"}|

<a name="2"></a>
## 2. OCR and Table Recognition

<a name="21"></a>
### 2.1 OCR

|model name| description | inference model size |download|
| --- |---|---| --- |
|en_ppocr_mobile_v2.0_table_det| Text detection model of English table scenes trained on PubTabNet dataset | 4.7M                |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_det_train.tar) |
|en_ppocr_mobile_v2.0_table_rec| Text recognition model of English table scenes trained on PubTabNet dataset | 6.9M                |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_rec_train.tar) |

If you need to use other OCR models, you can download the model in [PP-OCR model_list](../../doc/doc_ch/models_list.md) or use the model you trained yourself to configure to `det_model_dir`, `rec_model_dir` field.

<a name="22"></a>
### 2.2 Table Recognition

|model| description                                                                 |inference model size|download|
| --- |-----------------------------------------------------------------------------| --- | --- |
|en_ppocr_mobile_v2.0_table_structure| Table structure model for English table scenes trained on PubTabNet dataset |18.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |

<a name="3"></a>
## 3. VQA

|model| description                                                    |inference model size|download|
| --- |----------------------------------------------------------------| --- | --- |
|ser_LayoutXLM_xfun_zh| SER model trained on xfun Chinese dataset based on LayoutXLM   |1.4G|[inference model coming soon]() / [trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar) |
|re_LayoutXLM_xfun_zh| Re model trained on xfun Chinese dataset based on LayoutXLM    |1.4G|[inference model coming soon]() / [trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar) |
|ser_LayoutLMv2_xfun_zh| SER model trained on xfun Chinese dataset based on LayoutXLMv2 |778M|[inference model coming soon]() / [trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLMv2_xfun_zh.tar) |
|re_LayoutLMv2_xfun_zh| Re model trained on xfun Chinese dataset based on LayoutXLMv2  |765M|[inference model coming soon]() / [trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutLMv2_xfun_zh.tar) |
|ser_LayoutLM_xfun_zh| SER model trained on xfun Chinese dataset based on LayoutLM    |430M|[inference model coming soon]() / [trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLM_xfun_zh.tar) |

<a name="4"></a>
## 4. KIE

|model|description|model size|download|
| --- | --- | --- | --- |
|SDMGR|Key Information Extraction Model|78M|[inference model coming soon]() / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)|
