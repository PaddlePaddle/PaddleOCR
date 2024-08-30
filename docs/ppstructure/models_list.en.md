---
comments: true
---

# PP-Structure Model list

## 1. Layout Analysis

|model name| description                                                                                                                                             | inference model size                                                                                                                         |download|dict path|
| --- |----| --- | --- | --- |
| picodet_lcnet_x1_0_fgd_layout | The layout analysis English model trained on the PubLayNet dataset based on PicoDet LCNet_x1_0 and FGD . the model can recognition 5 types of areas such as **Text, Title, Table, Picture and List** | 9.7M | [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout.pdparams) | [PubLayNet dict](../../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt) |
| ppyolov2_r50vd_dcn_365e_publaynet | The layout analysis English model trained on the PubLayNet dataset based on PP-YOLOv2 | 221.0M | [inference_moel](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) / [trained model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams) | same as above |
| picodet_lcnet_x1_0_fgd_layout_cdla | The layout analysis Chinese model trained on the CDLA dataset, the model can recognition 10 types of areas such as **Table、Figure、Figure caption、Table、Table caption、Header、Footer、Reference、Equation** | 9.7M | [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams) | [CDLA dict](../../ppocr/utils/dict/layout_dict/layout_cdla_dict.txt) |
| picodet_lcnet_x1_0_fgd_layout_table | The layout analysis model trained on the table dataset, the model can detect tables in Chinese and English documents                     | 9.7M                                                  | [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table.pdparams) | [Table dict](../../ppocr/utils/dict/layout_dict/layout_table_dict.txt) |
| ppyolov2_r50vd_dcn_365e_tableBank_word | The layout analysis model trained on the TableBank Word dataset based on PP-YOLOv2, the model can detect  tables  in English documents | 221.0M | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar) | same as above |
| ppyolov2_r50vd_dcn_365e_tableBank_latex | The layout analysis model trained on the TableBank Latex dataset based on PP-YOLOv2, the model can detect  tables  in English documents | 221.0M                 | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar) | same as above |

## 2. OCR and Table Recognition

### 2.1 OCR

|model name| description | inference model size |download|
| --- |---|---| --- |
|en_ppocr_mobile_v2.0_table_det| Text detection model of English table scenes trained on PubTabNet dataset | 4.7M                |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_det_train.tar) |
|en_ppocr_mobile_v2.0_table_rec| Text recognition model of English table scenes trained on PubTabNet dataset | 6.9M                |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_rec_train.tar) |

If you need to use other OCR models, you can download the model in [PP-OCR model_list](../ppocr/model_list.en.md) or use the model you trained yourself to configure to `det_model_dir`, `rec_model_dir` field.

### 2.2 Table Recognition

|model| description |inference model size|download|
| --- |-----| --- | --- |
|en_ppocr_mobile_v2.0_table_structure| English table recognition model trained on PubTabNet dataset based on TableRec-RARE |6.8M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |
|en_ppstructure_mobile_v2.0_SLANet|English table recognition model trained on PubTabNet dataset based on SLANet|9.2M|[inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_train.tar) |
|ch_ppstructure_mobile_v2.0_SLANet|Chinese table recognition model based on SLANet|9.3M|[inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_train.tar) |

## 3. KIE

On XFUND_zh dataset, Accuracy and time cost of different models on V100 GPU are as follows.

|Model|Backbone|Task|Config|Hmean|Time cost(ms)|Download link|
| --- | --- |  --- | --- | --- | --- |--- |
|VI-LayoutXLM| VI-LayoutXLM-base | SER | [ser_vi_layoutxlm_xfund_zh_udml.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh_udml.yml)|**93.19%**| 15.49| [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | SER | [ser_layoutxlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutxlm_xfund_zh.yml)|90.38%| 19.49 |[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar)|
|LayoutLM| LayoutLM-base | SER | [ser_layoutlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutlm_xfund_zh.yml)|77.31%|-|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | SER | [ser_layoutlmv2_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutlmv2_xfund_zh.yml)|85.44%|31.46|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLMv2_xfun_zh.tar)|
|VI-LayoutXLM| VI-LayoutXLM-base | RE | [re_vi_layoutxlm_xfund_zh_udml.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh_udml.yml)|**83.92%**|15.49|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | RE | [re_layoutxlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/re_layoutxlm_xfund_zh.yml)|74.83%|19.49|[trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | RE | [re_layoutlmv2_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/re_layoutlmv2_xfund_zh.yml)|67.77%|31.46|[trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutLMv2_xfun_zh.tar)|

* Note: The above time cost information just considers inference time without preprocess or postprocess, test environment: `V100 GPU + CUDA 10.2 + CUDNN 8.1.1 + TRT 7.2.3.4`

On wildreceipt dataset, the algorithm result is as follows:

|Model|Backbone|Config|Hmean|Download link|
| --- | --- | --- | --- | --- |
|SDMGR|VGG6|[configs/kie/sdmgr/kie_unet_sdmgr.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/sdmgr/kie_unet_sdmgr.yml)|86.70%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)|
