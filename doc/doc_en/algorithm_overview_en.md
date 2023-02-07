# Algorithms

- [1. Two-stage OCR Algorithms](#1)
  - [1.1 Text Detection Algorithms](#11)
  - [1.2 Text Recognition Algorithms](#12)
  - [1.3 Text Super-Resolution Algorithms](#13)
  - [1.4 Formula Recognition Algorithm](#14)
- [2. End-to-end OCR Algorithms](#2)
- [3. Table Recognition Algorithms](#3)
- [4. Key Information Extraction Algorithms](#4)

This tutorial lists the OCR algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCRv3 models list](./models_list_en.md).

>>
Developers are welcome to contribute more algorithms! Please refer to [add new algorithm](./add_new_algorithm_en.md) guideline.


<a name="1"></a>

## 1. Two-stage OCR Algorithms

<a name="11"></a>

### 1.1 Text Detection Algorithms

Supported text detection algorithms (Click the link to get the tutorial):
- [x]  [DB && DB++](./algorithm_det_db_en.md)
- [x]  [EAST](./algorithm_det_east_en.md)
- [x]  [SAST](./algorithm_det_sast_en.md)
- [x]  [PSENet](./algorithm_det_psenet_en.md)
- [x]  [FCENet](./algorithm_det_fcenet_en.md)
- [x]  [DRRG](./algorithm_det_drrg_en.md)

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|88.71%|81.36%|84.88%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|78.20%|79.10%|78.65%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|
|PSE|ResNet50_vd|85.81%|79.53%|82.55%|[trianed model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE|MobileNetV3|82.20%|70.48%|75.89%|[trianed model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|
|DB++|ResNet50|90.89%|82.66%|86.58%|[pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_icdar15_train.tar)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

On CTW1500 dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean| Download link|
| --- | --- | --- | --- | --- |---|  
|FCE|ResNet50_dcn|88.39%|82.18%|85.27%| [trained model](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar) |
|DRRG|ResNet50_vd|89.92%|80.91%|85.18%|[trained model](https://paddleocr.bj.bcebos.com/contribution/det_r50_drrg_ctw_train.tar)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from:
* [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).
* [Google Drive](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)


<a name="12"></a>
### 1.2 Text Recognition Algorithms

Supported text recognition algorithms (Click the link to get the tutorial):
- [x]  [CRNN](./algorithm_rec_crnn_en.md)
- [x]  [Rosetta](./algorithm_rec_rosetta_en.md)
- [x]  [STAR-Net](./algorithm_rec_starnet_en.md)
- [x]  [RARE](./algorithm_rec_rare_en.md)
- [x]  [SRN](./algorithm_rec_srn_en.md)
- [x]  [NRTR](./algorithm_rec_nrtr_en.md)
- [x]  [SAR](./algorithm_rec_sar_en.md)
- [x]  [SEED](./algorithm_rec_seed_en.md)
- [x]  [SVTR](./algorithm_rec_svtr_en.md)
- [x]  [ViTSTR](./algorithm_rec_vitstr_en.md)
- [x]  [ABINet](./algorithm_rec_abinet_en.md)
- [x]  [VisionLAN](./algorithm_rec_visionlan_en.md)
- [x]  [SPIN](./algorithm_rec_spin_en.md)
- [x]  [RobustScanner](./algorithm_rec_robustscanner_en.md)
- [x]  [RFL](./algorithm_rec_rfl_en.md)

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|79.11%|rec_r34_vd_none_none_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|75.80%|rec_mv3_none_none_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|81.04%|rec_r34_vd_none_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|77.95%|rec_mv3_none_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|82.85%|rec_r34_vd_tps_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|79.28%|rec_mv3_tps_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|Resnet34_vd|83.98%|rec_r34_vd_tps_bilstm_att |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|RARE|MobileNetV3|81.76%|rec_mv3_tps_bilstm_att |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 86.31% | rec_r50fpn_vd_none_srn |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar)|
|NRTR|NRTR_MTB| 84.21% | rec_mtb_nrtr | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |
|SAR|Resnet31| 87.20% | rec_r31_sar | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) |
|SEED|Aster_Resnet| 85.35% | rec_resnet_stn_bilstm_att | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |
|SVTR|SVTR-Tiny| 89.25% | rec_svtr_tiny_none_ctc_en | [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) |
|ViTSTR|ViTSTR| 79.82% | rec_vitstr_none_ce | [trained model](https://paddleocr.bj.bcebos.com/rec_vitstr_none_ce_train.tar) |
|ABINet|Resnet45| 90.75% | rec_r45_abinet | [trained model](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar) |
|VisionLAN|Resnet45| 90.30% | rec_r45_visionlan | [trained model](https://paddleocr.bj.bcebos.com/VisionLAN/rec_r45_visionlan_train.tar) |
|SPIN|ResNet32| 90.00% | rec_r32_gaspin_bilstm_att | [trained model](https://paddleocr.bj.bcebos.com/contribution/rec_r32_gaspin_bilstm_att.tar) |
|RobustScanner|ResNet31| 87.77% | rec_r31_robustscanner | [trained model](https://paddleocr.bj.bcebos.com/contribution/rec_r31_robustscanner.tar)|
|RFL|ResNetRFL| 88.63% | rec_resnet_rfl_att | [trained model](https://paddleocr.bj.bcebos.com/contribution/rec_resnet_rfl_att_train.tar) |

<a name="13"></a>

### 1.3 Text Super-Resolution Algorithms

Supported text super-resolution algorithms (Click the link to get the tutorial):
- [x]  [Text Gestalt](./algorithm_sr_gestalt.md)
- [x]  [Text Telescope](./algorithm_sr_telescope.md)

On the TextZoom public dataset, the effect of the algorithm is as follows:

|Model|Backbone|PSNR_Avg|SSIM_Avg|Config|Download link|
|---|---|---|---|---|---|
|Text Gestalt|tsrn|19.28|0.6560| [configs/sr/sr_tsrn_transformer_strock.yml](../../configs/sr/sr_tsrn_transformer_strock.yml)|[trained model](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar)|
|Text Telescope|tbsrn|21.56|0.7411| [configs/sr/sr_telescope.yml](../../configs/sr/sr_telescope.yml)|[trained model](https://paddleocr.bj.bcebos.com/contribution/sr_telescope_train.tar)|

<a name="14"></a>

### 1.4 Formula Recognition Algorithm

Supported formula recognition algorithms (Click the link to get the tutorial):

- [x]  [CAN](./algorithm_rec_can_en.md)

On the CROHME handwritten formula dataset, the effect of the algorithm is as follows:

|Model    |Backbone|Config|ExpRate|Download link|
| ----- | ----- | ----- | ----- | ----- |
|CAN|DenseNet|[rec_d28_can.yml](../../configs/rec/rec_d28_can.yml)|51.72%|[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|


<a name="2"></a>

## 2. End-to-end OCR Algorithms

Supported end-to-end algorithms (Click the link to get the tutorial):
- [x]  [PGNet](./algorithm_e2e_pgnet_en.md)


<a name="3"></a>
## 3. Table Recognition Algorithms

Supported table recognition algorithms (Click the link to get the tutorial):
- [x]  [TableMaster](./algorithm_table_master_en.md)

On the PubTabNet dataset, the algorithm result is as follows:

|Model|Backbone|Config|Acc|Download link|
|---|---|---|---|---|
|TableMaster|TableResNetExtra|[configs/table/table_master.yml](../../configs/table/table_master.yml)|77.47%|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar) / [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_infer.tar)|


<a name="4"></a>

## 4. Key Information Extraction Algorithms

Supported KIE algorithms (Click the link to get the tutorial):

- [x]  [VI-LayoutXLM](./algorithm_kie_vi_layoutxlm_en.md)
- [x]  [LayoutLM](./algorithm_kie_layoutxlm_en.md)
- [x]  [LayoutLMv2](./algorithm_kie_layoutxlm_en.md)
- [x]  [LayoutXLM](./algorithm_kie_layoutxlm_en.md)
- [x]  [SDMGR](./algorithm_kie_sdmgr_en.md)

On wildreceipt dataset, the algorithm result is as follows:

|Model|Backbone|Config|Hmean|Download link|
| --- | --- | --- | --- | --- |
|SDMGR|VGG6|[configs/kie/sdmgr/kie_unet_sdmgr.yml](../../configs/kie/sdmgr/kie_unet_sdmgr.yml)|86.70%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)|

On XFUND_zh dataset, the algorithm result is as follows:

|Model|Backbone|Task|Config|Hmean|Download link|
| --- | --- |  --- | --- | --- | --- |
|VI-LayoutXLM| VI-LayoutXLM-base | SER | [ser_vi_layoutxlm_xfund_zh_udml.yml](../../configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh_udml.yml)|**93.19%**|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | SER | [ser_layoutxlm_xfund_zh.yml](../../configs/kie/layoutlm_series/ser_layoutxlm_xfund_zh.yml)|90.38%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar)|
|LayoutLM| LayoutLM-base | SER | [ser_layoutlm_xfund_zh.yml](../../configs/kie/layoutlm_series/ser_layoutlm_xfund_zh.yml)|77.31%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | SER | [ser_layoutlmv2_xfund_zh.yml](../../configs/kie/layoutlm_series/ser_layoutlmv2_xfund_zh.yml)|85.44%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLMv2_xfun_zh.tar)|
|VI-LayoutXLM| VI-LayoutXLM-base | RE | [re_vi_layoutxlm_xfund_zh_udml.yml](../../configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh_udml.yml)|**83.92%**|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | RE | [re_layoutxlm_xfund_zh.yml](../../configs/kie/layoutlm_series/re_layoutxlm_xfund_zh.yml)|74.83%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | RE | [re_layoutlmv2_xfund_zh.yml](../../configs/kie/layoutlm_series/re_layoutlmv2_xfund_zh.yml)|67.77%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutLMv2_xfun_zh.tar)|
