---
comments: true
---

# Algorithms

This tutorial lists the OCR algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCRv3 models list](../ppocr/model_list.en.md).

>
Developers are welcome to contribute more algorithms! Please refer to [add new algorithm](./add_new_algorithm.en.md) guideline.

## 1. Two-stage OCR Algorithms

### 1.1 Text Detection Algorithms

Supported text detection algorithms (Click the link to get the tutorial):

- [x]  [DB && DB++](./text_detection/algorithm_det_db.en.md)
- [x]  [EAST](./text_detection/algorithm_det_east.en.md)
- [x]  [SAST](./text_detection/algorithm_det_sast.en.md)
- [x]  [PSENet](./text_detection/algorithm_det_psenet.en.md)
- [x]  [FCENet](./text_detection/algorithm_det_fcenet.en.md)
- [x]  [DRRG](./text_detection/algorithm_det_drrg.en.md)
- [x]  [CT](./text_detection/algorithm_det_ct.en.md)

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|88.71%|81.36%|84.88%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|78.20%|79.10%|78.65%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|
|PSE|ResNet50_vd|85.81%|79.53%|82.55%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE|MobileNetV3|82.20%|70.48%|75.89%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|
|DB++|ResNet50|90.89%|82.66%|86.58%|[pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_icdar15_train.tar)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|
|CT|ResNet18_vd|88.68%|81.70%|85.05%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r18_ct_train.tar)|

On CTW1500 dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean| Download link|
| --- | --- | --- | --- | --- |---|
|FCE|ResNet50_dcn|88.39%|82.18%|85.27%| [trained model](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar) |
|DRRG|ResNet50_vd|89.92%|80.91%|85.18%|[trained model](https://paddleocr.bj.bcebos.com/contribution/det_r50_drrg_ctw_train.tar)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from:

- [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).
- [Google Drive](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)

### 1.2 Text Recognition Algorithms

Supported text recognition algorithms (Click the link to get the tutorial):

- [x]  [CRNN](./text_recognition/algorithm_rec_crnn.en.md)
- [x]  [Rosetta](./text_recognition/algorithm_rec_rosetta.en.md)
- [x]  [STAR-Net](./text_recognition/algorithm_rec_starnet.en.md)
- [x]  [RARE](./text_recognition/algorithm_rec_rare.en.md)
- [x]  [SRN](./text_recognition/algorithm_rec_srn.en.md)
- [x]  [NRTR](./text_recognition/algorithm_rec_nrtr.en.md)
- [x]  [SAR](./text_recognition/algorithm_rec_sar.en.md)
- [x]  [SEED](./text_recognition/algorithm_rec_seed.en.md)
- [x]  [SVTR](./text_recognition/algorithm_rec_svtr.en.md)
- [x]  [ViTSTR](./text_recognition/algorithm_rec_vitstr.en.md)
- [x]  [ABINet](./text_recognition/algorithm_rec_abinet.en.md)
- [x]  [VisionLAN](./text_recognition/algorithm_rec_visionlan.en.md)
- [x]  [SPIN](./text_recognition/algorithm_rec_spin.en.md)
- [x]  [RobustScanner](./text_recognition/algorithm_rec_robustscanner.en.md)
- [x]  [RFL](./text_recognition/algorithm_rec_rfl.en.md)
- [x]  [ParseQ](./text_recognition/algorithm_rec_parseq.md)
- [x]  [CPPD](./text_recognition/algorithm_rec_cppd.en.md)
- [x]  [SATRN](./text_recognition/algorithm_rec_satrn.en.md)

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
|ParseQ|VIT| 91.24% | rec_vit_parseq_synth | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_synth.tgz) |
|CPPD|SVTR-Base| 93.8% | rec_svtrnet_cppd_base_en | [trained model](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar) |
|SATRN|ShallowCNN| 88.05% | rec_satrn | [trained model](https://pan.baidu.com/s/10J-Bsd881bimKaclKszlaQ?pwd=lk8a) |

### 1.3 Text Super-Resolution Algorithms

Supported text super-resolution algorithms (Click the link to get the tutorial):

- [x]  [Text Gestalt](./super_resolution/algorithm_sr_gestalt.en.md)
- [x]  [Text Telescope](./super_resolution/algorithm_sr_telescope.en.md)

On the TextZoom public dataset, the effect of the algorithm is as follows:

|Model|Backbone|PSNR_Avg|SSIM_Avg|Config|Download link|
|---|---|---|---|---|---|
|Text Gestalt|tsrn|19.28|0.6560| [configs/sr/sr_tsrn_transformer_strock.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/sr/sr_tsrn_transformer_strock.yml)|[trained model](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar)|
|Text Telescope|tbsrn|21.56|0.7411| [configs/sr/sr_telescope.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/sr/sr_telescope.yml)|[trained model](https://paddleocr.bj.bcebos.com/contribution/sr_telescope_train.tar)|

### 1.4 Formula Recognition Algorithm

Supported formula recognition algorithms (Click the link to get the tutorial):

- [x]  [CAN](./formula_recognition/algorithm_rec_can.en.md)
- [x]  [LaTeX-OCR](./formula_recognition/algorithm_rec_latex_ocr.en.md)

On the CROHME handwritten formula dataset, the effect of the algorithm is as follows:

|Model    |Backbone|Config|ExpRate|Download link|
| ----- | ----- | ----- | ----- | ----- |
|CAN|DenseNet|[rec_d28_can.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_d28_can.yml)|51.72%|[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|

## 2. End-to-end OCR Algorithms

Supported end-to-end algorithms (Click the link to get the tutorial):

- [x]  [PGNet](./end_to_end/algorithm_e2e_pgnet.en.md)

## 3. Table Recognition Algorithms

Supported table recognition algorithms (Click the link to get the tutorial):

- [x]  [TableMaster](./table_recognition/algorithm_table_master.en.md)

On the PubTabNet dataset, the algorithm result is as follows:

|Model|Backbone|Config|Acc|Download link|
|---|---|---|---|---|
|TableMaster|TableResNetExtra|[configs/table/table_master.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/table/table_master.yml)|77.47%|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar) / [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_infer.tar)|

## 4. Key Information Extraction Algorithms

Supported KIE algorithms (Click the link to get the tutorial):

- [x]  [VI-LayoutXLM](./kie/algorithm_kie_vi_layoutxlm.en.md)
- [x]  [LayoutLM](./kie/algorithm_kie_layoutxlm.en.md)
- [x]  [LayoutLMv2](./kie/algorithm_kie_layoutxlm.en.md)
- [x]  [LayoutXLM](./kie/algorithm_kie_layoutxlm.en.md)
- [x]  [SDMGR](./kie/algorithm_kie_sdmgr.en.md)

On wildreceipt dataset, the algorithm result is as follows:

|Model|Backbone|Config|Hmean|Download link|
| --- | --- | --- | --- | --- |
|SDMGR|VGG6|[configs/kie/sdmgr/kie_unet_sdmgr.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/sdmgr/kie_unet_sdmgr.yml)|86.70%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)|

On XFUND_zh dataset, the algorithm result is as follows:

|Model|Backbone|Task|Config|Hmean|Download link|
| --- | --- |  --- | --- | --- | --- |
|VI-LayoutXLM| VI-LayoutXLM-base | SER | [ser_vi_layoutxlm_xfund_zh_udml.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh_udml.yml)|**93.19%**|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | SER | [ser_layoutxlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutxlm_xfund_zh.yml)|90.38%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar)|
|LayoutLM| LayoutLM-base | SER | [ser_layoutlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutlm_xfund_zh.yml)|77.31%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | SER | [ser_layoutlmv2_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutlmv2_xfund_zh.yml)|85.44%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLMv2_xfun_zh.tar)|
|VI-LayoutXLM| VI-LayoutXLM-base | RE | [re_vi_layoutxlm_xfund_zh_udml.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh_udml.yml)|**83.92%**|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | RE | [re_layoutxlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/re_layoutxlm_xfund_zh.yml)|74.83%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | RE | [re_layoutlmv2_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/re_layoutlmv2_xfund_zh.yml)|67.77%|[trained model](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutLMv2_xfun_zh.tar)|
