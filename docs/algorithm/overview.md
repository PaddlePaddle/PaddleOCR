---
comments: true
---

# 前沿算法与模型

本文给出了PaddleOCR已支持的OCR算法列表，以及每个算法在**英文公开数据集**上的模型和指标，主要用于算法简介和算法性能对比，更多包括中文在内的其他数据集上的模型请参考[PP-OCRv3 系列模型下载](../ppocr/model_list.md)。

PaddleOCR将**持续新增**支持OCR领域前沿算法与模型，**欢迎广大开发者合作共建，贡献更多算法。**

新增算法可参考教程：[使用PaddleOCR架构添加新算法](./add_new_algorithm.md)

## 1. 两阶段算法

### 1.1 文本检测算法

已支持的文本检测算法列表（戳链接获取使用教程）：

- [x]  [DB与DB++](./text_detection/algorithm_det_db.md)
- [x]  [EAST](./text_detection/algorithm_det_east.md)
- [x]  [SAST](./text_detection/algorithm_det_sast.md)
- [x]  [PSENet](./text_detection/algorithm_det_psenet.md)
- [x]  [FCENet](./text_detection/algorithm_det_fcenet.md)
- [x]  [DRRG](./text_detection/algorithm_det_drrg.md)
- [x]  [CT](./text_detection/algorithm_det_ct.md)

在ICDAR2015文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|88.71%|81.36%|84.88%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|78.20%|79.10%|78.65%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|
|PSE|ResNet50_vd|85.81%|79.53%|82.55%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE|MobileNetV3|82.20%|70.48%|75.89%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|
|DB++|ResNet50|90.89%|82.66%|86.58%|[合成数据预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_icdar15_train.tar)|

在Total-text文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|
|CT|ResNet18_vd|88.68%|81.70%|85.05%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r18_ct_train.tar)|

在CTW1500文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|FCE|ResNet50_dcn|88.39%|82.18%|85.27%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar)|
|DRRG|ResNet50_vd|89.92%|80.91%|85.18%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/det_r50_drrg_ctw_train.tar)|

**说明：** SAST模型训练额外加入了icdar2013、icdar2017、COCO-Text、ArT等公开数据集进行调优。PaddleOCR用到的经过整理格式的英文公开数据集下载：

- [百度云地址](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (提取码: 2bpi)
- [Google Drive下载地址](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)

### 1.2 文本识别算法

已支持的文本识别算法列表（戳链接获取使用教程）：

- [x]  [CRNN](./text_recognition/algorithm_rec_crnn.md)
- [x]  [Rosetta](./text_recognition/algorithm_rec_rosetta.md)
- [x]  [STAR-Net](./text_recognition/algorithm_rec_starnet.md)
- [x]  [RARE](./text_recognition/algorithm_rec_rare.md)
- [x]  [SRN](./text_recognition/algorithm_rec_srn.md)
- [x]  [NRTR](./text_recognition/algorithm_rec_nrtr.md)
- [x]  [SAR](./text_recognition/algorithm_rec_sar.md)
- [x]  [SEED](./text_recognition/algorithm_rec_seed.md)
- [x]  [SVTR](./text_recognition/algorithm_rec_svtr.md)
- [x]  [ViTSTR](./text_recognition/algorithm_rec_vitstr.md)
- [x]  [ABINet](./text_recognition/algorithm_rec_abinet.md)
- [x]  [VisionLAN](./text_recognition/algorithm_rec_visionlan.md)
- [x]  [SPIN](./text_recognition/algorithm_rec_spin.md)
- [x]  [RobustScanner](./text_recognition/algorithm_rec_robustscanner.md)
- [x]  [RFL](./text_recognition/algorithm_rec_rfl.md)
- [x]  [ParseQ](./text_recognition/algorithm_rec_parseq.md)
- [x]  [CPPD](./text_recognition/algorithm_rec_cppd.md)
- [x]  [SATRN](./text_recognition/algorithm_rec_satrn.md)

参考[DTRB](https://arxiv.org/abs/1904.01906) (3)文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|79.11%|rec_r34_vd_none_none_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|75.80%|rec_mv3_none_none_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|81.04%|rec_r34_vd_none_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|77.95%|rec_mv3_none_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|82.85%|rec_r34_vd_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|79.28%|rec_mv3_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|Resnet34_vd|83.98%|rec_r34_vd_tps_bilstm_att |[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|RARE|MobileNetV3|81.76%|rec_mv3_tps_bilstm_att |[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 86.31% | rec_r50fpn_vd_none_srn | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) |
|NRTR|NRTR_MTB| 84.21% | rec_mtb_nrtr | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |
|SAR|Resnet31| 87.20% | rec_r31_sar | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) |
|SEED|Aster_Resnet| 85.35% | rec_resnet_stn_bilstm_att | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |
|SVTR|SVTR-Tiny| 89.25% | rec_svtr_tiny_none_ctc_en | [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) |
|ViTSTR|ViTSTR| 79.82% | rec_vitstr_none_ce | [训练模型](https://paddleocr.bj.bcebos.com/rec_vitstr_none_ce_train.tar) |
|ABINet|Resnet45| 90.75% | rec_r45_abinet | [训练模型](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar) |
|VisionLAN|Resnet45| 90.30% | rec_r45_visionlan | [训练模型](https://paddleocr.bj.bcebos.com/VisionLAN/rec_r45_visionlan_train.tar) |
|SPIN|ResNet32| 90.00% | rec_r32_gaspin_bilstm_att | [训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_r32_gaspin_bilstm_att.tar) |
|RobustScanner|ResNet31| 87.77% | rec_r31_robustscanner | [训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_r31_robustscanner.tar)|
|RFL|ResNetRFL| 88.63% | rec_resnet_rfl_att | [训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_resnet_rfl_att_train.tar) |
|ParseQ|VIT| 91.24% | rec_vit_parseq_synth | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_synth.tgz) |
|CPPD|SVTR-Base| 93.8% | rec_svtrnet_cppd_base_en | [训练模型](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar) |
|SATRN|ShallowCNN| 88.05% | rec_satrn | [训练模型](https://pan.baidu.com/s/10J-Bsd881bimKaclKszlaQ?pwd=lk8a) |

### 1.3 文本超分辨率算法

已支持的文本超分辨率算法列表（戳链接获取使用教程）：

- [x]  [Text Gestalt](./super_resolution/algorithm_sr_gestalt.md)
- [x]  [Text Telescope](./super_resolution/algorithm_sr_telescope.md)

在TextZoom公开数据集上，算法效果如下：

|模型|骨干网络|PSNR_Avg|SSIM_Avg|配置文件|下载链接|
|---|---|---|---|---|---|
|Text Gestalt|tsrn|19.28|0.6560| [configs/sr/sr_tsrn_transformer_strock.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/sr/sr_tsrn_transformer_strock.yml)|[训练模型](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar)|
|Text Telescope|tbsrn|21.56|0.7411| [configs/sr/sr_telescope.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/sr/sr_telescope.yml)|[训练模型](https://paddleocr.bj.bcebos.com/contribution/sr_telescope_train.tar)|

### 1.4 公式识别算法

已支持的公式识别算法列表（戳链接获取使用教程）：

- [x]  [CAN](./formula_recognition/algorithm_rec_can.md)
- [x]  [LaTeX-OCR](./formula_recognition/algorithm_rec_latex_ocr.md)

在CROHME手写公式数据集上，算法效果如下：

|模型    |骨干网络|配置文件|ExpRate|下载链接|
| ----- | ----- | ----- | ----- | ----- |
|CAN|DenseNet|[rec_d28_can.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_d28_can.yml)|51.72%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|

## 2. 端到端算法

已支持的端到端OCR算法列表（戳链接获取使用教程）：

- [x]  [PGNet](./end_to_end/algorithm_e2e_pgnet.md)

## 3. 表格识别算法

已支持的表格识别算法列表（戳链接获取使用教程）：

- [x]  [TableMaster](./table_recognition/algorithm_table_master.md)

在PubTabNet表格识别公开数据集上，算法效果如下：

|模型|骨干网络|配置文件|acc|下载链接|
|---|---|---|---|---|
|TableMaster|TableResNetExtra|[configs/table/table_master.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/table/table_master.yml)|77.47%|[训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar) / [推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_infer.tar)|

## 4. 关键信息抽取算法

已支持的关键信息抽取算法列表（戳链接获取使用教程）：

- [x]  [VI-LayoutXLM](./kie/algorithm_kie_vi_layoutxlm.md)
- [x]  [LayoutLM](./kie/algorithm_kie_layoutxlm.md)
- [x]  [LayoutLMv2](./kie/algorithm_kie_layoutxlm.md)
- [x]  [LayoutXLM](./kie/algorithm_kie_layoutxlm.md)
- [x]  [SDMGR](./kie/algorithm_kie_sdmgr.md)

在wildreceipt发票公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|hmean|下载链接|
| --- | --- | --- | --- | --- |
|SDMGR|VGG6|[configs/kie/sdmgr/kie_unet_sdmgr.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/sdmgr/kie_unet_sdmgr.yml)|86.70%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)|

在XFUND_zh公开数据集上，算法效果如下：

|模型|骨干网络|任务|配置文件|hmean|下载链接|
| --- | --- |  --- | --- | --- | --- |
|VI-LayoutXLM| VI-LayoutXLM-base | SER | [ser_vi_layoutxlm_xfund_zh_udml.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh_udml.yml)|**93.19%**|[训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | SER | [ser_layoutxlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutxlm_xfund_zh.yml)|90.38%|[训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar)|
|LayoutLM| LayoutLM-base | SER | [ser_layoutlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutlm_xfund_zh.yml)|77.31%|[训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | SER | [ser_layoutlmv2_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/ser_layoutlmv2_xfund_zh.yml)|85.44%|[训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLMv2_xfun_zh.tar)|
|VI-LayoutXLM| VI-LayoutXLM-base | RE | [re_vi_layoutxlm_xfund_zh_udml.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh_udml.yml)|**83.92%**|[训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_pretrained.tar)|
|LayoutXLM| LayoutXLM-base | RE | [re_layoutxlm_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/re_layoutxlm_xfund_zh.yml)|74.83%|[训练模型](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar)|
|LayoutLMv2| LayoutLMv2-base | RE | [re_layoutlmv2_xfund_zh.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/layoutlm_series/re_layoutlmv2_xfund_zh.yml)|67.77%|[训练模型](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutLMv2_xfun_zh.tar)|
