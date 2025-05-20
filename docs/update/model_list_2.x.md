# PaddleOCR 2.x及更低版本支持的模型

由于串联逻辑和模型训练、推理时用到的配置不同，PaddleOCR 2.x 版本的 PP-OCRv4、PP-OCRv3 系列模型与 PaddleOCR 3.0 及以上版本的 PP-OCRv4、PP-OCRv3 系列模型不能互相使用。

## 检测模型

### 中文检测模型

* ch_PP-OCRv4_det
* ch_PP-OCRv4_server_det
* ch_PP-OCRv3_det_slim
* ch_PP-OCRv3_det
* ch_PP-OCRv2_det_slim
* ch_PP-OCRv2_det
* ch_ppocr_mobile_slim_v2.0_det
* ch_ppocr_mobile_v2.0_det
* ch_ppocr_server_v2.0_det

### 英文检测模型

* en_PP-OCRv3_det_slim
* en_PP-OCRv3_det

### 多语言检测模型

* ml_PP-OCRv3_det_slim
* ml_PP-OCRv3_det

## 识别模型

### 中文识别模型

* ch_PP-OCRv4_rec
* ch_PP-OCRv4_server_rec
* ch_PP-OCRv4_server_rec_doc
* ch_PP-OCRv3_rec_slim
* ch_PP-OCRv3_rec
* ch_PP-OCRv2_rec_slim
* ch_PP-OCRv2_rec
* ch_ppocr_mobile_slim_v2.0_rec
* ch_ppocr_mobile_v2.0_rec
* ch_ppocr_server_v2.0_rec
* SVTRv2(Rec Sever)
* RepSVTR(Mobile)

### 英文识别模型

* en_PP-OCRv4_rec
* en_PP-OCRv3_rec_slim
* en_PP-OCRv3_rec
* en_number_mobile_slim_v2.0_rec
* en_number_mobile_v2.0_rec

### 多语言识别模型

* korean_PP-OCRv3_rec
* japan_PP-OCRv3_rec
* chinese_cht_PP-OCRv3_rec
* te_PP-OCRv3_rec
* ka_PP-OCRv3_rec
* ta_PP-OCRv3_rec
* latin_PP-OCRv3_rec
* arabic_PP-OCRv3_rec
* cyrillic_PP-OCRv3_rec
* devanagari_PP-OCRv3_rec

## 端到端OCR模型

* PGNet

## 文本方向分类模型

* ch_ppocr_mobile_slim_v2.0_cls
* ch_ppocr_mobile_v2.0_cls

## 公式识别模型

* CAN
* UniMERNet
* LaTeX-OCR
* PP-FormulaNet-S
* PP-FormulaNet-L

## 表格识别模型

* TableMaster
* SLANet
* SLANeXt_wired
* SLANeXt_wireless
* en_ppocr_mobile_v2.0_table_structure
* en_ppstructure_mobile_v2.0_SLANet
* ch_ppstructure_mobile_v2.0_SLANet

## 表格OCR模型

* en_ppocr_mobile_v2.0_table_det
* en_ppocr_mobile_v2.0_table_rec

## 版面检测模型

* picodet_lcnet_x1_0_fgd_layout
* ppyolov2_r50vd_dcn_365e_publaynet
* picodet_lcnet_x1_0_fgd_layout_cdla
* picodet_lcnet_x1_0_fgd_layout_table
* ppyolov2_r50vd_dcn_365e_tableBank_word
* ppyolov2_r50vd_dcn_365e_tableBank_latex
