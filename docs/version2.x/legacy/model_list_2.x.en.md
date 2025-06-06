# Models Supported by PaddleOCR 2.x and Earlier Versions

Due to differences in the concatenation logic and configurations used during model training and inference, the PP-OCRv4 and PP-OCRv3 series models from the PaddleOCR 2.x branch cannot be used interchangeably with those from the PaddleOCR 3.0 and later branches.

## Detection Models

### Chinese Detection Models

* ch_PP-OCRv4_det
* ch_PP-OCRv4_server_det
* ch_PP-OCRv3_det_slim
* ch_PP-OCRv3_det
* ch_PP-OCRv2_det_slim
* ch_PP-OCRv2_det
* ch_ppocr_mobile_slim_v2.0_det
* ch_ppocr_mobile_v2.0_det
* ch_ppocr_server_v2.0_det

### English Detection Models

* en_PP-OCRv3_det_slim
* en_PP-OCRv3_det

### Multilingual Detection Models

* ml_PP-OCRv3_det_slim
* ml_PP-OCRv3_det

## Recognition Models

### Chinese Recognition Models

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

### English Recognition Models

* en_PP-OCRv4_rec
* en_PP-OCRv3_rec_slim
* en_PP-OCRv3_rec
* en_number_mobile_slim_v2.0_rec
* en_number_mobile_v2.0_rec

### Multilingual Recognition Models

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

## End-to-End OCR Models

* PGNet

## Text Direction Classification Models

* ch_ppocr_mobile_slim_v2.0_cls
* ch_ppocr_mobile_v2.0_cls

## Formula Recognition Models

* CAN
* UniMERNet
* LaTeX-OCR
* PP-FormulaNet-S
* PP-FormulaNet-L

## Table Recognition Models

* TableMaster
* SLANet
* SLANeXt_wired
* SLANeXt_wireless
* en_ppocr_mobile_v2.0_table_structure
* en_ppstructure_mobile_v2.0_SLANet
* ch_ppstructure_mobile_v2.0_SLANet

## Table OCR Models

* en_ppocr_mobile_v2.0_table_det
* en_ppocr_mobile_v2.0_table_rec

## Layout Detection Models

* picodet_lcnet_x1_0_fgd_layout
* ppyolov2_r50vd_dcn_365e_publaynet
* picodet_lcnet_x1_0_fgd_layout_cdla
* picodet_lcnet_x1_0_fgd_layout_table
* ppyolov2_r50vd_dcn_365e_tableBank_word
* ppyolov2_r50vd_dcn_365e_tableBank_latex
