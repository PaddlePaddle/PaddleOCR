# PP-Structure 系列模型列表

- [1. 版面分析模型](#1)
- [2. OCR和表格识别模型](#2)
    - [2.1 OCR](#21)
    - [2.2 表格识别模型](#22)
- [3. VQA模型](#3)
- [4. KIE模型](#4)


<a name="1"></a>
## 1. 版面分析模型

|模型名称|模型简介|下载地址|label_map|
| --- | --- | --- | --- |
| ppyolov2_r50vd_dcn_365e_publaynet | PubLayNet 数据集训练的版面分析模型，可以划分**文字、标题、表格、图片以及列表**5类区域 | [推理模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) / [训练模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams) |{0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}|
| ppyolov2_r50vd_dcn_365e_tableBank_word | TableBank Word 数据集训练的版面分析模型，只能检测表格 | [推理模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar) | {0:"Table"}|
| ppyolov2_r50vd_dcn_365e_tableBank_latex | TableBank Latex 数据集训练的版面分析模型，只能检测表格 | [推理模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar) | {0:"Table"}|

<a name="2"></a>
## 2. OCR和表格识别模型

<a name="21"></a>
### 2.1 OCR

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_det|PubLayNet数据集训练的英文表格场景的文字检测|4.7M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_det_train.tar) |
|en_ppocr_mobile_v2.0_table_rec|PubLayNet数据集训练的英文表格场景的文字识别|6.9M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_rec_train.tar) |

如需要使用其他OCR模型，可以在 [PP-OCR model_list](../../doc/doc_ch/models_list.md) 下载模型或者使用自己训练好的模型配置到 `det_model_dir`, `rec_model_dir`两个字段即可。

<a name="22"></a>
### 2.2 表格识别模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_structure|PubLayNet数据集训练的英文表格场景的表格结构预测|18.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |

<a name="3"></a>
## 3. VQA模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ser_LayoutXLM_xfun_zh|基于LayoutXLM在xfun中文数据集上训练的SER模型|1.4G|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar) |
|re_LayoutXLM_xfun_zh|基于LayoutXLM在xfun中文数据集上训练的RE模型|1.4G|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar) |
|ser_LayoutLMv2_xfun_zh|基于LayoutLMv2在xfun中文数据集上训练的SER模型|778M|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLMv2_xfun_zh.tar) |
|re_LayoutLMv2_xfun_zh|基于LayoutLMv2在xfun中文数据集上训练的RE模型|765M|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutLMv2_xfun_zh.tar) |
|ser_LayoutLM_xfun_zh|基于LayoutLM在xfun中文数据集上训练的SER模型|430M|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutLM_xfun_zh.tar) |

<a name="4"></a>
## 4. KIE模型

|模型名称|模型简介|模型大小|下载地址|
| --- | --- | --- | --- |
|SDMGR|关键信息提取模型|78M|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)|
