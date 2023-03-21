  [原项目说明文档](README_ch.md)

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">
<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/pypi/format/PaddleOCR?color=c77"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>

# 简介

本项目主要继承自[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)项目，主要对小语种OCR能力进行优化。

# 计划

## 文本检测

- [x] 【PP-OCRv30】通用多语言场景文档大模型
- [ ] 【PP-OCRv30】通用多语言文档大模型

## 文本识别

*阿拉伯语系*

- [x] 【PP-OCRv30】阿拉伯语文本行识别模型
- [x] 【PP-OCRv30】维吾尔语文本行识别

*斯拉夫语系*

- [x] 【PP-OCRv30】俄语文本行识别
- [x] 【PP-OCRv30】哈萨克语文本行识别

*元音附标语系*

- [x] 【PP-OCRv30】藏语文本行识别
- [x] 【PP-OCRv30】缅甸语文本行识别
- [x] 【PP-OCRv30】印地语文本行识别
- [ ] 【PP-OCRv30】高棉语文本行识别
- [ ] 【PP-OCRv30】老挝语文本行识别
- [ ] 【PP-OCRv30】泰语文本行识别

*拉丁语系*

- [ ] 【PP-OCRv30】越南语文本行识别
- [ ] 【PP-OCRv30】马来语文本行识别
- [ ] 【PP-OCRv30】印尼语文本行识别

# 模型

## **文字检测**

|       模型名称       |                   模型简介                   |                           配置文件                           | 模型大小 | hmean  |
| :------------------: | :------------------------------------------: | :----------------------------------------------------------: | :------: | ------ |
| ch_PP-OCRv30_det_dml | 文本行检测大模型，支持中英、多语言文本行检测 | [ch_PP-OCRv30_det_dml.yml](./configs/det/ch_PP-OCRv30/ch_PP-OCRv30_det_dml.yml) |   122M   | 89.87% |

## 文字识别

|     模型名称     |   语言   |                           配置文件                           |  acc   | norm_edit_dis | 备注   |
| :--------------: | :------: | :----------------------------------------------------------: | :----: | :-----------: | ------ |
| ar_PP-OCRv30_rec | 阿拉伯语 | [ar_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/ar_PP-OCRv30_rec.yml) | 73.33% |    97.83%     |        |
| ug_PP-OCRv30_rec | 维吾尔语 | [ug_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/ug_PP-OCRv30_rec.yml) | 70.97% |    93.83%     |        |
| ru_PP-OCRv30_rec |   俄语   | [ru_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/ru_PP-OCRv30_rec.yml) | 93.55% |    99.57%     |        |
| kk_PP-OCRv30_rec | 哈萨克语 | [kk_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/kk_PP-OCRv30_rec.yml) | 59.37% |    95.30%     |        |
| bo_PP-OCRv30_rec |   藏语   | [bo_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/bo_PP-OCRv30_rec.yml) | 67.09% |    93.07%     | 需优化 |
| my_PP-OCRv30_rec |  缅甸语  | [my_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/my_PP-OCRv30_rec.yml) | 74.19% |      97%      |        |
| hi_PP-OCRv30_rec |  印地语  | [hi_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/hi_PP-OCRv30_rec.yml) | 43.01% |    85.05%     | 需优化 |
| km_PP-OCRv30_rec |  高棉语  | [km_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/km_PP-OCRv30_rec.yml) |        |               |        |
| lo_PP-OCRv30_rec |  老挝语  | [lo_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/lo_PP-OCRv30_rec.yml) |        |               |        |
| th_PP-OCRv30_rec |   泰语   | [th_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/th_PP-OCRv30_rec.yml) |        |               |        |
| vi_PP-OCRv30_rec |   越南   |                                                              |        |               |        |
| ms_PP-OCRv30_rec |   马来   |                                                              |        |               |        |
| id_PP-OCRv30_rec |  印尼语  |                                                              |        |               |        |

注：测试数据采用场景数据。

## 📄 License

This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
