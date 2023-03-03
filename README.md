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

- [x] 【PP-OCRv30】【文本检测】通用多语言场景文档RestNet50模型
- [ ] 【PP-OCRv30】【文本检测】通用多语言文档RestNet50模型
- [x] 【PP-OCRv30】【文本识别】阿拉伯语文本行识别模型
- [ ] 【PP-OCRv30】【文本识别】维吾尔语文本行识别
- [ ] 【PP-OCRv30】【文本识别】俄语文本行识别
- [ ] 【PP-OCRv30】【文本识别】哈萨克语文本行识别
- [ ] 【PP-OCRv30】【文本识别】藏语文本行识别
- [ ] 【PP-OCRv30】【文本识别】缅甸语文本行识别
- [ ] 【PP-OCRv30】【文本识别】印地语文本行识别
- [ ] 【PP-OCRv30】【文本识别】越南语文本行识别
- [ ] 【PP-OCRv30】【文本识别】马来语文本行识别
- [ ] 【PP-OCRv30】【文本识别】印尼语文本行识别

# 模型

## **文字检测**

|       模型名称       |                   模型简介                   |                           配置文件                           | 推理模型大小 | hmean  |
| :------------------: | :------------------------------------------: | :----------------------------------------------------------: | :----------: | ------ |
| ch_PP-OCRv30_det_dml | 文本行检测大模型，支持中英、多语言文本行检测 | [ch_PP-OCRv30_det_dml.yml](./configs/det/ch_PP-OCRv30/ch_PP-OCRv30_det_dml.yml) |     122M     | 89.87% |

## 文字识别

| 模型名称         | 模型简介           | 配置文件                                                     | 推理模型大小 |  acc  | norm_edit_dis |
| ---------------- | ------------------ | ------------------------------------------------------------ | ------------ | :---: | :-----------: |
| ar_PP-OCRv30_rec | 阿拉伯语文本行识别 | [ar_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/ar_PP-OCRv30_rec.yml) | 8.69         | 73.33 |     97.83     |
| ug_PP-OCRv30_rec | 维吾尔语文本行识别 | [ug_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/ug_PP-OCRv30_rec.yml) |              |       |               |
| ru_PP-OCRv30_rec | 俄语文本行识别     | [ru_PP-OCRv30_rec.yml](./configs/rec/PP-OCRv30/ru_PP-OCRv30_rec.yml) |              |       |               |
| ka_PP-OCRv30_rec | 哈萨克语文本行识别 |                                                              |              |       |               |
| bo_PP-OCRv30_rec | 藏语文本行识别     |                                                              |              |       |               |
| my_PP-OCRv30_rec | 缅甸语文本行识别   |                                                              |              |       |               |
| hi_PP-OCRv30_rec | 印地语文本行识别   |                                                              |              |       |               |
| vi_PP-OCRv30_rec | 越南语文本行识别   |                                                              |              |       |               |
| ms_PP-OCRv30_rec | 马来语文本行识别   |                                                              |              |       |               |
| id_PP-OCRv30_rec | 印尼语文本行识别   |                                                              |              |       |               |

## 📄 License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
