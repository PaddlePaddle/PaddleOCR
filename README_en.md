English | [简体中文](README.md)

<p align="center">
 <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/PaddleOCR_logo.png" align="middle" width = "600"/>
<p align="center">
<p align="center">
    <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>

## Introduction

PaddleOCR aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice.

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/demo.gif" width="800">
</div>

## 🚀 Community

PaddleOCR is being oversight by a [PMC](https://github.com/PaddlePaddle/PaddleOCR/issues/12122). Issues and PRs will be reviewed on a best-effort basis. For a complete overview of PaddlePaddle community, please visit [community](https://github.com/PaddlePaddle/community).

⚠️ Note: The [Issues](https://github.com/PaddlePaddle/PaddleOCR/issues) module is only for reporting program 🐞 bugs, for the rest of the questions, please move to the [Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions). Please note that if the Issue mentioned is not a bug, it will be moved to the Discussions module.

## 📣 Recent updates ([more](https://paddlepaddle.github.io/PaddleOCR/en/update.html))

- **🔥2024.7 Added PaddleOCR Algorithm Model Challenge Champion Solutions**:
    - Challenge One, OCR End-to-End Recognition Task Champion Solution: [Scene Text Recognition Algorithm-SVTRv2](https://paddlepaddle.github.io/PaddleOCR/algorithm/text_recognition/algorithm_rec_svtrv2.html);
    - Challenge Two, General Table Recognition Task Champion Solution: [Table Recognition Algorithm-SLANet-LCNetV2](https://paddlepaddle.github.io/PaddleOCR/algorithm/table_recognition/algorithm_table_slanet.html).

## 📚 Documentation

Full documentation can be found on [docs](https://paddlepaddle.github.io/PaddleOCR/en/index.html).

## 🌟 Features

PaddleOCR support a variety of cutting-edge algorithms related to OCR, and developed industrial featured models/solution [PP-OCR](https://paddlepaddle.github.io/PaddleOCR/en/ppocr/overview.html)、 [PP-Structure](https://paddlepaddle.github.io/PaddleOCR/en/ppstructure/overview.html) and [PP-ChatOCR](https://aistudio.baidu.com/aistudio/projectdetail/6488689) on this basis, and get through the whole process of data production, model training, compression, inference and deployment.

<div align="center">
    <img src="./docs/images/ppocrv4.png">
</div>

> It is recommended to start with the “quick experience” in the document tutorial

## ⚡ [Quick Start](https://paddlepaddle.github.io/PaddleOCR/en/quick_start.html)

## 📖 Technical exchange and cooperation

PaddleX provides a one-stop full-process high-efficiency development platform for flying paddle ecological model training, pressure, and push. Its mission is to help AI technology quickly land, and its vision is to make everyone an AI Developer!

- PaddleX currently covers areas such as image classification, object detection, image segmentation, 3D, OCR, and time series prediction, and has built-in 36 basic single models, such as RP-DETR, PP-YOLOE, PP-HGNet, PP-LCNet, PP- LiteSeg, etc.; integrated 12 practical industrial solutions, such as PP-OCRv4, PP-ChatOCR, PP-ShiTu, PP-TS, vehicle-mounted road waste detection, identification of prohibited wildlife products, etc.
- PaddleX provides two AI development modes: "Toolbox" and "Developer". The toolbox mode can tune key hyperparameters without code, and the developer mode can perform single-model training, push and multi-model serial inference with low code, and supports both cloud and local terminals.
- PaddleX also supports joint innovation and development, profit sharing! At present, PaddleX is rapidly iterating, and welcomes the participation of individual developers and enterprise developers to create a prosperous AI technology ecosystem!

## 📚 E-book: *Dive Into OCR*

- [Dive Into OCR](./doc/doc_en/ocr_book_en.md)

## 🎖 Contributors

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>

## ⭐️ Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)

## 🇺🇳 Guideline for New Language Requests

If you want to request a new language support, a PR with 1 following files are needed：

- In folder [ppocr/utils/dict](./ppocr/utils/dict),
it is necessary to submit the dict text to this path and name it with `{language}_dict.txt` that contains a list of all characters. Please see the format example from other files in that folder.

If your language has unique elements, please tell me in advance within any way, such as useful links, wikipedia and so on.

More details, please refer to [Multilingual OCR Development Plan](https://github.com/PaddlePaddle/PaddleOCR/discussions/12734).

## 📄 License

This project is released under [Apache License Version 2.0](./LICENSE).
