English| [简体中文](README.md) 

# Application

PaddleOCR scene application covers general, manufacturing, finance, transportation industry of the main OCR vertical applications, on the basis of the general capabilities of PP-OCR, PP-Structure, in the form of notebook to show the use of scene data fine-tuning, model optimization methods, data augmentation and other content, for developers to quickly land OCR applications to provide demonstration and inspiration.

- [Tutorial](#1)
  - [General](#11)
  - [Manufacturing](#12)
  - [Finance](#13)
  - [Transportation](#14)

- [Model Download](#2)

<a name="1"></a>

## Tutorial

<a name="11"></a>

### General

| Case                                           | Feature          | Model Download       | Tutorial                                | Example                                                      |
| ---------------------------------------------- | ---------------- | -------------------- | --------------------------------------- | ------------------------------------------------------------ |
| High-precision Chineses recognition model SVTR | New model        | [Model Download](#2) | [中文](./高精度中文识别模型.md)/English | <img src="../doc/ppocr_v3/svtr_tiny.png" width=200>          |
| Chinese handwriting recognition                | New font support | [Model Download](#2) | [中文](./手写文字识别.md)/English       | <img src="https://ai-studio-static-online.cdn.bcebos.com/7a8865b2836f42d382e7c3fdaedc4d307d797fa2bcd0466e9f8b7705efff5a7b"  width = "200" height = "100" /> |

<a name="12"></a>

### Manufacturing

| Case                           | Feature                                                      | Model Download       | Tutorial                                                     | Example                                                      |
| ------------------------------ | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Digital tube                   | Digital tube data sythesis, recognition model fine-tuning    | [Model Download](#2) | [中文](./光功率计数码管字符识别/光功率计数码管字符识别.md)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/7d5774a273f84efba5b9ce7fd3f86e9ef24b6473e046444db69fa3ca20ac0986"  width = "200" height = "100" /> |
| LCD screen                     | Detection model distillation, serving deployment             | [Model Download](#2) | [中文](./液晶屏读数识别.md)/English                          | <img src="https://ai-studio-static-online.cdn.bcebos.com/901ab741cb46441ebec510b37e63b9d8d1b7c95f63cc4e5e8757f35179ae6373"  width = "200" height = "100" /> |
| Packaging production data      | Dot matrix character synthesis, overexposure and overdark text recognition | [Model Download](#2) | [中文](./包装生产日期识别.md)/English                        | <img src="https://ai-studio-static-online.cdn.bcebos.com/d9e0533cc1df47ffa3bbe99de9e42639a3ebfa5bce834bafb1ca4574bf9db684"  width = "200" height = "100" /> |
| PCB text recognition           | Small size text detection and recognition                    | [Model Download](#2) | [中文](./PCB字符识别/PCB字符识别.md)/English                 | <img src="https://ai-studio-static-online.cdn.bcebos.com/95d8e95bf1ab476987f2519c0f8f0c60a0cdc2c444804ed6ab08f2f7ab054880"  width = "200" height = "100" /> |
| Meter text recognition         | High-resolution image detection fine-tuning                  | [Model Download](#2) |                                                              |                                                              |
| LCD character defect detection | Non-text character recognition                               |                      |                                                              |                                                              |

<a name="13"></a>

### Finance

| Case                                | Feature                                            | Model Download       | Tutorial                              | Example                                                      |
| ----------------------------------- | -------------------------------------------------- | -------------------- | ------------------------------------- | ------------------------------------------------------------ |
| Form visual question and answer     | Multimodal general form structured extraction      | [Model Download](#2) | [中文](./多模态表单识别.md)/English   | <img src="https://ai-studio-static-online.cdn.bcebos.com/a3b25766f3074d2facdf88d4a60fc76612f51992fd124cf5bd846b213130665b"  width = "200" height = "200" /> |
| VAT invoice                         | Key information extraction, SER, RE task fine-tune | [Model Download](#2) | [中文](./发票关键信息抽取.md)/English | <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg"  width = "200"  /> |
| Seal detection and recognition      | End-to-end curved text recognition                 |                      |                                       |                                                              |
| Universal card recognition          | Universal structured extraction                    |                      |                                       |                                                              |
| ID card recognition                 | Structured extraction, image shading               |                      |                                       |                                                              |
| Contract key information extraction | Dense text detection, NLP concatenation            |                      |                                       |                                                              |

<a name="14"></a>

### Transportation

| Case                                            | Feature                                                      | Model Download       | Tutorial                            | Example                                                      |
| ----------------------------------------------- | ------------------------------------------------------------ | -------------------- | ----------------------------------- | ------------------------------------------------------------ |
| License plate recognition                       | Multi-angle images, lightweight models, edge-side deployment | [Model Download](#2) | [中文](./轻量级车牌识别.md)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/76b6a0939c2c4cf49039b6563c4b28e241e11285d7464e799e81c58c0f7707a7"  width = "200" height = "100" /> |
| Driver's license/driving license identification | coming soon                                                  |                      |                                     |                                                              |
| Express text recognition                        | coming soon                                                  |                      |                                     |                                                              |

<a name="2"></a>

## Model Download

- For international developers: We're building a way to download these trained models, and since the current tutorials are Chinese, if you are good at both Chinese and English, or willing to polish English documents, please let us know in [discussion](https://github.com/PaddlePaddle/PaddleOCR/discussions).
- For Chinese developer: If you want to download the trained application model in the above scenarios, scan the QR code below with your WeChat, follow the PaddlePaddle official account to fill in the questionnaire, and join the PaddleOCR official group to get the 20G OCR learning materials (including "Dive into OCR" e-book, course video, application models and other materials)

  <div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/dd721099bd50478f9d5fb13d8dd00fad69c22d6848244fd3a1d3980d7fefc63e"  width = "150" height = "150" />
  </div>

  If you are an enterprise developer and have not found a suitable solution in the above scenarios, you can fill in the [OCR Application Cooperation Survey Questionnaire](https://paddle.wjx.cn/vj/QwF7GKw.aspx) to carry out different levels of cooperation with the official team **for free**, including but not limited to problem abstraction, technical solution determination, project Q&A, joint research and development, etc. If you have already used paddleOCR in your project, you can also fill out this questionnaire to jointly  promote with the PaddlePaddle and enhance the technical publicity of enterprises. Looking forward to your submission!

<a href="https://trackgit.com">
<img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/l6u6aszdfexs2jnrlil6" alt="trackgit-views" />
</a>
