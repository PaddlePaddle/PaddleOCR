[English](README_en.md) | 简体中文

# 场景应用

PaddleOCR场景应用覆盖通用，制造、金融、交通行业的主要OCR垂类应用，在PP-OCR、PP-Structure的通用能力基础之上，以notebook的形式展示利用场景数据微调、模型优化方法、数据增广等内容，为开发者快速落地OCR应用提供示范与启发。

- [教程文档](#1)
  - [通用](#11)
  - [制造](#12)
  - [金融](#13)
  - [交通](#14)

- [模型下载](#2)

<a name="1"></a>

## 教程文档

<a name="11"></a>

### 通用

| 类别                   | 亮点                                                         | 模型下载       | 教程                                    | 示例图                                                       |
| ---------------------- | ------------------------------------------------------------ | -------------- | --------------------------------------- | ------------------------------------------------------------ |
| 高精度中文识别模型SVTR | 比PP-OCRv3识别模型精度高3%，<br />可用于数据挖掘或对预测效率要求不高的场景。 | [模型下载](#2) | [中文](./高精度中文识别模型.md)/English | <img src="../doc/ppocr_v3/svtr_tiny.png" width=200>          |
| 手写体识别             | 新增字形支持                                                 | [模型下载](#2) | [中文](./手写文字识别.md)/English       | <img src="https://ai-studio-static-online.cdn.bcebos.com/7a8865b2836f42d382e7c3fdaedc4d307d797fa2bcd0466e9f8b7705efff5a7b"  width = "200" height = "100" /> |

<a name="12"></a>

### 制造

| 类别           | 亮点                           | 模型下载       | 教程                                                         | 示例图                                                       |
| -------------- | ------------------------------ | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 数码管识别     | 数码管数据合成、漏识别调优     | [模型下载](#2) | [中文](./光功率计数码管字符识别/光功率计数码管字符识别.md)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/7d5774a273f84efba5b9ce7fd3f86e9ef24b6473e046444db69fa3ca20ac0986"  width = "200" height = "100" /> |
| 液晶屏读数识别 | 检测模型蒸馏、Serving部署      | [模型下载](#2) | [中文](./液晶屏读数识别.md)/English                          | <img src="https://ai-studio-static-online.cdn.bcebos.com/901ab741cb46441ebec510b37e63b9d8d1b7c95f63cc4e5e8757f35179ae6373"  width = "200" height = "100" /> |
| 包装生产日期   | 点阵字符合成、过曝过暗文字识别 | [模型下载](#2) | [中文](./包装生产日期识别.md)/English                        | <img src="https://ai-studio-static-online.cdn.bcebos.com/d9e0533cc1df47ffa3bbe99de9e42639a3ebfa5bce834bafb1ca4574bf9db684"  width = "200" height = "100" /> |
| PCB文字识别    | 小尺寸文本检测与识别           | [模型下载](#2) | [中文](./PCB字符识别/PCB字符识别.md)/English                 | <img src="https://ai-studio-static-online.cdn.bcebos.com/95d8e95bf1ab476987f2519c0f8f0c60a0cdc2c444804ed6ab08f2f7ab054880"  width = "200" height = "100" /> |
| 电表识别       | 大分辨率图像检测调优           | [模型下载](#2) |                                                              |                                                              |
| 液晶屏缺陷检测 | 非文字字符识别                 |                |                                                              |                                                              |

<a name="13"></a>

### 金融

| 类别           | 亮点                          | 模型下载       | 教程                                  | 示例图                                                       |
| -------------- | ----------------------------- | -------------- | ------------------------------------- | ------------------------------------------------------------ |
| 表单VQA        | 多模态通用表单结构化提取      | [模型下载](#2) | [中文](./多模态表单识别.md)/English   | <img src="https://ai-studio-static-online.cdn.bcebos.com/a3b25766f3074d2facdf88d4a60fc76612f51992fd124cf5bd846b213130665b"  width = "200" height = "200" /> |
| 增值税发票     | 关键信息抽取，SER、RE任务训练 | [模型下载](#2) | [中文](./发票关键信息抽取.md)/English | <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg"  width = "200"  /> |
| 印章检测与识别 | 端到端弯曲文本识别            |                |                                       |                                                              |
| 通用卡证识别   | 通用结构化提取                |                |                                       |                                                              |
| 身份证识别     | 结构化提取、图像阴影          |                |                                       |                                                              |
| 合同比对       | 密集文本检测、NLP串联         |                |                                       |                                                              |

<a name="14"></a>

### 交通

| 类别              | 亮点                           | 模型下载       | 教程                                | 示例图                                                       |
| ----------------- | ------------------------------ | -------------- | ----------------------------------- | ------------------------------------------------------------ |
| 车牌识别          | 多角度图像、轻量模型、端侧部署 | [模型下载](#2) | [中文](./轻量级车牌识别.md)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/76b6a0939c2c4cf49039b6563c4b28e241e11285d7464e799e81c58c0f7707a7"  width = "200" height = "100" /> |
| 驾驶证/行驶证识别 | 尽请期待                       |                |                                     |                                                              |
| 快递单识别        | 尽请期待                       |                |                                     |                                                              |

<a name="2"></a>

## 模型下载

如需下载上述场景中已经训练好的垂类模型，可以扫描下方二维码，关注公众号填写问卷后，加入PaddleOCR官方交流群获取20G OCR学习大礼包（内含《动手学OCR》电子书、课程回放视频、前沿论文等重磅资料）

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/dd721099bd50478f9d5fb13d8dd00fad69c22d6848244fd3a1d3980d7fefc63e"  width = "150" height = "150" />
</div>

如果您是企业开发者且未在上述场景中找到合适的方案，可以填写[OCR应用合作调研问卷](https://paddle.wjx.cn/vj/QwF7GKw.aspx)，免费与官方团队展开不同层次的合作，包括但不限于问题抽象、确定技术方案、项目答疑、共同研发等。如果您已经使用PaddleOCR落地项目，也可以填写此问卷，与飞桨平台共同宣传推广，提升企业技术品宣。期待您的提交！

<a href="https://trackgit.com">
<img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/l63cvzo0w09yxypc7ygl" alt="traffic" />
</a>
