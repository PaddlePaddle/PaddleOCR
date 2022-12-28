[English](README_en.md) | 简体中文

# OCR产业范例20讲

PaddleOCR场景应用覆盖通用，制造、金融、交通等行业的主要OCR垂类应用，基于PP-OCR、PP-Structure的通用能力和各类垂类场景中落地的经验，PaddleOCR联合**北京师范大学副教授柯永红、云南省能源投资集团财务有限公司智能化项目经理钟榆星、信雅达科技股份有限公司高级研发工程师张少华、郑州三晖电气股份有限公司工程师郭媛媛、福建中烟工业有限责任公司工程师顾茜、内蒙古阿尔泰电子信息技术有限公司CTO欧日乐克、安科私（北京）科技有限公司创始人柯双喜等产学研同仁共同开源《OCR产业范例20讲》电子书**，通过Notebook的形式系统展示OCR在产业界应用的具体场景的调优过程与落地经验，为开发者快速落地OCR应用提供示范与启发。该书包含以下特点：

- 20例OCR在工业、金融、教育、交通等行业的关键场景应用范例；
- 覆盖从问题抽象、数据处理、训练调优、部署应用的全流程AI落地环节，为开发者提供常见的OCR优化思路；
- 每个范例配有交互式Notebook教程，通过代码展示获得实际结果，便于学习修改与二次开发；
- GitHub和AI Studio上开源本书中涉及的范例内容和代码，方便开发者学习和使用。

<a name="1"></a>

## 教程文档

《OCR产业范例20讲》中包含如下教程。如需获取整合后的电子版，请参考[资料下载](#2)

<a name="11"></a>

### 通用

| 类别                   | 亮点                                                         | 模型下载       | 教程                                                         | 示例图                                                       |
| ---------------------- | ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 高精度中文识别模型SVTR | 比PP-OCRv3识别模型精度高3%，<br />可用于数据挖掘或对预测效率要求不高的场景。 | [模型下载](#2) | [中文](./高精度中文识别模型.md)/English                      | <img src="../doc/ppocr_v3/svtr_tiny.png" width=200>          |
| 手写体识别             | 新增字形支持                                                 | [模型下载](#2) | [中文](./手写文字识别.md)/English                            | <img src="https://ai-studio-static-online.cdn.bcebos.com/7a8865b2836f42d382e7c3fdaedc4d307d797fa2bcd0466e9f8b7705efff5a7b"  width = "200" height = "100" /> |
| 蒙文识别               | 新语种识别支持                                               | 即将开源       | [中文](./蒙古文书籍文字识别.md)/English                      | <img src="https://user-images.githubusercontent.com/50011306/206182391-431c2441-1d1d-4f25-931c-b0f663bf3285.png"  width = "200" height = "100" /> |
| 甲骨文识别             | 新语种识别支持                                               | [模型下载](#2) | [中文](https://aistudio.baidu.com/aistudio/projectdetail/5216041?contributionType=1)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/b973566a4897458cb4ed76ecbc8e4a838d68ac471a504c0daa57c17bc203c4e0"  width = "200" height = "100" /> |

<a name="12"></a>

### 制造

| 类别           | 亮点                           | 模型下载       | 教程                                                         | 示例图                                                       |
| -------------- | ------------------------------ | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 数码管识别     | 数码管数据合成、漏识别调优     | [模型下载](#2) | [中文](./光功率计数码管字符识别/光功率计数码管字符识别.md)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/7d5774a273f84efba5b9ce7fd3f86e9ef24b6473e046444db69fa3ca20ac0986"  width = "200" height = "100" /> |
| 液晶屏读数识别 | 检测模型蒸馏、Serving部署      | [模型下载](#2) | [中文](./液晶屏读数识别.md)/English                          | <img src="https://ai-studio-static-online.cdn.bcebos.com/901ab741cb46441ebec510b37e63b9d8d1b7c95f63cc4e5e8757f35179ae6373"  width = "200" height = "100" /> |
| 包装生产日期   | 点阵字符合成、过曝过暗文字识别 | [模型下载](#2) | [中文](./包装生产日期识别.md)/English                        | <img src="https://ai-studio-static-online.cdn.bcebos.com/d9e0533cc1df47ffa3bbe99de9e42639a3ebfa5bce834bafb1ca4574bf9db684"  width = "200" height = "100" /> |
| PCB文字识别    | 小尺寸文本检测与识别           | [模型下载](#2) | [中文](./PCB字符识别/PCB字符识别.md)/English                 | <img src="https://ai-studio-static-online.cdn.bcebos.com/95d8e95bf1ab476987f2519c0f8f0c60a0cdc2c444804ed6ab08f2f7ab054880"  width = "200" height = "100" /> |
| 电表识别       | 大分辨率图像检测调优           | [模型下载](#2) | [中文](https://aistudio.baidu.com/aistudio/projectdetail/5297312?forkThirdPart=1)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/9d4ebb5bf8544bbeabfacbfa539518c8e1ae68cbc3d74f67a3eb576ca94754a2"  width = "200" height = "100" /> |
| 液晶屏缺陷检测 | 非文字字符识别                 | [模型下载](#2) | [中文](https://aistudio.baidu.com/aistudio/projectdetail/4268015)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/c06b363d7ddb4b22b80701258c0a18003c40bca1d64a472698ee1bf746198e3a"  width = "200" height = "100" /> |

<a name="13"></a>

### 金融

| 类别               | 亮点                              | 模型下载       | 教程                                                         | 示例图                                                       |
| ------------------ | --------------------------------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 表单VQA            | 多模态通用表单结构化提取          | [模型下载](#2) | [中文](./多模态表单识别.md)/English                          | <img src="https://ai-studio-static-online.cdn.bcebos.com/a3b25766f3074d2facdf88d4a60fc76612f51992fd124cf5bd846b213130665b"  width = "200" height = "200" /> |
| 增值税发票         | 关键信息抽取，SER、RE任务训练     | [模型下载](#2) | [中文](./发票关键信息抽取.md)/English                        | <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg"  width = "200"  /> |
| 印章检测与识别     | 端到端弯曲文本识别                | [模型下载](#2) | [中文](./印章弯曲文字识别.md)/English                        | <img src="https://ai-studio-static-online.cdn.bcebos.com/498119182f0a414ab86ae2de752fa31c9ddc3a74a76847049cc57884602cb269"  width = "150"  /> |
| 通用卡证识别       | 通用结构化提取                    | [模型下载](#2) | [中文](./快速构建卡证类OCR.md)/English                       | <img src="https://ai-studio-static-online.cdn.bcebos.com/981640e17d05487e961162f8576c9e11634ca157f79048d4bd9d3bc21722afe8"  width = "300"  /> |
| 银行电子回单       | 回单关键信息抽取                  | ---            | [中文](https://aistudio.baidu.com/aistudio/projectdetail/5267489?contributionType=1)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/1c935a1e468e4911aadd1e8e9c30ca15420dc85fe95d49ce85c3c38ffff75adb"  width = "200"  /> |
| 身份证识别         | 结构化提取、图像阴影              | [模型下载](#2) | [中文](https://aistudio.baidu.com/aistudio/projectdetail/4255861?contributionType=1)/English | <img src='https://ai-studio-static-online.cdn.bcebos.com/4e2054032a9244a7a713e07e0dca00167685ecbc98ce484987e8c3c51208d08d' width='300'> |
| 合同比对           | 文本检测参数调整、NLP关键信息抽取 | ---            | [中文](./扫描合同关键信息提取.md)/English                    | <img src="https://ai-studio-static-online.cdn.bcebos.com/54f3053e6e1b47a39b26e757006fe2c44910d60a3809422ab76c25396b92e69b"  width = "300"  /> |
| 研报识别与实体统计 | 密集文本检测、NLP实体识别         | [模型下载](#2) | [中文](https://aistudio.baidu.com/aistudio/projectdetail/2574084)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/0bec003acb6444a69d8e3368962ca07452e9db6520ff44ceb5480011bc736609"  width = "300"  /> |
| 通用表格识别       | 表格数据生成                      | ---            | [中文](https://aistudio.baidu.com/aistudio/projectdetail/5099668?contributionType=1)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/da82ae8ef8fd479aaa38e1049eb3a681cf020dc108fa458eb3ec79da53b45fd1"  width = "300"  /> |

<a name="14"></a>

### 交通

| 类别              | 亮点                           | 模型下载       | 教程                                | 示例图                                                       |
| ----------------- | ------------------------------ | -------------- | ----------------------------------- | ------------------------------------------------------------ |
| 车牌识别          | 多角度图像、轻量模型、端侧部署 | [模型下载](#2) | [中文](./轻量级车牌识别.md)/English | <img src="https://ai-studio-static-online.cdn.bcebos.com/76b6a0939c2c4cf49039b6563c4b28e241e11285d7464e799e81c58c0f7707a7"  width = "200" height = "100" /> |
| 驾驶证/行驶证识别 | 尽请期待                       |                |                                     |                                                              |
| 快递单识别        | 尽请期待                       |                |                                     |                                                              |

<a name="2"></a>

## 资料下载

如需下载《OCR产业范例20讲》和上述场景中已经训练好的垂类模型，可以扫描下方二维码，关注公众号填写问卷后，加入PaddleOCR官方交流群获取20G OCR学习大礼包（内含《动手学OCR》电子书、课程回放视频、前沿论文等重磅资料）

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/dd721099bd50478f9d5fb13d8dd00fad69c22d6848244fd3a1d3980d7fefc63e"  width = "150" height = "150" />
</div>

如果您是企业开发者且未在上述场景中找到合适的方案，可以填写[OCR应用合作调研问卷](https://paddle.wjx.cn/vj/QwF7GKw.aspx)，免费与官方团队展开不同层次的合作，包括但不限于问题抽象、确定技术方案、项目答疑、共同研发等。如果您已经使用PaddleOCR落地项目，也可以填写此问卷，与飞桨平台共同宣传推广，提升企业技术品宣。期待您的提交！

<a href="https://trackgit.com"><img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/l63cvzo0w09yxypc7ygl" alt="traffic" /></a>
