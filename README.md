<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html" target="_blank">
      <img width="100%" src="./docs/images/Banner_cn.png" alt="PaddleOCR Banner"></a>
  </p>

<!-- language -->
[English](./README_en.md) | 简体中文| [日本語](./README_ja.md)

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8+-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)

[![Website](https://img.shields.io/badge/Website-PaddleOCR-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmmRkdj0AAAAASUVORK5CYII=)](https://www.paddleocr.ai/)
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)


</div>
<br>

## 🚀 简介
PaddleOCR自发布以来凭借学术前沿算法和产业落地实践，受到了产学研各方的喜爱，并被广泛应用于众多知名开源项目，例如：Umi-OCR、OmniParser、MinerU、RAGFlow等，已成为广大开发者心中的开源OCR领域的首选工具。2025年5月20日，飞桨团队发布**PaddleOCR 3.0**，全面适配**飞桨框架3.0正式版**，进一步**提升文字识别精度**，支持**多文字类型识别**和**手写体识别**，满足大模型应用对**复杂文档高精度解析**的旺盛需求，结合**文心大模型4.5 Turbo**显著提升关键信息抽取精度，并新增**对昆仑芯、昇腾等国产硬件**的支持。

PaddleOCR 3.0**新增**三大特色能力：：
- 🖼️全场景文字识别模型[PP-OCRv5](docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)：单模型支持五种文字类型和复杂手写体识别；整体识别精度相比上一代**提升13个百分点**。
- 🧮通用文档解析方案[PP-StructureV3](docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.md)：支持多场景、多版式 PDF 高精度解析，在公开评测集中**领先众多开源和闭源方案**。
- 📈智能文档理解方案[PP-ChatOCRv4](docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.md)：原生支持文心大模型4.5 Turbo，精度相比上一代**提升15个百分点**。

PaddleOCR 3.0除了提供优秀的模型库外，还提供好学易用的工具，覆盖模型训练、推理和服务化部署，方便开发者快速落地AI应用。
<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html" target="_blank">
      <img width="100%" src="./docs/images/Arch_cn.png" alt="PaddleOCR Architecture"></a>
  </p>
</div>


## 📣 最新动态
🔥🔥2025.05.20: **PaddleOCR 3.0** 正式发布，包含：
- **PP-OCRv5**: 全场景高精度文字识别

   1. 🌐 单模型支持**五种**文字类型(**简体中文**、**繁体中文**、**中文拼音**、**英文**和**日文**)。
   2. ✍️ 支持复杂**手写体**识别：复杂连笔、非规范字迹识别性能显著提升。
   3. 🎯 整体识别精度提升 - 多种应用场景达到 SOTA 精度, 相比上一版本PP-OCRv4，识别精度**提升13个百分点**！

- **PP-StructureV3**: 通用文档解析方案

   1. 🧮 支持多场景 PDF 高精度解析，在 OmniDocBench 基准测试中**领先众多开源和闭源方案**。
   2. 🧠 多项专精能力: **印章识别**、**图表转表格**、**嵌套公式/图片的表格识别**、**竖排文本解析**及**复杂表格结构分析**等。


- **PP-ChatOCRv4**: 智能文档理解方案
   1. 🔥 文档文件（PDF/PNG/JPG）关键信息提取精度相比上一代**提升15个百分点**！
   2. 💻 原生支持**文心大模型4.5 Turbo**，还兼容 PaddleNLP、Ollama、vLLM 等工具部署的大模型。
   3. 🤝 集成 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)，支持印刷文字、手写体文字、印章信息、表格、图表等常见的复杂文档信息抽取和理解的能力。

<details>
   <summary><strong>历史更新记录</strong></summary>

- 🔥🔥2025.03.07: **PaddleOCR v2.10** 发布：
  - 新增 **12 个自研模型**:
    - **[版式检测系列](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)**(3 模型): PP-DocLayout-L/M/S - 支持 23 类中英文文档版式检测（论文/报告/试卷/图书/期刊/合同等），最高达 **90.4% mAP@0.5**，轻量化设计支持每秒处理 100+ 页面
    - **[公式识别系列](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)**(2 模型): PP-FormulaNet-L/S - 支持 50,000+ LaTeX 公式识别，涵盖印刷体与手写体。PP-FormulaNet-L 精度提升 **6%**；PP-FormulaNet-S 速度提升 16 倍且精度相当
    - **[表格结构识别系列](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)**(2 模型): SLANeXt_wired/wireless - 新型模型复杂表格识别精度提升 **6%**
    - **[表格分类模型](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)**(1 模型): PP-LCNet_x1_0_table_cls - 超轻量有线/无线表格分类器

[更多详情，请查看](https://paddlepaddle.github.io/PaddleOCR/latest/en/update.html)

</details>

## ⚡ 快速开始
### 1. 在线体验无需安装
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. 本地安装指南

首先，请参考[安装指南](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)完成**PaddlePaddle 3.0**的安装。

然后，安装paddleocr
```bash
# 1. 安装 paddleocr
pip install paddleocr
# 2. 安装完毕后自检
paddleocr --version
```

### 3. 命令行方式推理
```bash
# 运行 PP-OCRv5 推理
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png

# 运行 PP-StructureV3 推理
paddleocr PP-StructureV3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png

# 运行 PP-ChatOCRv4 推理
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key

# 查看 "paddleocr ocr" 详细参数
paddleocr ocr --help
```
### 4. API方式推理

**4.1 PP-OCRv5 示例**
```python
from paddleocr import PaddleOCR
# 初始化 PaddleOCR 实例
ocr = PaddleOCR()
# 对示例图像执行 OCR 推理 
result = ocr.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")
# 可视化结果并保存 json 结果
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 PP-StructureV3 示例</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3()

# For Image
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png")

# 可视化结果并保存 json 结果
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output") 

# For PDF File
input_file = "./your_pdf_file.pdf"
output_path = Path("./output")

output = pipeline.predict(input_file)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)
```

</details>


<details>
   <summary><strong>4.3 PP-ChatOCRv4 示例</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

mllm_chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "PP-DocBee",
    "base_url": "http://127.0.0.1:8080/",  # your local mllm service url
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

pipeline = PPChatOCRv4Doc()

visual_predict_res = pipeline.visual_predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]

vector_info = pipeline.build_vector(
    visual_info_list, flag_save_bytes_vector=True, retriever_config=retriever_config
)
mllm_predict_res = pipeline.mllm_pred(
    input="vehicle_certificate-1.png",
    key_list=["驾驶室准乘人数"],
    mllm_chat_bot_config=mllm_chat_bot_config,
)
mllm_predict_info = mllm_predict_res["mllm_res"]
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)
```

</details>

### 5. **国产化硬件支持**
- [昆仑芯安装指南](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)
- [昇腾安装指南](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
  
## 😃 使用 PaddleOCR 的优秀项目
💗 PaddleOCR 的发展离不开社区贡献！衷心感谢所有开发者、合作伙伴与贡献者！
| 项目名称 | 简介 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|基于RAG的AI工作流引擎|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|多类型文档转换Markdown工具|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|开源批量离线OCR软件|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |基于纯视觉的GUI智能体屏幕解析工具|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |基于任意内容的问答系统|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|高效复杂PDF文档提取工具包|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |屏幕实时翻译工具|
| [更多项目](./awesome_projects.md) | [基于 PaddleOCR 的扩展项目](./awesome_projects.md)|

## 🔄 快速一览运行效果

<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html" target="_blank">
      <img width="100%" src="./docs/images/demo.gif" alt="PP-OCRv5 Demo"></a>
  </p>
</div>

<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html" target="_blank">
      <img width="100%" src="./docs/images/blue_v3.gif" alt="PP-StructureV3 Demo"></a>
  </p>
</div>

## 👩‍👩‍👧‍👦 开发者社区

## 📄 许可协议
本项目采用 [Apache 2.0 协议](./LICENSE) 开源发布。
