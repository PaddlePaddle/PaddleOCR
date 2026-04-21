---
comments: true
---

# 安装

## 1. 安装推理引擎、PaddleOCR Python 库及可选功能依赖

本节说明如何按需安装推理引擎、`paddleocr` 分发包，以及按能力域启用的可选依赖组。这条路径适用于在本地调用预训练产线完成推理，或是使用文档格式转换等辅助功能。**模型训练与模型导出**见第 2 节，与上述安装路径相互独立。

### 1.1 安装推理引擎（按需）

PaddleOCR 3.5 采用统一推理引擎配置，底层可对接飞桨、Transformers 等。若要实际执行模型推理，请参考 [推理引擎与配置说明](./inference_engine.md) 安装所选推理引擎。

### 1.2 安装 paddleocr

从 PyPI 安装最新版本的 `paddleocr`：

```bash
# 仅需要通用 OCR 与文档图像预处理等默认能力
python -m pip install paddleocr
# 需要文档解析、文档理解、文档翻译、关键信息抽取等全部可选能力
# python -m pip install "paddleocr[all]"
```

或从源码安装（默认跟踪仓库当前默认分支）：

```bash
# 仅需要通用 OCR 与文档图像预处理等默认能力
python -m pip install "paddleocr@git+https://github.com/PaddlePaddle/PaddleOCR.git"
# 需要文档解析、文档理解、文档翻译、关键信息抽取等全部可选能力
# python -m pip install "paddleocr[all]@git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

### 1.3 按功能选择依赖组

除 `all` 外，可通过指定依赖组启用部分可选能力。依赖组表示能力域（文档解析、信息抽取、文档翻译等）。各依赖组如下：

| 依赖组名称 | 对应的功能 |
| - | - |
| `doc-parser` | 文档解析，可用于提取文档中的表格、公式、印章、图片等版面元素，包含 PP-StructureV3 等模型方案 |
| `ie` | 信息抽取，可用于从文档中提取关键信息，如姓名、日期、地址、金额等，包含 PP-ChatOCRv4 等模型方案 |
| `trans` | 文档翻译，可用于将文档从一种语言翻译为另一种语言，包含 PP-DocTranslation 等模型方案 |
| `doc2md` | 文档转 MarkDown，可用于将 Word、Excel、PowerPoint 文件快速转为可读文本 |
| `all` | 完整功能 |

通用 OCR 产线与文档图像预处理产线无需额外依赖组；文档解析、信息抽取、文档翻译等按上表安装对应组。各产线所属依赖组见对应产线文档；单功能模块在任一包含该模块的依赖组安装后即可调用其基础能力。

## 2. 安装训练与导出依赖

若要进行模型训练或模型导出，需另行安装训练相关依赖。该路径与第 1 节的 `paddleocr` 包及可选依赖组属于不同安装维度；同一环境中可同时存在，无需强制隔离。

训练与导出依赖飞桨框架，请先参考[飞桨框架安装](./paddlepaddle_installation.md)完成 PaddlePaddle 安装。

将本仓库克隆到本地后安装其余依赖：

```bash
# 推荐方式
git clone https://github.com/PaddlePaddle/PaddleOCR

# （可选）切换到指定分支
git checkout release/3.5

# 如果因为网络问题无法克隆成功，也可选择使用码云上的仓库：
git clone https://gitee.com/paddlepaddle/PaddleOCR

# 注：码云托管代码可能无法实时同步本 GitHub 项目更新，存在3~5天延时，请优先使用推荐方式。
```

执行如下命令安装其余训练依赖：

```bash
python -m pip install -r requirements.txt
```
