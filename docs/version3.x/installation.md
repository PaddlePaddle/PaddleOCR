---
comments: true
---

# 安装

## 1. 安装 PaddleOCR 推理依赖

### 1.1 安装推理引擎

PaddleOCR 3.5 引入了统一的推理引擎配置概念，底层支持飞桨框架、Transformers 等不同推理引擎。如果希望执行产线/模型推理，请参考 [推理引擎与配置说明](./inference_engine.md) 安装相关推理引擎；如果只希望使用 PaddleOCR 推理以外的功能，可跳过此步骤。

### 1.2 安装 PaddleOCR 推理包

从 PyPI 安装最新版本 PaddleOCR 推理包：

```bash
# 只希望使用基础文字识别功能（返回文字位置坐标和文本内容）
python -m pip install paddleocr
# 希望使用文档解析、文档理解、文档翻译、关键信息抽取等全部功能
# python -m pip install "paddleocr[all]"
```

或者从源码安装（默认为开发分支）：

```bash
# 只希望使用基础文字识别功能（返回文字位置坐标和文本内容）
python -m pip install "paddleocr@git+https://github.com/PaddlePaddle/PaddleOCR.git"
# 希望使用文档解析、文档理解、文档翻译、关键信息抽取等全部功能
# python -m pip install "paddleocr[all]@git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

除了上面演示的 `all` 依赖组以外，PaddleOCR 也支持通过指定其它依赖组，安装部分可选功能。PaddleOCR 提供的所有依赖组如下：

| 依赖组名称 | 对应的功能 |
| - | - |
| `doc-parser` | 文档解析，可用于提取文档中的表格、公式、印章、图片等版面元素，包含 PP-StructureV3 等模型方案 |
| `ie` | 信息抽取，可用于从文档中提取关键信息，如姓名、日期、地址、金额等，包含 PP-ChatOCRv4 等模型方案 |
| `trans` | 文档翻译，可用于将文档从一种语言翻译为另一种语言，包含 PP-DocTranslation 等模型方案 |
| `all` | 完整功能 |

通用 OCR 产线（如 PP-OCRv3/v4/v5）、文档图像预处理产线的功能无需安装额外的依赖组即可使用。除了这两条产线外，每一条产线属于且仅属于一个依赖组。在各产线的使用文档中可以了解产线属于哪一依赖组。对于单功能模块，安装任意包含该模块的产线对应的依赖组后即可使用相关的基础功能。

## 2. 安装训练依赖

如果您希望进行模型训练、导出等开发工作，需要安装训练依赖。在同一环境中安装推理包和训练依赖是允许的，无需进行环境隔离。

训练、导出等开发场景依赖飞桨框架，请先参考[飞桨框架安装](./paddlepaddle_installation.md)完成 PaddlePaddle 安装。

要进行模型训练、导出等，需要将仓库克隆到本地：

```bash
# 推荐方式
git clone https://github.com/PaddlePaddle/PaddleOCR

# （可选）切换到指定分支
git checkout release/3.2

# 如果因为网络问题无法克隆成功，也可选择使用码云上的仓库：
git clone https://gitee.com/paddlepaddle/PaddleOCR

# 注：码云托管代码可能无法实时同步本 GitHub 项目更新，存在3~5天延时，请优先使用推荐方式。
```

执行如下命令安装其余训练依赖：

```bash
python -m pip install -r requirements.txt
```
