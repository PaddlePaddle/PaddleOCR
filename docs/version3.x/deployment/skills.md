---
comments: true
---

# PaddleOCR Agent Skills

PaddleOCR 提供官方 Agent Skills，将 OCR 与文档解析任务的触发规则、调用步骤、配置要求和结果处理最佳实践打包为可按需加载的模块化能力，帮助支持 Skills 的 AI 应用更稳定地完成文字识别与版面解析任务。

## 先选合适的 Skill

| 需求 | 推荐 Skill | 输出 |
| --- | --- | --- |
| 只想提取图片或 PDF 中的纯文本 | `paddleocr-text-recognition` | 纯文本 |
| 需要保留标题、段落、表格、公式等文档结构 | `paddleocr-doc-parsing` | Markdown / 结构化结果 |

## 包含的 Skills

- `paddleocr-text-recognition`：用于识别图片、扫描件与 PDF 中的文字。
- `paddleocr-doc-parsing`：用于解析复杂文档版面，并转换为 Markdown 或结构化结果。

当前各 Skill 支持的底层模型如下：

- `paddleocr-text-recognition`：`PP-OCRv5`
- `paddleocr-doc-parsing`：`PP-StructureV3`、`PaddleOCR-VL`、`PaddleOCR-VL-1.5`

## 安装前准备

1. 请确保执行 Skill 的设备已安装 Python 3.9 或以上版本，以及 [uv](https://docs.astral.sh/uv/)。
2. 所有脚本均以 [PEP 723](https://peps.python.org/pep-0723/) 格式内联声明依赖，`uv run` 会自动解析依赖。
3. Skills 依赖 PaddleOCR 官方 API。请前往 [PaddleOCR 官网](https://www.paddleocr.com) 点击 **API**，选择对应模型后复制 `API_URL` 和 `Token`。

## 安装到 AI 应用

> 以下说明涵盖两个 Skill。只需安装并配置您需要的 Skill 即可。

### 方式一：通过 `skills` CLI 安装

`skills` CLI 可将 Skill 全局安装到设备上，安装后各 AI 应用均可使用。需要先安装 [Node.js](https://nodejs.org/en/download)。

```shell
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
```

> 由于 PaddleOCR 仓库较大，在网络较慢的环境下 `npx skills add` 可能因超时而失败。如遇此情况，可先将仓库克隆到本地，再从本地路径安装：
>
> ```shell
> git clone https://github.com/PaddlePaddle/PaddleOCR.git
> npx skills add ./PaddleOCR/skills/paddleocr-text-recognition
> npx skills add ./PaddleOCR/skills/paddleocr-doc-parsing
> ```

### 方式二：通过 `clawhub` 安装（OpenClaw）

```shell
clawhub install paddleocr-text-recognition
clawhub install paddleocr-doc-parsing
```

详见 [OpenClaw Skills 文档](https://docs.openclaw.ai/tools/skills)。

### 方式三：手动安装

如果上述方式不适用，也可以先克隆仓库，再手动将 Skill 目录拷贝到 AI 应用指定的位置：

```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

Skill 源码位于 `PaddleOCR/skills` 目录。请参考对应 AI 应用的安装文档：

- Claude Code：<https://code.claude.com/docs/en/skills>
- claude.ai：<https://support.claude.com/en/articles/12512180-use-skills-in-claude>
- OpenClaw：<https://docs.openclaw.ai/tools/skills>

## 配置环境变量 {#配置环境变量}

安装完成后，需要配置环境变量以便 Skill 正常工作：

| Skill | 必填 | 可选 |
| --- | --- | --- |
| `paddleocr-text-recognition` | `PADDLEOCR_OCR_API_URL`（完整端点 URL，须以 `/ocr` 结尾）、`PADDLEOCR_ACCESS_TOKEN`（access token） | `PADDLEOCR_OCR_TIMEOUT`（API 请求超时时间） |
| `paddleocr-doc-parsing` | `PADDLEOCR_DOC_PARSING_API_URL`（完整端点 URL，须以 `/layout-parsing` 结尾）、`PADDLEOCR_ACCESS_TOKEN`（access token） | `PADDLEOCR_DOC_PARSING_TIMEOUT`（API 请求超时时间） |

部分 AI 应用的配置方式如下：

- **Claude Code**：在项目的 `.claude/settings.local.json` 中添加 `env` 字段：

  ```json
  {
    "env": {
      "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>",
      "PADDLEOCR_OCR_API_URL": "<OCR_API_URL>",
      "PADDLEOCR_DOC_PARSING_API_URL": "<DOC_PARSING_API_URL>"
    }
  }
  ```

- **OpenClaw**：在 `~/.openclaw/openclaw.json` 中添加 Skill 配置：

  ```json
  {
    "skills": {
      "entries": {
        "paddleocr-text-recognition": {
          "enabled": true,
          "env": {
            "PADDLEOCR_OCR_API_URL": "<OCR_API_URL>",
            "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
          }
        },
        "paddleocr-doc-parsing": {
          "enabled": true,
          "env": {
            "PADDLEOCR_DOC_PARSING_API_URL": "<DOC_PARSING_API_URL>",
            "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
          }
        }
      }
    }
  }
  ```

## 使用示例

配置完成后，可以直接用自然语言描述任务，并附上文件 URL 或本地路径，让 AI 应用调用对应 Skill。

### `paddleocr-text-recognition`

解析 URL 示例：

```text
提取这个文件中的全部文本：https://example.com/invoice.jpg
```

解析本地文件示例：

```text
提取本地文件 C:\docs\invoice.pdf 中的全部文本。
```

### `paddleocr-doc-parsing`

解析 URL 示例：

```text
解析这个 PDF，并返回主体内容和全部表格：https://example.com/report.pdf
```

解析本地文件示例：

```text
解析本地文件 C:\docs\report.pdf，并返回完整结构化结果。
```

## 本地验证

本节介绍如何在本地运行冒烟测试，以验证 Skill 配置是否正常。

> 以下示例覆盖两个 Skill。如果只需使用其中一个，只执行对应命令即可。

执行前，请确保工作目录位于本文档所在目录。所有脚本均以内联依赖声明，`uv` 会自动解析，无需单独安装依赖。

1. 配置环境变量（见上文 [配置环境变量](#配置环境变量)）。

   ```shell
   export PADDLEOCR_OCR_API_URL="<OCR_API_URL>"
   export PADDLEOCR_ACCESS_TOKEN="<ACCESS_TOKEN>"
   export PADDLEOCR_DOC_PARSING_API_URL="<DOC_PARSING_API_URL>"
   ```

2. 运行冒烟测试脚本。

   ```shell
   cd paddleocr-text-recognition && uv run scripts/smoke_test.py && cd ..
   cd paddleocr-doc-parsing && uv run scripts/smoke_test.py && cd ..
   ```

   使用 `--skip-api-test` 可只做配置检查（不发网络请求）。使用 `--test-url "https://..."` 可指定自定义测试文档或图片 URL。
