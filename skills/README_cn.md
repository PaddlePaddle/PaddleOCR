# PaddleOCR Skills

本目录包含 PaddleOCR 官方提供的 Agent Skills，可与 Claude Code 等 AI 应用结合，实现图片/PDF 文档的文字识别和版面解析。

## 技能列表

- `paddleocr-text-recognition`：可用于识别图片/PDF 中的文字。
- `paddleocr-doc-parsing`：版面解析，可将图片/PDF 转换成 Markdown。

## 准备工作

1. 请确保执行 skill 的设备安装有 Python 3.8 或以上版本。
2. Skill 底层依赖于 PaddleOCR 官方 API，因此需要配置相关凭证才能使用。可以在 [PaddleOCR 官网](https://www.paddleocr.com) 点击 **API**，选择需要用到的算法，然后复制 `API_URL` 和 `Token`，它们分别对应服务的 API URL 和用户鉴权使用的 access token。各 skill 支持的算法如下：
   - `paddleocr-text-recognition`：`PP-OCRv5`
   - `paddleocr-doc-parsing`：`PP-StructureV3`、`PaddleOCR-VL`、`PaddleOCR-VL-1.5`

## 在 AI 应用中使用

> 以下说明涵盖两个 skill。只需安装并配置你需要的 skill 即可。

### 安装到 AI 应用

#### 方式一：通过 `skills` CLI 安装

`skills` CLI 可将 skill 全局安装到设备上，安装后各 AI 应用均可使用。需要首先安装 [Node.js](https://nodejs.org/en/download)。

```shell
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
```

> 由于 PaddleOCR 仓库较大，在网络较慢的环境下 `npx skills add` 可能因超时而失败。如遇此情况，可先将仓库克隆到本地，再从本地路径安装：
>
> ```shell
> git clone https://github.com/PaddlePaddle/PaddleOCR.git
> npx skills add ./PaddleOCR/skills/paddleocr-text-recognition
> ```

#### 方式二：通过 `clawhub` 安装（OpenClaw）

```shell
clawhub install paddleocr-text-recognition
clawhub install paddleocr-doc-parsing
```

详见 [OpenClaw Skills 文档](https://docs.openclaw.ai/tools/skills)。

#### 方式三：手动安装

如果上述方式不适用，也可以将仓库克隆到本地后，手动将 skill 目录拷贝到 AI 应用指定的位置（需要安装 [Git](https://git-scm.com/downloads)）：

```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

克隆完成后，skill 源码位于 `PaddleOCR/skills` 目录下。请参考各 AI 应用的文档完成安装：

- Claude Code：<https://code.claude.com/docs/en/skills>
- claude.ai：<https://support.claude.com/en/articles/12512180-use-skills-in-claude>
- OpenClaw：<https://docs.openclaw.ai/tools/skills>

### 配置环境变量

安装完成后，需要配置环境变量，以便 skill 正确工作。各 skill 需要的环境变量如下：

| Skill | 必填 | 可选 |
| --- | --- | --- |
| `paddleocr-text-recognition` | `PADDLEOCR_OCR_API_URL`（API URL）、`PADDLEOCR_ACCESS_TOKEN`（access token） | `PADDLEOCR_OCR_TIMEOUT`（API 请求超时时间） |
| `paddleocr-doc-parsing` | `PADDLEOCR_DOC_PARSING_API_URL`（API URL）、`PADDLEOCR_ACCESS_TOKEN`（access token） | `PADDLEOCR_DOC_PARSING_TIMEOUT`（API 请求超时时间） |

以下是部分 AI 应用的配置方式：

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

- **OpenClaw**：在 `~/.openclaw/openclaw.json` 中添加 skill 配置：

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

### 使用示例

配置完成后，可以直接用自然语言描述 OCR 或文档解析需求，并附上文件 URL 或本地路径，让 AI 应用调用对应 skill。以下是部分提示词示例：

**paddleocr-text-recognition**

解析 URL 示例：

```text
提取这个文件中的全部文本：https://example.com/invoice.jpg
```

解析本地文件示例：

```text
提取本地文件 C:\docs\invoice.pdf 中的全部文本。
```

**paddleocr-doc-parsing**

解析 URL 示例：

```text
解析这个 PDF，并返回主体内容和全部表格：https://example.com/report.pdf
```

解析本地文件示例：

```text
解析本地文件 C:\docs\report.pdf，并返回完整结构化结果。
```

## 本地测试

本节介绍如何在本地运行冒烟测试，以验证 skill 功能是否正常。

> 以下示例覆盖多个 skill。如果只需使用某一个 skill，只需执行该 skill 对应的命令。

执行前，请确保工作目录位于本文档所在的目录下。

1. 安装依赖库。

   ```shell
   python -m pip install -r paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements.txt
   # 可选依赖，仅在优化文档文件大小时需要
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```

2. 配置环境变量（需要配置的变量参见[配置环境变量](#配置环境变量)一节）。

   ```shell
   export PADDLEOCR_OCR_API_URL="<OCR_API_URL>"
   export PADDLEOCR_ACCESS_TOKEN="<ACCESS_TOKEN>"
   export PADDLEOCR_DOC_PARSING_API_URL="<DOC_PARSING_API_URL>"
   ```

3. 运行冒烟测试脚本。

   ```shell
   python paddleocr-text-recognition/scripts/smoke_test.py
   python paddleocr-doc-parsing/scripts/smoke_test.py
   ```
