---
comments: true
---

# 文档转 Markdown 使用教程

## 1. 功能介绍

文档转 Markdown（doc2md）是 PaddleOCR 内置的轻量级 Office 文档结构化转换功能，**无需 OCR 推理**，直接解析文档结构并输出规范的 Markdown 文本。适用于需要将 Word、Excel、PowerPoint 文件快速转为可读文本的场景，如知识库构建、文档检索、内容提取等。

**支持格式**：`.docx`（Word）/ `.xlsx`（Excel）/ `.pptx`（PowerPoint）

**核心能力一览：**

<table>
<thead>
<tr>
<th>功能</th>
<th>Word (.docx)</th>
<th>Excel (.xlsx)</th>
<th>PowerPoint (.pptx)</th>
</tr>
</thead>
<tbody>
<tr>
<td>标题层级</td>
<td>✅ 内置样式 + 字号启发式 + 中文编号</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>文本格式化（粗体/斜体/下划线/删除线）</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
<tr>
<td>上标 / 下标</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
<tr>
<td>超链接</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
<tr>
<td>列表（有序 / 无序 / 嵌套）</td>
<td>✅</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>表格（含合并单元格）</td>
<td>✅ HTML table</td>
<td>✅ HTML table</td>
<td>✅ HTML table</td>
</tr>
<tr>
<td>图片</td>
<td>✅ 按比例宽度</td>
<td>✅ 浮动图片</td>
<td>✅ 按比例宽度</td>
</tr>
<tr>
<td>数学公式（OMML → LaTeX）</td>
<td>✅ 行内 / 显示公式</td>
<td>✅ drawing 层公式</td>
<td>✅</td>
</tr>
<tr>
<td>代码块</td>
<td>✅ 等宽字体自动识别</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>文本框</td>
<td>✅</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>图表（Chart）</td>
<td>✅ → HTML table</td>
<td>— </td>
<td>✅ 14 种图表类型</td>
</tr>
<tr>
<td>页眉 / 页脚</td>
<td>✅ 多节 + 奇偶页</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>多 sheet / 多幻灯片</td>
<td>— </td>
<td>✅</td>
<td>✅ <code>---</code> 分隔</td>
</tr>
<tr>
<td>演讲者备注</td>
<td>— </td>
<td>— </td>
<td>✅</td>
</tr>
</tbody>
</table>

## 2. 快速开始

在使用 doc2md 功能前，请确保已按照[安装教程](./installation.md)完成 PaddleOCR 基础安装，然后安装 doc2md 可选依赖：

```bash
pip install "paddleocr[doc2md]"
```

**依赖说明：**

<table>
<thead>
<tr>
<th>包名</th>
<th>版本约束</th>
<th>用途</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>python-docx</code></td>
<td><code>&gt;=0.8.11</code></td>
<td>Word (.docx) 文档解析</td>
</tr>
<tr>
<td><code>python-pptx</code></td>
<td><code>&gt;=0.6.21</code></td>
<td>PowerPoint (.pptx) 文档解析</td>
</tr>
<tr>
<td><code>openpyxl</code></td>
<td><code>&gt;=3.0.0</code></td>
<td>Excel (.xlsx) 文档解析</td>
</tr>
<tr>
<td><code>pylatexenc</code></td>
<td><code>&gt;=2.10,&lt;3</code></td>
<td>数学公式 Unicode → LaTeX 符号映射</td>
</tr>
</tbody>
</table>

### 2.1 命令行方式

```bash
# 转换 Word 文档，输出到文件
paddleocr doc2md -i report.docx -o output.md

# 转换 Excel 表格，输出到文件
paddleocr doc2md -i data.xlsx -o output.md

# 转换 PowerPoint 演示文稿，输出到文件
paddleocr doc2md -i slides.pptx -o output.md

# 不指定输出路径，结果打印到终端
paddleocr doc2md -i report.docx

# 查看支持的格式列表
paddleocr doc2md --formats
```

<details><summary><b>命令行支持更多参数设置，点击展开以查看命令行参数的详细说明</b></summary>
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>-i</code>, <code>--input</code></td>
<td><b>含义：</b>输入文件路径，必填（使用 <code>--formats</code> 时可省略）。<br/>
<b>说明：</b>支持 <code>.docx</code>、<code>.xlsx</code>、<code>.pptx</code> 格式。</td>
<td><code>str</code></td>
<td>必填</td>
</tr>
<tr>
<td><code>-o</code>, <code>--output</code></td>
<td><b>含义：</b>输出 Markdown 文件路径。<br/>
<b>说明：</b>如果不设置，转换结果将打印到标准输出（stdout）。设置后，Markdown 写入指定文件，图片保存到同目录下的 <code>images/</code> 文件夹。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>-q</code>, <code>--quiet</code></td>
<td><b>含义：</b>静默模式。<br/>
<b>说明：</b>启用后不打印耗时、保存路径等提示信息。</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--formats</code></td>
<td><b>含义：</b>列出当前支持的文件格式后退出。<br/>
<b>说明：</b>启用后不需要 <code>--input</code>，仅输出支持的扩展名列表。</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--no-drawings</code></td>
<td><b>含义：</b>跳过文本框和 drawing 层内容提取。<br/>
<b>说明：</b>适用于 <code>.docx</code> 和 <code>.xlsx</code>，启用后将跳过文本框（docx）和 drawing 层数学公式（xlsx）的提取。</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--no-headers-footers</code></td>
<td><b>含义：</b>跳过页眉页脚内容提取。<br/>
<b>说明：</b>仅适用于 <code>.docx</code>，启用后不提取文档的页眉和页脚内容。</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--sheet-name</code></td>
<td><b>含义：</b>仅转换指定名称的 sheet。<br/>
<b>说明：</b>仅适用于 <code>.xlsx</code>，不设置则转换所有 sheet。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>--max-rows</code></td>
<td><b>含义：</b>每个 sheet 的最大转换行数。<br/>
<b>说明：</b>仅适用于 <code>.xlsx</code>，用于限制大表格的输出行数。</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />

### 2.2 Python API

**基础用法：**

```python
from paddleocr._doc2md import convert

# 转换文档，返回结果对象
result = convert("report.docx")

# 访问 Markdown 文本
print(result.markdown)

# 查看提取的图片（字典，key 为相对路径，value 为图片字节）
print(list(result.images.keys()))

# 查看文档标题
print(result.title)

# 查看元信息（格式、sheet 数等）
print(result.metadata)
```

**`ConvertResult` 字段说明：**

<table>
<thead>
<tr>
<th>字段</th>
<th>类型</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>markdown</code></td>
<td><code>str</code></td>
<td>转换后的 Markdown 文本</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>dict[str, bytes]</code></td>
<td>提取的图片字典，key 为相对路径（如 <code>images/image1.png</code>），value 为图片字节</td>
</tr>
<tr>
<td><code>title</code></td>
<td><code>Optional[str]</code></td>
<td>文档标题，可能为 <code>None</code></td>
</tr>
<tr>
<td><code>metadata</code></td>
<td><code>dict</code></td>
<td>文档元信息，如格式类型、sheet 数量等</td>
</tr>
</tbody>
</table>

**指定输出路径（自动保存文件和图片）：**

```python
from paddleocr._doc2md import convert

# 指定 output 后，Markdown 写入文件，图片保存到同目录 images/ 下
result = convert("report.docx", output="output/report.md")
```

**各格式可用的 kwargs 参数：**

<table>
<thead>
<tr>
<th>参数</th>
<th>类型</th>
<th>默认值</th>
<th>适用格式</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>extract_drawings</code></td>
<td><code>bool</code></td>
<td><code>True</code></td>
<td>docx, xlsx</td>
<td>是否提取文本框（docx）/ drawing 层数学公式（xlsx）</td>
</tr>
<tr>
<td><code>extract_headers_footers</code></td>
<td><code>bool</code></td>
<td><code>True</code></td>
<td>docx</td>
<td>是否提取页眉页脚</td>
</tr>
<tr>
<td><code>sheet_name</code></td>
<td><code>Optional[str]</code></td>
<td><code>None</code></td>
<td>xlsx</td>
<td>仅转换指定名称的 sheet，<code>None</code> 表示转换全部</td>
</tr>
<tr>
<td><code>max_rows</code></td>
<td><code>Optional[int]</code></td>
<td><code>None</code></td>
<td>xlsx</td>
<td>每个 sheet 的最大转换行数</td>
</tr>
</tbody>
</table>

**按格式传入 kwargs 示例：**

```python
from paddleocr._doc2md import convert

# Word：不提取文本框和页眉页脚
result = convert("report.docx", extract_drawings=False, extract_headers_footers=False)

# Excel：仅转换名为 "Sheet1" 的 sheet，最多 100 行
result = convert("data.xlsx", sheet_name="Sheet1", max_rows=100)
```

## 3. 各格式支持特性

### 3.1 Word (.docx)

**标题识别**支持三种方式：
- **内置 Heading 样式**：Word 内置的 Heading 1–6 样式直接映射为 `#`–`######`
- **字号启发式**：字号大于正文 1.5 倍且段落较短时，自动识别为标题
- **中文编号**："一、"格式识别为 H2，"（一）"格式识别为 H3

**文本格式化**：粗体（`**`）、斜体（`*`）、下划线（`<u>`）、删除线（`~~`）、上标（`<sup>`）、下标（`<sub>`）

**列表**：有序列表、无序列表、嵌套列表，自动识别缩进层级

**表格**：输出为 HTML `<table>` 格式，支持 `rowspan`/`colspan` 合并单元格

**图片**：按文档内容区宽度百分比计算，输出 `<img width="75%">` 形式

**数学公式**：OMML 格式公式转为 LaTeX，行内公式为 `$...$`，显示公式为 `$$...$$`

**代码块**：自动检测等宽字体（Courier New、Consolas 等 9 种字体），输出为 fenced code block（` ``` `）

**其他**：文本框内容（`wps:txbx`）、图表（Chart → HTML table）、超链接（两种格式）、页眉页脚（多节 + 奇偶页）

### 3.2 Excel (.xlsx)

**多 sheet**：每个 sheet 输出一个以 `## sheet名称` 开头的章节

**数据边界裁剪**：自动去除尾部空行和空列，只输出有效数据区域

**合并单元格**：使用 `rowspan`/`colspan` 还原单元格合并结构

**字体格式化**：粗体、斜体、下划线、删除线、上标、下标

**超链接**：支持单元格级超链接

**浮动图片**：支持 `OneCellAnchor` 和 `TwoCellAnchor` 两种锚定方式的浮动图片

**数学公式**：通过解析 drawing 层 XML 提取 OMML 格式公式并转为 LaTeX

### 3.3 PowerPoint (.pptx)

**多幻灯片**：每张幻灯片内容以 `---` 分隔

**文本格式化**：粗体、斜体、下划线、删除线、上标、下标

**图片**：按幻灯片宽度百分比计算，输出带宽度的 `<img>` 标签

**表格**：HTML `<table>` 格式，支持合并单元格，支持带背景图片的表格

**图表**：支持 14 种图表类型，转换为 HTML table 输出

**分组形状（GroupShape）**：递归处理嵌套的形状组合

**数学公式**：从 `mc:AlternateContent` 中提取 OMML 格式公式并转为 LaTeX

**演讲者备注**：附加在每张幻灯片内容末尾

## 4. FAQ

**Q：转换时提示 `RuntimeError: python-docx is required`？**

doc2md 采用延迟导入，缺少对应格式的解析库时会抛出此错误。请根据提示安装对应依赖：
```bash
pip install python-docx     # Word (.docx)
pip install python-pptx     # PowerPoint (.pptx)
pip install openpyxl        # Excel (.xlsx)
pip install pylatexenc      # 数学公式支持
```
或直接安装全部依赖：`pip install "paddleocr[doc2md]"`

---

**Q：格式不支持，提示 `ValueError`？**

运行 `paddleocr doc2md --formats` 查看当前支持的扩展名。doc2md 仅支持 `.docx`、`.xlsx`、`.pptx`，不支持 `.doc`（旧版 Word）、`.csv`、`.pdf` 等格式。

---

**Q：Excel 转换后表格行数很多，输出太长？**

使用 `--max-rows` 限制每个 sheet 的行数：
```bash
paddleocr doc2md -i data.xlsx -o output.md --max-rows 100
```

---

**Q：只想转换 Excel 中的某一个 sheet？**

使用 `--sheet-name` 指定 sheet 名称：
```bash
paddleocr doc2md -i data.xlsx -o output.md --sheet-name "Sheet1"
```

---

**Q：Word 文档中的页眉页脚内容不需要，如何跳过？**

使用 `--no-headers-footers` 参数：
```bash
paddleocr doc2md -i report.docx -o output.md --no-headers-footers
```

---

**Q：图片输出到哪里？**

使用 `-o` 指定输出文件时，图片自动保存在输出文件同目录的 `images/` 文件夹下。Markdown 文件中的图片引用路径也会相应更新为相对路径。

---

**Q：doc2md 与 PaddleOCR 的 OCR 功能有什么区别？**

doc2md 直接解析 Office 文档的 XML 结构，**不使用任何 OCR 模型**，速度快、零 GPU 依赖。适用于有原始 Office 文件的场景。PaddleOCR 的 OCR 功能则针对图片或扫描件进行文字识别，适用于没有原始文档的场景。
