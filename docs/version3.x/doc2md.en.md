---
comments: true
---

# Document to Markdown Tutorial

## 1. Introduction

Document to Markdown (doc2md) is a lightweight Office document conversion feature built into PaddleOCR. It **requires no OCR inference** — it directly parses document structure and outputs well-formed Markdown text. It is well-suited for scenarios where Word, Excel, or PowerPoint files need to be quickly converted to readable text, such as knowledge base construction, document retrieval, and content extraction.

**Supported formats**: `.docx` (Word) / `.xlsx` (Excel) / `.pptx` (PowerPoint)

**Feature overview:**

<table>
<thead>
<tr>
<th>Feature</th>
<th>Word (.docx)</th>
<th>Excel (.xlsx)</th>
<th>PowerPoint (.pptx)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Heading levels</td>
<td>✅ Built-in styles + font-size heuristic + Chinese numbering</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>Text formatting (bold / italic / underline / strikethrough)</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
<tr>
<td>Superscript / subscript</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
<tr>
<td>Hyperlinks</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
<tr>
<td>Lists (ordered / unordered / nested)</td>
<td>✅</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>Tables (with merged cells)</td>
<td>✅ HTML table</td>
<td>✅ HTML table</td>
<td>✅ HTML table</td>
</tr>
<tr>
<td>Images</td>
<td>✅ Proportional width</td>
<td>✅ Floating images</td>
<td>✅ Proportional width</td>
</tr>
<tr>
<td>Math formulas (OMML → LaTeX)</td>
<td>✅ Inline / display formulas</td>
<td>✅ Drawing-layer formulas</td>
<td>✅</td>
</tr>
<tr>
<td>Code blocks</td>
<td>✅ Monospace font auto-detection</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>Text boxes</td>
<td>✅</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>Charts</td>
<td>✅ → HTML table</td>
<td>— </td>
<td>✅ 14 chart types</td>
</tr>
<tr>
<td>Headers / footers</td>
<td>✅ Multi-section + odd/even pages</td>
<td>— </td>
<td>— </td>
</tr>
<tr>
<td>Multiple sheets / slides</td>
<td>— </td>
<td>✅</td>
<td>✅ Separated by <code>---</code></td>
</tr>
<tr>
<td>Speaker notes</td>
<td>— </td>
<td>— </td>
<td>✅</td>
</tr>
</tbody>
</table>

## 2. Quick Start

Before using doc2md, make sure you have completed the PaddleOCR base installation as described in the [installation guide](./installation.en.md), then install the doc2md optional dependencies:

```bash
pip install "paddleocr[doc2md]"
```

**Dependency details:**

<table>
<thead>
<tr>
<th>Package</th>
<th>Version constraint</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>python-docx</code></td>
<td><code>&gt;=0.8.11</code></td>
<td>Word (.docx) document parsing</td>
</tr>
<tr>
<td><code>python-pptx</code></td>
<td><code>&gt;=0.6.21</code></td>
<td>PowerPoint (.pptx) document parsing</td>
</tr>
<tr>
<td><code>openpyxl</code></td>
<td><code>&gt;=3.0.0</code></td>
<td>Excel (.xlsx) document parsing</td>
</tr>
<tr>
<td><code>pylatexenc</code></td>
<td><code>&gt;=2.10,&lt;3</code></td>
<td>Math formula Unicode → LaTeX symbol mapping</td>
</tr>
</tbody>
</table>

### 2.1 Command Line

```bash
# Convert a Word document to a file
paddleocr doc2md -i report.docx -o output.md

# Convert an Excel spreadsheet to a file
paddleocr doc2md -i data.xlsx -o output.md

# Convert a PowerPoint presentation to a file
paddleocr doc2md -i slides.pptx -o output.md

# Omit output path to print to stdout
paddleocr doc2md -i report.docx

# List supported formats and exit
paddleocr doc2md --formats
```

<details><summary><b>Click to expand the full list of command line parameters</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>-i</code>, <code>--input</code></td>
<td><b>Meaning:</b> Input file path. Required (may be omitted when <code>--formats</code> is used).<br/>
<b>Note:</b> Supports <code>.docx</code>, <code>.xlsx</code>, and <code>.pptx</code> formats.</td>
<td><code>str</code></td>
<td>required</td>
</tr>
<tr>
<td><code>-o</code>, <code>--output</code></td>
<td><b>Meaning:</b> Output Markdown file path.<br/>
<b>Note:</b> If not set, the result is printed to stdout. When set, Markdown is written to the specified file and images are saved to an <code>images/</code> subdirectory alongside it.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>-q</code>, <code>--quiet</code></td>
<td><b>Meaning:</b> Quiet mode.<br/>
<b>Note:</b> When enabled, timing and save-path messages are suppressed.</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--formats</code></td>
<td><b>Meaning:</b> Print the list of supported file formats and exit.<br/>
<b>Note:</b> <code>--input</code> is not required when this flag is used.</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--no-drawings</code></td>
<td><b>Meaning:</b> Skip text box and drawing-layer content extraction.<br/>
<b>Note:</b> Applies to <code>.docx</code> and <code>.xlsx</code>. When enabled, text boxes (docx) and drawing-layer math formulas (xlsx) are not extracted.</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--no-headers-footers</code></td>
<td><b>Meaning:</b> Skip header and footer extraction.<br/>
<b>Note:</b> Applies to <code>.docx</code> only.</td>
<td>flag</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>--sheet-name</code></td>
<td><b>Meaning:</b> Convert only the sheet with the given name.<br/>
<b>Note:</b> Applies to <code>.xlsx</code> only. All sheets are converted when not set.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>--max-rows</code></td>
<td><b>Meaning:</b> Maximum number of rows to convert per sheet.<br/>
<b>Note:</b> Applies to <code>.xlsx</code> only. Use this to limit output for large spreadsheets.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />

### 2.2 Python API

**Basic usage:**

```python
from paddleocr._doc2md import convert

# Convert a document and get the result object
result = convert("report.docx")

# Access the Markdown text
print(result.markdown)

# List extracted images (dict mapping relative path → bytes)
print(list(result.images.keys()))

# Get the document title
print(result.title)

# Get metadata (format, sheet count, etc.)
print(result.metadata)
```

**`ConvertResult` fields:**

<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>markdown</code></td>
<td><code>str</code></td>
<td>The converted Markdown text</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>dict[str, bytes]</code></td>
<td>Extracted images dict; key is a relative path (e.g. <code>images/image1.png</code>), value is raw bytes</td>
</tr>
<tr>
<td><code>title</code></td>
<td><code>Optional[str]</code></td>
<td>Document title; may be <code>None</code></td>
</tr>
<tr>
<td><code>metadata</code></td>
<td><code>dict</code></td>
<td>Document metadata such as format type and sheet count</td>
</tr>
</tbody>
</table>

**Writing output to a file (images saved automatically):**

```python
from paddleocr._doc2md import convert

# When output is specified, Markdown is written to the file
# and images are saved to images/ in the same directory
result = convert("report.docx", output="output/report.md")
```

**Available kwargs by format:**

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Formats</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>extract_drawings</code></td>
<td><code>bool</code></td>
<td><code>True</code></td>
<td>docx, xlsx</td>
<td>Whether to extract text boxes (docx) / drawing-layer math formulas (xlsx)</td>
</tr>
<tr>
<td><code>extract_headers_footers</code></td>
<td><code>bool</code></td>
<td><code>True</code></td>
<td>docx</td>
<td>Whether to extract headers and footers</td>
</tr>
<tr>
<td><code>sheet_name</code></td>
<td><code>Optional[str]</code></td>
<td><code>None</code></td>
<td>xlsx</td>
<td>Convert only the named sheet; <code>None</code> converts all sheets</td>
</tr>
<tr>
<td><code>max_rows</code></td>
<td><code>Optional[int]</code></td>
<td><code>None</code></td>
<td>xlsx</td>
<td>Maximum number of rows to convert per sheet</td>
</tr>
</tbody>
</table>

**Passing kwargs by format:**

```python
from paddleocr._doc2md import convert

# Word: skip text boxes and headers/footers
result = convert("report.docx", extract_drawings=False, extract_headers_footers=False)

# Excel: convert only "Sheet1", up to 100 rows
result = convert("data.xlsx", sheet_name="Sheet1", max_rows=100)
```

## 3. Supported Features by Format

### 3.1 Word (.docx)

**Heading detection** uses three strategies:
- **Built-in Heading styles**: Word's built-in Heading 1–6 styles map directly to `#`–`######`
- **Font-size heuristic**: paragraphs with a font size more than 1.5× the body text and short length are promoted to headings
- **Chinese numbering**: "一、" patterns are treated as H2; "（一）" patterns as H3

**Text formatting**: bold (`**`), italic (`*`), underline (`<u>`), strikethrough (`~~`), superscript (`<sup>`), subscript (`<sub>`)

**Lists**: ordered, unordered, and nested lists; indentation level is detected automatically

**Tables**: output as HTML `<table>` with `rowspan`/`colspan` for merged cells

**Images**: width is calculated as a percentage of the content area; output as `<img width="75%">`

**Math formulas**: OMML formulas are converted to LaTeX; inline formulas use `$...$` and display formulas use `$$...$$`

**Code blocks**: monospace fonts (Courier New, Consolas, and 7 others) are detected automatically and output as fenced code blocks (` ``` `)

**Other**: text boxes (`wps:txbx`), charts (Chart → HTML table), hyperlinks (two formats supported), headers and footers (multi-section + odd/even pages)

### 3.2 Excel (.xlsx)

**Multiple sheets**: each sheet is output as a section beginning with `## sheet_name`

**Data boundary trimming**: trailing empty rows and columns are removed; only the effective data range is output

**Merged cells**: cell merges are reproduced using `rowspan`/`colspan`

**Text formatting**: bold, italic, underline, strikethrough, superscript, subscript

**Hyperlinks**: cell-level hyperlinks are supported

**Floating images**: both `OneCellAnchor` and `TwoCellAnchor` anchor types are supported

**Math formulas**: OMML formulas are extracted from the drawing-layer XML and converted to LaTeX

### 3.3 PowerPoint (.pptx)

**Multiple slides**: slide content is separated by `---`

**Text formatting**: bold, italic, underline, strikethrough, superscript, subscript

**Images**: width is calculated as a percentage of the slide width; output as `<img>` tags with a width attribute

**Tables**: HTML `<table>` format with merged cell support; tables with background images are handled

**Charts**: 14 chart types are supported; all are converted to HTML table output

**Group shapes**: nested shape groups are processed recursively

**Math formulas**: OMML formulas are extracted from `mc:AlternateContent` and converted to LaTeX

**Speaker notes**: appended at the end of each slide's content

## 4. FAQ

**Q: I get `RuntimeError: python-docx is required` during conversion.**

doc2md uses lazy imports — if a required parsing library is missing, this error is raised. Install the dependency for the format you need:
```bash
pip install python-docx     # Word (.docx)
pip install python-pptx     # PowerPoint (.pptx)
pip install openpyxl        # Excel (.xlsx)
pip install pylatexenc      # Math formula support
```
Or install all dependencies at once: `pip install "paddleocr[doc2md]"`

---

**Q: I get a `ValueError` saying the format is not supported.**

Run `paddleocr doc2md --formats` to see the currently supported extensions. doc2md only supports `.docx`, `.xlsx`, and `.pptx`. Formats such as `.doc` (legacy Word), `.csv`, and `.pdf` are not supported.

---

**Q: The Excel output is very long because the table has too many rows.**

Use `--max-rows` to limit the number of rows per sheet:
```bash
paddleocr doc2md -i data.xlsx -o output.md --max-rows 100
```

---

**Q: I only want to convert one specific sheet from an Excel file.**

Use `--sheet-name` to specify the sheet by name:
```bash
paddleocr doc2md -i data.xlsx -o output.md --sheet-name "Sheet1"
```

---

**Q: How do I skip headers and footers in a Word document?**

Use the `--no-headers-footers` flag:
```bash
paddleocr doc2md -i report.docx -o output.md --no-headers-footers
```

---

**Q: Where are the extracted images saved?**

When `-o` is specified, images are automatically saved to an `images/` subdirectory in the same directory as the output file. Image references in the Markdown file use relative paths accordingly.

---

**Q: What is the difference between doc2md and PaddleOCR's OCR feature?**

doc2md parses the XML structure of Office documents directly — **no OCR models are used**, making it fast with zero GPU requirement. It is suitable when the original Office file is available. PaddleOCR's OCR feature recognizes text from images or scanned documents, and is suitable when no original document exists.
