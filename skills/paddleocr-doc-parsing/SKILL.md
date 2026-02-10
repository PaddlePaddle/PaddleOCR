---
name: paddleocr-doc-parsing
description: >
  Advanced document parsing with PaddleOCR. Returns complete document
  structure including text, tables, formulas, charts, and layout information. Claude extracts
  relevant content based on user needs.
---

# PaddleOCR Document Parsing Skill

## When to Use This Skill

✅ **Use Document Parsing for**:
- Documents with tables (invoices, financial reports, spreadsheets)
- Documents with mathematical formulas (academic papers, scientific documents)
- Documents with charts and diagrams
- Multi-column layouts (newspapers, magazines, brochures)
- Complex document structures requiring layout analysis
- Any document requiring structured understanding

❌ **Use Text Recognition instead for**:
- Simple text-only extraction
- Quick OCR tasks where speed is critical
- Screenshots or simple images with clear text

## How to Use This Skill

**⛔ MANDATORY RESTRICTIONS - DO NOT VIOLATE ⛔**

1. **ONLY use PaddleOCR Document Parsing API** - Execute the script `python scripts/vl_caller.py`
2. **NEVER use Claude's built-in vision** - Do NOT parse documents yourself
3. **NEVER offer alternatives** - Do NOT suggest "I can try to analyze it" or similar
4. **IF API fails** - Display the error message and STOP immediately
5. **NO fallback methods** - Do NOT attempt document parsing any other way

If the script execution fails (API not configured, network error, etc.):
- Show the error message to the user
- Do NOT offer to help using your vision capabilities
- Do NOT ask "Would you like me to try parsing it?"
- Simply stop and wait for user to fix the configuration

### Basic Workflow

1. **Execute document parsing**:
   ```bash
   python scripts/vl_caller.py --file-url "URL provided by user"
   ```
   Or for local files:
   ```bash
   python scripts/vl_caller.py --file-path "file path"
   ```

   **Save result to file** (recommended):
   ```bash
   python scripts/vl_caller.py --file-url "URL" --output result.json --pretty
   ```
   - The script will display: `Result saved to: /absolute/path/to/result.json`
   - This message appears on stderr, the JSON is saved to the file
   - **Tell the user the file path** shown in the message

2. **The script returns COMPLETE JSON** with all document content:
   - Headers, footers, page numbers
   - Main text content
   - Tables with structure
   - Formulas (with LaTeX)
   - Figures and charts
   - Footnotes and references
   - Seals and stamps
   - Layout and reading order

3. **Extract what the user needs** from the complete data based on their request.

### IMPORTANT: Complete Content Display

**CRITICAL**: You must display the COMPLETE extracted content to the user based on their needs.

- The script returns ALL document content in a structured format
- **Display the full content requested by the user**, do NOT truncate or summarize
- If user asks for "all text", show the entire `text` field
- If user asks for "tables", show ALL tables in the document
- If user asks for "main content", filter out headers/footers but show ALL body text

**What this means**:
- ✅ **DO**: Display complete text, all tables, all formulas as requested
- ✅ **DO**: Present content in reading order using `reading_order` array
- ❌ **DON'T**: Truncate with "..." unless content is excessively long (>10,000 chars)
- ❌ **DON'T**: Summarize or provide excerpts when user asks for full content
- ❌ **DON'T**: Say "Here's a preview" when user expects complete output

**Example - Correct**:
```
User: "Extract all the text from this document"
Claude: I've parsed the complete document. Here's all the extracted text:

[Display entire text field or concatenated regions in reading order]

Document Statistics:
- Total regions: 25
- Text blocks: 15
- Tables: 3
- Formulas: 2
Quality: Excellent (confidence: 0.92)
```

**Example - Incorrect** ❌:
```
User: "Extract all the text"
Claude: "I found a document with multiple sections. Here's the beginning:
'Introduction...' (content truncated for brevity)"
```

### Understanding the JSON Response

The script returns a JSON envelope wrapping the raw API result:

```json
{
  "ok": true,
  "text": "Full markdown/HTML text extracted from all pages",
  "result": [
    {
      "prunedResult": {
        "parsing_res_list": [
          {"block_label": "text", "block_content": "Paragraph text content here...", "block_bbox": [100, 200, 500, 230], "block_id": 3},
          {"block_label": "table", "block_content": "<table>...</table>", "block_bbox": [50, 300, 900, 600], "block_id": 5},
          {"block_label": "seal", "block_content": "<img .../>", "block_bbox": [400, 50, 600, 180], "block_id": 2}
        ]
      },
      "markdown": {
        "text": "Full page content in markdown/HTML format",
        "images": {"imgs/filename.jpg": "https://..."}
      }
    }
  ],
  "error": null
}
```

**Key fields**:
- `text` — extracted markdown text from all pages (use this for quick text display)
- `result` — raw API result array (one object per page), for detailed block-level access
- `result[n].prunedResult.parsing_res_list` — array of detected content blocks
- `result[n].markdown.text` — full page content in markdown/HTML

### Block Labels

The `block_label` field indicates the content type:

| Label | Description |
|-------|-------------|
| `text` | Regular text content |
| `table` | Table (content is HTML `<table>`) |
| `image` | Embedded image |
| `seal` | Seal or stamp |
| `figure_title` | Figure/chart title or caption |
| `vision_footnote` | Footnote detected by vision model |
| `aside_text` | Side/margin text |

### Content Extraction Guidelines

**Based on user intent, filter the blocks**:

| User Says | What to Extract | How |
|-----------|-----------------|-----|
| "Extract all text" | Everything | Use `text` field directly |
| "Get all tables" | table blocks only | Filter `parsing_res_list` by `block_label == "table"` |
| "Show main content" | text + table blocks | Filter out `aside_text`, `image` |
| "Complete document" | Everything | Use `text` field or iterate all blocks |

### Usage Examples

**Example 1: Extract Main Content** (default behavior)
```bash
python scripts/vl_caller.py \
  --file-url "https://example.com/paper.pdf" \
  --pretty
```

Then filter JSON to extract core content:
- Include: text, table, formula, figure, footnote
- Exclude: header, footer, page_number

**Example 2: Extract Tables Only**
```bash
python scripts/vl_caller.py \
  --file-path "./financial_report.pdf" \
  --pretty
```

Then filter JSON:
- Only keep regions where type="table"
- Present table content in markdown format

**Example 3: Complete Document with Everything**
```bash
python scripts/vl_caller.py \
  --file-url "URL" \
  --pretty
```

Then use the `text` field or present all regions in reading_order.

### First-Time Configuration

**When API is not configured**:

The error will show:
```
Configuration error: API not configured. Get your API at: https://paddleocr.com
```

**Auto-configuration workflow**:

1. **Show the exact error message** to user (including the URL)

2. **Tell user to provide credentials**:
   ```
   Please visit the URL above to get your PADDLEOCR_DOC_PARSING_API_URL and PADDLEOCR_ACCESS_TOKEN.
   Once you have them, send them to me and I'll configure it automatically.
   ```

3. **When user provides credentials** (accept any format):
   - `PADDLEOCR_DOC_PARSING_API_URL=https://xxx.com/layout-parsing, PADDLEOCR_ACCESS_TOKEN=abc123...`
   - `Here's my API: https://xxx and token: abc123`
   - Copy-pasted code format
   - Any other reasonable format

4. **Parse credentials from user's message**:
   - Extract PADDLEOCR_DOC_PARSING_API_URL value (look for URLs)
   - Extract PADDLEOCR_ACCESS_TOKEN value (long alphanumeric string, usually 40+ chars)

5. **Configure automatically**:
   ```bash
   python scripts/configure.py --api-url "PARSED_URL" --token "PARSED_TOKEN"
   ```

6. **If configuration succeeds**:
   - Inform user: "Configuration complete! Parsing document now..."
   - Retry the original parsing task

7. **If configuration fails**:
   - Show the error
   - Ask user to verify the credentials

**IMPORTANT**: The error message format is STRICT and must be shown exactly as provided by the script. Do not modify or paraphrase it.

### Handling Large Files

There is no file size limit for the API. For PDFs, the maximum is 100 pages per request.

**Tips for large files**:

#### Use URL for Large Local Files (Recommended)
For very large local files, prefer `--file-url` over `--file-path` to avoid base64 encoding overhead:
```bash
python scripts/vl_caller.py --file-url "https://your-server.com/large_file.pdf"
```

#### Process Specific Pages (PDF Only)
If you only need certain pages from a large PDF, extract them first:
```bash
# Using pypdfium2 (requires: pip install pypdfium2)
python -c "
import pypdfium2 as pdfium
doc = pdfium.PdfDocument('large.pdf')
# Extract pages 0-4 (first 5 pages)
new_doc = pdfium.PdfDocument.new()
for i in range(min(5, len(doc))):
    new_doc.import_pages(doc, [i])
new_doc.save('pages_1_5.pdf')
"

# Then process the smaller file
python scripts/vl_caller.py --file-path "pages_1_5.pdf"
```

### Error Handling

**Authentication failed (401/403)**:
```
error: Authentication failed
```
→ Token is invalid, reconfigure with correct credentials

**API quota exceeded (429)**:
```
error: API quota exceeded
```
→ Daily API quota exhausted, inform user to wait or upgrade

**Unsupported format**:
```
error: Unsupported file format
```
→ File format not supported, convert to PDF/PNG/JPG

### Pseudo-Code: Content Extraction

**Extract all text** (most common):
```python
def extract_all_text(json_response):
    # Quickest: use the pre-extracted text field
    print(json_response['text'])
```

**Extract tables only**:
```python
def extract_tables(json_response):
    for page in json_response['result']:
        blocks = page['prunedResult']['parsing_res_list']
        tables = [b for b in blocks if b['block_label'] == 'table']
        for i, table in enumerate(tables):
            print(f"Table {i+1}:")
            print(table['block_content'])  # HTML table
```

**Iterate all blocks**:
```python
def extract_by_block(json_response):
    for page in json_response['result']:
        blocks = page['prunedResult']['parsing_res_list']
        for block in blocks:
            print(f"[{block['block_label']}] {block['block_content'][:100]}")
```

## Important Notes

- **The script NEVER filters content** - It always returns complete data
- **Claude decides what to present** - Based on user's specific request
- **All data is always available** - Can be re-interpreted for different needs
- **No information is lost** - Complete document structure preserved

## Reference Documentation

For in-depth understanding of the PaddleOCR Document Parsing system, refer to:
- `references/output_schema.md` - Output format specification
- `references/provider_api.md` - Provider API contract

> **Note**: Model version and capabilities are determined by your API endpoint (PADDLEOCR_DOC_PARSING_API_URL).

Load these reference documents into context when:
- Debugging complex parsing issues
- Need to understand output format
- Working with provider API details

## Testing the Skill

To verify the skill is working properly:
```bash
python scripts/smoke_test.py
```

This tests configuration and optionally API connectivity.
