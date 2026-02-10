---
name: paddleocr-text-recognition
description: >
  Use this skill when users need to extract text from images, PDFs, or documents. Supports URLs and local files.
  Returns structured JSON containing recognized text.
---

# PaddleOCR Text Recognition Skill

## When to Use This Skill

Invoke this skill in the following situations:
- Extract text from images (screenshots, photos, scans, charts)
- Read text from PDF or document images
- Perform OCR on any visual content containing text
- Parse structured documents (invoices, receipts, forms, tables)
- Recognize text in photos taken by mobile phones
- Extract text from URLs pointing to images or PDFs

Do not use this skill in the following situations:
- Plain text files that can be read directly with the Read tool
- Code files or markdown documents
- Tasks that do not involve image-to-text conversion

## How to Use This Skill

**MANDATORY RESTRICTIONS - DO NOT VIOLATE**

1. **ONLY use PaddleOCR Text Recognition API** - Execute the script `python scripts/ocr_caller.py`
2. **NEVER use Claude's built-in vision** - Do NOT read images yourself
3. **NEVER offer alternatives** - Do NOT suggest "I can try to read it" or similar
4. **IF API fails** - Display the error message and STOP immediately
5. **NO fallback methods** - Do NOT attempt OCR any other way

If the script execution fails (API not configured, network error, etc.):
- Show the error message to the user
- Do NOT offer to help using your vision capabilities
- Do NOT ask "Would you like me to try reading it?"
- Simply stop and wait for user to fix the configuration

### Basic Workflow

1. **Identify the input source**:
   - User provides URL: Use the `--file-url` parameter
   - User provides local file path: Use the `--file-path` parameter
   - User uploads image: Save it first, then use `--file-path`

2. **Execute OCR**:
   ```bash
   python scripts/ocr_caller.py --file-url "URL provided by user" --pretty
   ```
   Or for local files:
   ```bash
   python scripts/ocr_caller.py --file-path "file path" --pretty
   ```

   **Save result to file** (recommended):
   ```bash
   python scripts/ocr_caller.py --file-url "URL" --output result.json --pretty
   ```

3. **Parse JSON response**:
   - Check the `ok` field: `true` means success, `false` means error
   - Extract text: `text` field contains all recognized text
   - Handle errors: If `ok` is false, display `error.message`

4. **Present results to user**:
   - Display extracted text in a readable format
   - If the text is empty, the image may contain no text

### IMPORTANT: Complete Output Display

**CRITICAL**: Always display the COMPLETE recognized text to the user. Do NOT truncate or summarize the OCR results.

- The script returns the full JSON with complete text content in `text` field
- **You MUST display the entire `text` content to the user**, no matter how long it is
- Do NOT use phrases like "Here's a summary" or "The text begins with..."
- Do NOT truncate with "..." unless the text truly exceeds reasonable display limits
- The user expects to see ALL the recognized text, not a preview or excerpt

**Correct approach**:
```
I've extracted the text from the image. Here's the complete content:

[Display the entire text here]
```

**Incorrect approach**:
```
I found some text in the image. Here's a preview:
"The quick brown fox..." (truncated)
```

### Usage Examples

**URL OCR**:
```bash
python scripts/ocr_caller.py --file-url "https://example.com/invoice.jpg" --pretty
```

**Local File OCR**:
```bash
python scripts/ocr_caller.py --file-path "./document.pdf" --pretty
```

### Understanding the Output

The script outputs JSON structure as follows:
```json
{
  "ok": true,
  "text": "All recognized text here...",
  "result": { ... },
  "error": null
}
```

**Key fields**:
- `ok`: `true` for success, `false` for error
- `text`: Complete recognized text
- `result`: Raw API response (for debugging)
- `error`: Error details if `ok` is false

### First-Time Configuration

**When API is not configured**:

The error will show:
```
CONFIG_ERROR: PADDLEOCR_OCR_API_URL not configured. Get your API at: https://paddleocr.com
```

**Configuration workflow**:

1. **Show the exact error message** to user (including the URL)

2. **Tell user to provide credentials**:
   ```
   Please visit the URL above to get your API_URL and TOKEN.
   Once you have them, send them to me and I'll configure it automatically.
   ```

3. **When user provides credentials** (accept any format):
   - `API_URL=https://xxx.paddleocr.com/ocr, TOKEN=abc123...`
   - `Here's my API: https://xxx and token: abc123`
   - Copy-pasted code format
   - Any other reasonable format

4. **Parse credentials from user's message**:
   - Extract API_URL value (look for URLs with paddleocr.com or similar)
   - Extract TOKEN value (long alphanumeric string, usually 40+ chars)

5. **Configure automatically**:
   ```bash
   python scripts/configure.py --api-url "PARSED_URL" --token "PARSED_TOKEN"
   ```

6. **If configuration succeeds**:
   - Inform user: "Configuration complete! Running OCR now..."
   - Retry the original OCR task

7. **If configuration fails**:
   - Show the error
   - Ask user to verify the credentials

### Error Handling

**Authentication failed**:
```
API_ERROR: Authentication failed (403). Check your token.
```
- Token is invalid, reconfigure with correct credentials

**Quota exceeded**:
```
API_ERROR: API rate limit exceeded (429)
```
- Daily API quota exhausted, inform user to wait or upgrade

**No text detected**:
- `text` field is empty
- Image may be blank, corrupted, or contain no text

### Tips for Better Results

If recognition quality is poor, suggest:
- Check if the image is clear and contains text
- Provide a higher resolution image if possible

## Reference Documentation

For in-depth understanding of the OCR system, refer to:
- `references/output_schema.md` - Output format specification
- `references/provider_api.md` - Provider API contract

> **Note**: Model version and capabilities are determined by your API endpoint (PADDLEOCR_OCR_API_URL).

## Testing the Skill

To verify the skill is working properly:
```bash
python scripts/smoke_test.py
```

This tests configuration and API connectivity.
