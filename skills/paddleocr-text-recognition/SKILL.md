---
name: paddleocr-text-recognition
description: Extracts text (with locations) from images and PDF documents using PaddleOCR.
metadata:
  openclaw:
    requires:
      env:
        - PADDLEOCR_OCR_API_URL
        - PADDLEOCR_ACCESS_TOKEN
        - PADDLEOCR_OCR_TIMEOUT
      bins:
        - python
    primaryEnv: PADDLEOCR_ACCESS_TOKEN
    emoji: "🔤"
    homepage: https://github.com/PaddlePaddle/PaddleOCR/tree/main/skills/paddleocr-text-recognition
---

# PaddleOCR Text Recognition Skill

## When to Use This Skill

Invoke this skill in the following situations:
- Extract text from images (screenshots, photos, scans)
- Extract text from PDFs or document images
- Extract text and positions from structured documents (invoices, receipts, forms, tables)
- Extract text from URLs or local files that point to images/PDFs

Do not use this skill in the following situations:
- Plain text files that can be read directly with the Read tool
- Code files or markdown documents
- Tasks that do not involve image-to-text conversion

## How to Use This Skill

**⛔ MANDATORY RESTRICTIONS - DO NOT VIOLATE ⛔**

1. **ONLY use PaddleOCR Text Recognition API** - Execute the script `python scripts/ocr_caller.py`
2. **NEVER read images directly** - Do NOT read images yourself
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

   **Input type note**:
   - Supported file types depend on the model and endpoint configuration.
   - Follow the official endpoint/API documentation for the exact supported formats.

2. **Execute OCR**:
   ```bash
   python scripts/ocr_caller.py --file-url "URL provided by user" --pretty
   ```
   Or for local files:
   ```bash
   python scripts/ocr_caller.py --file-path "file path" --pretty
   ```

   **Default behavior: save raw JSON to a temp file**:
   - If `--output` is omitted, the script saves automatically under the system temp directory
   - Default path pattern: `<system-temp>/paddleocr/text-recognition/results/result_<timestamp>_<id>.json`
   - If `--output` is provided, it overrides the default temp-file destination
   - If `--stdout` is provided, JSON is printed to stdout and no file is saved
   - In save mode, the script prints the absolute saved path on stderr: `Result saved to: /absolute/path/...`
   - In default/custom save mode, read and parse the saved JSON file before responding
   - Use `--stdout` only when you explicitly want to skip file persistence

3. **Parse JSON response**:
   - In default/custom save mode, load JSON from the saved file path shown by the script
   - Check the `ok` field: `true` means success, `false` means error
   - Extract text: `text` field contains all recognized text
   - If `--stdout` is used, parse the stdout JSON directly
   - Handle errors: If `ok` is false, display `error.message`

4. **Present results to user**:
   - Display extracted text in a readable format
   - If the text is empty, the image may contain no text
   - In save mode, always tell the user the saved file path and that full raw JSON is available there

### IMPORTANT: Complete Output Display

**CRITICAL**: Always display the COMPLETE recognized text to the user. Do NOT truncate or summarize the OCR results.

- The output JSON contains complete output, including full text in `text` field
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

**Example 1: URL OCR**:
```bash
python scripts/ocr_caller.py --file-url "https://example.com/invoice.jpg" --pretty
```

**Example 2: Local File OCR**:
```bash
python scripts/ocr_caller.py --file-path "./document.pdf" --pretty
```

**Example 3: OCR With Explicit File Type**:
```bash
python scripts/ocr_caller.py --file-url "https://example.com/input" --file-type 1 --pretty
```

**Example 4: Print JSON Without Saving**:
```bash
python scripts/ocr_caller.py --file-url "https://example.com/input" --stdout --pretty
```

### Understanding the Output

The output JSON structure is as follows:
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

> Raw result location (default): the temp-file path printed by the script on stderr

### First-Time Configuration

You can generally assume that the required environment variables have already been configured. Only when an OCR task fails should you analyze the error message to determine whether it is caused by a configuration issue. If it is indeed a configuration problem, you should notify the user to fix it.

**When API is not configured**:

The error will show:
```
CONFIG_ERROR: PADDLEOCR_OCR_API_URL not configured. Get your API at: https://paddleocr.com
```

**Configuration workflow**:

1. **Show the exact error message** to the user (including the URL).

2. **Guide the user to configure securely**:
   - Recommend configuring through the host application's standard method (e.g., settings file, environment variable UI) rather than pasting credentials in chat.
   - List the required environment variables:
     ```
     - PADDLEOCR_OCR_API_URL
     - PADDLEOCR_ACCESS_TOKEN
     - Optional: PADDLEOCR_OCR_TIMEOUT
     ```

3. **If the user provides credentials in chat anyway** (accept any reasonable format), for example:
   - `PADDLEOCR_OCR_API_URL=https://xxx.paddleocr.com/ocr, PADDLEOCR_ACCESS_TOKEN=abc123...`
   - `Here's my API: https://xxx and token: abc123`
   - Copy-pasted code format
   - Any other reasonable format
   - **Security note**: Warn the user that credentials shared in chat may be stored in conversation history. Recommend setting them through the host application's configuration instead when possible.

   Then parse and validate the values:
   - Extract `PADDLEOCR_OCR_API_URL` (look for URLs with `paddleocr.com` or similar)
   - Confirm `PADDLEOCR_OCR_API_URL` is a full endpoint ending with `/ocr`
   - Extract `PADDLEOCR_ACCESS_TOKEN` (long alphanumeric string, usually 40+ chars)

4. **Ask the user to confirm the environment is configured**.

5. **Retry only after confirmation**:
   - Once the user confirms the environment variables are available, retry the original OCR task

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

> **Note**: Model version, capabilities, and supported file formats are determined by your API endpoint (`PADDLEOCR_OCR_API_URL`) and its official API documentation.

## Testing the Skill

To verify the skill is working properly:
```bash
python scripts/smoke_test.py
```

This tests configuration and API connectivity.
