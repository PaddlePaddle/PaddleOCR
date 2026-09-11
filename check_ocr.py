import os
import sys
import subprocess


def extract_pdf_text(pdf_path: str) -> str:
    """Extract embedded text from a PDF."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return ""

        return result.stdout.strip()

    except Exception as exc:
        print(f"Text extraction error: {exc}")
        return ""


def get_pdf_page_count(pdf_path: str) -> int:
    """Get PDF page count."""
    try:
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())

    except Exception:
        pass

    return 0


def check_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return

    extension = os.path.splitext(pdf_path)[1].lower()

    print("=" * 60)
    print("DOCUMENT OCR CHECK")
    print("=" * 60)
    print(f"File: {pdf_path}")
    print(f"Type: {extension}")

    if extension != ".pdf":
        print("\nRESULT: OCR REQUIRED")
        print("Reason: This is an image/document file without a PDF text layer.")
        return

    pages = get_pdf_page_count(pdf_path)
    text = extract_pdf_text(pdf_path)

    character_count = len(text)
    words = len(text.split())

    print(f"Pages: {pages}")
    print(f"Extracted characters: {character_count}")
    print(f"Extracted words: {words}")

    print("\n" + "-" * 60)

    if character_count >= 50:
        print("RESULT: OCR NOT REQUIRED")
        print("Reason: The PDF contains an existing text layer.")
        print("\nExtracted text preview:")
        print(text[:1000])

    else:
        print("RESULT: OCR REQUIRED")
        print("Reason: The PDF contains little or no embedded text.")
        print("The pages should be rendered as images and processed using OCR.")

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 check_ocr.py <document.pdf>")
        sys.exit(1)

    check_document(sys.argv[1])
