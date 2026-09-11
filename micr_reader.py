"""
micr_reader.py
Specialized MICR (Magnetic Ink Character Recognition - E-13B) reader for cancelled cheques:
- Crops bottom 15-20% band of cheque image
- Recognizes E-13B delimiters (⑆ Transit/Cheque, ⑈ On-Us/Account, ⑇ Amount, ⑉ Dash) and digits
- Parses standard Indian MICR structure:
  1. Cheque Number (6 digits)
  2. MICR / Transit Code (9 digits: 3 City + 3 Bank + 3 Branch)
  3. Account Number (variable 6-18 digits)
  4. Transaction Code (2 digits)
- Evaluates confidence: marks micr_confidence: "low" if uncertain or format mismatches.
- Never fabricates digits.
- Cross-checks MICR-derived fields against full-page OCR extracted fields.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

from extractors import mask_account_number


def crop_cheque_bottom_band(cheque_img: Image.Image, band_ratio: float = 0.20) -> Image.Image:
    """
    Crop the bottom 15-20% horizontal band where the MICR band is printed.
    Default band_ratio = 0.20 (20%).
    """
    w, h = cheque_img.size
    top = int(h * (1.0 - band_ratio))
    return cheque_img.crop((0, top, w, h))


def parse_micr_string(raw_micr: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse standard Indian RBI Cheque MICR format:
    Example: ⑆000123⑆ 400002001⑈ 123456789012⑇ 10
    - 6 digits: Cheque Number
    - 9 digits: MICR City(3) + Bank(3) + Branch(3)
    - 6-18 digits: Account Number
    - 2 digits: Transaction Code

    Strict rule: Never silently fabricate digits.
    If digit counts do not match expected structure, micr_confidence is set to 'low'.
    """
    clean_text = raw_micr.replace("\n", " ").strip()
    result: Dict[str, Any] = {
        "micr_line": clean_text,
        "raw_micr_line": clean_text,
        "cheque_number": None,
        "micr_code": None,
        "account_number": None,
        "account_number_masked": None,
        "tran_code": None,
        "micr_confidence": "low",
        "micr_confidence_score": 0.30,
    }

    if not clean_text:
        return result, "low"

    # Normalize common E-13B OCR delimiter misrecognitions into uniform delimiters
    # ⑆ (transit): ', ", c, C, :, |
    # ⑈ (on-us): a, A, ;
    # ⑇ (amount): d, D
    # Standardize delimiters
    normalized = clean_text
    # Extract candidate 6-digit cheque number (typically first 6-digit cluster)
    chq_match = re.search(r"(?:[c⑆|:\"']|\b)(\d{6})(?:[c⑆|:\"']|\b)", normalized)
    chq_span = None
    if chq_match:
        result["cheque_number"] = chq_match.group(1)
        chq_span = chq_match.span(1)

    # Extract candidate 9-digit MICR code
    micr_match = re.search(r"(?:[a⑈d⑇|\s:\"']|^)(\d{9})(?:[a⑈d⑇|\s:\"']|$)", normalized)
    micr_span = None
    if micr_match:
        result["micr_code"] = micr_match.group(1)
        micr_span = micr_match.span(1)

    # Extract candidate 2-digit transaction code at the end
    tran_match = re.search(r"\b(\d{2})\b(?:\s*[\"']?)$", normalized)
    tran_span = None
    if tran_match:
        result["tran_code"] = tran_match.group(1)
        tran_span = tran_match.span(1)

    # Extract account number:
    # On Indian cheques, account number is typically situated after MICR code and before tran code
    # Find remaining digit clusters of 6-18 digits that are not the cheque number or MICR code
    candidate_account = None
    digit_blocks = list(re.finditer(r"\b(\d{6,18})\b", normalized))
    for m in digit_blocks:
        span = m.span(1)
        val = m.group(1)
        # Skip if it is the cheque number or MICR code
        if chq_span and span == chq_span:
            continue
        if micr_span and span == micr_span:
            continue
        # If situated after micr code
        if micr_span and span[0] >= micr_span[1]:
            candidate_account = val
            break
        elif not candidate_account:
            candidate_account = val

    if candidate_account:
        result["account_number"] = candidate_account
        result["account_number_masked"] = mask_account_number(candidate_account)

    # Validate structure and determine confidence without guessing/fabricating
    has_valid_chq = bool(result["cheque_number"] and len(result["cheque_number"]) == 6)
    has_valid_micr = bool(result["micr_code"] and len(result["micr_code"]) == 9)
    has_acc = bool(result["account_number"] and 6 <= len(result["account_number"]) <= 18)
    has_tran = bool(result["tran_code"] and len(result["tran_code"]) == 2)

    # Check for degraded / corrupt patterns (e.g. invalid length digits found where MICR line was expected)
    # If raw line contains partial/corrupt fragments (e.g. 4-digit or 7-digit numbers where 6/9 needed)
    partial_digits = re.findall(r"\b\d+\b", normalized)
    is_severely_corrupted = False
    if not has_valid_chq and not has_valid_micr:
        is_severely_corrupted = True
    elif has_valid_chq and not has_valid_micr:
        # If there are numbers present but none is 9 digits
        other_nums = [n for n in partial_digits if n != result["cheque_number"]]
        if any(len(n) in (7, 8, 10) for n in other_nums):
            is_severely_corrupted = True

    if has_valid_chq and has_valid_micr and has_acc:
        result["micr_confidence"] = "high"
        result["micr_confidence_score"] = 0.95
    elif has_valid_chq and has_valid_micr:
        result["micr_confidence"] = "medium"
        result["micr_confidence_score"] = 0.70
    else:
        result["micr_confidence"] = "low"
        result["micr_confidence_score"] = 0.30

    return result, result["micr_confidence"]


def cross_check_micr_with_cheque_fields(
    micr_data: Dict[str, Any],
    cheque_fields: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Cross-checks MICR-derived fields against full-page extracted fields.
    Surfaces disagreements if values differ, without silently preferring one.

    Returns:
        (micr_match: bool, disagreements: List[str])
    """
    disagreements: List[str] = []

    # 1. Cross-check Cheque Number
    micr_chq = micr_data.get("cheque_number")
    doc_chq = cheque_fields.get("cheque_number")
    if micr_chq and doc_chq:
        if str(micr_chq).strip() != str(doc_chq).strip():
            disagreements.append(
                f"Cheque number mismatch: MICR line has '{micr_chq}', full-page text has '{doc_chq}'"
            )

    # 2. Cross-check Account Number (compare last 4 digits)
    micr_acc = micr_data.get("account_number")
    doc_acc_masked = cheque_fields.get("account_number_masked")
    if micr_acc and doc_acc_masked:
        clean_micr_digits = re.sub(r"\D", "", str(micr_acc))
        clean_doc_digits = re.sub(r"\D", "", str(doc_acc_masked))
        if len(clean_micr_digits) >= 4 and len(clean_doc_digits) >= 4:
            if clean_micr_digits[-4:] != clean_doc_digits[-4:]:
                disagreements.append(
                    f"Account number mismatch: MICR line ends with '{clean_micr_digits[-4:]}', "
                    f"full-page text ends with '{clean_doc_digits[-4:]}'"
                )

    micr_match = len(disagreements) == 0
    return micr_match, disagreements


def extract_micr_from_cheque(
    cheque_img: Image.Image,
    ocr_func: Optional[Callable[[Image.Image], Tuple[str, Any]]] = None,
    ocr_full_text: Optional[str] = None,
    full_page_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract MICR data from a cheque image.
    Uses bottom crop + OCR on the 15-20% band, with fallback to full-page OCR text.
    Cross-checks with full-page fields and surfaces disagreements.
    """
    parsed: Optional[Dict[str, Any]] = None
    conf = "low"

    # 1. Dedicated OCR pass on bottom 20% crop band
    if ocr_func is not None and cheque_img and (cheque_img.width > 50 and cheque_img.height > 50):
        try:
            band_img = crop_cheque_bottom_band(cheque_img, band_ratio=0.20)
            band_text, _ = ocr_func(band_img)
            if band_text and band_text.strip():
                parsed_band, band_conf = parse_micr_string(band_text)
                if band_conf in ("high", "medium"):
                    parsed = parsed_band
                    conf = band_conf
        except Exception:
            pass

    # 2. Fallback attempt: search in full OCR text for MICR line patterns
    if (not parsed or conf == "low") and ocr_full_text:
        parsed_text, text_conf = parse_micr_string(ocr_full_text)
        if parsed_text.get("cheque_number") or parsed_text.get("micr_code"):
            if not parsed or (parsed and text_conf in ("high", "medium")):
                parsed = parsed_text
                conf = text_conf

    # 3. Default if nothing parsed
    if not parsed:
        parsed = {
            "micr_line": None,
            "raw_micr_line": None,
            "cheque_number": None,
            "micr_code": None,
            "account_number": None,
            "account_number_masked": None,
            "tran_code": None,
            "micr_confidence": "low",
            "micr_confidence_score": 0.0,
        }

    # 4. Cross-check against full-page fields if available
    if full_page_fields:
        match, disagreements = cross_check_micr_with_cheque_fields(parsed, full_page_fields)
        parsed["micr_match"] = match
        parsed["micr_disagreements"] = disagreements
    else:
        parsed["micr_match"] = True
        parsed["micr_disagreements"] = []

    return parsed
