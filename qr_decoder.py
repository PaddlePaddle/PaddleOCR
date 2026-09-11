"""
qr_decoder.py
Performs QR code detection and decoding alongside OCR:
- Decodes QR using OpenCV QRCodeDetector or pyzbar
- Extracts structured data for Aadhaar, Udyam, and FSSAI
- Compares with OCR fields, prioritizes QR (higher trust),
  and explicitly surfaces disagreements.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def decode_qr_from_image(pil_img: Image.Image) -> List[str]:
    """Decode any QR codes present in the PIL image."""
    decoded_texts: List[str] = []

    # 1. Try OpenCV QRCodeDetector if cv2 is available
    if HAS_CV2:
        try:
            cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            detector = cv2.QRCodeDetector()
            # Try detectAndDecode
            data, bbox, _ = detector.detectAndDecode(cv_img)
            if data:
                decoded_texts.append(data)
            else:
                # Try detectAndDecodeMulti
                retval, decoded_info, points, _ = detector.detectAndDecodeMulti(cv_img)
                if retval and decoded_info:
                    for text in decoded_info:
                        if text and text not in decoded_texts:
                            decoded_texts.append(text)
        except Exception:
            pass

    # 2. Try pyzbar if installed
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        results = pyzbar_decode(pil_img)
        for r in results:
            text = r.data.decode("utf-8", errors="ignore")
            if text and text not in decoded_texts:
                decoded_texts.append(text)
    except (ImportError, Exception):
        pass

    return decoded_texts


def parse_qr_payload(doc_type: str, raw_text: str) -> Dict[str, Any]:
    """
    Parse document-specific payloads from decoded QR code data.
    """
    parsed: Dict[str, Any] = {"raw_qr": raw_text}

    if doc_type == "aadhaar":
        # Aadhaar QR codes often contain XML, JSON, or secure string formats
        # e.g., <?xml version="1.0"?> <PrintLetterBarcodeData uid="123412341234" name="..." dob="..." />
        uid_match = re.search(r'uid="?(\d{12})"?', raw_text)
        if uid_match:
            parsed["aadhaar_number"] = uid_match.group(1)
        name_match = re.search(r'name="?([^"\n>]+)"?', raw_text, re.IGNORECASE)
        if name_match:
            parsed["name"] = name_match.group(1).strip()
        dob_match = re.search(r'dob="?([^"\n>]+)"?', raw_text, re.IGNORECASE)
        if dob_match:
            parsed["dob"] = dob_match.group(1).strip()
        gender_match = re.search(r'gender="?([MFmf]|Male|Female)"?', raw_text, re.IGNORECASE)
        if gender_match:
            parsed["gender"] = "Male" if gender_match.group(1).upper().startswith("M") else "Female"

    elif doc_type == "udyam":
        # Udyam QR contains verification URL: https://udyamregistration.gov.in/print/verify.aspx?urn=UDYAM-XX-00-0000000
        urn_match = re.search(r"UDYAM-[A-Z]{2}-\d{2}-\d{7}", raw_text, re.IGNORECASE)
        if urn_match:
            parsed["udyam_registration_number"] = urn_match.group(0).upper()

    elif doc_type == "fssai":
        # FSSAI QR contains verification link with 14-digit license number
        fssai_match = re.search(r"\b(\d{14})\b", raw_text)
        if fssai_match:
            parsed["fssai_licence_number"] = fssai_match.group(1)

    return parsed


def reconcile_ocr_and_qr(
    ocr_fields: Dict[str, Any],
    qr_fields: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Combine OCR and QR data.
    Treat QR data as higher-trust when both are present and disagree;
    surface the disagreement explicitly.
    """
    merged = dict(ocr_fields)
    disagreements: List[Dict[str, Any]] = []

    for field, qr_val in qr_fields.items():
        if field == "raw_qr":
            continue
        ocr_val = ocr_fields.get(field)
        if ocr_val is not None and qr_val is not None:
            # Check if they disagree (normalizing whitespace and case)
            str_ocr = str(ocr_val).strip().upper()
            str_qr = str(qr_val).strip().upper()
            if str_ocr != str_qr:
                disagreements.append({
                    "field": field,
                    "ocr_value": ocr_val,
                    "qr_value": qr_val,
                    "resolution": "qr_preferred",
                    "reason": "QR payload is cryptographically/digitally signed or higher trust",
                })
                # Override with QR value
                merged[field] = qr_val
        elif qr_val is not None and ocr_val is None:
            merged[field] = qr_val

    return merged, disagreements
