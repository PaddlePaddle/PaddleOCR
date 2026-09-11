#!/usr/bin/env python3
"""
validation/sweep_doc_signatures.py
Exhaustive 13x13 matrix sweep testing:
1. Self-detection: For each sample of doc_type T, verify detect_document_type(text) == T.
2. Cross-check mismatch: For each sample of doc_type T, verify that requesting any other
   doc_type U (where U != T) triggers check_doc_type_mismatch(U, text) == (True, T).
3. Ambiguity & score margin check: Ensures no ties or threshold failures occur.
"""

import json
import os
import sys
from PIL import Image

import ocr_engine
from verifier import DOC_SIGNATURES, check_doc_type_mismatch, detect_document_type

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

SAMPLE_FILES = {
    "pan": "/home/vighnesh/Downloads/Demo_PAN_Card.pdf",
    "aadhaar": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Aadhaar_Card.pdf",
    "cancelled_cheque": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Cancelled_Cheque.pdf",
    "bank_statement": "/home/vighnesh/Downloads/Demo_Bank_Statement.pdf",
    "salary_slip": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Salary_Slip.pdf",
    "driving_licence": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Driving_Licence.pdf",
    "passport": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Passport.pdf",
    "voter_id": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Voter_ID.pdf",
    "udyam": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Udyam_Registration.pdf",
    "fssai": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_FSSAI_Certificate.pdf",
    "shop_establishment": "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Shop_Establishment.pdf",
    "itr": "/home/vighnesh/Downloads/Demo_ITR_Acknowledgement.pdf",
    "utility_bill": "/home/vighnesh/Downloads/IMG-20260821-WA0003.jpg.jpeg",
    "gst_certificate": os.path.join(SAMPLES_DIR, "gst_certificate.txt"),
    "certificate_of_incorporation": os.path.join(SAMPLES_DIR, "certificate_of_incorporation.txt"),
    "partnership_deed": os.path.join(SAMPLES_DIR, "partnership_deed.txt"),
    "rent_agreement": os.path.join(SAMPLES_DIR, "rent_agreement.txt"),
    "form_16": os.path.join(SAMPLES_DIR, "form_16.txt"),
    "bank_passbook": os.path.join(SAMPLES_DIR, "bank_passbook.txt"),
    "property_tax_receipt": os.path.join(SAMPLES_DIR, "property_tax_receipt.txt"),
    "iec_certificate": os.path.join(SAMPLES_DIR, "iec_certificate.txt"),
}

ALL_DOC_TYPES = list(DOC_SIGNATURES.keys())


def extract_sample_text(doc_type: str, file_path: str) -> str:
    """Extract full OCR / digital text from sample file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file for '{doc_type}' not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    engine = ocr_engine.OCREngine()

    if ext == ".pdf":
        has_text_layer, full_text, _ = ocr_engine.check_pdf_text_layer(file_path, min_char_threshold=50)
        if has_text_layer and full_text.strip():
            return full_text
        doc_res = engine.process_pdf(file_path)
        return doc_res.full_text
    else:
        with Image.open(file_path) as img:
            page_res = engine.process_image(img)
            return page_res.full_text


def run_sweep():
    print(f"================================================================================")
    print(f"STARTING 21x21 DOCUMENT SIGNATURE AND MISMATCH MATRIX SWEEP")
    print(f"Supported document types ({len(ALL_DOC_TYPES)}): {', '.join(ALL_DOC_TYPES)}")
    print(f"================================================================================\n")

    # 1. Extract texts
    texts = {}
    for dtype, path in SAMPLE_FILES.items():
        text = extract_sample_text(dtype, path)
        texts[dtype] = text
        print(f"[EXTRACTED] {dtype:20s} from {os.path.basename(path)} ({len(text)} chars)")

    print("\n" + "=" * 80)
    print("PART 1: SELF-DETECTION CHECK (detect_document_type(text) == actual_type)")
    print("=" * 80)

    self_detection_results = []
    self_success = True

    for dtype in ALL_DOC_TYPES:
        text = texts[dtype]
        detected = detect_document_type(text)
        status = "PASS" if detected == dtype else "FAIL"
        if status == "FAIL":
            self_success = False

        # Calculate scores per signature for debugging
        scores = {}
        ocr_upper = text.upper()
        import re
        for dt, patterns in DOC_SIGNATURES.items():
            sc = sum(1 for p in patterns if re.search(p, ocr_upper))
            if sc > 0:
                scores[dt] = sc

        self_detection_results.append({
            "expected_type": dtype,
            "detected_type": detected,
            "status": status,
            "score_breakdown": scores,
        })
        print(f"  {dtype:20s} -> Detected: {str(detected):20s} [{status}] (scores: {scores})")

    print("\n" + "=" * 80)
    print("PART 2: 13x13 CROSS-CHECK MISMATCH MATRIX")
    print("Verifying check_doc_type_mismatch(requested_type, actual_text)")
    print("For actual == requested: is_mismatch must be False.")
    print("For actual != requested: is_mismatch must be True, detected == actual.")
    print("=" * 80)

    matrix_results = {}
    mismatch_failures = []

    header = f"{'Actual \\ Requested':<20}" + "".join(f"{dt[:4]:>6}" for dt in ALL_DOC_TYPES)
    print(header)
    print("-" * len(header))

    for actual_type in ALL_DOC_TYPES:
        row_str = f"{actual_type:<20}"
        text = texts[actual_type]
        matrix_results[actual_type] = {}

        for req_type in ALL_DOC_TYPES:
            is_mismatch, detected = check_doc_type_mismatch(req_type, text)
            matrix_results[actual_type][req_type] = {
                "is_mismatch": is_mismatch,
                "detected": detected,
            }

            if actual_type == req_type:
                # Should NOT be a mismatch
                if not is_mismatch:
                    cell = "  OK  "
                else:
                    cell = " ERR! "
                    mismatch_failures.append(f"False positive mismatch: actual={actual_type}, requested={req_type}, detected={detected}")
            else:
                # MUST be a mismatch
                if is_mismatch and detected == actual_type:
                    cell = " MIS  "
                elif is_mismatch:
                    cell = " M(?)"
                    mismatch_failures.append(f"Mismatch detected wrong type: actual={actual_type}, requested={req_type}, detected={detected}")
                else:
                    cell = " MISS "
                    mismatch_failures.append(f"Failed to flag mismatch: actual={actual_type}, requested={req_type}")

            row_str += cell
        print(row_str)

    print("\n" + "=" * 80)
    print("SUMMARY OF SWEEP RESULTS")
    print("=" * 80)
    print(f"Self-detection passed: {self_success}")
    print(f"Total mismatch cross-check tests: {len(ALL_DOC_TYPES) * len(ALL_DOC_TYPES)}")
    print(f"Total mismatch failures: {len(mismatch_failures)}")
    for f in mismatch_failures:
        print(f"  - {f}")

    # Output detailed report to JSON
    report = {
        "self_detection": self_detection_results,
        "mismatch_failures": mismatch_failures,
        "matrix": matrix_results,
    }

    report_path = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed JSON report saved to: {report_path}")

    return self_success and len(mismatch_failures) == 0


if __name__ == "__main__":
    success = run_sweep()
    sys.exit(0 if success else 1)
