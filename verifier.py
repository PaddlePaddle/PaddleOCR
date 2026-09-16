"""
verifier.py
Comprehensive verification logic for Company-Server OCR service:
- Format & checksum validation:
  - PAN format (including entity character)
  - Aadhaar Verhoeff checksum algorithm
  - IFSC format and bank-prefix validation
- Cross-check against optional expected applicant data (fuzzy match & date normalization)
- Document-type mismatch detection
"""

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# 1. Checksum & Format Validation
# ==============================================================================

# Verhoeff algorithm multiplication and permutation tables
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff_checksum(number_str: str) -> bool:
    """
    Validate standard Verhoeff checksum.
    Returns True if valid 12-digit Aadhaar number checksum evaluates to 0.
    """
    clean_num = re.sub(r"\D", "", number_str)
    if len(clean_num) != 12:
        return False
    # Aadhaar numbers cannot begin with 0 or 1
    if clean_num[0] in ("0", "1"):
        return False

    c = 0
    # Process digits in reverse order
    for i, digit_char in enumerate(reversed(clean_num)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit_char)]]
    return c == 0


def generate_verhoeff_checksum_digit(number_str_11: str) -> str:
    """Helper to compute the 12th Verhoeff checksum digit for testing."""
    c = 0
    for i, digit_char in enumerate(reversed(number_str_11)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(digit_char)]]
    return str(_VERHOEFF_INV[c])


def validate_pan_format(pan_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Indian Permanent Account Number (PAN):
    Format: 5 letters, 4 digits, 1 letter.
    4th character designates entity type:
    P - Individual, C - Company, H - HUF, A - AOP, T - Trust,
    B - BOI, L - Local Authority, J - Artificial Juridical Person, G - Government.
    """
    clean_pan = pan_str.strip().upper()
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", clean_pan):
        return False, "invalid_pan_pattern"
    entity_char = clean_pan[3]
    if entity_char not in "PCHFATBLJG":
        return False, f"invalid_pan_entity_type_{entity_char}"
    return True, None


# Recognized RBI Bank Codes (4-letter prefixes)
# Covers public sector, private sector, payment banks, small finance banks, and RRBs.
# Notice: Synthetic test codes like "DEMO" are strictly excluded from production code.
VALID_BANK_CODES = {
    "SBIN", "HDFC", "ICIC", "UTIB", "PUNB", "BARB", "KKBK", "CNRB", "UBIN",
    "IOBA", "BKID", "IDIB", "CBIN", "MAHB", "PSIB", "UCOB", "INDB", "YESB",
    "IDFB", "BAND", "CSBK", "DCBL", "DLXB", "FDRL", "JSFB", "KVBL", "RBLN",
    "SIBL", "TMBL", "AUBL", "ESFB", "ESMF", "FINO", "NESF", "SURY", "UCBA",
    "AIRP", "IPOS", "PYTM", "JIOP", "KANG", "APBL", "AGCX", "ALLA", "ANDB",
    "BDBL", "CORP", "DBSX", "DEUT", "HSBC", "IBKL", "JAKA", "ORBC", "SCBL",
    "SYNB", "VIJB", "VIJY",
}


def load_valid_bank_codes() -> set:
    """
    Load valid bank codes from VALID_BANK_CODES_FILE or IFSC_BANK_CODES_FILE if configured.
    - If configured: file MUST exist, be valid JSON, and contain a non-empty list of codes.
      Any error raises RuntimeError (fail-fast startup behavior).
    - If not configured (env var unset): explicitly falls back to bundled default VALID_BANK_CODES set.
    """
    codes_file = os.getenv("VALID_BANK_CODES_FILE") or os.getenv("IFSC_BANK_CODES_FILE")
    if not codes_file:
        return VALID_BANK_CODES

    if not os.path.exists(codes_file):
        raise RuntimeError(
            f"VALID_BANK_CODES_FILE is configured as '{codes_file}', but the file does not exist."
        )

    try:
        with open(codes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as ex:
        raise RuntimeError(
            f"VALID_BANK_CODES_FILE '{codes_file}' is invalid or unreadable: {ex}"
        )

    if not isinstance(data, (list, set)) or len(data) == 0:
        raise RuntimeError(
            f"VALID_BANK_CODES_FILE '{codes_file}' must contain a non-empty list of bank codes."
        )

    return set(data)


def get_valid_bank_codes() -> set:
    """Return valid bank codes via load_valid_bank_codes()."""
    return load_valid_bank_codes()


def validate_ifsc_code(ifsc_str: str, custom_bank_codes: Optional[set] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate Indian Financial System Code (IFSC):
    Format: 4 letters (bank code), 5th char is '0', 6 alphanumeric characters (branch code).
    If format is invalid: returns (False, 'invalid_ifsc_format').
    If format is valid but bank code is not in registry: returns (False, 'ifsc_needs_review')
    flagging it for manual review rather than hard-failing as an invalid document.
    """
    clean_ifsc = re.sub(r"\s+", "", ifsc_str).upper()
    if not re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", clean_ifsc):
        return False, "invalid_ifsc_format"
    bank_code = clean_ifsc[:4]
    codes_set = custom_bank_codes if custom_bank_codes is not None else get_valid_bank_codes()
    if bank_code not in codes_set:
        # Well-formatted IFSC, but bank prefix is outside the known list:
        # Failure mode is 'needs review', not hard invalid rejection.
        return False, "ifsc_needs_review"
    return True, None


def calculate_gstin_checksum_digit(gstin_14: str) -> str:
    """
    Calculate the 15th check digit of an Indian GSTIN using the official GSTN
    Luhn mod-36 / ISO 7064 Mod 36, 36 algorithm.
    """
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    clean = gstin_14.strip().upper()[:14]
    total = 0
    for i, c in enumerate(clean):
        val = chars.index(c)
        weight = 1 if (i % 2 == 0) else 2
        prod = val * weight
        quotient = prod // 36
        remainder = prod % 36
        total += (quotient + remainder)
    check_val = (36 - (total % 36)) % 36
    return chars[check_val]


def validate_gstin_format(gstin_str: str, verify_checksum: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate Indian Goods and Services Tax Identification Number (GSTIN):
    15 characters: 2 digits (State Code) + 10 chars (PAN) + 1 entity code + 'Z' + 1 checksum char.
    Validates structural pattern, embedded PAN validity, and optionally the 15th check digit
    using the official GSTN Luhn mod-36 algorithm.
    """
    clean_gstin = re.sub(r"\s+", "", gstin_str).upper()
    if not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", clean_gstin):
        return False, "invalid_gstin_format"
    pan_part = clean_gstin[2:12]
    valid_pan, pan_err = validate_pan_format(pan_part)
    if not valid_pan:
        return False, f"gstin_{pan_err}"
    if verify_checksum:
        expected_check = calculate_gstin_checksum_digit(clean_gstin[:14])
        if clean_gstin[14] != expected_check:
            return False, "invalid_gstin_checksum"
    return True, None


def validate_cin_format(cin_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Indian Corporate Identification Number (CIN):
    21 characters: [UL] + 5 digits (Industry) + 2 letters (State) + 4 digits (Year) + 3 letters (Company Type) + 6 digits (Reg No).
    """
    clean_cin = re.sub(r"\s+", "", cin_str).upper()
    if not re.match(r"^[UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}$", clean_cin):
        return False, "invalid_cin_format"
    return True, None


def validate_document_checksums(doc_type: str, extracted_fields: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Run domain-specific checksum/format validation based on doc_type.
    Returns (is_valid, reason_if_invalid).
    """
    if doc_type == "pan":
        pan = extracted_fields.get("pan_number")
        if pan:
            valid, err = validate_pan_format(pan)
            if not valid:
                return False, err

    elif doc_type == "aadhaar":
        raw_aadhaar = extracted_fields.get("raw_aadhaar") or extracted_fields.get("aadhaar_number")
        if raw_aadhaar:
            # If raw unmasked Aadhaar is available (e.g. from QR code or before masking)
            digits = re.sub(r"\D", "", str(raw_aadhaar))
            if len(digits) == 12:
                if not validate_verhoeff_checksum(digits):
                    return False, "invalid_aadhaar_checksum"

    elif doc_type in ("cancelled_cheque", "bank_statement", "bank_passbook"):
        ifsc = extracted_fields.get("ifsc")
        if ifsc:
            valid, err = validate_ifsc_code(ifsc)
            if not valid:
                return False, err

    elif doc_type == "gst_certificate":
        gstin = extracted_fields.get("gstin")
        if gstin:
            valid, err = validate_gstin_format(gstin)
            if not valid:
                return False, err

    elif doc_type == "certificate_of_incorporation":
        cin = extracted_fields.get("cin")
        if cin:
            valid, err = validate_cin_format(cin)
            if not valid:
                return False, err

    elif doc_type in ("form_16", "iec_certificate"):
        pan = extracted_fields.get("pan_number")
        if pan:
            valid, err = validate_pan_format(pan)
            if not valid:
                return False, err

    return True, None



# ==============================================================================
# 2. Document-Type Mismatch Detection
# ==============================================================================

DOC_SIGNATURES = {
    "pan": [
        r"INCOME TAX DEPARTMENT",
        r"PERMANENT ACCOUNT NUMBER",
        r"GOVT\. OF INDIA.*INCOME TAX|INCOME TAX.*GOVT\. OF INDIA",
        r"FATHER'?S NAME",
        r"[A-Z]{5}[0-9]{4}[A-Z]",
    ],
    "aadhaar": [
        r"UNIQUE IDENTIFICATION AUTHORITY OF INDIA|UIDAI",
        r"AADHAAR|AADHAR|माझे\s*आधार|मेरा\s*आधार",
        r"MERA AADHAAR|MERA AADHAR",
        r"GOVERNMENT OF INDIA.*AADHAAR|GOVT OF INDIA.*AADHAAR|भारतीय\s*विशिष्ट\s*(?:ओळख|पहचान)\s*प्राधिकरण",
        r"\bXXXX\s+XXXX\s+[0-9]{4}\b|\b[2-9][0-9]{3}\s+[0-9]{4}\s+[0-9]{4}\b",
    ],
    "cancelled_cheque": [
        r"CANCELLED",
        r"PAY\s+(?:AGAINST\s+CHEQUE|TO\s+ORDER|TO\s+[A-Z]|THE\s+SUM|BEARER)",
        r"A/C\s*(?:NO|NUM|NUMBER)[\.:]?\s*[0-9Xx]+",
        r"IFS\s*CODE|IFSC",
        r"[0-9]{6}\s+[0-9]{9}\s+[0-9]{6}",
        r"OR\s+BEARER",
    ],
    "passport": [
        r"PASSPORT",
        r"REPUBLIC OF INDIA.*PASSPORT|PASSPORT.*REPUBLIC OF INDIA",
        r"SURNAME.*GIVEN NAME|GIVEN NAME.*SURNAME",
        r"NATIONALITY\s*:\s*INDIAN",
        r"PASSPORT\s*(?:NO|NUMBER)",
    ],
    "voter_id": [
        r"ELECTION COMMISSION OF INDIA",
        r"ELECTOR PHOTO IDENTITY CARD",
        r"EPIC\s*(?:NO|NUMBER)",
        r"BHARAT NIRVACHAN AYOG",
    ],
    "driving_licence": [
        r"DRIVING LICEN[CS]E",
        r"LICEN[CS]E\s*(?:NO|NUMBER)",
        r"VEHICLE CLASS",
        r"TRANSPORT DEPARTMENT|MOTOR VEHICLES ACT|UNION OF INDIA.*DRIVING",
    ],
    "udyam": [
        r"UDYAM REGISTRATION|उद्यम\s*नोंदणी|उद्यम\s*पंजीकरण",
        r"UDYAM-[A-Z]{2}-[0-9]{2}-[0-9]+",
        r"ENTERPRISE NAME|NAME OF ENTERPRISE|उद्यमाचे\s*नाव|उद्यम\s*का\s*नाम",
        r"TYPE OF ENTERPRISE|ENTERPRISE TYPE|उद्यमाचा\s*प्रकार",
        r"MINISTRY OF MICRO.*SMALL AND MEDIUM|सूक्ष्म,\s*लघु\s*(?:व|आणि|एवं)\s*मध्यम\s*उद्योग",
    ],
    "fssai": [
        r"FOOD SAFETY AND STANDARDS AUTHORITY",
        r"FSSAI",
        r"LICENSE UNDER FSS ACT|REGISTRATION UNDER FSS ACT",
        r"KIND OF BUSINESS",
    ],
    "shop_establishment": [
        r"SHOP & ESTABLISHMENT|SHOPS & ESTABLISHMENTS|SHOPS AND ESTABLISHMENTS|दुकान\s*(?:आणि|व|एवं)\s*(?:आस्थापना|स्थापना)|दु\s*क\s*ाने\s*(?:आणि|व)\s*आ\s*(?:थापना|स्थापना)",
        r"SHOPS AND COMMERCIAL|आस्थापना\s*नोंदणी|स्थापना\s*पंजीकरण|नमु\s*न\s*ा\s*[\"'\u201c\u201d]?[फगFG][\"'\u201c\u201d]?|Form\s*[-–]\s*[\"'\u2018\u2019]?[फगFG][\"'\u2018\u2019]?",
        r"ESTABLISHMENT REGISTRATION|महाराष्ट्र\s*शासन|कामगार\s*आयुक्त|Registration\s*Certificate\s*/\s*Intimation|पावती\s*(?:क्रमांक|मांक)",
        r"NATURE OF BUSINESS|व्यवसायाचे\s*स्वरूप|Category\s*Of\s*Establishment",
    ],
    "bank_statement": [
        r"ACCOUNT STATEMENT",
        r"STATEMENT OF ACCOUNT",
        r"CLOSING BALANCE",
        r"TRANSACTION DETAILS|TRANSACTION DATE",
        r"WITHDRAWAL.*DEPOSIT.*BALANCE",
    ],
    "salary_slip": [
        r"SALARY SLIP|PAYSLIP|वेतन\s*पावती|वेतन\s*प्रमाणपत्र|मासिक\s*वेतन\s*पावती|वेतन\s*पर्ची",
        r"EARNINGS.*DEDUCTIONS|DEDUCTIONS.*EARNINGS|मिळकत.*कपात|कपात.*मिळकत|एकूण\s*मिळकत|एकूण\s*कपात|उपार्जन.*कटौती",
        r"NET SALARY|NET PAY|निव्वळ\s*वेतन|निव्वळ\s*देय|शुद्ध\s*वेतन|हाती\s*येणारे\s*वेतन",
        r"BASIC SALARY|BASIC PAY|मूळ\s*वेतन|मूल\s*वेतन",
        r"PAY PERIOD|PAY SLIP FOR THE MONTH|वेतन\s*महिना|माहे\s*:\s*[A-Za-z\u0900-\u097F]+|माह\s*:\s*[A-Za-z\u0900-\u097F]+",
    ],
    "utility_bill": [
        r"ELECTRICITY BILL|WATER BILL|GAS BILL|ENERGY BILL|BILL OF SUPPLY|POWER DISTRIBUTION|MSEDCL|MAHADISCOM|महावितरण|महािवतरण|वीज\s*देयक|विद्युत\s*देयक|पाणी\s*पट्टी|गॅस\s*बिल",
        r"CONSUMER\s*(?:NO|NUMBER|ID|PORTAL)|ग्राहक\s*(?:क्रमांक|क्र\.?|नंबर)|साहक\s*(?:क्रमांक|कमांक|कमक)|उपभोक्ता\s*(?:संख्या|क्रमांक)",
        r"BILL\s*AMOUNT|TOTAL AMOUNT DUE|देयक\s*रक्कम|एकूण\s*रक्कम|भरणा\s*रक्कम|देय\s*रक्कम|आतमतारीख",
        r"DUE DATE|देय\s*दिनांक|अंतिम\s*(?:तारीख|दिनांक)|या\s*तारखेपर्यंत\s*भरल्यास",
    ],
    "itr": [
        r"INDIAN INCOME TAX RETURN",
        r"ITR-V|ITR-1|ITR-2|ITR-4",
        r"ACKNOWLEDGEMENT NUMBER",
        r"ASSESSMENT YEAR",
    ],
    "gst_certificate": [
        r"GOODS AND SERVICES TAX",
        r"FORM GST REG-06|FORM GST REG-02|FORM GST REG",
        r"REGISTRATION CERTIFICATE.*GST|GST.*REGISTRATION CERTIFICATE",
        r"GOVERNMENT OF INDIA.*GOODS AND SERVICES|GOODS AND SERVICES.*GOVERNMENT OF INDIA",
        r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b",
        r"CONSTITUTION OF BUSINESS",
    ],
    "certificate_of_incorporation": [
        r"CERTIFICATE OF INCORPORATION",
        r"CORPORATE IDENTITY NUMBER|CORPORATE IDENTIFICATION NUMBER",
        r"REGISTRAR OF COMPANIES",
        r"COMPANIES ACT,\s*(?:1956|2013)|THE COMPANIES ACT",
        r"MINISTRY OF CORPORATE AFFAIRS",
    ],
    "partnership_deed": [
        r"PARTNERSHIP DEED|DEED OF PARTNERSHIP",
        r"PARTNERS OF THE FIRM|PARTNERS OF FIRST PART|PARTY OF THE FIRST PART",
        r"PROFIT SHARING RATIO|PROFIT AND LOSS.*SHARING",
        r"INDIAN PARTNERSHIP ACT",
    ],
    "rent_agreement": [
        r"RENT AGREEMENT|LEASE AGREEMENT|TENANCY AGREEMENT|भाडेकरार|भाडे\s*करार|परवाना\s*करार",
        r"LESSOR AND LESSEE|LANDLORD AND TENANT|परवाना\s*देणारा|घरमालक",
        r"MONTHLY RENT|PREMISES ON LEASE|मासिक\s*भाडे",
        r"SECURITY DEPOSIT.*RENT|RENT.*SECURITY DEPOSIT|REFUNDABLE SECURITY DEPOSIT|डिपॉझिट",
    ],
    "form_16": [
        r"FORM NO\.?\s*16\b",
        r"CERTIFICATE UNDER SECTION 203",
        r"TAX DEDUCTED AT SOURCE",
        r"CENTRAL BOARD OF DIRECT TAXES",
    ],
    "bank_passbook": [
        r"PASS\s*BOOK|PASSBOOK|पास\s*बुक|पासबुक|बचत\s*खाते\s*पासबुक",
        r"ACCOUNT HOLDER NAME|NAME OF ACCOUNT HOLDER|खातेदाराचे\s*नाव|खाताधारक\s*का\s*नाम",
        r"ACCOUNT NUMBER.*IFSC|IFSC.*ACCOUNT NUMBER|खाते\s*(?:क्रमांक|क्र\.?).*(?:आयएफएससी|शाखा)|शाखा.*खाते\s*(?:क्रमांक|क्र\.?)",
        r"CUSTOMER ID|CIF NO|CUSTOMER NO|ग्राहक\s*(?:क्रमांक|क्र\.?|आयडी)|सीआयएफ",
    ],
    "property_tax_receipt": [
        r"PROPERTY TAX RECEIPT|PROPERTY TAX PAYMENT|मालमत्ता\s*कर|घरपट्टी",
        r"MUNICIPAL CORPORATION|NAGAR NIGAM|MUNICIPAL COUNCIL|महानगरपालिका|नगरपरिषद|नगर\s*पंचायत",
        r"PROPERTY TAX PAID|PROPERTY TAX ASSESSMENT|मालमत्ता\s*कर\s*पावती|कर\s*पावती",
        r"PROPERTY ID|ASSESSMENT NO|INDEX NO|TAX BILL NO|मालमत्ता\s*(?:क्रमांक|क्र\.?)|पावती\s*(?:क्रमांक|क्र\.?)",
    ],
    "iec_certificate": [
        r"IMPORT EXPORT CODE|IMPORTER EXPORTER CODE",
        r"DIRECTORATE GENERAL OF FOREIGN TRADE|DGFT",
        r"GOVERNMENT OF INDIA.*MINISTRY OF COMMERCE|MINISTRY OF COMMERCE.*DIRECTORATE GENERAL",
        r"IEC CERTIFICATE|IEC ISSUANCE",
    ],
    "income_certificate": [
        r"INCOME CERTIFICATE|CERTIFICATE OF INCOME|उत्पन्नाचा\s*दाखला|उत्पन्नाचे\s*प्रमाणपत्र|उलपञाचे\s*पमाणपऋ|उतनाचा\s*दाखला|वार्षिक\s*उत्पन्नाचा\s*दाखला|उत्पन्न\s*दाखला|उत्पन्न\s*प्रमाणपत्र",
        r"TAHSILDAR|REVENUE DEPARTMENT|तहसीलदार|तहसील\s*कार्यालय|उपविभागीय\s*अधिकारी|प्रांत\s*अधिकारी|नायब\s*तहसीलदार",
        r"ANNUAL INCOME|TOTAL INCOME.*CERTIFIED|मिळालेले\s*वार्षिक\s*उत्पन्न|वार्षिक\s*उत्पन्न|एकूण\s*वार्षिक\s*उत्पन्न|उत्पन्न\s*खालीलप्रमाणे|जलनखालीलपमाणे",
        r"THIS IS TO CERTIFY THAT|CERTIFIED THAT|प्रमाणित\s*करण्यात\s*येते\s*की|अमाणतकरणयात|अमािणतकरण|सदरचा\s*दाखला",
    ],
}



def detect_document_type(ocr_text: str, min_score: int = 2, min_margin: int = 1) -> Optional[str]:
    """
    Identify the likely document type based on signature keywords.
    Requires top score >= min_score and a margin >= min_margin over the second-highest score
    to prevent single ambiguous keyword hits from falsely identifying a document type.
    """
    ocr_upper = ocr_text.upper()
    scores = {}
    for doc_type, patterns in DOC_SIGNATURES.items():
        score = sum(1 for p in patterns if re.search(p, ocr_upper))
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return None

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type, top_score = sorted_scores[0]
    if top_score < min_score:
        return None

    if len(sorted_scores) > 1:
        second_type, second_score = sorted_scores[1]
        if (top_score - second_score) < min_margin:
            return None

    return top_type


def check_doc_type_mismatch(
    requested_type: str, ocr_text: str, min_score: int = 2, min_margin: int = 1
) -> Tuple[bool, Optional[str]]:
    """
    Check if the document uploaded completely contradicts the requested doc_type.
    Returns (is_mismatch, detected_type).
    A mismatch is flagged if:
    1. A distinct document type is detected with top_score >= min_score and clear margin.
    2. The detected type != requested_type.
    3. The requested document type signatures do not meet min_score (i.e. req_score < min_score),
       confirming the document lacks genuine features of requested_type.
    """
    detected = detect_document_type(ocr_text, min_score=min_score, min_margin=min_margin)
    if detected and detected != requested_type:
        requested_patterns = DOC_SIGNATURES.get(requested_type, [])
        ocr_upper = ocr_text.upper()
        req_score = sum(1 for p in requested_patterns if re.search(p, ocr_upper))
        if req_score < min_score:
            return True, detected
    return False, None


# ==============================================================================
# 3. Cross-Check Capability
# ==============================================================================

def normalize_date_str(date_str: str) -> Optional[str]:
    """Attempt parsing multiple date formats to YYYY-MM-DD."""
    clean = re.sub(r"[\s\.,]+", "/", date_str.strip())
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%y", "%d-%m-%y", "%d %b %Y", "%d %B %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(clean, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate normalized token-sorted string similarity."""
    t1 = " ".join(sorted(re.sub(r"[^\w\s]", "", s1.lower()).split()))
    t2 = " ".join(sorted(re.sub(r"[^\w\s]", "", s2.lower()).split()))
    return SequenceMatcher(None, t1, t2).ratio()


def perform_cross_check(extracted_fields: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare extracted fields against expected applicant data.
    Does not fail if fields are missing; returns score and matched status per field.
    """
    results = {}
    for key, expected_val in expected.items():
        if expected_val is None:
            continue
        expected_str = str(expected_val).strip()
        extracted_val = extracted_fields.get(key)
        if extracted_val is None:
            # Check aliases (e.g. full_name -> name, employee_name -> name)
            if key in ("name", "full_name", "employee_name", "applicant_name"):
                extracted_val = (
                    extracted_fields.get("name")
                    or extracted_fields.get("employee_name")
                    or extracted_fields.get("raw_employee_name")
                    or extracted_fields.get("account_holder")
                    or extracted_fields.get("account_holder_name")
                    or extracted_fields.get("given_name")
                    or extracted_fields.get("lessee_name")
                    or extracted_fields.get("lessor_name")
                    or extracted_fields.get("owner_name")
                    or extracted_fields.get("raw_partner_names")
                    or extracted_fields.get("partner_names")
                    or extracted_fields.get("partner_names_masked")
                    or extracted_fields.get("legal_name")
                    or extracted_fields.get("company_name")
                    or extracted_fields.get("firm_name")
                    or extracted_fields.get("employer_name")
                    or extracted_fields.get("entity_name")
                )
            elif key in ("partner_name", "partner_names", "partners"):
                extracted_val = (
                    extracted_fields.get("raw_partner_names")
                    or extracted_fields.get("partner_names")
                    or extracted_fields.get("partner_names_masked")
                    or extracted_fields.get("partner_name")
                )
            elif key in ("dob", "date_of_birth"):
                extracted_val = extracted_fields.get("dob") or extracted_fields.get("date_of_birth")
            elif key in ("identifier", "id_number"):
                extracted_val = (
                    extracted_fields.get("pan_number")
                    or extracted_fields.get("raw_aadhaar")
                    or extracted_fields.get("aadhaar_number")
                    or extracted_fields.get("passport_number")
                    or extracted_fields.get("epic_number")
                    or extracted_fields.get("licence_number")
                    or extracted_fields.get("udyam_registration_number")
                    or extracted_fields.get("fssai_licence_number")
                    or extracted_fields.get("gstin")
                    or extracted_fields.get("cin")
                    or extracted_fields.get("iec_number")
                    or extracted_fields.get("property_id")
                    or extracted_fields.get("tan_number")
                )

        if extracted_val is None:
            results[key] = {
                "expected": expected_str,
                "extracted": None,
                "matched": False,
                "score": 0.0,
                "reason": "field_not_found_in_document",
            }
            continue

        # Handle list-valued expectations or extracted values (e.g. partner_names)
        if isinstance(expected_val, (list, tuple)) or isinstance(extracted_val, (list, tuple)):
            exp_items = [str(x).strip() for x in expected_val] if isinstance(expected_val, (list, tuple)) else [expected_str]
            ext_items = [str(x).strip() for x in extracted_val] if isinstance(extracted_val, (list, tuple)) else [str(extracted_val).strip()]

            if not ext_items or not exp_items:
                results[key] = {
                    "expected": expected_val if isinstance(expected_val, (list, tuple)) else expected_str,
                    "extracted": extracted_val if isinstance(extracted_val, (list, tuple)) else str(extracted_val).strip(),
                    "matched": False,
                    "score": 0.0,
                    "reason": "empty_list_in_field",
                }
                continue

            # For each expected item, find best matching extracted item
            matched_count = 0
            item_scores = []
            best_extracted_matches = []
            for exp_item in exp_items:
                best_sim = 0.0
                best_ext = None
                for ext_item in ext_items:
                    sim = calculate_similarity(ext_item, exp_item)
                    if sim > best_sim:
                        best_sim = sim
                        best_ext = ext_item
                item_scores.append(best_sim)
                if best_sim >= 0.82:
                    matched_count += 1
                    best_extracted_matches.append(best_ext)

            avg_score = sum(item_scores) / len(item_scores) if item_scores else 0.0
            is_matched = (matched_count == len(exp_items))
            results[key] = {
                "expected": expected_val if isinstance(expected_val, (list, tuple)) else expected_str,
                "extracted": best_extracted_matches[0] if (len(exp_items) == 1 and best_extracted_matches) else extracted_val,
                "matched": is_matched,
                "score": round(avg_score, 4),
            }
            continue

        extracted_str = str(extracted_val).strip()

        # Date comparison
        if "dob" in key or "date" in key:
            norm_exp = normalize_date_str(expected_str)
            norm_ext = normalize_date_str(extracted_str)
            if norm_exp and norm_ext:
                matched = (norm_exp == norm_ext)
                results[key] = {
                    "expected": norm_exp,
                    "extracted": norm_ext,
                    "matched": matched,
                    "score": 1.0 if matched else 0.0,
                }
                continue

        # Text / Identifier comparison
        score = calculate_similarity(extracted_str, expected_str)
        matched = (score >= 0.82)
        results[key] = {
            "expected": expected_str,
            "extracted": extracted_str,
            "matched": matched,
            "score": round(score, 4),
        }

    return results


def determine_document_status(
    checksum_valid: bool,
    average_confidence: float,
    confidence_threshold: float = 0.70,
    vault_mode: bool = True,
) -> str:
    """
    Unified status determination function used by /api/upload, execute_ocr_pipeline,
    and migration/reprocessing scripts to prevent logic drift.

    - If checksum_valid is False: returns 'warning' (vault_mode=True) or 'low_confidence' (vault_mode=False)
    - If checksum_valid is True:
        - returns 'completed' (vault_mode=True) or 'success' (vault_mode=False) if average_confidence >= confidence_threshold
        - returns 'low_confidence' if average_confidence < confidence_threshold
    """
    if not checksum_valid:
        return "warning" if vault_mode else "low_confidence"
    if average_confidence >= confidence_threshold:
        return "completed" if vault_mode else "success"
    return "low_confidence"
