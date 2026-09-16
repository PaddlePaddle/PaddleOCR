"""
extractors.py
Domain-specific extractors for 13 document types in Company-Server OCR service:
- PAN, Aadhaar, Cancelled Cheque, Udyam, FSSAI, Shop & Establishment,
  Bank Statement, Salary Slip, Utility Bill, Passport, Voter ID, Driving Licence, ITR.
- Enforces strict PII minimisation allowlists at source.
- Tracks per-field confidence from PaddleOCR lines.
- Handles multi-page table merges for bank statements.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from ocr_engine import OCRDocumentResult, OCRLine


# ==============================================================================
# Helper Masking & Normalization Functions
# ==============================================================================

def mask_account_number(acc: Optional[str]) -> Optional[str]:
    """Mask bank account number leaving only the last 4 digits visible."""
    if not acc:
        return None
    clean = re.sub(r"\D", "", acc)
    if len(clean) >= 4:
        prefix_len = max(6, len(clean) - 4)
        return "X" * prefix_len + clean[-4:]
    return "XXXXXX" + clean


def mask_aadhaar(uid: Optional[str]) -> Optional[str]:
    """Mask Aadhaar number to XXXXXXXX1234."""
    if not uid:
        return None
    clean = re.sub(r"\D", "", uid)
    if len(clean) >= 4:
        return "XXXXXXXX" + clean[-4:]
    return "XXXXXXXX" + clean


def mask_person_name(name: Optional[str]) -> Optional[str]:
    """Partially mask person name if requested (e.g. for salary slip)."""
    if not name:
        return None
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0][0] + "*" * max(1, len(parts[0]) - 1)
    return parts[0] + " " + parts[-1][0] + "*" * max(1, len(parts[-1]) - 1)


def mask_address(addr: Optional[str]) -> Optional[str]:
    """Partially mask residential or property address to protect PII.
    Redacts specific unit/flat/door/building/street numbers while preserving
    locality, city, state, and postal code for downstream verification.
    """
    if not addr:
        return None
    clean = addr.strip()
    clean = re.sub(r"[ \t]+", " ", clean)

    # If comma-separated address
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    if len(parts) >= 2:
        premise_kw = re.compile(
            r"^(?:flat|plot|house|door|shop|unit|room|bldg|building|wing|tower|block|floor|no\.?|#|\d+)",
            re.IGNORECASE,
        )
        cut_idx = 0
        while cut_idx < len(parts) - 1:
            if premise_kw.search(parts[cut_idx]) or cut_idx == 0:
                cut_idx += 1
                if cut_idx < len(parts) - 1 and re.search(r"^(?:bldg|building|wing|tower|block|floor)\b", parts[cut_idx], re.I):
                    cut_idx += 1
                break
            else:
                break
        preserved = ", ".join(parts[cut_idx:])
        return f"XXXX, {preserved}" if preserved else f"XXXX {clean[-10:]}"

    # Non-comma separated: replace leading unit/door/number
    masked = re.sub(
        r"^(?:(?:flat|plot|house|door|shop|unit|room|bldg|building|no\.?|#)\s*)?[0-9A-Za-z\-\/]+(?:\s+(?:road|marg|street|lane|bldg|building|floor))?\s*",
        "XXXX ",
        clean,
        flags=re.IGNORECASE,
    )
    if masked == clean:
        masked = "XXXX " + clean[10:].strip() if len(clean) > 15 else "XXXX"
    return masked.strip()


def normalize_devanagari_numbers(s: Optional[str]) -> Optional[str]:
    """Convert Devanagari numerals (०-९) to standard ASCII digits (0-9)."""
    if not s:
        return s
    devanagari_digits = "०१२३४५६७८९"
    trans = str.maketrans({char: str(i) for i, char in enumerate(devanagari_digits)})
    return s.translate(trans)


def find_line_confidence(pattern: str, lines: List[OCRLine], default_conf: float = 0.95) -> float:
    """Find the confidence score of the OCR line matching a specific regex pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    for line in lines:
        if regex.search(line.text):
            return round(line.confidence, 4)
    return default_conf


# Known field labels that might be captured as residual bleed on a subsequent line
RESIDUAL_LABEL_LINES = {
    "purpose", "sno", "sno.", "s.no", "s.no.", "sr no", "sr. no", "sr.no",
    "classification year", "enterprise type", "type of enterprise", "major activity",
    "social category", "date of incorporation", "date of commencement",
    "permanent account number", "pan", "date of birth", "dob", "father's name", "father name",
    "mother's name", "husband's name", "address", "status", "form number",
    "acknowledgement number", "assessment year", "ay", "total income", "taxes paid",
    "gender", "sex", "nationality", "expiry date", "date of expiry", "issue date", "date of issue",
    "valid till", "valid from", "kind of business", "nature of business", "registration number",
    "reg no", "licence number", "license number", "fssai licence number",
    "ifsc", "ifsc code", "branch", "bank name", "account number", "account holder",
    "consumer number", "bill date", "due date", "bill amount", "total amount",
    "current year business loss", "book profit", "net tax payable",
    "gstin", "trade name", "legal name", "constitution of business", "date of registration",
    "additional trade names", "additional trade name", "trade names", "trade name if any", "trade name, if any", "if any",
    "name of deductor", "tan of deductor", "name of employer", "name of buyer", "name of collector", "name of deductee", "name of seller",
    "corporate identity number", "cin", "registrar of companies", "company name",
    "partnership deed", "firm name", "partner", "partner name", "profit sharing ratio",
    "lessor", "lessee", "landlord", "tenant", "monthly rent", "security deposit", "lease period",
    "gross salary", "tax deducted", "tan", "tan number", "employer name", "employee name",
    "passbook", "account holder name", "cif no", "customer id",
    "property id", "property tax", "tax paid", "assessment no", "tax amount",
    "iec", "iec number", "importer exporter code", "dgft", "entity name",
    # Marathi & Hindi field labels
    "नोंदणी क्रमांक", "नोंदणी क्र", "आस्थापनेचे नाव", "दुकानाचे नाव", "मालकाचे नाव",
    "मालकाचा तपशील", "व्यवसायाचे स्वरूप", "उद्यमाचे नाव", "उद्यम नोंदणी क्रमांक",
    "उद्यमाचा प्रकार", "मुख्य कार्यकलाप", "मालमत्ता क्रमांक", "मालमत्ता कर",
    "कर आकारणी", "भरलेली रक्कम", "पावती क्रमांक", "परवाना देणारा", "परवाना घेणारा",
    "भाडेकरार", "मासिक भाडे", "डिपॉझिट", "जन्मतारीख", "जन्म तारीख", "लिंग", "पत्ता", "आधार क्रमांक",
    # Salary Slip labels (Marathi & Hindi)
    "कार्यालयाचे नाव", "संस्थेचे नाव", "विभागाचे नाव", "कंपनीचे नाव", "नियोक्त्याचे नाव",
    "कर्मचाऱ्याचे नाव", "कर्मचारी नाव", "कर्मचारी का नाम", "अधिकाऱ्याचे नाव", "सेवकाचे नाव",
    "निव्वळ वेतन", "निव्वळ देय रक्कम", "निव्वळ देय", "हाती येणारे वेतन", "शुद्ध वेतन", "कुल शुद्ध देय",
    "वेतन महिना", "माहे", "कालावधी",
    # Bank Passbook labels (Marathi & Hindi)
    "बँकेचे नाव", "बैंक का नाम", "शाखेचे नाव", "शाखा कोड", "शाखा",
    "खाते क्रमांक", "खाते क्र", "खाते नं", "बचत खाते क्रमांक", "खाता संख्या", "खाता क्रमांक",
    "खातेदाराचे नाव", "खातेदार नाव", "ग्राहकाचे नाव", "खाताधारक का नाम", "खाताधारी का नाम",
    "पासबुक", "बचत खाते पासबुक",

    # Income Certificate labels (English & Marathi)
    "income certificate", "certificate number", "applicant name", "annual income", "financial year", "issuing authority",
    "उत्पन्नाचे प्रमाणपत्र", "उत्पन्नाचा दाखला", "दाखला क्रमांक", "प्रमाणपत्र क्रमांक", "वार्षिक उत्पन्न", "अर्जदाराचे नाव",
}



import logging
logger = logging.getLogger(__name__)


def clean_field_value(val: Any, field_name: Optional[str] = None, doc_type: Optional[str] = None) -> Any:
    """
    Sanitize extracted field value to prevent boundary bleeds into subsequent field labels:
    1. Cut off at first double newline.
    2. Discard subsequent lines if they match known section/field headers.
    3. Strip trailing whitespace and colons/dashes.
    4. Structured logging (NO PII) when trimming occurs, with warning on suspicious leftovers.
    """
    if not isinstance(val, str):
        if isinstance(val, list):
            return [clean_field_value(item, field_name=field_name, doc_type=doc_type) for item in val]
        return val
    # 1. Immediately cut off at double newline (multi-field gap in OCR/PDF text)
    first_block = re.split(r"\r?\n\s*\r?\n", val.strip())[0].strip()
    if not first_block:
        if val.strip():
            logger.info(
                "clean_field_value trimmed residual bleed: field=%s doc_type=%s",
                field_name or "unknown",
                doc_type or "unknown",
            )
        return ""

    # 2. Inspect individual lines
    lines = [l.strip() for l in first_block.splitlines() if l.strip()]
    if not lines:
        return ""

    cleaned_lines = [lines[0]]
    for line in lines[1:]:
        norm = re.sub(r"[:\-_.]", " ", line).strip().lower()
        norm_words = norm.split()

        is_label = False
        if norm in RESIDUAL_LABEL_LINES:
            is_label = True
        else:
            for lbl in RESIDUAL_LABEL_LINES:
                if norm == lbl or norm.startswith(lbl + " ") or (len(norm_words) <= 3 and lbl in norm):
                    is_label = True
                    break
        if is_label:
            break
        cleaned_lines.append(line)

    res = " ".join(cleaned_lines).strip()
    # Strip any trailing colons, hyphens, or commas
    res = re.sub(r"[\s,:;\-]+$", "", res).strip()

    # Log when clean_field_value actually trims something (no PII: field name and doc_type only)
    if res != val:
        logger.info(
            "clean_field_value trimmed residual bleed: field=%s doc_type=%s",
            field_name or "unknown",
            doc_type or "unknown",
        )

    # Suspicious leftover check (contains newline or unusually long > 80 chars)
    if "\n" in res or len(res) > 80:
        logger.warning(
            "clean_field_value output suspicious (length=%d, has_newline=%s): field=%s doc_type=%s",
            len(res),
            "\n" in res,
            field_name or "unknown",
            doc_type or "unknown",
        )

    return res


# ==============================================================================
# 1. PAN Extractor
# ==============================================================================

def extract_pan(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # PAN Number pattern: 5 letters, 4 digits, 1 letter
    pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
    if pan_match:
        fields["pan_number"] = pan_match.group(1)
        confidences["pan_number"] = find_line_confidence(pan_match.group(1), all_lines)

    # DOB pattern: DD/MM/YYYY
    dob_match = re.search(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", text)
    if dob_match:
        fields["dob"] = dob_match.group(1).replace("-", "/").replace(".", "/")
        confidences["dob"] = find_line_confidence(r"\b\d{2}[/\-\.]\d{2}[/\-\.]\d{4}\b", all_lines)

    # Name extraction
    name_match = re.search(
        r"(?:Name|NAME)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Father|DOB|Date|Permanent|PAN|Purpose)\b))",
        text,
        re.IGNORECASE,
    )
    if name_match:
        cand = clean_field_value(name_match.group(1))
        if not re.search(r"^(?:Father|DOB|Date|Permanent|PAN|Purpose|Demo Value|Field)\b", cand, re.I):
            fields["name"] = cand
            confidences["name"] = find_line_confidence(fields["name"], all_lines)

    # Father's Name
    father_match = re.search(
        r"(?:Father['’]?s?\s*Name)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Purpose|DOB|Date|Name|Permanent|PAN)\b))",
        text,
        re.IGNORECASE,
    )
    if father_match:
        cand = clean_field_value(father_match.group(1))
        if not re.search(r"^(?:Purpose|DOB|Date|Name|Permanent|PAN)\b", cand, re.I):
            fields["father_name"] = cand
            confidences["father_name"] = find_line_confidence(fields["father_name"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 2. Aadhaar Extractor (with Masking & PII Strip)
# ==============================================================================

def extract_aadhaar(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Check for Aadhaar number pattern: 4 digits + 4 digits + 4 digits, or masked XXXX XXXX 1234
    norm_text = normalize_devanagari_numbers(text) or text
    uid_match = re.search(r"\b(\d{4}\s\d{4}\s\d{4}|\d{12})\b", norm_text)
    masked_match = re.search(r"\b([X\d]{4}\s[X\d]{4}\s\d{4})\b", norm_text, re.IGNORECASE)

    raw_uid = None
    if uid_match:
        raw_uid = uid_match.group(1).replace(" ", "")
        fields["raw_aadhaar"] = raw_uid  # Used internally by verifier, stripped before output
        masked_val = mask_aadhaar(raw_uid)
        fields["aadhaar_number"] = masked_val
        fields["aadhaar_number_masked"] = masked_val
        conf = find_line_confidence(r"\d{4}\s\d{4}\s\d{4}|\d{12}", all_lines)
        confidences["aadhaar_number"] = conf
        confidences["aadhaar_number_masked"] = conf
    elif masked_match:
        masked_val = normalize_devanagari_numbers(masked_match.group(1).upper())
        fields["aadhaar_number"] = masked_val
        fields["aadhaar_number_masked"] = masked_val
        conf = find_line_confidence(r"[X\d]{4}\s[X\d]{4}\s\d{4}", all_lines)
        confidences["aadhaar_number"] = conf
        confidences["aadhaar_number_masked"] = conf

    # Name (English or Devanagari)
    name_match = re.search(
        r"(?:Name|नाव|नाम)[\s:]*(?:\r?\n)?[\s:]*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:Aadhaar|DOB|Date|Gender|Address|Father|Mother|Husband|Year|YOB|आधार|जन्म|लिंग|पत्ता)\b))",
        text,
        re.IGNORECASE,
    )
    if name_match:
        cand = clean_field_value(name_match.group(1))
        if not re.search(r"^(?:Aadhaar|DOB|Gender|Address|Date|Year|YOB|आधार|जन्म|लिंग|पत्ता)\b", cand, re.I):
            fields["name"] = cand
            confidences["name"] = find_line_confidence(fields["name"], all_lines)

    # DOB / Year of Birth
    dob_match = re.search(r"(?:DOB|Date of Birth|जन्मतारीख|जन्म\s*तारीख|जन्म\s*तिथि)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if dob_match:
        fields["dob"] = dob_match.group(1).replace("-", "/").replace(".", "/")
        confidences["dob"] = find_line_confidence(dob_match.group(1), all_lines)
    else:
        yob_match = re.search(r"(?:Year of Birth|YOB|जन्म\s*वर्ष)[\s:]+(\d{4})", text, re.IGNORECASE)
        if yob_match:
            fields["dob"] = yob_match.group(1)
            confidences["dob"] = find_line_confidence(yob_match.group(1), all_lines)

    # Gender
    gender_match = re.search(r"\b(MALE|FEMALE|TRANSGENDER)\b|(पुरुष|स्त्री|महिला|तृतीयपंथी)", text, re.IGNORECASE)
    if gender_match:
        val = gender_match.group(1) or gender_match.group(2)
        dev_gen_map = {"पुरुष": "Male", "स्त्री": "Female", "महिला": "Female", "तृतीयपंथी": "Transgender"}
        fields["gender"] = dev_gen_map.get(val, val.capitalize())
        confidences["gender"] = find_line_confidence(val, all_lines)

    # NOTE: Address is intentionally stripped from response to prevent PII leakage.
    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 3. Cancelled Cheque Extractor
# ==============================================================================

def extract_cancelled_cheque(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Marking
    if re.search(r"\bCANCELLED\b", text, re.IGNORECASE):
        fields["marking"] = "CANCELLED"
        confidences["marking"] = 1.0

    # Account Number
    acc_match = re.search(r"(?:A/C\s*No\.?|Account\s*Number)[\s:]*([X\d]{6,18})", text, re.IGNORECASE)
    if acc_match:
        raw_acc = acc_match.group(1)
        fields["account_number_masked"] = mask_account_number(raw_acc)
        confidences["account_number_masked"] = find_line_confidence(raw_acc, all_lines)

    # IFSC
    ifsc_match = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", text)
    if ifsc_match:
        fields["ifsc"] = ifsc_match.group(1)
        confidences["ifsc"] = find_line_confidence(ifsc_match.group(1), all_lines)

    # Bank Name & Branch
    bank_match = re.search(r"(?:Bank\s*Name|Bank)[\s:]*([A-Za-z][A-Za-z \t.&'-]+?BANK)\b", text, re.IGNORECASE)
    if bank_match:
        fields["bank_name"] = clean_field_value(bank_match.group(1))
    branch_match = re.search(
        r"(?:Branch)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t,.'-]{1,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:IFSC|IFS|A/C|Account|Cheque|Chq|Payee)\b))",
        text,
        re.IGNORECASE,
    )
    if branch_match:
        fields["branch"] = clean_field_value(branch_match.group(1))

    # Cheque Number
    chq_match = re.search(r"(?:Cheque\s*Number|Chq\s*No)[\s:]*(\d{6})", text, re.IGNORECASE)
    if chq_match:
        fields["cheque_number"] = chq_match.group(1)
        confidences["cheque_number"] = find_line_confidence(chq_match.group(1), all_lines)

    # Account Holder
    holder_match = re.search(
        r"(?:Account\s*Holder|Payee|Name)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Branch|IFSC|IFS|A/C|Account|Cheque|Chq|CANCELLED)\b))",
        text,
        re.IGNORECASE,
    )
    if holder_match:
        cand = clean_field_value(holder_match.group(1))
        if not re.search(r"^(?:Branch|IFSC|A/C|Account|Cheque|CANCELLED)\b", cand, re.I):
            fields["account_holder"] = cand
            confidences["account_holder"] = find_line_confidence(fields["account_holder"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 4. Udyam Registration Extractor
# ==============================================================================

def extract_udyam(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Udyam Registration Number: UDYAM-XX-00-0000000
    urn_match = re.search(r"\b(UDYAM-[A-Z]{2}-\d{2}-\d{7})\b", text, re.IGNORECASE)
    if urn_match:
        fields["udyam_registration_number"] = urn_match.group(1).upper()
        confidences["udyam_registration_number"] = find_line_confidence(urn_match.group(1), all_lines)

    # Enterprise Name
    name_match = re.search(
        r"(?:उद्यमाचे\s*नाव|उद्यम\s*का\s*नाम|Enterprise\s*Name|Name of Enterprise)[\s:]*([^\r\n:]{1,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:SNo|S\.No|Type of Enterprise|उद्यमाचा\s*प्रकार|Classification|Major Activity|मुख्य\s*कार्यकलाप|Social Category|Date|Official Address)\b))",
        text,
        re.IGNORECASE,
    )
    if name_match:
        fields["enterprise_name"] = clean_field_value(name_match.group(1))
        confidences["enterprise_name"] = find_line_confidence(fields["enterprise_name"], all_lines)

    # Type of Enterprise (Micro / Small / Medium / सूक्ष्म / लघु / मध्यम)
    type_words = r"(Micro|Small|Medium|सूक्ष्म|लघु|मध्यम)"
    type_map = {
        "सूक्ष्म": "Micro",
        "लघु": "Small",
        "मध्यम": "Medium",
        "micro": "Micro",
        "small": "Small",
        "medium": "Medium",
    }
    # Anchor to Enterprise Type / Type of Enterprise label to prevent false match against Ministry header
    type_line = re.search(
        r"(?:Type\s*of\s*Enterprise|Enterprise\s*Type|उद्यमाचा\s*प्रकार|उद्यम\s*का\s*प्रकार)[\s:]*([^\r\n:]{1,30})",
        text,
        re.IGNORECASE,
    )
    if type_line:
        m_cand = re.search(type_words, type_line.group(1), re.IGNORECASE)
        if m_cand:
            val = m_cand.group(1).lower()
            fields["enterprise_type"] = type_map.get(val, val.capitalize())
            confidences["enterprise_type"] = 0.98

    if "enterprise_type" not in fields:
        type_match_win = re.search(
            r"(?:Type\s*of\s*Enterprise|Enterprise\s*Type|उद्यमाचा\s*प्रकार|उद्यम\s*का\s*प्रकार)[\s\S]{0,60}?(?:\b|(?<=[\s:]))" + type_words + r"(?=\b|[\s\r\n:]|$)",
            text,
            re.IGNORECASE,
        )
        if not type_match_win:
            type_match_win = re.search(
                r"(?:\b|(?<=[\s:]))" + type_words + r"(?=\b|[\s\r\n:]|$)[\s\S]{0,50}?(?:Type\s*of\s*Enterprise|Enterprise\s*Type|उद्यमाचा\s*प्रकार)",
                text,
                re.IGNORECASE,
            )
        if not type_match_win:
            type_match_win = re.search(r"(?:\b|(?<=[\s:]))" + type_words + r"(?=\b|[\s\r\n:]|$)", text, re.IGNORECASE)
        if type_match_win:
            val = type_match_win.group(1).lower()
            fields["enterprise_type"] = type_map.get(val, val.capitalize())
            confidences["enterprise_type"] = 0.98

    # Major Activity (Trading / Services / Manufacturing / व्यापार / ट्रेडिंग / सेवाएं / सेवा / उत्पादन / विनिर्माण)
    # Anchor strictly to MAJOR ACTIVITY banner/label to avoid bleeding from NIC Classification table's "Activity" column
    act_words = r"(Trading|Services|Manufacturing|व्यापार|ट्रेडिंग|सेवाएं|सेवा|उत्पादन|विनिर्माण)"
    act_map = {
        "trading": "Trading",
        "services": "Services",
        "manufacturing": "Manufacturing",
        "व्यापार": "Trading",
        "ट्रेडिंग": "Trading",
        "सेवा": "Services",
        "सेवाएं": "Services",
        "उत्पादन": "Manufacturing",
        "विनिर्माण": "Manufacturing",
    }
    # 1. Direct label line: MAJOR ACTIVITY: TRADING or मुख्य कार्यकलाप: व्यापार
    act_line = re.search(
        r"(?:MAJOR\s*ACTIVITY|मुख्य\s*(?:कार्यकलाप|गतिविधि|कामकाज))[\s:]*([^\r\n:]{1,40})",
        text,
        re.IGNORECASE,
    )
    if act_line:
        m_cand = re.search(act_words, act_line.group(1), re.IGNORECASE)
        if m_cand:
            val = m_cand.group(1).lower()
            fields["major_activity"] = act_map.get(val, val.capitalize())
            confidences["major_activity"] = find_line_confidence(m_cand.group(1), all_lines) or 0.98

    # 2. Window matching preceding or following (for multi-column table cells like Poppler outputs)
    if "major_activity" not in fields:
        act_match_pre = re.search(
            r"(?:\b|(?<=[\s:]))" + act_words + r"(?=\b|[\s\r\n:]|$)[\s\S]{0,60}?(?:MAJOR\s*ACTIVITY|मुख्य\s*(?:कार्यकलाप|गतिविधि|कामकाज))",
            text,
            re.IGNORECASE,
        )
        act_match_post = re.search(
            r"(?:MAJOR\s*ACTIVITY|मुख्य\s*(?:कार्यकलाप|गतिविधि|कामकाज))[\s\S]{0,100}?(?:\b|(?<=[\s:]))" + act_words + r"(?=\b|[\s\r\n:]|$)",
            text,
            re.IGNORECASE,
        )
        if act_match_pre:
            val = act_match_pre.group(1).lower()
            fields["major_activity"] = act_map.get(val, val.capitalize())
            confidences["major_activity"] = find_line_confidence(act_match_pre.group(1), all_lines) or 0.98
        elif act_match_post:
            val = act_match_post.group(1).lower()
            fields["major_activity"] = act_map.get(val, val.capitalize())
            confidences["major_activity"] = find_line_confidence(act_match_post.group(1), all_lines) or 0.98

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 5. FSSAI Certificate Extractor
# ==============================================================================

def extract_fssai(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # 14-digit FSSAI License Number
    lic_match = re.search(r"\b(\d{14})\b", text)
    if lic_match:
        fields["fssai_licence_number"] = lic_match.group(1)
        confidences["fssai_licence_number"] = find_line_confidence(lic_match.group(1), all_lines)

    # Business Name
    biz_match = re.search(
        r"(?:Business\s*Name|Business\s*Operator\s*(?:\(FBO\))?|Name of Food Business Operator)[\s:]+([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{1,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:FSSAI|Licence|License|Kind of Business|Valid|Period|Address|SR\s*NO)\b))",
        text,
        re.IGNORECASE,
    )
    if biz_match:
        fields["business_name"] = clean_field_value(biz_match.group(1))
        confidences["business_name"] = find_line_confidence(fields["business_name"], all_lines)

    # Kind of Business
    kind_match = re.search(
        r"(?:Kind of Business)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{1,50}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Valid|Licence|License|Business Name)\b))",
        text,
        re.IGNORECASE,
    )
    if kind_match:
        fields["kind_of_business"] = clean_field_value(kind_match.group(1))

    # Validity
    valid_from = re.search(r"(?:Valid\s*From|Issued\s*On)[\s:/]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if valid_from:
        fields["valid_from"] = valid_from.group(1)
    valid_till = re.search(r"(?:Valid\s*Till|Fee\s*Paid\s*Upto)[\s:/]+.*?(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if valid_till:
        fields["valid_till"] = valid_till.group(1)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 6. Shop & Establishment Extractor (Per-State Template Registry)
# ==============================================================================

SHOP_ESTABLISHMENT_STATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "MH": {
        "state_name": "Maharashtra",
        "state_code": "MH",
        "verified": True,
        "authority_patterns": [
            r"Government\s+of\s+Maharashtra",
            r"Labour\s+Department[,\s]+Maharashtra",
            r"Aaple\s*Sarkar",
            r"Municipal\s+Corporation\s+of\s+Greater\s+Mumbai",
            r"Pune\s+Municipal\s+Corporation",
            r"\bMAHARASHTRA\b",
            r"महाराष्ट्र\s*शासन",
            r"कामगार\s*आयुक्त",
            r"बृहन्मुंबई\s*महानगरपालिका",
            r"पुणे\s*महानगरपालिका",
            r"आपले\s*सरकार",
            r"दुकान\s*(?:आणि|व)\s*आस्थापना",
            r"महारा\s+(?:दु\s*क\s*ाने|दुकान|दुकाने)\s*(?:आणि|व)\s*आ\s*थ?स्थापना",
            r"नमु\s*न\s*ा\s*[\"'\u201c\u201d]?[फगFG][\"'\u201c\u201d]?",
            r"Form\s*[-–]\s*[\"'\u2018\u2019]?[फगFG][\"'\u2018\u2019]?",
        ],
        "reg_no_patterns": [
            r"(?:नोंदणी\s*(?:क्रमांक|क्र\.?)|Registration\s*Number|Reg\s*No\.?|Certificate\s*No\.?)[\s:]*([A-Za-z0-9\-\/]{5,30})",
            r"(?:पावती\s*(?:क्रमांक|क्र\.?|मांक)|Registration\s*Certificate\s*/\s*Intimation)[\s:]*([A-Za-z0-9\-\/]{5,30})",
            r"\b(SHOP-[A-Z0-9\-]+)\b",
            r"\b(MH[0-9A-Z\-\/]{6,25})\b",
        ],
        "issuing_authority_default": "Government of Maharashtra",
    },
    "DL": {
        "state_name": "Delhi",
        "state_code": "DL",
        "verified": False,
        "authority_patterns": [
            r"Government\s+of\s+NCT\s+of\s+Delhi",
            r"Labour\s+Department[,\s]+Delhi",
            r"NCT\s+of\s+Delhi",
            r"\bDELHI\b",
        ],
        "reg_no_patterns": [
            r"(?:Registration\s*Number|Reg\s*No\.?|Certificate\s*No\.?)[\s:]*([A-Za-z0-9\-\/]{5,30})",
            r"\b(DL[0-9A-Z\-\/]{5,25})\b",
            r"\b(D-SE\/[0-9A-Z\-\/]+)\b",
        ],
        "issuing_authority_default": "Government of NCT of Delhi",
    },
    "KA": {
        "state_name": "Karnataka",
        "state_code": "KA",
        "verified": False,
        "authority_patterns": [
            r"Government\s+of\s+Karnataka",
            r"Labour\s+Department[,\s]+Karnataka",
            r"e-Karmika",
            r"\bKARNATAKA\b",
            r"\bBENGALURU\b",
            r"\bBANGALORE\b",
        ],
        "reg_no_patterns": [
            r"(?:Registration\s*Number|Reg\s*No\.?|Certificate\s*No\.?)[\s:]*([A-Za-z0-9\-\/]{5,30})",
            r"\b(KA[0-9A-Z\-\/]{5,25})\b",
            r"\b(KSC\/[0-9A-Z\-\/]+)\b",
        ],
        "issuing_authority_default": "Government of Karnataka",
    },
}


def extract_shop_establishment(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # State template detection
    matched_state = None
    for state_code, state_cfg in SHOP_ESTABLISHMENT_STATE_REGISTRY.items():
        for pat in state_cfg["authority_patterns"]:
            if re.search(pat, text, re.IGNORECASE):
                matched_state = state_code
                break
        if matched_state:
            break

    if matched_state:
        cfg = SHOP_ESTABLISHMENT_STATE_REGISTRY[matched_state]
        fields["state"] = matched_state
        fields["state_name"] = cfg["state_name"]
        fields["issuing_authority"] = cfg["issuing_authority_default"]
        fields["template_matched"] = True
        fields["template_verified"] = bool(cfg.get("verified", False))
        confidences["state"] = 0.98
        confidences["template_matched"] = 1.0
        confidences["template_verified"] = 1.0

        # State-specific registration number patterns
        for reg_pat in cfg["reg_no_patterns"]:
            reg_match = re.search(reg_pat, text, re.IGNORECASE)
            if reg_match:
                fields["registration_number"] = reg_match.group(1).strip()
                confidences["registration_number"] = find_line_confidence(fields["registration_number"], all_lines)
                break
    else:
        fields["state"] = None
        fields["template_matched"] = False
        fields["template_verified"] = False

    # Fallback / Generic Registration Number if not matched by state template
    if not fields.get("registration_number"):
        reg_match = re.search(r"(?:नोंदणी\s*(?:क्रमांक|क्र\.?)|Registration\s*Number|Reg\s*No\.?)[\s:]*([A-Za-z0-9\-\/]{5,25})", text, re.IGNORECASE)
        if reg_match:
            fields["registration_number"] = reg_match.group(1).strip()
            confidences["registration_number"] = find_line_confidence(reg_match.group(1), all_lines)

    # Establishment Name
    est_match = re.search(
        r"(?:Name\s*of\s*(?:the\s*)?establishment|आ\s*थापनेचे\s*नाव|आस्थापनेचे\s*नाव|दुकानाचे\s*नाव|(?<!&\s)(?<!and\s)(?<!of\s)\bEstablishment\b)[\s:/]+([^\r\n:]{2,70}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:Registration|Reg|Employer|नोंदणी|मालक|Nature\s*of\s*Business|व्यवसाय|Address|पत्ता|Previous)\b))",
        text,
        re.IGNORECASE,
    )
    if est_match:
        est_val = est_match.group(1).strip()
        est_val = re.sub(r"^[\s/]*(?:आ\s*(?:थापनेचे|स्थापनेचे)(?:\s*नाव)?|नाव)[\s:]*", "", est_val).strip()
        est_val = re.split(r"(?<=[a-zA-Z])\s+(?=[\u0900-\u097F])", est_val)[0].strip()
        fields["establishment_name"] = clean_field_value(est_val, field_name="establishment_name", doc_type="shop_establishment")
        confidences["establishment_name"] = find_line_confidence(fields["establishment_name"], all_lines)

    # Employer Name
    emp_match = re.search(
        r"(?:Name\s*of\s*(?:the\s*)?Employer(?:[\s/]*(?:मालकाचे\s*नाव)?)?|मालकाचे\s*नाव|Employer)[\s:]*([^\r\n:]{2,50}?)(?=[ \t]{2,}|\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:Registration|Reg|Establishment|नोंदणी|आस्थापना|Nature\s*of\s*Business|व्यवसाय|Residential|Address|पत्ता)\b)",
        text,
        re.IGNORECASE,
    )
    if emp_match:
        emp_val = emp_match.group(1).strip()
        emp_val = re.sub(r"^[\s/]*(?:मालकाचे(?:\s*नाव)?|नाव)[\s:]*", "", emp_val).strip()
        emp_val = re.split(r"(?<=[a-zA-Z])\s+(?=[\u0900-\u097F])", emp_val)[0].strip()
        fields["employer_name"] = clean_field_value(emp_val, field_name="employer_name", doc_type="shop_establishment")
        confidences["employer_name"] = find_line_confidence(fields["employer_name"], all_lines)

    # Nature of Business
    nature_match = re.search(
        r"(?:व्यवसायाचे\s*स्वरूप|Nature\s*of\s*Business|Category\s*Of\s*Establishment\s*Type(?:[\s/]*(?:आ\s*थापनेचे\s*उपवगवार)?)?)[\s:/]+([^\r\n:]{2,50}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:Registration|Reg|Employer|Date|तारीख|दिनांक|Type\s*of)\b))",
        text,
        re.IGNORECASE,
    )
    if nature_match:
        fields["nature_of_business"] = clean_field_value(nature_match.group(1), field_name="nature_of_business", doc_type="shop_establishment")
        confidences["nature_of_business"] = find_line_confidence(fields["nature_of_business"], all_lines)

    cleaned_fields = {
        k: (clean_field_value(v, field_name=k, doc_type="shop_establishment") if isinstance(v, str) else v)
        for k, v in fields.items()
    }
    return cleaned_fields, confidences


# ==============================================================================
# 7. Bank Statement Extractor (Multi-Page Table Merge & Strict PII Allowlist)
# ==============================================================================

def extract_bank_statement(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Bank statement extractor:
    - Merges transaction tables across ALL pages (including multi-line and multi-column formats)
    - Applies strict PII allowlist:
      ALLOWLIST = {bank_name, account_number_masked, statement_period, opening_balance, closing_balance, transactions}
      NO full account numbers, NO residential addresses, NO full DOB!
    - Reliable closing balance determination:
      1. Explicit 'Closing Balance:' field or Account Summary table
      2. Running balance from the LAST valid transaction in the statement
      3. Strict guardrail: Never returns opening balance as closing balance when transactions occurred
      4. Never defaults to first generic 'Balance' match
    """
    all_lines = [line for page in doc_res.pages for line in page.lines]
    full_text = doc_res.full_text
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # 1. Masked Account Number (matches 'Account Number' as well as 'Account No.')
    acc_match = re.search(
        r"(?:Account\s*(?:Number|No\.?)|A/C\s*(?:Number|No\.?))[\s:]*([X\d]{6,18})",
        full_text,
        re.IGNORECASE,
    )
    if acc_match:
        raw_acc = acc_match.group(1)
        fields["account_number_masked"] = mask_account_number(raw_acc)
        confidences["account_number_masked"] = find_line_confidence(raw_acc, all_lines)

    # 2. Bank Name
    bank_match = re.search(
        r"(?:Statement\s+(?:of\s+)?|Account\s+Statement\s+(?:of\s+)?|Branch\s*Office\s*:|Bank\s*Name|Bank)[\s:]*([A-Za-z][A-Za-z \t.&'-]+?\b(?:BANK(?:\s+OF\s+[A-Za-z]+)?|PAYMENTS\s+BANK)|BANK\s+OF\s+[A-Za-z]+)\b|^[\s:]*([A-Za-z][A-Za-z \t.&'-]+?\b(?:BANK(?:\s+OF\s+[A-Za-z]+)?|PAYMENTS\s+BANK)|BANK\s+OF\s+[A-Za-z]+)\b",
        full_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if bank_match:
        cand_bank = (bank_match.group(1) or bank_match.group(2) or "").strip()
        cand_bank = re.sub(r"^(?:Statement\s+(?:of\s+)?|Account\s+Statement\s+(?:of\s+)?|Branch\s*Office\s*:?)\s*", "", cand_bank, flags=re.IGNORECASE).strip()
        if cand_bank:
            fields["bank_name"] = clean_field_value(cand_bank)
            confidences["bank_name"] = find_line_confidence(cand_bank, all_lines)

    # 3. Statement Period (supports numeric and alphanumeric month ranges e.g. '28-Mar-2026 to 27-Apr-2026' or '( From : 30/07/2025 To : 30/07/2026 )')
    period_match = re.search(
        r"(?:Transaction\s*Period|Statement\s*Period|Period)[\s:]*(?:\(\s*)?(?:From\s*[:]\s*)?([A-Za-z0-9\/\-\.]+)\s*(?:to|-|To\s*[:])\s*([A-Za-z0-9\/\-\.]+)",
        full_text,
        re.IGNORECASE,
    )
    if period_match:
        fields["statement_period"] = {
            "from_date": period_match.group(1).strip(),
            "to_date": period_match.group(2).strip(),
        }

    # 4. Opening Balance
    opening_bal: Optional[str] = None
    op_match = re.search(
        r"Opening\s*Balance[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)",
        full_text,
        re.IGNORECASE,
    )
    if op_match:
        opening_bal = op_match.group(1).replace(",", "")
        fields["opening_balance"] = opening_bal
        confidences["opening_balance"] = find_line_confidence(op_match.group(1), all_lines)

    # 5. Multi-page transactions table parsing and chronological merging
    transactions: List[Dict[str, Any]] = []
    header_pattern = re.compile(
        r"\b(?:Transaction\s*Details|ACCOUNT\s*SUMMARY|END\s*OF\s*REPORT|DISCLAIMER|DATE\s+TRAN|Guidelines\s+for|Remember\s+that|Branch\s+Office|Customer\s+Address|Registered\s+Mobile|Account\s+Number|Nomination|Account\s+Type|Customer\s+ID|IFSC|MICR)\b",
        re.IGNORECASE,
    )
    prev_balance: Optional[float] = None
    if opening_bal:
        try:
            prev_balance = float(opening_bal)
        except ValueError:
            pass

    # Process all pages sequentially
    pages_to_process = doc_res.pages if doc_res.pages else []
    if not pages_to_process and full_text.strip():
        # Fallback if no pages list provided
        from ocr_engine import OCRPageResult, parse_text_into_ocr_lines
        pages_to_process = [OCRPageResult(page_num=1, full_text=full_text, lines=parse_text_into_ocr_lines(full_text), average_confidence=1.0)]

    for page in pages_to_process:
        # Strategy 1: Multi-line / Grid Finacle Block Parsing (e.g. Axis Bank, PNB, Canara Bank)
        # Transactions are grouped in blocks separated by double newlines, with S.No, Date, Particulars, and Amounts interleaved.
        raw_blocks = re.split(r"\n\s*\n+", page.full_text.strip()) if page.full_text else []
        page_block_txns = []
        for blk in raw_blocks:
            if header_pattern.search(blk) and not re.search(r"^\s*\d+\s+\d{2}[/\-\.]", blk, re.M):
                continue
            m_date = re.search(r"(?:^|\n)\s*(?:(\d{1,6})\s+)?(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})\b", blk)
            m_fin = re.search(r"([\d,]+\.\d{2})\s+(CR|DR|Cr|Dr)\s+([\d,]+\.\d{2})", blk)
            if m_date and m_fin:
                date_str = m_date.group(2)
                amt_str = m_fin.group(1).replace(",", "")
                t_type = m_fin.group(2).upper()
                bal_str = m_fin.group(3).replace(",", "")

                desc_parts = []
                for b_line in blk.splitlines():
                    if re.search(r"Branch Name|Debit/Credit|Balance\(INR\)|Transaction\s+Date", b_line, re.I):
                        continue
                    l_fin = re.search(r"([\d,]+\.\d{2})\s+(?:CR|DR|Cr|Dr)\s+([\d,]+\.\d{2})", b_line)
                    if l_fin:
                        prefix = b_line[:l_fin.start()].strip()
                        if prefix:
                            desc_parts.append(prefix)
                    else:
                        part_slice = b_line[35:110].strip() if len(b_line) > 35 else b_line.strip()
                        if part_slice and not re.match(r"^(?:\d{1,6}\s+)?\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}", part_slice):
                            desc_parts.append(part_slice)
                desc = " ".join(desc_parts)
                desc = re.sub(r"\s+", " ", desc).strip()
                if bal_str:
                    try:
                        prev_balance = float(bal_str)
                    except ValueError:
                        pass
                page_block_txns.append({
                    "date": date_str,
                    "description": desc,
                    "amount": amt_str,
                    "type": t_type,
                    "balance": bal_str,
                })

        if page_block_txns:
            transactions.extend(page_block_txns)
            continue

        # Strategy 2: Line-by-Line Parsing (e.g. India Post Payments Bank, HDFC, SBI standard format)
        lines = [l.strip() for l in page.full_text.splitlines() if l.strip()] if page.full_text else [l.text.strip() for l in page.lines if l.text.strip()]
        date_pattern = re.compile(r"^\s*(?:(\d{1,6})\s+)?(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})\b")
        i = 0
        while i < len(lines):
            line = lines[i]
            m_date = date_pattern.match(line)
            if m_date and not header_pattern.search(line):
                date_str = m_date.group(2)
                rest = line[m_date.end():]

                # Lookahead for wrapped multiline description lines
                extra_desc = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if not next_line.strip() or date_pattern.match(next_line) or header_pattern.search(next_line):
                        break
                    # If the next line does not look like an amounts/balance line, it is description continuation
                    if not re.search(r"\d+\.\d{2}", next_line):
                        extra_desc.append(next_line.strip())
                        j += 1
                    else:
                        break
                i = j - 1

                # Parse row:
                # Format A: Single-line format with explicit CR/DR (e.g. '01/01/2026 SALARY CREDIT 50000 CR 50000')
                m_std = re.search(r"^(.*?)\s+([\d,]+\.?\d*)\s+(CR|DR|Cr|Dr)\s*([\d,]+\.?\d*)?\s*$", rest, re.IGNORECASE)
                if m_std:
                    desc = m_std.group(1).strip()
                    if extra_desc:
                        desc += " " + " ".join(extra_desc)
                    amt_str = m_std.group(2).replace(",", "")
                    t_type = m_std.group(3).upper()
                    bal_str = m_std.group(4).replace(",", "") if m_std.group(4) else None
                    if bal_str:
                        try:
                            prev_balance = float(bal_str)
                        except ValueError:
                            pass
                    transactions.append({
                        "date": date_str,
                        "description": re.sub(r"[ \t]+", " ", desc),
                        "amount": amt_str,
                        "type": t_type,
                        "balance": bal_str,
                    })
                else:
                    # Format B: Multi-column tabular format with running balance at end (e.g. 'S52193252 UPI... 50.00 55.63 Cr.')
                    bal_m = re.search(r"([\d,]+\.\d{2})\s*(?:Cr\.?|Dr\.?|CR|DR)?\s*$", rest, re.IGNORECASE)
                    if bal_m:
                        balance_val = bal_m.group(1).replace(",", "")
                        rest_before = rest[:bal_m.start()].rstrip()
                        amt_m = re.search(r"([\d,]+\.\d{2})\s*$", rest_before)
                        if amt_m:
                            amount_val = amt_m.group(1).replace(",", "")
                            desc = rest_before[:amt_m.start()].strip()
                            if extra_desc:
                                desc += " " + " ".join(extra_desc)

                            try:
                                bal_float = float(balance_val)
                            except ValueError:
                                bal_float = None

                            # Determine CR vs DR:
                            # Prefer mathematical delta if previous balance is known
                            txn_type = "DR"
                            if prev_balance is not None and bal_float is not None:
                                if bal_float > prev_balance:
                                    txn_type = "CR"
                                elif bal_float < prev_balance:
                                    txn_type = "DR"
                                else:
                                    txn_type = "CR" if re.search(r"~CR~|\bCR\b|\bCREDIT\b|\bDEPOSIT\b|\bNEFT-IN\b", desc, re.I) else "DR"
                            else:
                                txn_type = "CR" if re.search(r"~CR~|\bCR\b|\bCREDIT\b|\bDEPOSIT\b|\bNEFT-IN\b", desc, re.I) else "DR"

                            if bal_float is not None:
                                prev_balance = bal_float

                            transactions.append({
                                "date": date_str,
                                "description": re.sub(r"[ \t]+", " ", desc),
                                "amount": amount_val,
                                "type": txn_type,
                                "balance": balance_val,
                            })
            i += 1

    fields["transactions"] = transactions
    if transactions:
        confidences["transactions"] = round(sum(find_line_confidence(t["date"], all_lines) for t in transactions) / len(transactions), 4)

    # 6. Closing Balance Determination (Prioritized Strategy)
    closing_bal: Optional[str] = None

    # Priority 1: Explicit 'Closing Balance: <val>' label
    cl_match = re.search(
        r"(?<!Opening\s)\bClosing\s*Balance[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)",
        full_text,
        re.IGNORECASE,
    )
    if cl_match:
        closing_bal = cl_match.group(1).replace(",", "")

    # Priority 1b: Tabular Account Summary block (headers row followed by values row)
    if not closing_bal:
        m_tbl = re.search(r"ACCOUNT\s*SUMMARY\s*\n\s*(.*?)\n\s*(.*?)(?:\n|$)", full_text, re.IGNORECASE)
        if m_tbl:
            h_line = m_tbl.group(1).strip()
            v_line = m_tbl.group(2).strip()
            positions = []
            for col_kw in ["OPENING BALANCE", "TOTAL WITHDRAWALS", "WITHDRAWALS", "TOTAL DEPOSITS", "DEPOSITS", "CLOSING BALANCE", "NO. OF TRANSACTIONS", "NO OF TRANSACTIONS"]:
                m_kw = re.search(r"\b" + re.escape(col_kw) + r"\b", h_line, re.I)
                if m_kw:
                    positions.append((m_kw.start(), col_kw.upper()))
            positions.sort()
            # Filter sub-matches at same position
            filtered_cols = []
            for pos, name in positions:
                if not any(pos >= f_pos and pos < f_pos + len(f_name) for f_pos, f_name in filtered_cols):
                    filtered_cols.append((pos, name))

            nums = re.findall(r"[\d,]+\.?\d*", v_line)
            for i, (_, col_name) in enumerate(filtered_cols):
                if i < len(nums):
                    val_str = nums[i].replace(",", "")
                    if "CLOSING" in col_name:
                        closing_bal = val_str
                    elif "OPENING" in col_name and not opening_bal:
                        opening_bal = val_str
                        fields["opening_balance"] = opening_bal

    # Priority 2: Running Balance from LAST valid transaction row
    final_txn_balance: Optional[str] = None
    if transactions:
        for t in reversed(transactions):
            if t.get("balance") is not None:
                final_txn_balance = t["balance"]
                break

    if not closing_bal and final_txn_balance:
        closing_bal = final_txn_balance

    # Priority 3 & 4: Strict Guardrail - Never accept opening balance as closing balance when transactions occurred
    if closing_bal and opening_bal and closing_bal == opening_bal and transactions:
        if final_txn_balance and final_txn_balance != opening_bal:
            closing_bal = final_txn_balance

    if closing_bal:
        fields["closing_balance"] = closing_bal
        confidences["closing_balance"] = find_line_confidence(closing_bal, all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 8. Salary Slip Extractor (Strict PII Allowlist)
# ==============================================================================

def extract_salary_slip(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Salary slip extractor:
    - Applies strict PII allowlist:
      ALLOWLIST = {employer_name, employee_name_masked, net_pay, pay_period}
      NO full account numbers, NO residential addresses, NO full DOB!
    - Supports bilingual / Devanagari salary slips (state government bodies, Zilla Parishad,
      municipal corporations, police, MSRTC) with Marathi and Hindi field labels
      and Devanagari numerals.
    """
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = normalize_devanagari_numbers(doc_res.full_text) or ""
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Employer Name
    emp_match = re.search(
        r"(?:Company(?:\s*Name)?|Employer(?:\s*Name)?|कार्यालयाचे\s*नाव|संस्थेचे\s*नाव|विभागाचे\s*नाव|कंपनीचे\s*नाव|नियोक्त्याचे\s*नाव)[\s:：]+([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F \t,\.\-&]{2,80}?(?:LIMITED|PRIVATE\s+LIMITED|PVT\.?\s*LTD\.?|LTD\.?|CORP|INC|DEMO|मर्यादित)?)(?=[ \t]*(?:\r?\n|$|(?:Employee|Name|Emp\s*ID|Month|Net|Basic|HRA|कर्मचाऱ्याचे|कर्मचारी|माहे|निव्वळ)[\s:：]))",
        text,
        re.IGNORECASE,
    )
    if emp_match:
        cand = clean_field_value(emp_match.group(1), field_name="employer_name", doc_type="salary_slip")
        if not re.search(r"^(?:LIMITED|PRIVATE|PVT|LTD|मर्यादित)\b", cand, re.I):
            fields["employer_name"] = cand
            confidences["employer_name"] = find_line_confidence(cand, all_lines)

    if not fields.get("employer_name"):
        # Check first prominent header line (e.g. "DEMO COMPANY PRIVATE LIMITED - SALARY SLIP" or "DEMO COMPANY PVT LTD")
        header_match = re.search(
            r"^[ \t]*([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F \t,\.\-&]{2,80}(?:LIMITED|PRIVATE\s+LIMITED|PVT\.?\s*LTD\.?|LTD\.?|CORPORATION|SERVICES|ENTERPRISES|TECHNOLOGIES|COMPANY|महाराष्ट्र\s*शासन|जिल्हा\s*परिषद|महानगरपालिका|नगरपरिषद|महामंडळ|मर्यादित))",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if header_match:
            cand = clean_field_value(header_match.group(1), field_name="employer_name", doc_type="salary_slip")
            fields["employer_name"] = cand
            confidences["employer_name"] = find_line_confidence(cand, all_lines)

    if not fields.get("employer_name"):
        gov_match = re.search(
            r"^[ \t]*((?:महाराष्ट्र\s*शासन|जिल्हा\s*परिषद|महानगरपालिका|नगरपरिषद|राज्य\s*परिवहन\s*महामंडळ)[^\r\n]{0,60})",
            text,
            re.MULTILINE,
        )
        if gov_match:
            cand = clean_field_value(gov_match.group(1), field_name="employer_name", doc_type="salary_slip")
            fields["employer_name"] = cand
            confidences["employer_name"] = find_line_confidence(cand, all_lines)

    # Employee Name (Raw and Masked)
    name_match = re.search(
        r"(?:Employee(?:\s*Name)?|कर्मचाऱ्याचे\s*नाव|कर्मचारी\s*नाव|कर्मचारी\s*का\s*नाम|अधिकाऱ्याचे\s*नाव|सेवकाचे\s*नाव|(?<!Employer\s)(?<!Company\s)(?<!कार्यालयाचे\s)(?<!संस्थेचे\s)\bName)[\s:：]*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F \t.'-]{1,40}?)(?=[ \t]*(?:\r?\n|$|(?:ID|Emp\s*ID|Designation|Department|Net|Gross|Month|Pay\s*Period|पदनाम|पद|विभाग|वेतन|माहे|रक्कम|भविष्य\s*निर्वाह)[\s:：]))",
        text,
        re.IGNORECASE,
    )
    if name_match:
        raw_name = clean_field_value(name_match.group(1), field_name="employee_name", doc_type="salary_slip")
        fields["raw_employee_name"] = raw_name
        fields["employee_name"] = raw_name
        fields["employee_name_masked"] = mask_person_name(raw_name)
        confidences["employee_name_masked"] = find_line_confidence(raw_name, all_lines)

    # Net Pay
    net_match = re.search(
        r"(?:Net\s*Salary|Net\s*Pay|Net\s*Amount|निव्वळ\s*वेतन|निव्वळ\s*देय\s*(?:रक्कम)?|निव्वळ\s*रक्कम|हाती\s*येणारे\s*वेतन|शुद्ध\s*वेतन|कुल\s*शुद्ध\s*देय)[\s:：]+(?:Rs\.?|INR|₹|रु\.?|रुपये)?\s*([\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if net_match:
        fields["net_pay"] = net_match.group(1).replace(",", "")
        confidences["net_pay"] = find_line_confidence(net_match.group(1), all_lines)

    # Pay Period / Month
    month_match = re.search(
        r"(?:Month|Pay\s*Period|वेतन\s*महिना|माहे(?:\s*महिना)?|माहे|कालावधी)[\s:：]+([A-Za-z\u0900-\u097F]+\s*\d{4}|\d{1,2}[\/\-]\d{4})",
        text,
        re.IGNORECASE,
    )
    if month_match:
        fields["pay_period"] = month_match.group(1).strip()
        confidences["pay_period"] = find_line_confidence(fields["pay_period"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 9. Utility Bill Extractor (Strict PII Allowlist)
# ==============================================================================

def extract_utility_bill(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Utility bill extractor:
    - Applies strict PII allowlist:
      ALLOWLIST = {utility_provider, consumer_number, bill_date, due_date, bill_amount}
      NO residential address, NO sensitive private identifiers!
    - Supports bilingual / Devanagari bills (MSEDCL/Mahavitaran, state electricity boards,
      municipal water/gas) with Marathi and Hindi field labels and Devanagari numerals.
    """
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = normalize_devanagari_numbers(doc_res.full_text) or ""
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Utility Provider
    provider_match = re.search(
        r"(?:Provider|Company|Board|प्रदाता)[\s]*[:：][\s]*([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F \t,\.\-&]{1,50}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:Consumer|Bill|Due|Date|Amount|CA\s*No|Connection|ग्राहक|देयक|देय|रक्कम)\b))",
        text,
        re.IGNORECASE,
    )
    if provider_match:
        fields["utility_provider"] = clean_field_value(provider_match.group(1))
    else:
        text_upper = text.upper()
        if any(k in text for k in ["महावितरण", "महािवतरण", "महाराष्ट्र राज्य विद्युत"]) or any(k in text_upper for k in ["MSEDCL", "MAHADISCOM", "MAHAVITARAN"]):
            fields["utility_provider"] = "Mahavitaran (MSEDCL)"
        elif "टाटा पॉवर" in text or "TATA POWER" in text_upper:
            fields["utility_provider"] = "Tata Power"
        elif "अदानी" in text or "ADANI ELECTRICITY" in text_upper:
            fields["utility_provider"] = "Adani Electricity"
        elif "बेस्ट" in text or "BEST UNDERTAKING" in text_upper:
            fields["utility_provider"] = "BEST Undertaking"
        elif "टॉरेंट पॉवर" in text or "TORRENT POWER" in text_upper:
            fields["utility_provider"] = "Torrent Power"
        elif any(k in text_upper for k in ["ELECTRICITY", "POWER"]) or any(k in text for k in ["विद्युत", "वीज"]):
            fields["utility_provider"] = "Electricity Distribution Board"
        elif "WATER" in text_upper or any(k in text for k in ["पाणी", "जल"]):
            fields["utility_provider"] = "Water Supply Department"
        elif "GAS" in text_upper or "गॅस" in text:
            fields["utility_provider"] = "Natural Gas Corporation"

    # Consumer Number
    # Verified Marathi: ग्राहक क्रमांक, ग्राहक क्र., ग्राहक नंबर, साहकक्रमांक (OCR variant of ग्राहकक्रमांक)
    # Verified Hindi: उपभोक्ता संख्या, उपभोक्ता क्रमांक, खाता संख्या
    # English: Consumer No, CA No, Account No, Connection ID, K No
    consumer_match = re.search(
        r"(?:ग्राहक\s*(?:क्रमांक|क्र\.?|नंबर|सं\.?)|साहक\s*(?:क्रमांक|कमांक|कमक|कमिक)|उपभोक्ता\s*(?:संख्या|क्रमांक|क्र\.?)|खाता\s*(?:संख्या|क्रमांक)|Consumer\s*(?:No\.?|Number|ID)|CA\s*No\.?|Account\s*No\.?|Connection\s*ID|K\s*No\.?)[\s:：]*([A-Za-z0-9\-]{6,25})",
        text,
        re.IGNORECASE,
    )
    if consumer_match:
        c_val = normalize_devanagari_numbers(consumer_match.group(1).strip())
        fields["consumer_number"] = c_val
        confidences["consumer_number"] = find_line_confidence(consumer_match.group(1), all_lines)
    else:
        # Check RTGS/NEFT payment virtual account on bills (e.g. Beneficiaryaccountno.:MSEDCL01177453132860)
        neft_match = re.search(r"Beneficiary\s*account\s*no\.?[\s:：]*([A-Za-z0-9]{8,25})", text, re.IGNORECASE)
        if neft_match:
            fields["consumer_number"] = neft_match.group(1)
            confidences["consumer_number"] = find_line_confidence(neft_match.group(1), all_lines)

    # Bill Date
    # Verified Marathi: देयक दिनांक, देयक तारीख, देयक दि., बिल दिनांक, बिलाचा दिनांक
    # Verified Hindi: बिल तिथि, बिल दिनांक, देयक तिथि
    # English: Bill Date, Date of Bill, Billing Date, Invoice Date
    bdate_match = re.search(
        r"(?:Bill\s*Date|Date\s*of\s*Bill|Billing\s*Date|Invoice\s*Date|देयक\s*(?:दिनांक|तारीख|तिथि|दि\.?)|बिल\s*(?:दिनांक|तारीख|तिथि)|बिलाचा\s*दिनांक)[\s:：]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if bdate_match:
        fields["bill_date"] = bdate_match.group(1).replace("-", "/").replace(".", "/")
        confidences["bill_date"] = find_line_confidence(bdate_match.group(1), all_lines)

    # Due Date
    # Verified Marathi: देय दिनांक, देय तारीख, अंतिम तारीख, अंतिम दिनांक, आतमतारीख (OCR variant of अंतिम तारीख)
    # Verified Hindi: देय तिथि, अंतिम तिथि, भुगतान तिथि
    # English: Due Date, Payment Due Date, Pay By Date, Last Date
    ddate_match = re.search(
        r"(?:Due\s*Date|Payment\s*Due\s*Date|Pay\s*By\s*Date|देय\s*(?:दिनांक|तारीख|तिथि)|अंतिम\s*(?:दिनांक|तारीख|तिथि)|आत[मम]\s*(?:तारीख|तारीस|दिनांक))[\s:：]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if ddate_match:
        fields["due_date"] = ddate_match.group(1).replace("-", "/").replace(".", "/")
        confidences["due_date"] = find_line_confidence(ddate_match.group(1), all_lines)
    else:
        # Prompt payment date fallback (या तारखेपर्यंत भरल्यास)
        prompt_date_match = re.search(
            r"(?:या\s*तारखेपर्यंत\s*भरल्यास|यातारखेपय[ंंत]+[^\n\r]*भरल[याा]स)[\s:：]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
            text,
            re.IGNORECASE,
        )
        if prompt_date_match:
            fields["due_date"] = prompt_date_match.group(1).replace("-", "/").replace(".", "/")
            confidences["due_date"] = find_line_confidence(prompt_date_match.group(1), all_lines)

    # Bill Amount
    # Verified Marathi: देयक रक्कम, एकूण रक्कम, एकूण देयक रक्कम, निव्वळ देयक रक्कम, देय रक्कम, भरणा रक्कम, बिल रक्कम
    # Verified Hindi: कुल राशि, देय राशि, बिल राशि, कुल देय राशि
    # English: Total Amount, Amount Due, Bill Amount, Net Amount, Total Due
    amt_match = re.search(
        r"(?:Total\s*Amount|Amount\s*Due|Bill\s*Amount|Net\s*Amount(?:\s*Payable)?|Total\s*Due|देयक\s*रक्कम|एकूण\s*रक्कम|एकूण\s*देयक|निव्वळ\s*(?:देयक\s*)?रक्कम|देय\s*रक्कम|भरणा\s*रक्कम|बिल\s*रक्कम|कुल\s*(?:देय\s*)?राशि)[\s:：]+(?:Rs\.?|INR|₹|रु\.?|रुपये)?\s*([\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if amt_match:
        raw_amt = normalize_devanagari_numbers(amt_match.group(1)).replace(",", "").rstrip(".")
        fields["bill_amount"] = raw_amt
        confidences["bill_amount"] = find_line_confidence(amt_match.group(1), all_lines)
    else:
        # Payment coupon / prompt payment slip pattern:
        # e.g. "यातारखेपयंत भरलास 28-10-2024 Rs.1200.00" or "आतमतारीख 07-11-2024 Rs. 1200.00"
        slip_amt_match = re.search(
            r"(?:यातारखेपय[ंंत]+[^\n\r]*भरल[याा]स|या\s*तारखेपर्यंत\s*भरल्यास|आत[मम]\s*(?:तारीख|तारीस)|अंतिम\s*(?:तारीख|दिनांक)|देय\s*दिनांक)[\s\S]{1,80}?(?:Rs\.?|INR|₹|रु\.?|रुपये)\s*([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if slip_amt_match:
            raw_amt = normalize_devanagari_numbers(slip_amt_match.group(1)).replace(",", "").rstrip(".")
            fields["bill_amount"] = raw_amt
            confidences["bill_amount"] = find_line_confidence(slip_amt_match.group(1), all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 10. Passport Extractor
# ==============================================================================

def extract_passport(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = doc_res.full_text
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Passport Number (1 letter + 7 digits)
    pass_match = re.search(r"\b([A-PR-WYa-pr-wy][0-9]{7}|[A-Z][0-9]{7})\b", text)
    if pass_match:
        fields["passport_number"] = pass_match.group(1).upper()
        confidences["passport_number"] = find_line_confidence(pass_match.group(1), all_lines)

    # Surname
    sur_match = re.search(
        r"(?:Surname)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,30}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Given|Given\s*Name|Nationality|Passport|DOB|Date)\b))",
        text,
        re.IGNORECASE,
    )
    if sur_match:
        fields["surname"] = clean_field_value(sur_match.group(1))
        confidences["surname"] = find_line_confidence(fields["surname"], all_lines)

    # Given Name
    given_match = re.search(
        r"(?:Given\s*Name)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,30}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Surname|Nationality|Passport|DOB|Date)\b))",
        text,
        re.IGNORECASE,
    )
    if given_match:
        fields["given_name"] = clean_field_value(given_match.group(1))
        confidences["given_name"] = find_line_confidence(fields["given_name"], all_lines)

    # Combine into name for unified cross-checking
    if "given_name" in fields and "surname" in fields:
        fields["name"] = f"{fields['given_name']} {fields['surname']}"
    elif "given_name" in fields:
        fields["name"] = fields["given_name"]

    # Nationality
    nat_match = re.search(r"(?:Nationality)[\s:]+([A-Za-z]+)", text, re.IGNORECASE)
    if nat_match:
        fields["nationality"] = nat_match.group(1).strip()

    # DOB
    dob_match = re.search(r"(?:Date of Birth|DOB)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if dob_match:
        fields["dob"] = dob_match.group(1).replace("-", "/").replace(".", "/")
        confidences["dob"] = find_line_confidence(dob_match.group(1), all_lines)

    # Expiry Date
    exp_match = re.search(r"(?:Date of Expiry|Expiry\s*Date)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if exp_match:
        fields["expiry_date"] = exp_match.group(1).replace("-", "/").replace(".", "/")

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 11. Voter ID Extractor
# ==============================================================================

def extract_voter_id(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = doc_res.full_text
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # EPIC Number: 3 letters + 7 digits (or alphanumeric formats)
    epic_match = re.search(r"\b([A-Z]{3}[0-9]{7})\b", text)
    if epic_match:
        fields["epic_number"] = epic_match.group(1)
        confidences["epic_number"] = find_line_confidence(epic_match.group(1), all_lines)

    # Name
    name_match = re.search(
        r"(?:Name|Elector['’]?s?\s*Name)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Father|Husband|Relative|DOB|Date|Age|Gender|EPIC)\b))",
        text,
        re.IGNORECASE,
    )
    if name_match:
        fields["name"] = clean_field_value(name_match.group(1))
        confidences["name"] = find_line_confidence(fields["name"], all_lines)

    # Relative / Father Name
    rel_match = re.search(
        r"(?:Father['’]?s?\s*Name|Husband['’]?s?\s*Name|Relative['’]?s?\s*Name)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:DOB|Date|Age|Gender|EPIC|Address)\b))",
        text,
        re.IGNORECASE,
    )
    if rel_match:
        fields["relative_name"] = clean_field_value(rel_match.group(1))

    # DOB / Age
    dob_match = re.search(r"(?:DOB|Date of Birth)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if dob_match:
        fields["dob"] = dob_match.group(1).replace("-", "/").replace(".", "/")
    else:
        age_match = re.search(r"(?:Age)[\s:]+(\d{1,2})", text, re.IGNORECASE)
        if age_match:
            fields["age"] = int(age_match.group(1))

    # Gender
    gender_match = re.search(r"\b(MALE|FEMALE)\b", text, re.IGNORECASE)
    if gender_match:
        fields["gender"] = gender_match.group(1).capitalize()

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 12. Driving Licence Extractor
# ==============================================================================

def extract_driving_licence(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = doc_res.full_text
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Driving Licence Number: State code (2) + RTO code (2) + Year (4) + 7 digits (or with spaces)
    dl_match = re.search(r"\b([A-Z]{2}[0-9]{2}\s?[0-9]{4}[0-9]{7}|[A-Z]{2}[0-9]{2}\s?[0-9A-Z]{11,15})\b", text)
    if dl_match:
        fields["licence_number"] = dl_match.group(1).strip()
        confidences["licence_number"] = find_line_confidence(fields["licence_number"], all_lines)

    # Name
    name_match = re.search(
        r"(?:Name)[\s:]*([A-Za-z][A-Za-z \t.'-]{1,35}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:DOB|Date|Licence|License|Issue|Valid|Vehicle)\b))",
        text,
        re.IGNORECASE,
    )
    if name_match:
        fields["name"] = clean_field_value(name_match.group(1))
        confidences["name"] = find_line_confidence(fields["name"], all_lines)

    # DOB
    dob_match = re.search(r"(?:DOB|Date of Birth)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if dob_match:
        fields["dob"] = dob_match.group(1).replace("-", "/").replace(".", "/")
        confidences["dob"] = find_line_confidence(dob_match.group(1), all_lines)

    # Issue Date
    issue_match = re.search(r"(?:Issue Date|Date of Issue)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if issue_match:
        fields["issue_date"] = issue_match.group(1).replace("-", "/").replace(".", "/")

    # Valid Till
    valid_match = re.search(r"(?:Valid Till|Validity)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", text, re.IGNORECASE)
    if valid_match:
        fields["valid_till"] = valid_match.group(1).replace("-", "/").replace(".", "/")

    # Vehicle Class
    class_match = re.search(r"\b(LMV|MCWG|MCWOG|TRANS|HGMV|HPMV)\b", text)
    if class_match:
        fields["vehicle_class"] = class_match.group(1)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 13. ITR (Income Tax Return) Extractor
# ==============================================================================

INVALID_ITR_NAME_LABELS = {
    "of deductor", "of employer", "of collector", "of buyer", "of seller",
    "of deductee", "of bank", "of premises", "of assessee", "of taxpayer",
    "deductor", "employer", "collector", "buyer", "seller", "deductee",
    "assessee", "taxpayer", "bank", "address", "status", "pan", "pan number",
    "acknowledgement", "acknowledgement number", "form", "form number",
    "total income", "taxes paid", "assessment year", "financial year",
}


def is_valid_itr_name(val: Optional[str]) -> bool:
    """Validate extracted ITR assessee name against third-party field labels and fragments."""
    if not val or not isinstance(val, str):
        return False
    clean = re.sub(r"^\d+[\.\)]\s*", "", val.strip()).strip(" :,-")
    if len(clean) < 2:
        return False
    norm = re.sub(r"[\s\W_]+", " ", clean).strip().lower()
    if not norm:
        return False
    if norm in INVALID_ITR_NAME_LABELS:
        return False
    if norm.startswith("of "):
        return False
    if any(norm.startswith(hdr) for hdr in [
        "name of", "tan of", "pan of", "total amount", "sr no", "date of",
        "form no", "assessment year", "financial year", "details of", "part "
    ]):
        return False
    return True


def extract_itr(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = doc_res.full_text
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Acknowledgement Number (15 digits)
    ack_match = re.search(r"(?:Acknowledgement\s*Number|Ack\s*No\.?)[\s:]*(\d{15})", text, re.IGNORECASE)
    if ack_match:
        fields["acknowledgement_number"] = ack_match.group(1)
        confidences["acknowledgement_number"] = find_line_confidence(ack_match.group(1), all_lines)

    # Assessment Year (20XX-YY)
    ay_match = re.search(r"(?:Assessment\s*Year|AY)[\s:]*(\d{4}[-\/]\d{2,4})", text, re.IGNORECASE)
    if ay_match:
        fields["assessment_year"] = ay_match.group(1)

    # PAN
    pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
    if pan_match:
        fields["pan_number"] = pan_match.group(1)
        confidences["pan_number"] = find_line_confidence(pan_match.group(1), all_lines)

    # Name (Assessee Name)
    name_val: Optional[str] = None

    # Determine cover/acknowledgement page text slices if multi-page
    search_slices: List[str] = []
    if doc_res.pages:
        for p in doc_res.pages[:3]:
            if any(k in p.full_text.lower() for k in ["acknowledgement", "itr-v", "income tax return", "pan", "assessee"]):
                search_slices.append(p.full_text)
    if not search_slices:
        search_slices.append(text)

    for stext in search_slices:
        # Strategy 1: Explicit "Name of Assessee" or "Assessee Name"
        m_assessee = re.search(
            r"(?:Name\s*of\s*Assessee|Assessee\s*Name)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t.,'\-&_]{1,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:PAN|Address|Status|Ward|Assessment|Financial|D\.O\.I|DOI|Return|Date)\b))",
            stext,
            re.IGNORECASE,
        )
        if m_assessee:
            cand = clean_field_value(m_assessee.group(1))
            if is_valid_itr_name(cand):
                name_val = cand
                break

        # Strategy 2: CA cover title block: "Income Tax Return ... Of \n <Assessee> \n Pan"
        m_cover_of = re.search(
            r"(?:Assessment\s*Year[^\n]*\n\s*Of|\bReturn\b[^\n]*\n[^\n]*\bOf)[\s:\r\n]+([A-Za-z0-9][A-Za-z0-9 \t.,'\-&_]{1,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:PAN|Address|Prepared)\b))",
            stext,
            re.IGNORECASE,
        )
        if m_cover_of:
            cand = clean_field_value(m_cover_of.group(1))
            if is_valid_itr_name(cand):
                name_val = cand
                break

        # Strategy 3: Standard ITR-V "Name" label (negative lookahead strictly blocks "Name of Deductor/Employer/...")
        m_name = re.search(
            r"\bName\b(?!\s*of\s*(?:Deductor|Employer|Bank|Buyer|Collector|Seller|Deductee|Premises|Branch))[\s:]*([A-Za-z0-9][A-Za-z0-9 \t.,'\-&_]{1,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:PAN|Address|Status|Form|Acknowledgement|Ack|Date|Father|Total|Taxes|Current)\b))",
            stext,
            re.IGNORECASE,
        )
        if m_name:
            cand = clean_field_value(m_name.group(1))
            if is_valid_itr_name(cand):
                name_val = cand
                break

        # Strategy 4: Multiline "Name \n <Assessee> \n Address"
        lines = [l.strip() for l in stext.splitlines() if l.strip()]
        for idx, line in enumerate(lines):
            if re.match(r"^Name\s*[:\-]?$", line, re.IGNORECASE):
                if idx + 1 < len(lines):
                    next_l = lines[idx + 1].strip()
                    if not re.search(r"^(?:PAN|Address|Status|Form|Acknowledgement|Ack|Date|Father|Total|Taxes)\b", next_l, re.IGNORECASE):
                        cand = clean_field_value(next_l)
                        if is_valid_itr_name(cand):
                            name_val = cand
                            break
        if name_val:
            break

    # Fallback to full text if slices did not yield a valid name
    if not name_val:
        m_name = re.search(
            r"\bName\b(?!\s*of\s*(?:Deductor|Employer|Bank|Buyer|Collector|Seller|Deductee|Premises|Branch))[\s:]*([A-Za-z0-9][A-Za-z0-9 \t.,'\-&_]{1,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:PAN|Address|Status|Form|Acknowledgement|Ack|Date|Father|Total|Taxes|Current)\b))",
            text,
            re.IGNORECASE,
        )
        if m_name:
            cand = clean_field_value(m_name.group(1))
            if is_valid_itr_name(cand):
                name_val = cand

    if name_val:
        fields["name"] = name_val
        confidences["name"] = find_line_confidence(fields["name"], all_lines)

    # Total Income
    inc_match = re.search(
        r"(?:Total\s*Income(?:[^\n\d]*?round[^\n]*)?)[\s:]+(?:(?:(?:Row\s*)?\d+[A-Za-z]?)\b[\s:]+)?(?:Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)",
        text,
        re.IGNORECASE,
    )
    if inc_match:
        fields["total_income"] = inc_match.group(1).replace(",", "")
        confidences["total_income"] = find_line_confidence(inc_match.group(1), all_lines)

    # Taxes Paid
    tax_match = re.search(
        r"(?:Taxes\s*Paid|Total\s*Tax(?:es)?\s*Paid)[\s:]+(?:(?:(?:Row\s*)?\d+[A-Za-z]?)\b[\s:]+)?(?:Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)",
        text,
        re.IGNORECASE,
    )
    if tax_match:
        fields["taxes_paid"] = tax_match.group(1).replace(",", "")

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 14. GST Certificate Extractor
# ==============================================================================

GST_TRADE_NAME_LABEL_WORDS = {"trade", "name", "names", "if", "any", "additional", "s"}
GST_BLANK_OR_NA_VALUES = {"na", "n a", "not applicable", "nil", "none", "null", "blank", "x", "xx", "xxx", "-"}
GST_NEXT_SECTION_HEADERS = ("constitution", "legal name", "gstin", "date of", "registration", "type of", "address", "period")


def is_valid_gst_trade_name(val: Optional[str]) -> bool:
    """Validate extracted GST trade name against field label fragments, blanks, and OCR artifacts."""
    if not val or not isinstance(val, str):
        return False
    # Strip leading item numbering like "2." or "2)"
    clean = re.sub(r"^\d+[\.\)]\s*", "", val.strip()).strip(" :,-")
    if len(clean) < 2:
        return False
    norm = re.sub(r"[\s\W_]+", " ", clean).strip().lower()
    if not norm:
        return False
    if norm in GST_BLANK_OR_NA_VALUES:
        return False
    words = norm.split()
    # Reject if all constituent words belong to GST label words (e.g. "s, if", "if any", "Trade Name, if any")
    if all(w in GST_TRADE_NAME_LABEL_WORDS for w in words):
        return False
    # Reject if it starts with another GST certificate section header
    if any(norm.startswith(hdr) for hdr in GST_NEXT_SECTION_HEADERS):
        return False
    return True


def extract_gst_certificate(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # GSTIN: 15 alphanumeric characters
    gstin_match = re.search(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b", text)
    if gstin_match:
        fields["gstin"] = gstin_match.group(1).upper()
        confidences["gstin"] = find_line_confidence(gstin_match.group(1), all_lines)

    # Legal Name
    legal_match = re.search(
        r"(?:Legal\s*Name(?:[\s/]*(?:of\s+Taxpayer)?)?)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t,\.\-&_]{1,70}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Trade\s*Name|GSTIN|Constitution|Date|Address|Period)\b))",
        text,
        re.IGNORECASE,
    )
    if legal_match:
        fields["legal_name"] = clean_field_value(legal_match.group(1), "legal_name", "gst_certificate")
        confidences["legal_name"] = find_line_confidence(fields["legal_name"], all_lines)

    # Trade Name
    trade_name_val: Optional[str] = None
    # 1. Inline regex match (strictly excluding "Additional trade names" with negative lookbehind)
    trade_match = re.search(
        r"(?<!Additional\s)(?<!Additional\s\s)\bTrade\s*Name(?:\s*,\s*if\s*any)?[\s:]+([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{1,70}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Constitution|Legal\s*Name|GSTIN|Date|Address|Period|Additional)\b))",
        text,
        re.IGNORECASE,
    )
    if trade_match:
        cand = clean_field_value(trade_match.group(1), "trade_name", "gst_certificate")
        if is_valid_gst_trade_name(cand):
            trade_name_val = cand

    # 2. Multiline fallback if inline regex did not capture a valid trade name
    if not trade_name_val:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for idx, line in enumerate(lines):
            if re.search(r"(?<!Additional\s)(?<!Additional\s\s)\bTrade\s*Name\b", line, re.IGNORECASE):
                inline = re.sub(r"^.*?\bTrade\s*Name(?:\s*,\s*if\s*any)?\s*[:\-]?\s*", "", line, flags=re.IGNORECASE).strip()
                inline = re.split(r"(?:\b|(?<=[a-z0-9A-Z]))(?:Constitution|Legal\s*Name|GSTIN|Date|Address|Period|Additional)\b", inline, flags=re.IGNORECASE)[0].strip()
                inline = clean_field_value(inline, "trade_name", "gst_certificate")
                if is_valid_gst_trade_name(inline):
                    trade_name_val = inline
                    break
                # Check next line if present
                if idx + 1 < len(lines):
                    next_l = lines[idx + 1].strip()
                    if not re.search(r"^(?:\d+[\.\)]\s*)?(?:Additional|Constitution|Legal\s*Name|GSTIN|Date|Address|Period)\b", next_l, re.IGNORECASE):
                        cand_next = clean_field_value(next_l, "trade_name", "gst_certificate")
                        if is_valid_gst_trade_name(cand_next):
                            trade_name_val = cand_next
                break

    if trade_name_val:
        fields["trade_name"] = trade_name_val
        confidences["trade_name"] = find_line_confidence(fields["trade_name"], all_lines)
    else:
        fields["trade_name"] = None

    # Registration Date
    reg_date = re.search(
        r"(?:Date\s*of\s*(?:liability|Registration|Validity|issue\s*of\s*Certificate)|From)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if reg_date:
        fields["registration_date"] = reg_date.group(1)
        confidences["registration_date"] = find_line_confidence(reg_date.group(1), all_lines)

    # Constitution of Business
    const_match = re.search(
        r"(?:Constitution\s*of\s*Business)[\s:]*([A-Za-z][A-Za-z \t,\.\-&]{1,50}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Type|Date|Address|Particulars|Period)\b))",
        text,
        re.IGNORECASE,
    )
    if const_match:
        fields["constitution_of_business"] = clean_field_value(const_match.group(1), "constitution_of_business", "gst_certificate")
        confidences["constitution_of_business"] = find_line_confidence(fields["constitution_of_business"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 15. Certificate of Incorporation Extractor
# ==============================================================================

def extract_certificate_of_incorporation(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # CIN: 21 alphanumeric characters
    cin_match = re.search(r"\b([UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})\b", text)
    if cin_match:
        fields["cin"] = cin_match.group(1).upper()
        confidences["cin"] = find_line_confidence(cin_match.group(1), all_lines)

    # Company Name
    co_match = re.search(
        r"(?:hereby\s*certifies\s*that\s+|Name\s*of\s*the\s*Company[\s:]*)([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{2,80}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:is\s+incorporated|under\s+the|CIN|Corporate|Date|Given\s+under)\b))",
        text,
        re.IGNORECASE,
    )
    if co_match:
        fields["company_name"] = clean_field_value(co_match.group(1), "company_name", "certificate_of_incorporation")
        confidences["company_name"] = find_line_confidence(fields["company_name"], all_lines)

    # Date of Incorporation
    doi_match = re.search(
        r"(?:incorporated\s*(?:under\s*the\s*Companies\s*Act.*?)?on\s*this|Date\s*of\s*Incorporation[\s:]*|dated\s*this\s+)\s*([A-Za-z0-9 \t]{3,50}?\b\d{4}|\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if doi_match:
        fields["date_of_incorporation"] = clean_field_value(doi_match.group(1), "date_of_incorporation", "certificate_of_incorporation")
        confidences["date_of_incorporation"] = find_line_confidence(fields["date_of_incorporation"], all_lines)

    # Registrar Office
    roc_match = re.search(
        r"(?:Registrar\s*of\s*Companies|ROC)[\s,:-]*([A-Za-z][A-Za-z \t,\.\-]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Ministry|Government|Companies\s*Act|CIN|Corporate)\b))",
        text,
        re.IGNORECASE,
    )
    if roc_match:
        fields["registrar_office"] = clean_field_value(roc_match.group(1), "registrar_office", "certificate_of_incorporation")
        confidences["registrar_office"] = find_line_confidence(fields["registrar_office"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 16. Partnership Deed Extractor
# ==============================================================================

def extract_partnership_deed(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Firm Name
    firm_match = re.search(
        r"(?:name\s*(?:and|&)\s*style\s*of\s*[:\s]*|firm\s*name[\s:]*|M/S[\s\.]*)([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{2,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:hereinafter|having|place\s+of\s+business|partners|date|ratio)\b))",
        text,
        re.IGNORECASE,
    )
    if firm_match:
        fields["firm_name"] = clean_field_value(firm_match.group(1), "firm_name", "partnership_deed")
        confidences["firm_name"] = find_line_confidence(fields["firm_name"], all_lines)

    # Partner Names: Schema difference - returns List[str]
    partner_names: List[str] = []
    partner_patterns = [
        r"(?:Party\s*of\s*(?:the\s*)?(?:First|Second|Third|Fourth|[1-4])\s*Part|Partner\s*[0-9]+|Between\s+(?:Mr\.|Shri|Smt\.)?)[\s,:-]*([A-Za-z][A-Za-z \t.'-]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:hereinafter|son\s+of|daughter\s+of|wife\s+of|residing|party|and)\b))",
        r"(?:(?:^|\n)\s*(?:[0-9]+[\.\)]|Partner\s*[0-9]+[\.:]))\s*([A-Za-z][A-Za-z \t.'-]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:hereinafter|son\s+of|daughter\s+of|wife\s+of|residing|party|and)\b))",
    ]
    seen_partners = set()
    for pat in partner_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw_cand = clean_field_value(m.group(1), "partner_names", "partnership_deed")
            cand_norm = raw_cand.strip().upper()
            if cand_norm and cand_norm not in seen_partners and len(cand_norm) > 2:
                if not re.search(r"^(?:PARTNERSHIP|DEED|FIRM|THE|PROFIT|CAPITAL)\b", cand_norm):
                    seen_partners.add(cand_norm)
                    partner_names.append(raw_cand.strip())
    fields["partner_names"] = partner_names
    fields["raw_partner_names"] = partner_names
    conf_val = 0.95 if partner_names else 0.5
    confidences["partner_names"] = conf_val
    confidences["raw_partner_names"] = conf_val

    # Masked variant for PII minimisation: First name + initial per mask_person_name
    partner_names_masked = [mask_person_name(p) for p in partner_names if p]
    fields["partner_names_masked"] = partner_names_masked
    confidences["partner_names_masked"] = conf_val

    # Date of Deed
    deed_date = re.search(
        r"(?:executed\s*on\s*(?:this)?|dated\s*(?:this)?|date\s*of\s*(?:deed|execution)[\s:]*)(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Za-z]+\s+\d{4}|\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if deed_date:
        fields["date_of_deed"] = clean_field_value(deed_date.group(1), "date_of_deed", "partnership_deed")
        confidences["date_of_deed"] = find_line_confidence(fields["date_of_deed"], all_lines)

    # Profit Sharing Ratio
    ratio_match = re.search(
        r"(?:Profit\s*(?:and|&)\s*Loss\s*(?:sharing)?\s*ratio|profit\s*sharing\s*ratio)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t%:,\.\-/]{1,50}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:The\s+parties|Capital|Bank|Accounts|Duration)\b))",
        text,
        re.IGNORECASE,
    )
    if ratio_match:
        fields["profit_sharing_ratio"] = clean_field_value(ratio_match.group(1), "profit_sharing_ratio", "partnership_deed")
        confidences["profit_sharing_ratio"] = find_line_confidence(fields["profit_sharing_ratio"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 17. Rent Agreement Extractor
# ==============================================================================

def extract_rent_agreement(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Lessor Name
    lessor_match = re.search(
        r"(?:परवाना\s*देणारा|पट्टा\s*देणारा|घरमालक|LESSOR|LANDLORD|FIRST\s*PARTY)[\s,:-]*([^\r\n:]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:hereinafter|son\s+of|daughter\s+of|wife\s+of|residing|and|lessee|tenant|second\s+party|परवाना\s*घेणारा|पट्टा\s*घेणारा|भाडेकरू)\b))",
        text,
        re.IGNORECASE,
    )
    lessor_name = clean_field_value(lessor_match.group(1), "lessor_name", "rent_agreement") if lessor_match else None
    if lessor_name:
        fields["raw_lessor_name"] = lessor_name
        fields["lessor_name_masked"] = mask_person_name(lessor_name)
        confidences["lessor_name_masked"] = find_line_confidence(lessor_name, all_lines)

    # Lessee Name
    lessee_match = re.search(
        r"(?:परवाना\s*घेणारा|पट्टा\s*घेणारा|भाडेकरू|LESSEE|TENANT|SECOND\s*PARTY)[\s,:-]*([^\r\n:]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:hereinafter|son\s+of|daughter\s+of|wife\s+of|residing|and|lessor|premises|rent|परवाना\s*देणारा|भाडे)\b))",
        text,
        re.IGNORECASE,
    )
    lessee_name = clean_field_value(lessee_match.group(1), "lessee_name", "rent_agreement") if lessee_match else None
    if lessee_name:
        fields["raw_lessee_name"] = lessee_name
        fields["lessee_name_masked"] = mask_person_name(lessee_name)
        confidences["lessee_name_masked"] = find_line_confidence(lessee_name, all_lines)

    # Property Address
    addr_match = re.search(
        r"(?:Premises\s*(?:situated\s*at|at)|Demised\s*Premises|Property\s*Address)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t,\.\-\/]{5,100}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:Monthly\s*Rent|Rent|Period|Term|Security\s*Deposit)\b))",
        text,
        re.IGNORECASE,
    )
    prop_addr = clean_field_value(addr_match.group(1), "property_address", "rent_agreement") if addr_match else None
    if prop_addr:
        fields["raw_property_address"] = prop_addr
        fields["property_address_masked"] = mask_address(prop_addr)
        confidences["property_address_masked"] = find_line_confidence(prop_addr, all_lines)

    # Monthly Rent
    rent_match = re.search(
        r"(?:मासिक\s*भाडे|Monthly\s*Rent|Rent\s*per\s*month)[\s:]*(?:Rs\.?|INR|रु\.?)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        re.IGNORECASE,
    )
    if rent_match:
        fields["monthly_rent"] = rent_match.group(1).replace(",", "")
        confidences["monthly_rent"] = find_line_confidence(rent_match.group(1), all_lines)

    # Agreement Start Date
    start_match = re.search(
        r"(?:commencing\s*(?:from|on)?|lease\s*period\s*from|start\s*date)[\s:]*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if start_match:
        fields["agreement_start_date"] = clean_field_value(start_match.group(1), "agreement_start_date", "rent_agreement")
        confidences["agreement_start_date"] = find_line_confidence(fields["agreement_start_date"], all_lines)

    # Agreement End Date
    end_match = re.search(
        r"(?:expiring\s*(?:on)?|ending\s*(?:on)?|lease\s*period\s*to|valid\s*(?:till|to)|end\s*date)[\s:]*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if end_match:
        fields["agreement_end_date"] = clean_field_value(end_match.group(1), "agreement_end_date", "rent_agreement")
        confidences["agreement_end_date"] = find_line_confidence(fields["agreement_end_date"], all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 18. Form 16 Extractor
# ==============================================================================

def extract_form_16(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Employee PAN
    pan_match = re.search(
        r"(?:PAN\s*(?:of\s*(?:the\s*)?Employee)?|Employee\s*PAN)[\s:]*([A-Z]{5}[0-9]{4}[A-Z])",
        text,
        re.IGNORECASE,
    )
    if pan_match:
        fields["pan_number"] = pan_match.group(1).upper()
        confidences["pan_number"] = find_line_confidence(pan_match.group(1), all_lines)

    # Employer TAN
    tan_match = re.search(
        r"(?:TAN\s*(?:of\s*(?:the\s*)?Deductor|Employer)?|Deductor\s*TAN)[\s:]*([A-Z]{4}[0-9]{5}[A-Z])",
        text,
        re.IGNORECASE,
    )
    if tan_match:
        fields["tan_number"] = tan_match.group(1).upper()
        confidences["tan_number"] = find_line_confidence(tan_match.group(1), all_lines)

    # Employer Name
    er_match = re.search(
        r"(?:Name\s*and\s*address\s*of\s*the\s*Employer|Name\s*of\s*(?:the\s*)?(?:Employer|Deductor))[\s:]*([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{2,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:PAN|TAN|Employee|Address|Assessment|Period)\b))",
        text,
        re.IGNORECASE,
    )
    if er_match:
        fields["employer_name"] = clean_field_value(er_match.group(1), "employer_name", "form_16")
        confidences["employer_name"] = find_line_confidence(fields["employer_name"], all_lines)

    # Employee Name
    ee_match = re.search(
        r"(?:Name\s*of\s*(?:the\s*)?Employee)[\s:]*([A-Za-z][A-Za-z \t.'-]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:PAN|Designation|Address|Assessment|Period)\b))",
        text,
        re.IGNORECASE,
    )
    ee_name = clean_field_value(ee_match.group(1), "employee_name", "form_16") if ee_match else None
    if ee_name:
        fields["raw_employee_name"] = ee_name
        fields["employee_name_masked"] = mask_person_name(ee_name)
        confidences["employee_name_masked"] = find_line_confidence(ee_name, all_lines)

    # Assessment Year
    ay_match = re.search(
        r"(?:Assessment\s*Year|AY)[\s:]*([0-9]{4}\s*-\s*(?:[0-9]{4}|[0-9]{2}))",
        text,
        re.IGNORECASE,
    )
    if ay_match:
        fields["assessment_year"] = re.sub(r"\s+", "", ay_match.group(1))
        confidences["assessment_year"] = find_line_confidence(ay_match.group(1), all_lines)

    # Gross Salary
    gross_match = re.search(
        r"(?:Gross\s*Salary|Total\s*Salary|Gross\s*Total\s*Income)[\s:]*(?:Rs\.?|INR)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        re.IGNORECASE,
    )
    if gross_match:
        fields["gross_salary"] = gross_match.group(1).replace(",", "")
        confidences["gross_salary"] = find_line_confidence(gross_match.group(1), all_lines)

    # Tax Deducted
    tds_match = re.search(
        r"(?:Total\s*Tax\s*Deducted|Tax\s*Deducted\s*at\s*Source|TDS\s*Deducted)[\s:]*(?:Rs\.?|INR)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        re.IGNORECASE,
    )
    if tds_match:
        fields["tax_deducted"] = tds_match.group(1).replace(",", "")
        confidences["tax_deducted"] = find_line_confidence(tds_match.group(1), all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 19. Bank Passbook Extractor
# ==============================================================================

def extract_bank_passbook(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = normalize_devanagari_numbers(doc_res.full_text) or ""
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Bank Name
    bank_match = re.search(
        r"(?:(?:^|\r?\n)\s*([A-Za-z0-9\u0900-\u097F \t]{3,50}?(?:\bBANK(?:\s+OF\s+[A-Za-z \t]+)?|सहकारी\s*बँक|ग्रामीण\s*बँक|नागरी\s*सहकारी\s*बँक|विकास\s*बँक|को(?:-|ऑ)परेटीव्ह\s*बँक|बँक\s*मर्यादित|बैंक\s*लिमिटेड))|Bank\s*Name[\s:：]*([A-Za-z][A-Za-z \t,\.\-&]{2,40}?)|(?:बँकेचे\s*नाव|बैंक\s*का\s*नाम|बँक|बैंक)[\s:：]+([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F \t,\.\-&]{2,60}?))(?=[ \t]*(?:\r?\n|$|(?:Branch|IFSC|A/C|Account|शाखा|आयएफएससी|खाते|खाता)))",
        text,
        re.IGNORECASE,
    )
    if bank_match:
        cand = (bank_match.group(1) or bank_match.group(2) or bank_match.group(3) or "").strip()
        cand = cand.splitlines()[0].strip()
        cand = re.sub(r"\s*(?:SAVINGS\s+BANK\s+PASS\s*BOOK|PASS\s*BOOK|बचत\s*खाते\s*पासबुक|पासबुक).*", "", cand, flags=re.I).strip()
        if cand:
            fields["bank_name"] = clean_field_value(cand, "bank_name", "bank_passbook")
            confidences["bank_name"] = find_line_confidence(fields["bank_name"], all_lines)

    if not fields.get("bank_name"):
        # Check first prominent line with bank keywords
        header_bank = re.search(
            r"^[ \t]*([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F \t,\.\-&]{2,60}?(?:बँक|सहकारी\s*बँक|ग्रामीण\s*बँक|BANK))(?=[ \t]*(?:\r?\n|$))",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if header_bank:
            cand = header_bank.group(1).strip()
            cand = re.sub(r"\s*(?:SAVINGS\s+BANK\s+PASS\s*BOOK|PASS\s*BOOK|बचत\s*खाते\s*पासबुक|पासबुक).*", "", cand, flags=re.I).strip()
            if cand:
                fields["bank_name"] = clean_field_value(cand, "bank_name", "bank_passbook")
                confidences["bank_name"] = find_line_confidence(fields["bank_name"], all_lines)

    # Branch
    branch_match = re.search(
        r"(?:Branch\s*(?:Name)?|Branch\s*Code|शाखेचे\s*नाव|शाखा\s*कोड|शाखा)[\s:：]*([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F \t,\.\-&]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:IFSC|Account|A/C|MICR|CIF|Customer|आयएफएससी|खाते|खाता|ग्राहक|सीआयएफ)))",
        text,
        re.IGNORECASE,
    )
    if branch_match:
        fields["branch"] = clean_field_value(branch_match.group(1), "branch", "bank_passbook")
        confidences["branch"] = find_line_confidence(fields["branch"], all_lines)

    # IFSC
    ifsc_match = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", text, re.IGNORECASE)
    if ifsc_match:
        fields["ifsc"] = ifsc_match.group(1).upper()
        confidences["ifsc"] = find_line_confidence(fields["ifsc"], all_lines)

    # Account Number
    acc_match = re.search(
        r"(?:Account\s*(?:No|Number)|A/C\s*(?:No|Number)?|बचत\s*खाते\s*(?:क्रमांक|क्र|नं)?|खाते\s*(?:क्रमांक|क्र\.?|नं\.?)|खाता\s*(?:संख्या|क्रमांक|क्र\.?|नं\.?))[\s:：]*([0-9]{9,18})",
        text,
        re.IGNORECASE,
    )
    if acc_match:
        raw_acc = acc_match.group(1)
        fields["raw_account_number"] = raw_acc
        fields["account_number_masked"] = mask_account_number(raw_acc)
        confidences["account_number_masked"] = find_line_confidence(raw_acc, all_lines)

    # Account Holder Name
    holder_match = re.search(
        r"(?:Account\s*Holder(?:\s*Name)?|Name\s*of\s*Account\s*Holder|Customer\s*Name|खातेदाराचे\s*नाव|खातेदार\s*नाव|ग्राहकाचे\s*नाव|खाताधारक\s*का\s*नाम|खाताधारी\s*का\s*नाम|(?<!Branch\s)(?<!Bank\s)(?<!शाखेचे\s)(?<!बँकेचे\s)(?<!शाखा\s)(?<!बँक\s)\bName)[\s:：]*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F \t.'-]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:Account|A/C|IFSC|CIF|Customer|Branch|खाते|खाता|आयएफएससी|सीआयएफ|शाखा)))",
        text,
        re.IGNORECASE,
    )
    holder_name = clean_field_value(holder_match.group(1), "account_holder_name", "bank_passbook") if holder_match else None
    if holder_name:
        fields["raw_account_holder_name"] = holder_name
        fields["account_holder_name_masked"] = mask_person_name(holder_name)
        confidences["account_holder_name_masked"] = find_line_confidence(holder_name, all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 20. Property Tax Receipt Extractor
# ==============================================================================

def extract_property_tax_receipt(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # Property ID / Number
    pid_match = re.search(
        r"(?:मालमत्ता\s*(?:क्रमांक|क्र\.?)|इंडेक्स\s*(?:क्रमांक|क्र\.?)|Property\s*(?:ID|No|Number)|Assessment\s*No|Index\s*No|Tax\s*Bill\s*No)[\s:]*([A-Za-z0-9\-\/\u0966-\u096F]{4,25})",
        text,
        re.IGNORECASE,
    )
    if pid_match:
        raw_pid = normalize_devanagari_numbers(pid_match.group(1))
        fields["property_id"] = raw_pid
        confidences["property_id"] = find_line_confidence(pid_match.group(1), all_lines)

    # Owner Name
    owner_match = re.search(
        r"(?:मालकाचे\s*नाव|करदात्याचे\s*नाव|भोगवटादाराचे\s*नाव|Owner\s*Name|Name\s*of\s*(?:the\s*)?Owner|Tax\s*Payer\s*Name)[\s:]*([^\r\n:]{2,40}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z\u0900-\u097F]))(?:Property|Ward|Zone|Address|Assessment|Tax|Amount|मालमत्ता|प्रभाग|कर|रक्कम)\b))",
        text,
        re.IGNORECASE,
    )
    owner_name = clean_field_value(owner_match.group(1), "owner_name", "property_tax_receipt") if owner_match else None
    if owner_name:
        fields["raw_owner_name"] = owner_name
        fields["owner_name_masked"] = mask_person_name(owner_name)
        confidences["owner_name_masked"] = find_line_confidence(owner_name, all_lines)

    # Tax Amount Paid
    tax_match = re.search(
        r"(?:भरलेली\s*रक्कम|एकूण\s*रक्कम|Tax\s*Amount\s*Paid|Total\s*Amount\s*Paid|Amount\s*Paid)[\s:]*(?:Rs\.?|INR|रु\.?)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        re.IGNORECASE,
    )
    if tax_match:
        fields["tax_amount_paid"] = tax_match.group(1).replace(",", "")
        confidences["tax_amount_paid"] = find_line_confidence(tax_match.group(1), all_lines)

    # Payment Date
    date_match = re.search(
        r"(?:पावती\s*(?:दिनांक|तारीख)|दिनांक|Payment\s*Date|Receipt\s*Date|Date\s*of\s*Payment)[\s:]*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if date_match:
        fields["payment_date"] = date_match.group(1)
        confidences["payment_date"] = find_line_confidence(date_match.group(1), all_lines)

    # Assessment Year
    ay_match = re.search(
        r"(?:कर\s*आकारणी\s*वर्ष|आकारणी\s*वर्ष|Assessment\s*Year|AY)[\s:]*([0-9]{4}\s*-\s*(?:[0-9]{4}|[0-9]{2}))",
        text,
        re.IGNORECASE,
    )
    if ay_match:
        fields["assessment_year"] = re.sub(r"\s+", "", ay_match.group(1))
        confidences["assessment_year"] = find_line_confidence(ay_match.group(1), all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 21. IEC Certificate Extractor
# ==============================================================================

def extract_iec_certificate(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    text = doc_res.full_text
    all_lines = [line for page in doc_res.pages for line in page.lines]
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # IEC Number (10 alphanumeric / PAN)
    iec_match = re.search(r"(?:IEC\s*(?:Number|No)?|Import\s*Export\s*Code)[\s:]*([A-Z0-9]{10})", text, re.IGNORECASE)
    if iec_match:
        fields["iec_number"] = iec_match.group(1).upper()
        confidences["iec_number"] = find_line_confidence(iec_match.group(1), all_lines)

    # Entity Name
    ent_match = re.search(
        r"(?:Name\s*of\s*(?:the\s*)?(?:Firm|Entity|Company)|Entity\s*Name)[\s:]*([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{2,60}?)(?=[ \t]*(?:\r?\n|$|(?:\b|(?<=[a-z0-9A-Z]))(?:IEC|PAN|Date|Address|Branch|Director)\b))",
        text,
        re.IGNORECASE,
    )
    if ent_match:
        fields["entity_name"] = clean_field_value(ent_match.group(1), "entity_name", "iec_certificate")
        confidences["entity_name"] = find_line_confidence(fields["entity_name"], all_lines)

    # Issue Date
    date_match = re.search(
        r"(?:Date\s*of\s*Issue|Issue\s*Date)[\s:]*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if date_match:
        fields["issue_date"] = date_match.group(1)
        confidences["issue_date"] = find_line_confidence(date_match.group(1), all_lines)

    # PAN Number
    pan_match = re.search(r"(?:PAN|Permanent\s*Account\s*Number)[\s:]*([A-Z]{5}[0-9]{4}[A-Z])", text, re.IGNORECASE)
    if pan_match:
        fields["pan_number"] = pan_match.group(1).upper()
        confidences["pan_number"] = find_line_confidence(pan_match.group(1), all_lines)

    cleaned_fields = {k: clean_field_value(v) for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# 22. Income Certificate Normalization Helpers & Extractor
# ==============================================================================

def normalize_issue_date(raw_date: Optional[str]) -> Optional[str]:
    """
    Strict calendar-validated date normalizer for issue_date.
    Accepts:
    - DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
    - YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    - OCR variants where one extra trailing digit was attached (e.g. 2025-06-266 -> 2025-06-26).
    Valid normal dates remain unchanged.
    Invalid dates (e.g. 99/99/2025, 2025-02-30) are rejected (return None).
    """
    if not raw_date or not isinstance(raw_date, str):
        return None
    raw = raw_date.strip()

    def _is_valid(y: int, m: int, d: int) -> bool:
        if not (1900 <= y <= 2100):
            return False
        try:
            import datetime
            datetime.date(y, m, d)
            return True
        except (ValueError, OverflowError):
            return False

    # Check YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    m_iso = re.match(r"^(\d{4})([-\/\.])(\d{1,2})[-\/\.](\d{1,3})$", raw)
    if m_iso:
        y_s, sep, m_s, d_s = m_iso.groups()
        y, m = int(y_s), int(m_s)
        # Try as-is
        if len(d_s) <= 2 and _is_valid(y, m, int(d_s)):
            return raw
        # Try removing trailing digit if 3 digits
        if len(d_s) == 3 and _is_valid(y, m, int(d_s[:2])):
            return f"{y_s}{sep}{m_s}{sep}{d_s[:2]}"

    # Check DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    m_dmy = re.match(r"^(\d{1,3})([-\/\.])(\d{1,2})[-\/\.](\d{4,5})$", raw)
    if m_dmy:
        d_s, sep, m_s, y_s = m_dmy.groups()
        m = int(m_s)
        # Try as-is
        if len(d_s) <= 2 and len(y_s) == 4 and _is_valid(int(y_s), m, int(d_s)):
            return raw
        # Try removing trailing digit from year if 5 digits
        if len(d_s) <= 2 and len(y_s) == 5 and _is_valid(int(y_s[:4]), m, int(d_s)):
            return f"{d_s}{sep}{m_s}{sep}{y_s[:4]}"
        # Try removing trailing digit from day if 3 digits
        if len(d_s) == 3 and len(y_s) == 4 and _is_valid(int(y_s), m, int(d_s[:2])):
            return f"{d_s[:2]}{sep}{m_s}{sep}{y_s}"

    # Also handle month with extra digit YYYY-MMM-DD
    m_month_iso = re.match(r"^(\d{4})([-\/\.])(\d{3})[-\/\.](\d{1,2})$", raw)
    if m_month_iso:
        y_s, sep, m_s, d_s = m_month_iso.groups()
        if _is_valid(int(y_s), int(m_s[:2]), int(d_s)):
            return f"{y_s}{sep}{m_s[:2]}{sep}{d_s}"

    return None


MARATHI_LOCATION_NORMALIZATION: Dict[str, str] = {
    # Talukas / Tahsils
    "जुनर": "जुन्नर",
    "जुनर्": "जुन्नर",
    "जुन्नर": "जुन्नर",
    "हवेली": "हवेली",
    "खेड": "खेड",
    "आंबेगाव": "आंबेगाव",
    "शिरूर": "शिरूर",
    "शिरुर": "शिरूर",
    "बारामती": "बारामती",
    "इंदापूर": "इंदापूर",
    "इंदापुर": "इंदापूर",
    "दौंड": "दौंड",
    "भोर": "भोर",
    "वेल्हे": "वेल्हे",
    "मुळशी": "मुळशी",
    "मावळ": "मावळ",
    "पुरंदर": "पुरंदर",
    # Districts
    "पुण": "पुणे",
    "पुणे": "पुणे",
    "मुंबई": "मुंबई",
    "ठाणे": "ठाणे",
    "ठाण": "ठाणे",
    "नाशिक": "नाशिक",
    "नासिक": "नाशिक",
    "नागपूर": "नागपूर",
    "नागपुर": "नागपूर",
    "छत्रपती संभाजीनगर": "छत्रपती संभाजीनगर",
    "औरंगाबाद": "छत्रपती संभाजीनगर",
    "सातारा": "सातारा",
    "सांगली": "सांगली",
    "कोल्हापूर": "कोल्हापूर",
    "कोल्हापुर": "कोल्हापूर",
    "सोलापूर": "सोलापूर",
    "सोलापुर": "सोलापूर",
    "अहमदनगर": "अहमदनगर",
    "अहिल्यानगर": "अहिल्यानगर",
}


def normalize_marathi_location(name: Optional[str]) -> str:
    """Normalize common Marathi OCR misrecognitions of taluka / district names."""
    if not name or not isinstance(name, str):
        return ""
    clean = re.sub(r"^[:\s\.\-]+|[\s,\.:\-]+$", "", name).strip()
    return MARATHI_LOCATION_NORMALIZATION.get(clean, clean)


def normalize_issuing_authority(auth: Optional[str], taluka: Optional[str] = None) -> str:
    """Normalize issuing authority, resolving 'तहसीलदार <taluka>' where available."""
    if not auth or not isinstance(auth, str):
        return ""
    val = auth.strip()
    if "तहसीलदार" in val:
        m = re.search(r"तहसीलदार[\s:]*([A-Za-z\u0900-\u097F]+)", val)
        if m and m.group(1).strip():
            raw_sub = m.group(1).strip()
            norm_sub = normalize_marathi_location(raw_sub)
            return f"तहसीलदार {norm_sub}"
        elif taluka:
            norm_tal = normalize_marathi_location(taluka)
            return f"तहसीलदार {norm_tal}"
        return "तहसीलदार"
    return val


MARATHI_NAME_STOPS = (
    r"[ \t,]*(?:"
    r"या(?:ंना|ंस|ना|ंता|ंचे|ंचा|ंची|ंच्या|ंसाठी|स)|"
    r"राहणार|रा\.|राहणारे|"
    r"तह्सील|तहसील|तालुका|ता\.|जिल्हा|िजल्हा|िजला|"
    r"शैक्षणिक|शीमिणक|शिक्षणासाठी|कारणासाठी|कारणासावी|कारणास्तव|कामासाठी|"
    r"देण्यात|देपयाता|येत\s*आहे|नाही|"
    r"यांचा\s*मुलगा|यांची\s*मुलगी|"
    r"$|\r?\n"
    r")"
)


DEV_NUMS_TRANS = str.maketrans("०१२३४५६७८९", "0123456789")


def validate_financial_year(candidate: Optional[str]) -> Optional[str]:
    """
    Validate and normalize a candidate financial year string.
    Rules:
    - Normalizes Devanagari numerals (०-९ -> 0-9).
    - Unambiguous OCR substitutions: O/o -> 0, I/l/| -> 1, S/s -> 5.
    - Matches YYYY-YYYY or YYYY-YY.
    - Start year must be 1900-2100.
    - End year must be start_year + 1.
    - Normalizes YYYY-YY to YYYY-YYYY.
    - Rejects invalid candidates, arbitrary numeric garbage (e.g. Q028-2034),
      barcode fragments, dates, and hallucinations.
    """
    if not candidate or not isinstance(candidate, str):
        return None
    s = candidate.translate(DEV_NUMS_TRANS).strip()
    s = re.sub(r"(?<=[0-9\-/])[oO](?=[0-9\-/]|$)", "0", s)
    s = re.sub(r"(?<=[0-9\-/])[Il\|](?=[0-9\-/]|$)", "1", s)
    s = re.sub(r"(?<=[0-9\-/])[sS](?=[0-9\-/]|$)", "5", s)
    s = re.sub(r"^[oO](?=\d)", "0", s)
    s = re.sub(r"^[Il\|](?=\d)", "1", s)
    s = re.sub(r"^[sS](?=\d)", "5", s)

    m = re.search(r"\b(\d{4})\s*[-/]\s*(\d{2,4})\b", s)
    if not m:
        return None
    y1_str, y2_str = m.group(1), m.group(2)
    try:
        y1 = int(y1_str)
        if not (1900 <= y1 <= 2100):
            return None
        if len(y2_str) == 4:
            y2 = int(y2_str)
            if y2 == y1 + 1:
                return f"{y1}-{y2}"
        elif len(y2_str) == 2:
            y2_2d = int(y2_str)
            if y2_2d == (y1 + 1) % 100:
                return f"{y1}-{y1 + 1}"
    except (ValueError, TypeError):
        pass
    return None


def find_valid_financial_years(text: str) -> List[str]:
    """Find all valid, chronologically sorted financial years in text."""
    if not text:
        return []
    valid = []
    for m in re.finditer(r"\b([A-Za-z0-9०-९]{4}\s*[-/]\s*[A-Za-z0-9०-९]{2,4})\b", text):
        norm_fy = validate_financial_year(m.group(1))
        if norm_fy and norm_fy not in valid:
            valid.append(norm_fy)
    valid.sort(key=lambda y: int(y.split("-")[0]))
    return valid


def infer_financial_years_from_issue_date(issue_date_str: Optional[str], count: int = 3) -> List[str]:
    """
    Infer certified financial years from the certificate's issue date.
    Income certificates certify income for past completed financial years.
    If issued in FY Y-(Y+1) (months April-December, mm >= 4):
      The latest completed financial year is (Y-1)-Y.
    If issued in months January-March (mm < 4):
      The latest completed financial year is (Y-2)-(Y-1).
    Returns list of 'count' consecutive financial years ending at the latest completed FY.
    """
    if not issue_date_str or not isinstance(issue_date_str, str):
        return []
    m = re.search(r"\b(19\d{2}|20\d{2})[-/\.](\d{1,2})[-/\.](\d{1,2})\b", issue_date_str)
    if not m:
        m2 = re.search(r"\b(\d{1,2})[-/\.](\d{1,2})[-/\.](19\d{2}|20\d{2})\b", issue_date_str)
        if m2:
            year, month = int(m2.group(3)), int(m2.group(2))
        else:
            return []
    else:
        year, month = int(m.group(1)), int(m.group(2))

    if month >= 4:
        latest_end_year = year
    else:
        latest_end_year = year - 1

    years = []
    for i in range(count - 1, -1, -1):
        end_y = latest_end_year - i
        start_y = end_y - 1
        years.append(f"{start_y}-{end_y}")
    return years


def extract_income_certificate(doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Income Certificate extractor supporting English and Devanagari (Marathi/Hindi).
    Extracts:
    - certificate_number: 15-22 digits or labeled certificate/application number
    - applicant_name: applicant or beneficiary name
    - applicant_name_marathi: original Marathi name if Devanagari
    - address: resident address / village
    - district: district name
    - taluka: taluka / tahsil name
    - financial_year: financial year (e.g. '2024-2025' or '3 वर्षासाठी')
    - annual_income: annual income amount in digits (latest financial year)
    - income_amount_words: amount in words
    - issuing_authority: tahsildar / revenue authority
    - issue_date: issue or digital signature date
    """
    all_lines = [line for page in doc_res.pages for line in page.lines]
    text = doc_res.full_text
    norm_text = normalize_devanagari_numbers(text)
    fields: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    # 1. Certificate Number (15-22 digit barcode/app no or labeled)
    cert_m = re.search(
        r"(?:Certificate\s*(?:No\.?|Number)|Application\s*(?:No\.?|Number)|दाखला\s*(?:क्रमांक|क्र\.?|नं\.?)|प्रमाणपत्र\s*(?:क्रमांक|क्र\.?|नं\.?))[\s:]*([A-Za-z0-9\/-]{10,25})",
        text,
        re.IGNORECASE,
    )
    if cert_m:
        fields["certificate_number"] = cert_m.group(1).strip()
        confidences["certificate_number"] = find_line_confidence(fields["certificate_number"], all_lines)
    else:
        # Standalone numeric barcode / application number (15 to 22 digits)
        num_m = re.search(r"\b(\d{15,22})\b", norm_text)
        if num_m:
            fields["certificate_number"] = num_m.group(1).strip()
            confidences["certificate_number"] = find_line_confidence(fields["certificate_number"], all_lines)

    # 2. Issue Date (extracted early to assist financial_year validation/inference)
    excluded_years = {"2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035"}
    date_m = re.search(
        r"(?:Date(?:\s*of\s*Issue)?|Dated|दिनांक|तारीख)[\s:\.]*(\d{2,4}[-\/\.]\d{2}[-\/\.]\d{2,5})",
        norm_text,
        re.IGNORECASE,
    )
    if date_m:
        norm_d = normalize_issue_date(date_m.group(1).strip())
        if norm_d:
            fields["issue_date"] = norm_d
            confidences["issue_date"] = find_line_confidence(date_m.group(1), all_lines)
    if "issue_date" not in fields:
        for cand_d in re.findall(r"\b(\d{2,4}[-\/\.]\d{2}[-\/\.]\d{2,5})\b", norm_text):
            norm_d = normalize_issue_date(cand_d)
            if norm_d and norm_d not in excluded_years:
                fields["issue_date"] = norm_d
                confidences["issue_date"] = find_line_confidence(cand_d, all_lines)
                break

    # 3. Annual Income from tabular structure or text
    norm_text_income = re.sub(r"[yY][oO0]{1,2}[,\.][oO0]{3}", "50,000", norm_text)
    norm_text_income = re.sub(r"[gG][oO0]{1,2}[,\.][oO0]{3}", "40,000", norm_text_income)

    # Look for tabular rows e.g. '2023-2024 85,000' or '2024-2025 40,000'
    inc_rows = re.findall(
        r"\b([A-Za-z0-9०-९]\w{2,4}\s*[-/]\s*[A-Za-z0-9०-९]?\w{2,4})\b[ \t\n]*(?:Rs\.?|INR|रुपये)?\s*([\d,]{4,10}(?:\.\d{2})?)[ \t]*([^\n\d]+)?",
        norm_text_income,
        re.IGNORECASE,
    )
    if inc_rows:
        latest_row = inc_rows[-1]
        fields["annual_income"] = latest_row[1].replace(",", "")
        if latest_row[2] and any(w in latest_row[2] for w in ("हजार", "लाख", "मात्र", "Only", "Thousand", "Lakh", "मान", "फक्त")):
            fields["income_amount_words"] = latest_row[2].strip()
        confidences["annual_income"] = find_line_confidence(latest_row[1], all_lines)

    # If table block is present (e.g. between 'खालीलप्रमाणे' and 'सदरचा दाखला')
    if "annual_income" not in fields:
        table_m = re.search(
            r"(?:उलनखालीलममाणे|खालीलप्रमाणे|खालील\s*माणे|वार्षिक\s*उत्पन्न)[^\n]*?\n(.*?)(?=सदरचादाखला|सदरचा\s*दाखला|कारणासाठी|स्वाक्षरी|$)",
            norm_text_income,
            re.DOTALL,
        )
        if table_m:
            block = table_m.group(1)
            tbl_amounts = [
                a.replace(",", "")
                for a in re.findall(r"\b([\d,]{4,10})\b", block)
                if a.replace(",", "") not in excluded_years
            ]
            valid_tbl = [a for a in tbl_amounts if a.isdigit() and int(a) >= 5000]
            if valid_tbl:
                latest_amt = valid_tbl[-1]
                fields["annual_income"] = latest_amt
                confidences["annual_income"] = find_line_confidence(latest_amt, all_lines)

    # 4. Financial Year: Collect all valid candidates and select latest chronologically
    valid_fys = find_valid_financial_years(norm_text)
    if valid_fys:
        fields["financial_year"] = valid_fys[-1]
        confidences["financial_year"] = find_line_confidence(valid_fys[-1], all_lines)
    else:
        # Check explicit labeled year e.g. "Financial Year: 2024-2025" or "वर्ष 2024-25"
        fy_m = re.search(
            r"(?:Financial\s*Year|Year|वर्ष|आर्थिक\s*वर्ष)[\s:]*([A-Za-z0-9०-९]{4}\s*[-/]\s*[A-Za-z0-9०-९]{2,4})",
            norm_text,
            re.IGNORECASE,
        )
        if fy_m:
            cand_fy = validate_financial_year(fy_m.group(1))
            if cand_fy:
                fields["financial_year"] = cand_fy
                confidences["financial_year"] = find_line_confidence(fy_m.group(1), all_lines)

    # If valid financial year not found in text, infer from issue date
    if "financial_year" not in fields and fields.get("issue_date"):
        inferred_fys = infer_financial_years_from_issue_date(fields["issue_date"], count=3)
        if inferred_fys:
            fields["financial_year"] = inferred_fys[-1]
            confidences["financial_year"] = confidences.get("issue_date", 0.85)

    # Check period clause e.g. '3 वर्षासाठी'
    if "financial_year" not in fields:
        per_m = re.search(r"([१२३४५0-9]\s*वर्षांसाठी|[१२३४५0-9]\s*वर्षासाठी)", text)
        if per_m:
            fields["financial_year"] = per_m.group(1).strip()
            confidences["financial_year"] = find_line_confidence(fields["financial_year"], all_lines)

    if "annual_income" not in fields:
        # A. Explicit currency symbol/word prefix
        rs_m = re.search(r"(?:Rs\.?|INR|रुपये|रु\.?)\s*([\d,]{4,10}(?:\.\d{2})?)", norm_text_income, re.IGNORECASE)
        if rs_m:
            val = rs_m.group(1).replace(",", "")
            if val not in excluded_years:
                fields["annual_income"] = val
                confidences["annual_income"] = find_line_confidence(rs_m.group(1), all_lines)

    if "annual_income" not in fields:
        # B. Keyword with optional currency
        inc_m = re.search(
            r"(?:Annual\s*Income|Total\s*Income|वार्षिक\s*उत्पन्न|एकूण\s*उत्पन्न)[^\n\d]{0,40}?(?:Rs\.?|INR|रुपये|रु\.?)?[\s:\.]*([\d,]{4,10}(?:\.\d{2})?)",
            norm_text_income,
            re.IGNORECASE,
        )
        if inc_m:
            val = inc_m.group(1).replace(",", "")
            if val not in excluded_years:
                fields["annual_income"] = val
                confidences["annual_income"] = find_line_confidence(inc_m.group(1), all_lines)

    if "annual_income" not in fields:
        # C. Standalone amount (e.g. 50,000 from normalized Devanagari numerals)
        stand_amounts = re.findall(r"\b([\d,]{4,10})\b", norm_text_income)
        for s in stand_amounts:
            val = s.replace(",", "")
            if val not in excluded_years:
                try:
                    if int(float(val)) >= 5000:
                        fields["annual_income"] = val
                        confidences["annual_income"] = find_line_confidence(s, all_lines)
                        break
                except ValueError:
                    pass

    # Income amount in words
    if "income_amount_words" not in fields:
        words_m = re.search(
            r"(?:Rupees|रुपये|अक्षरी)[\s:]+([A-Za-z\u0900-\u097F\s]+?(?:Only|मात्र|मान|फक्त))\b",
            text,
            re.IGNORECASE,
        )
        if words_m:
            fields["income_amount_words"] = words_m.group(1).strip()
        else:
            standalone_words = re.search(r"([A-Za-z\u0900-\u097F\s]{2,30}?(?:हजार|लाख|Thousand|Lakh)[^\n]*?(?:मान|मात्र|फक्त|Only|\b))", text)
            if standalone_words:
                fields["income_amount_words"] = standalone_words.group(1).strip()

    # 3. Applicant Name
    eng_name_m = re.search(
        r"(?:certify\s*that|Name\s*of\s*Applicant|Applicant\s*Name|Applicant)[\s:]*(?:Mr\.?|Mrs\.?|Ms\.?|Kumar|Kumari)?\s*([A-Za-z][A-Za-z\s\.\'-]{2,40}?)(?=[ \t]*(?:\r?\n|$|Address|Residing|Village|Taluka|District|Son|Daughter))",
        text,
        re.IGNORECASE,
    )
    if eng_name_m and eng_name_m.group(1).strip():
        fields["applicant_name"] = eng_name_m.group(1).strip()
        confidences["applicant_name"] = find_line_confidence(fields["applicant_name"], all_lines)
    else:
        # Beneficiary or applicant in Marathi
        beneficiary_m = re.search(
            r"(?:यांचा\s*मुलगा|यांची\s*मुलगी|मुलगा|मुलगी|विनंतीवरून|विनंतीवरुन|दाखला)?\s*"
            r"((?:कुमार|कुमारी)[A-Za-z\u0900-\u097F\s]{2,35}?)"
            r"(?=" + MARATHI_NAME_STOPS + r")",
            text,
        )
        if not beneficiary_m:
            beneficiary_m = re.search(
                r"(?:यांचा\s*मुलगा|यांची\s*मुलगी|मुलगा|मुलगी)\s*"
                r"((?:श्री\.?|श्रीमती|ञी\.?)[A-Za-z\u0900-\u097F\s]{2,35}?)"
                r"(?=" + MARATHI_NAME_STOPS + r")",
                text,
            )
        if beneficiary_m:
            cand_name = beneficiary_m.group(1).strip()
            fields["applicant_name"] = cand_name
            fields["applicant_name_marathi"] = cand_name
            confidences["applicant_name"] = find_line_confidence(cand_name, all_lines)
        else:
            head_m = re.search(
                r"(?:प्रमाणित\s*करण्यात\s*येते\s*की|अमािणतकरणयात|अमाणतकरणयात|दाखला\s*देण्यात\s*येतो\s*की)[^\n]*?"
                r"(?:की[\.\s]*)?((?:श्री\.?|श्रीमती|कुमार|कुमारी|ञी\.?|शी\.)[A-Za-z\u0900-\u097F\s\.\'-]{2,35}?)"
                r"(?=" + MARATHI_NAME_STOPS + r")",
                text,
            )
            if head_m:
                cand_name = head_m.group(1).strip()
                fields["applicant_name"] = cand_name
                fields["applicant_name_marathi"] = cand_name
                confidences["applicant_name"] = find_line_confidence(cand_name, all_lines)

    # 4. Address & Village
    addr_m = re.search(
        r"(?:राहणार\s*गाव|राहणार\s*मु\.|राहणार|रा\.|Residing\s*at|Village)[\s:]*"
        r"([A-Za-z\u0900-\u097F\s,\.\'-]{2,50}?)"
        r"(?=[ \t,]*(?:तह्सील|तहसील|तहसिल|तालुका|ता\.|(?<=\s)ता(?=\s)|जिल्हा|िजल्हा|िजलहा|िजला|जि\.|Taluka|District|Tahsil|$|\n))",
        text,
        re.IGNORECASE,
    )
    if addr_m:
        raw_addr = addr_m.group(1).strip()
        clean_addr = re.sub(r"^(?:गाव|मु\.|मु|ग्राम)[\s:\.]*", "", raw_addr).strip()
        clean_addr = re.sub(r"[\s,]+(?:तह्सील|तहसील|तालुका|जिल्हा|िजला).*$", "", clean_addr).strip()
        if clean_addr:
            fields["address"] = clean_addr
            confidences["address"] = find_line_confidence(clean_addr, all_lines)

    # 5. Taluka & District
    tal_m = re.search(
        r"(?:\b|(?<=[^A-Za-z\u0900-\u097F]))(?:तह्सील(?!दार)|तहसील(?!दार)|तहसिल(?!दार)|तालुका|ता\.|(?<=\s)ता(?=\s)|Taluka\b|Tahsil(?!dar)\b)[\s:\.]*([A-Za-z\u0900-\u097F]{2,20}?)(?=[ \t,]*(?:िजल्हा|जिल्हा|िजलहा|िजला|जि\.|District|,|$|\n))",
        text,
        re.IGNORECASE,
    )
    if tal_m:
        raw_tal = tal_m.group(1).strip()
        fields["taluka"] = normalize_marathi_location(raw_tal)
        confidences["taluka"] = find_line_confidence(raw_tal, all_lines)

    dist_m = re.search(
        r"(?:\b|(?<=[^A-Za-z\u0900-\u097F]))(?:िजल्हा|जिल्हा|िजलहा|िजला|जि\.|District)[\s:\.]*([A-Za-z\u0900-\u097F]{2,20})",
        text,
        re.IGNORECASE,
    )
    if dist_m:
        raw_dist = re.sub(r"[,\.\s]+$", "", dist_m.group(1).strip())
        fields["district"] = normalize_marathi_location(raw_dist)
        confidences["district"] = find_line_confidence(raw_dist, all_lines)

    # 6. Issuing Authority
    if "तहसीलदार" in text:
        fields["issuing_authority"] = normalize_issuing_authority("तहसीलदार", taluka=fields.get("taluka"))
        confidences["issuing_authority"] = find_line_confidence("तहसीलदार", all_lines)
    else:
        auth_m = re.search(
            r"(?:Digitally\s*signed\s*by[\s:]*([A-Za-z\s]+)|Tahsildar(?:[ \t]+([A-Za-z]+))?|Executive\s*Magistrate|Sub-Divisional\s*Officer)",
            text,
            re.IGNORECASE,
        )
        if auth_m:
            val = auth_m.group(1) or auth_m.group(0)
            fields["issuing_authority"] = re.sub(r"[ \t]+", " ", val.strip())
            confidences["issuing_authority"] = find_line_confidence(fields["issuing_authority"], all_lines)

    # 7. Issue Date
    if "issue_date" not in fields:
        date_m = re.search(
            r"(?:Date(?:\s*of\s*Issue)?|Dated|दिनांक|तारीख)[\s:\.]*(\d{2,4}[-\/\.]\d{2}[-\/\.]\d{2,5})",
            norm_text,
            re.IGNORECASE,
        )
        if date_m:
            norm_d = normalize_issue_date(date_m.group(1).strip())
            if norm_d:
                fields["issue_date"] = norm_d
                confidences["issue_date"] = find_line_confidence(date_m.group(1), all_lines)
    if "issue_date" not in fields:
        for cand_d in re.findall(r"\b(\d{2,4}[-\/\.]\d{2}[-\/\.]\d{2,5})\b", norm_text):
            norm_d = normalize_issue_date(cand_d)
            if norm_d and norm_d not in excluded_years:
                fields["issue_date"] = norm_d
                confidences["issue_date"] = find_line_confidence(cand_d, all_lines)
                break

    # 8. Document Type and Review requirement
    fields["document_type"] = "Income Certificate"
    confidences["document_type"] = 1.0

    core_fields = ["certificate_number", "applicant_name", "annual_income"]
    has_all_core = all(fields.get(f) and str(fields[f]).strip() for f in core_fields)
    low_confidence = any(confidences.get(f, 0.0) < 0.60 for f in core_fields if f in fields)
    fields["needs_manual_review"] = (not has_all_core) or low_confidence

    fields.setdefault("certificate_number", "")
    fields.setdefault("applicant_name", "")
    fields.setdefault("annual_income", "")
    fields.setdefault("financial_year", "")

    cleaned_fields = {k: clean_field_value(v, k, "income_certificate") if isinstance(v, str) else v for k, v in fields.items()}
    return cleaned_fields, confidences


# ==============================================================================
# Dispatcher & PII Minimisation Enforcer
# ==============================================================================

EXTRACTOR_REGISTRY = {
    "pan": extract_pan,
    "aadhaar": extract_aadhaar,
    "cancelled_cheque": extract_cancelled_cheque,
    "udyam": extract_udyam,
    "fssai": extract_fssai,
    "shop_establishment": extract_shop_establishment,
    "bank_statement": extract_bank_statement,
    "salary_slip": extract_salary_slip,
    "utility_bill": extract_utility_bill,
    "passport": extract_passport,
    "voter_id": extract_voter_id,
    "driving_licence": extract_driving_licence,
    "itr": extract_itr,
    "gst_certificate": extract_gst_certificate,
    "certificate_of_incorporation": extract_certificate_of_incorporation,
    "partnership_deed": extract_partnership_deed,
    "rent_agreement": extract_rent_agreement,
    "form_16": extract_form_16,
    "bank_passbook": extract_bank_passbook,
    "property_tax_receipt": extract_property_tax_receipt,
    "iec_certificate": extract_iec_certificate,
    "income_certificate": extract_income_certificate,
}


# Strict PII allowlist: fields not in allowlist are strictly stripped before leaving the service
PII_ALLOWLIST = {
    "bank_statement": {"bank_name", "account_number_masked", "statement_period", "opening_balance", "closing_balance", "transactions"},
    "salary_slip": {"employer_name", "employee_name_masked", "net_pay", "pay_period"},
    "utility_bill": {"utility_provider", "consumer_number", "bill_date", "due_date", "bill_amount"},
    "rent_agreement": {"lessor_name_masked", "lessee_name_masked", "property_address_masked", "monthly_rent", "agreement_start_date", "agreement_end_date"},
    "form_16": {"employer_name", "employee_name_masked", "pan_number", "tan_number", "assessment_year", "gross_salary", "tax_deducted"},
    "bank_passbook": {"bank_name", "branch", "ifsc", "account_number_masked", "account_holder_name_masked"},
    "property_tax_receipt": {"property_id", "owner_name_masked", "tax_amount_paid", "payment_date", "assessment_year"},
    "partnership_deed": {"firm_name", "partner_names_masked", "date_of_deed", "profit_sharing_ratio"},
}


LANGUAGE_META_FIELDS = {
    "detected_languages",
    "partial_language_coverage",
    "language_review_required",
    "language_coverage_notes",
}

CORE_FIELDS_PER_DOC_TYPE = {
    "shop_establishment": {"registration_number", "establishment_name"},
    "udyam": {"udyam_registration_number", "enterprise_name"},
    "property_tax_receipt": {"property_id", "tax_amount_paid"},
    "rent_agreement": {"monthly_rent", "lessor_name_masked"},
    "aadhaar": {"aadhaar_number", "name"},
    "utility_bill": {"consumer_number", "bill_amount"},
    "salary_slip": {"employer_name", "net_pay"},
    "bank_passbook": {"account_number_masked", "bank_name"},
    "income_certificate": {"certificate_number", "applicant_name", "annual_income"},
}



def check_language_coverage(doc_type: str, text: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if the document contains Devanagari script content and whether
    core fields were successfully extracted.
    If core fields are missing on a Devanagari document, flags:
    partial_language_coverage=True, language_review_required=True
    to avoid hallucinating unverified semantic structures.
    """
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    has_devanagari = devanagari_chars >= 10

    detected_languages = ["en"]
    if has_devanagari:
        # Note: Hindi and Marathi share the Devanagari script
        detected_languages.append("devanagari")

    res: Dict[str, Any] = {
        "detected_languages": detected_languages,
        "partial_language_coverage": False,
        "language_review_required": False,
        "language_coverage_notes": None,
    }

    if not has_devanagari:
        return res

    core_req = CORE_FIELDS_PER_DOC_TYPE.get(doc_type)
    if not core_req:
        return res

    missing = []
    for f in core_req:
        # check both masked and unmasked variants
        raw_key = f"raw_{f.replace('_masked', '')}" if f.endswith("_masked") else f
        if not fields.get(f) and not fields.get(raw_key):
            missing.append(f)

    if missing:
        res["partial_language_coverage"] = True
        res["language_review_required"] = True
        res["language_coverage_notes"] = (
            f"Document contains Devanagari script text ({devanagari_chars} characters), but core field(s) "
            f"could not be extracted: {', '.join(sorted(missing))}. Manual review recommended."
        )
    else:
        res["language_coverage_notes"] = (
            f"Devanagari script text detected ({devanagari_chars} characters); all core fields extracted successfully."
        )

    return res


def extract_document_fields_raw(doc_type: str, doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Run appropriate extractor for doc_type and return RAW fields (including unmasked
    values like raw_aadhaar and employee_name) for verification and cross-checking.
    """
    extractor = EXTRACTOR_REGISTRY.get(doc_type)
    if not extractor:
        return {}, {}
    fields, confidences = extractor(doc_res)
    cleaned_fields = {
        k: (clean_field_value(v, field_name=k, doc_type=doc_type) if isinstance(v, str) else v)
        for k, v in fields.items()
    }
    # Run language coverage check
    coverage_meta = check_language_coverage(doc_type, doc_res.full_text, cleaned_fields)
    cleaned_fields.update(coverage_meta)
    for k in coverage_meta:
        confidences[k] = 1.0
    return cleaned_fields, confidences


def sanitize_extracted_fields(doc_type: str, fields: Dict[str, Any], confidences: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Sanitize extracted fields AFTER verification and cross-check.
    - Applies strict PII allowlist for bank_statement, salary_slip, utility_bill.
    - Preserves language coverage metadata fields.
    - Strictly strips all 'raw_' prefix keys and transient unmasked identifiers.
    """
    # 1. Apply PII allowlist filtering if applicable
    if doc_type in PII_ALLOWLIST:
        allowed = PII_ALLOWLIST[doc_type] | LANGUAGE_META_FIELDS
        filtered_fields = {k: v for k, v in fields.items() if k in allowed}
        filtered_conf = {k: v for k, v in confidences.items() if k in allowed}
        return filtered_fields, filtered_conf

    # 2. General cleanup: strip any key starting with 'raw_' or transient unmasked names
    cleaned_fields = {}
    cleaned_conf = {}
    transient_sensitive_keys = {
        "raw_aadhaar", "raw_account_number", "raw_employee_name",
        "raw_lessor_name", "raw_lessee_name", "raw_property_address",
        "raw_owner_name", "raw_account_holder_name",
        "raw_partner_names", "partner_names",
    }

    for k, v in fields.items():
        if k.startswith("raw_") or k in transient_sensitive_keys:
            continue
        cleaned_fields[k] = v

    for k, v in confidences.items():
        if k.startswith("raw_") or k in transient_sensitive_keys:
            continue
        cleaned_conf[k] = v

    return cleaned_fields, cleaned_conf


def extract_document_fields(doc_type: str, doc_res: OCRDocumentResult) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Convenience method: runs raw extraction and immediately sanitizes fields.
    For internal workflows, prefer extract_document_fields_raw -> verify -> sanitize_extracted_fields.
    """
    raw_fields, confidences = extract_document_fields_raw(doc_type, doc_res)
    return sanitize_extracted_fields(doc_type, raw_fields, confidences)

