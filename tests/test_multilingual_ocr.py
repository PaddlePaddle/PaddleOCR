"""
tests/test_multilingual_ocr.py
Comprehensive test suite for Marathi/Hindi/English multi-language OCR support:
- Per-document-type language configuration (DOC_TYPE_LANGUAGES)
- Devanagari neural recognizer availability & RapidOCR parameter patching
- Dual-pass line merging with Devanagari character priority
- Devanagari numeral normalization
- Field extraction for Marathi & bilingual documents:
  - Shop & Establishment (MH Form-G)
  - Udyam Registration (Marathi/Hindi)
  - Property Tax Receipt (BMC/PMC Marathi)
  - Rent Agreement (Maharashtra Leave & License)
  - Aadhaar Card (Bilingual Devanagari)
- Partial language coverage & language review flagging for unverified structures
- Zero regression on English-only documents
- Document signature detection on Devanagari text
"""

import pytest
from ocr_engine import (
    DOC_TYPE_LANGUAGES,
    OCREngine,
    OCRDocumentResult,
    OCRLine,
    OCRPageResult,
    get_devanagari_ocr_instance,
    get_languages_for_doc_type,
    get_ocr_engine_info,
    merge_ocr_lines,
)
from extractors import (
    check_language_coverage,
    extract_aadhaar,
    extract_bank_passbook,
    extract_document_fields,
    extract_document_fields_raw,
    extract_pan,
    extract_property_tax_receipt,
    extract_rent_agreement,
    extract_salary_slip,
    extract_shop_establishment,
    extract_udyam,
    extract_utility_bill,
    mask_account_number,
    mask_person_name,
    normalize_devanagari_numbers,
)
from verifier import detect_document_type


def _make_doc_res(text: str, default_conf: float = 0.95) -> OCRDocumentResult:
    lines = [OCRLine(text=l, confidence=default_conf) for l in text.splitlines() if l.strip()]
    page = OCRPageResult(page_num=1, full_text=text, lines=lines, average_confidence=default_conf)
    return OCRDocumentResult(pages=[page], full_text=text, average_confidence=default_conf, ocr_required=True)


# ==============================================================================
# 1. Language Configuration & Engine Initialization Tests
# ==============================================================================

def test_doc_type_languages_configuration():
    """Verify DOC_TYPE_LANGUAGES returns Devanagari for target docs and English-only for others."""
    assert get_languages_for_doc_type("shop_establishment") == ["en", "mr"]
    assert get_languages_for_doc_type("udyam") == ["en", "hi", "mr"]
    assert get_languages_for_doc_type("property_tax_receipt") == ["en", "mr"]
    assert get_languages_for_doc_type("rent_agreement") == ["en", "mr"]
    assert get_languages_for_doc_type("aadhaar") == ["en", "hi"]
    assert get_languages_for_doc_type("utility_bill") == ["en", "hi", "mr"]
    assert get_languages_for_doc_type("salary_slip") == ["en", "hi", "mr"]
    assert get_languages_for_doc_type("bank_passbook") == ["en", "hi", "mr"]

    # English-only documents
    assert get_languages_for_doc_type("pan") == ["en"]
    assert get_languages_for_doc_type("bank_statement") == ["en"]
    assert get_languages_for_doc_type("gst_certificate") == ["en"]
    assert get_languages_for_doc_type(None) == ["en"]
    assert get_languages_for_doc_type("unknown_type") == ["en"]

    # OCREngine staticmethod parity
    assert OCREngine.get_languages_for_doc_type("shop_establishment") == ["en", "mr"]
    assert OCREngine.get_languages_for_doc_type("salary_slip") == ["en", "hi", "mr"]
    assert OCREngine.get_languages_for_doc_type("bank_passbook") == ["en", "hi", "mr"]
    assert OCREngine.get_languages_for_doc_type("utility_bill") == ["en", "hi", "mr"]


def test_devanagari_ocr_engine_initialization():
    """Verify that the Devanagari neural recognizer loads without error."""
    engine = get_devanagari_ocr_instance()
    assert engine is not None, "Devanagari OCR instance must be successfully initialized"


# ==============================================================================
# 2. Dual-Pass Line Merging Tests
# ==============================================================================

def test_merge_ocr_lines_devanagari_priority():
    """
    Verify that when English model outputs ASCII gibberish over Devanagari text,
    the Devanagari candidate is prioritized when confidence is adequate.
    """
    # Overlapping bounding box: [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    bbox_common = [[10, 10], [200, 10], [200, 40], [10, 40]]

    lines_en = [
        OCRLine(text="HRI 9 MAHARASHTRA", confidence=0.55, bbox=bbox_common),
        OCRLine(text="REGISTRATION NO: 12345", confidence=0.96, bbox=[[10, 60], [200, 60], [200, 90], [10, 90]]),
    ]
    lines_dev = [
        OCRLine(text="महाराष्ट्र शासन", confidence=0.88, bbox=bbox_common),
    ]

    merged = merge_ocr_lines(lines_en, lines_dev)
    merged_texts = [l.text for l in merged]

    # Devanagari line should be preferred over "HRI 9 MAHARASHTRA"
    assert "महाराष्ट्र शासन" in merged_texts
    assert "HRI 9 MAHARASHTRA" not in merged_texts
    # Non-overlapping line should be preserved
    assert "REGISTRATION NO: 12345" in merged_texts
    assert len(merged) == 2


def test_merge_ocr_lines_non_overlapping():
    """Verify non-overlapping lines from both passes are kept and sorted top-to-bottom."""
    lines_en = [
        OCRLine(text="GOVERNMENT OF MAHARASHTRA", confidence=0.95, bbox=[[10, 10], [200, 10], [200, 30], [10, 30]]),
    ]
    lines_dev = [
        OCRLine(text="दुकान आणि आस्थापना नोंदणी", confidence=0.92, bbox=[[10, 50], [200, 50], [200, 70], [10, 70]]),
    ]

    merged = merge_ocr_lines(lines_en, lines_dev)
    assert len(merged) == 2
    assert merged[0].text == "GOVERNMENT OF MAHARASHTRA"
    assert merged[1].text == "दुकान आणि आस्थापना नोंदणी"


# ==============================================================================
# 3. Devanagari Numerals Normalization Tests
# ==============================================================================

def test_normalize_devanagari_numbers():
    """Verify conversion of Devanagari digits to ASCII digits."""
    assert normalize_devanagari_numbers("०१२३४५६७८९") == "0123456789"
    assert normalize_devanagari_numbers("नोंदणी क्र. १८३९/२०२३") == "नोंदणी क्र. 1839/2023"
    assert normalize_devanagari_numbers("Rs. 15000") == "Rs. 15000"
    assert normalize_devanagari_numbers(None) is None


# ==============================================================================
# 4. Field Extraction Tests (Marathi & Bilingual Documents)
# ==============================================================================

def test_extract_shop_establishment_marathi():
    """Verify extraction of Maharashtra Form-G / Gumasta certificate in Marathi."""
    marathi_text = """
    महाराष्ट्र शासन
    कामगार आयुक्त कार्यालय, मुंबई
    दुकान आणि आस्थापना नोंदणी प्रमाणपत्र
    नोंदणी क्रमांक: MH-40-2023-009823
    आस्थापनेचे नाव: सह्याद्री किराणा स्टोअर्स
    मालकाचे नाव: विठ्ठल तुकाराम पाटील
    व्यवसायाचे स्वरूप: किरकोळ किराणा मालाची विक्री
    पत्ता: दुकान क्र. ४, स्टेशन रोड, दादर, मुंबई - ४०००२८
    """
    doc_res = _make_doc_res(marathi_text)
    fields, confs = extract_shop_establishment(doc_res)

    assert fields.get("state") == "MH"
    assert fields.get("state_name") == "Maharashtra"
    assert fields.get("registration_number") == "MH-40-2023-009823"
    assert fields.get("establishment_name") == "सह्याद्री किराणा स्टोअर्स"
    assert fields.get("employer_name") == "विठ्ठल तुकाराम पाटील"
    assert fields.get("nature_of_business") == "किरकोळ किराणा मालाची विक्री"

    # End-to-end check with language coverage
    e2e_fields, _ = extract_document_fields("shop_establishment", doc_res)
    assert e2e_fields["partial_language_coverage"] is False
    assert e2e_fields["language_review_required"] is False
    assert "devanagari" in e2e_fields["detected_languages"]


def test_extract_udyam_marathi_hindi():
    """Verify extraction of Udyam registration in Marathi/Hindi with enum mappings."""
    udyam_text = """
    भारत सरकार
    सूक्ष्म, लघु व मध्यम उद्योग मंत्रालय
    उद्यम नोंदणी प्रमाणपत्र
    उद्यम नोंदणी क्रमांक: UDYAM-MH-01-0045678
    उद्यमाचे नाव: स्वराज इंजिनिअरिंग वर्क्स
    उद्यमाचा प्रकार: सूक्ष्म
    मुख्य कार्यकलाप: उत्पादन
    """
    doc_res = _make_doc_res(udyam_text)
    fields, confs = extract_udyam(doc_res)

    assert fields.get("udyam_registration_number") == "UDYAM-MH-01-0045678"
    assert fields.get("enterprise_name") == "स्वराज इंजिनिअरिंग वर्क्स"
    assert fields.get("enterprise_type") == "Micro"  # mapped from सूक्ष्म
    assert fields.get("major_activity") == "Manufacturing"  # mapped from उत्पादन

    e2e_fields, _ = extract_document_fields("udyam", doc_res)
    assert e2e_fields["partial_language_coverage"] is False
    assert e2e_fields["language_review_required"] is False


def test_extract_property_tax_receipt_marathi():
    """Verify extraction of Property Tax Receipt in Marathi with PII masking."""
    prop_tax_text = """
    बृहन्मुंबई महानगरपालिका
    मालमत्ता कर पावती
    मालमत्ता क्रमांक: PROP-MUM-88219
    करदात्याचे नाव: गजानन भिकाजी जोशी
    भरलेली रक्कम: 15,450.00
    कर आकारणी वर्ष: 2024-2025
    पावती दिनांक: 12/04/2024
    """
    doc_res = _make_doc_res(prop_tax_text)
    fields, confs = extract_property_tax_receipt(doc_res)

    assert fields.get("property_id") == "PROP-MUM-88219"
    assert fields.get("tax_amount_paid") == "15450.00"
    assert fields.get("assessment_year") == "2024-2025"
    assert fields.get("payment_date") == "12/04/2024"
    assert "owner_name_masked" in fields

    # Sanitized pipeline verification
    sanitized, _ = extract_document_fields("property_tax_receipt", doc_res)
    assert "raw_owner_name" not in sanitized
    assert sanitized["property_id"] == "PROP-MUM-88219"
    assert sanitized["partial_language_coverage"] is False
    assert sanitized["language_review_required"] is False


def test_extract_rent_agreement_marathi():
    """Verify extraction of Maharashtra Leave & License rent agreement in Marathi."""
    rent_text = """
    भाडेकरार (LEAVE AND LICENSE AGREEMENT)
    परवाना देणारा: सुहास रामचंद्र कुलकर्णी
    परवाना घेणारा: अनिकेत संजय शिंदे
    मासिक भाडे: Rs. 22,000/-
    करार कालावधी: 01/05/2024 ते 31/03/2025
    """
    doc_res = _make_doc_res(rent_text)
    fields, confs = extract_rent_agreement(doc_res)

    assert fields.get("monthly_rent") == "22000"
    assert "lessor_name_masked" in fields
    assert "lessee_name_masked" in fields

    sanitized, _ = extract_document_fields("rent_agreement", doc_res)
    assert "raw_lessor_name" not in sanitized
    assert "raw_lessee_name" not in sanitized
    assert sanitized["monthly_rent"] == "22000"
    assert sanitized["partial_language_coverage"] is False


def test_extract_aadhaar_bilingual():
    """Verify extraction of bilingual Aadhaar card with Devanagari labels and name."""
    aadhaar_text = """
    भारतीय विशिष्ट ओळख प्राधिकरण
    माझे आधार, माझी ओळख
    नाव: रमेश वामन जोशी
    जन्मतारीख: 15/08/1985
    लिंग: पुरुष
    आधार क्रमांक: 9876 5432 1098
    """
    doc_res = _make_doc_res(aadhaar_text)
    fields, confs = extract_aadhaar(doc_res)

    assert fields.get("aadhaar_number_masked") == "XXXXXXXX1098"
    assert fields.get("name") == "रमेश वामन जोशी"
    assert fields.get("dob") == "15/08/1985"
    assert fields.get("gender") == "Male"  # mapped from पुरुष

    sanitized, _ = extract_document_fields("aadhaar", doc_res)
    assert "raw_aadhaar" not in sanitized
    assert sanitized["aadhaar_number_masked"] == "XXXXXXXX1098"
    assert sanitized["partial_language_coverage"] is False


# ==============================================================================
# 5. Language Review & Coverage Guardrail Tests
# ==============================================================================

def test_partial_language_coverage_flags_manual_review_when_fields_missing():
    """
    When a document has significant Devanagari text but core fields cannot be
    parsed with high confidence, partial_language_coverage and language_review_required
    must be flagged True for human reviewer intervention.
    """
    unrecognized_marathi_text = """
    महाराष्ट्र शासन कामगार विभाग
    काही मजकूर येथे उपलब्ध आहे परंतु कोणताही अधिकृत नोंदणी क्रमांक किंवा आस्थापनेचे नाव नाही.
    तसेच इतर कोणत्याही अधिकृत माहितीचा अभाव आहे.
    """
    doc_res = _make_doc_res(unrecognized_marathi_text)
    fields, _ = extract_document_fields("shop_establishment", doc_res)

    assert fields["partial_language_coverage"] is True
    assert fields["language_review_required"] is True
    assert "Manual review recommended" in fields["language_coverage_notes"]
    assert "registration_number" in fields["language_coverage_notes"]


def test_language_coverage_flags_review_when_extraction_incomplete():
    """
    True-positive test case:
    When a document has significant Devanagari content (>= 10 characters),
    but one or more core extractable fields fail to extract (missing, garbled, or
    unrecognized layout), the guardrail must set:
    - partial_language_coverage: True
    - language_review_required: True
    - language_coverage_notes: naming the missing core fields.
    """
    # 1. Udyam Registration with Devanagari content but missing registration number and enterprise name
    incomplete_udyam_text = """
    भारत सरकार
    सूक्ष्म, लघु एवं मध्यम उद्यम मंत्रालय
    उद्यम पंजीकरण मार्गदर्शन व सूचना
    हे केवळ माहितीसाठी जारी केलेले परिपत्रक आहे.
    सर्व उद्योजकांनी अधिकृत पोर्टलवर जाऊन नोंदणी करावी.
    """
    doc_res_udyam = _make_doc_res(incomplete_udyam_text)
    fields_udyam, _ = extract_document_fields("udyam", doc_res_udyam)

    assert "devanagari" in fields_udyam["detected_languages"]
    assert fields_udyam["partial_language_coverage"] is True
    assert fields_udyam["language_review_required"] is True
    assert "Manual review recommended" in fields_udyam["language_coverage_notes"]
    assert "enterprise_name" in fields_udyam["language_coverage_notes"]
    assert "udyam_registration_number" in fields_udyam["language_coverage_notes"]

    # 2. Udyam Registration with registration number present, but enterprise name missing
    udyam_missing_name = """
    भारत सरकार
    सूक्ष्म, लघु एवं मध्यम उद्यम मंत्रालय
    UDYAM REGISTRATION NUMBER: UDYAM-MH-12-0044556
    उद्यमाचा प्रकार: सूक्ष्म
    मुख्य कार्यकलाप: विनिर्माण
    """
    doc_res_name_missing = _make_doc_res(udyam_missing_name)
    fields_missing_name, _ = extract_document_fields("udyam", doc_res_name_missing)

    assert fields_missing_name.get("udyam_registration_number") == "UDYAM-MH-12-0044556"
    assert "devanagari" in fields_missing_name["detected_languages"]
    assert fields_missing_name["partial_language_coverage"] is True
    assert fields_missing_name["language_review_required"] is True
    assert "enterprise_name" in fields_missing_name["language_coverage_notes"]

    # 3. Shop & Establishment document with Devanagari content but missing registration number
    shop_incomplete_text = """
    महाराष्ट्र शासन
    कामगार आयुक्त कार्यालय
    दुकान व आस्थापना अधिनियम अंतर्गत सूचना
    आस्थापनेचे नाव: श्री गणेश स्टोअर्स
    पत्ता: दादर, मुंबई
    """
    doc_res_shop = _make_doc_res(shop_incomplete_text)
    fields_shop, _ = extract_document_fields("shop_establishment", doc_res_shop)

    assert "devanagari" in fields_shop["detected_languages"]
    assert fields_shop["partial_language_coverage"] is True
    assert fields_shop["language_review_required"] is True
    assert "registration_number" in fields_shop["language_coverage_notes"]


def test_language_coverage_stays_clear_when_extraction_succeeds():
    """
    True-negative test case:
    When a document has significant Devanagari content (>= 10 characters),
    and all mandatory core fields are successfully extracted, the guardrail
    must stay false:
    - partial_language_coverage: False
    - language_review_required: False
    - language_coverage_notes: noting successful extraction.
    """
    successful_udyam_text = """
    भारत सरकार
    सूक्ष्म, लघु एवं मध्यम उद्यम मंत्रालय
    UDYAM REGISTRATION CERTIFICATE
    UDYAM REGISTRATION NUMBER: UDYAM-MH-26-0804117
    NAME OF ENTERPRISE: ROYAL CAKE HOUSE
    TYPE OF ENTERPRISE: Micro
    MAJOR ACTIVITY: TRADING
    """
    doc_res = _make_doc_res(successful_udyam_text)
    fields, _ = extract_document_fields("udyam", doc_res)

    assert fields.get("udyam_registration_number") == "UDYAM-MH-26-0804117"
    assert fields.get("enterprise_name") == "ROYAL CAKE HOUSE"
    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is False
    assert fields["language_review_required"] is False
    assert "all core fields extracted successfully" in fields["language_coverage_notes"]


def test_english_document_has_no_language_review_flags():
    """English documents must not flag partial_language_coverage or language_review_required."""
    pan_text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    Permanent Account Number: ABCDE1234F
    Name: JOHN DOE
    Father's Name: ROBERT DOE
    Date of Birth: 01/01/1990
    """
    doc_res = _make_doc_res(pan_text)
    fields, _ = extract_document_fields("pan", doc_res)

    assert fields.get("pan_number") == "ABCDE1234F"
    assert fields.get("partial_language_coverage") is False
    assert fields.get("language_review_required") is False
    assert fields.get("detected_languages") == ["en"]


# ==============================================================================
# 6. Devanagari Signature Detection Tests
# ==============================================================================

def test_devanagari_signatures_detect_document_type():
    """Verify that detect_document_type correctly classifies pure Devanagari text."""
    # Aadhaar Devanagari signature
    aadhaar_text = "भारतीय विशिष्ट ओळख प्राधिकरण माझे आधार 9876 5432 1098"
    assert detect_document_type(aadhaar_text) == "aadhaar"

    # Udyam Devanagari signature
    udyam_text = "उद्यम नोंदणी प्रमाणपत्र सूक्ष्म, लघु व मध्यम उद्योग UDYAM-MH-01-0012345"
    assert detect_document_type(udyam_text) == "udyam"

    # Shop & Establishment Devanagari signature
    shop_text = "महाराष्ट्र शासन दुकान आणि आस्थापना नोंदणी प्रमाणपत्र आस्थापना नोंदणी"
    assert detect_document_type(shop_text) == "shop_establishment"

    # Property Tax Receipt Devanagari signature
    prop_text = "बृहन्मुंबई महानगरपालिका मालमत्ता कर पावती मालमत्ता क्रमांक"
    assert detect_document_type(prop_text) == "property_tax_receipt"

    # Rent Agreement Devanagari signature
    rent_text = "भाडेकरार परवाना देणारा मासिक भाडे डिपॉझिट"
    assert detect_document_type(rent_text) == "rent_agreement"

    # Utility Bill Devanagari signature
    util_text = "महाराष्ट्र राज्य विद्युत वितरण कंपनी मर्यादित महावितरण वीज देयक ग्राहक क्रमांक देयक रक्कम अंतिम तारीख"
    assert detect_document_type(util_text) == "utility_bill"

    # Salary Slip Devanagari signature
    salary_text = "महाराष्ट्र शासन वेतन पावती मिळकत कपात निव्वळ वेतन माहे ऑगस्ट 2024"
    assert detect_document_type(salary_text) == "salary_slip"

    # Bank Passbook Devanagari signature
    passbook_text = "पुणे जिल्हा मध्यवर्ती सहकारी बँक मर्यादित बचत खाते पासबुक खातेदाराचे नाव खाते क्रमांक"
    assert detect_document_type(passbook_text) == "bank_passbook"


# ==============================================================================
# 7. Udyam Major Activity & Devanagari Header Regression Tests
# ==============================================================================

def test_udyam_major_activity_trading_vs_nic_manufacturing():
    """
    Regression test for Udyam certificate where declared Major Activity is TRADING,
    while the NIC classification table's Activity column mentions Manufacturing.
    The extractor must correctly return 'Trading', anchored to the MAJOR ACTIVITY label.
    """
    real_sample_text = """
    भारत सरकार
    सूक्ष्म, लघु एवं मध्यम उद्यम मंत्रालय
    UDYAM REGISTRATION CERTIFICATE
    UDYAM REGISTRATION NUMBER: UDYAM-MH-26-0804117
    NAME OF ENTERPRISE: ROYAL CAKE HOUSE
    TYPE OF ENTERPRISE: Micro
    Classification Date: 14/11/2025
    TRADING
    MAJOR ACTIVITY
    [For availing benefits of Priority Sector Lending(PSL) ONLY]
    SOCIAL CATEGORY OF ENTREPRENEUR: GENERAL
    NATIONAL INDUSTRY CLASSIFICATION CODE(S)
    SNo. NIC 2 Digit NIC 4 Digit NIC 5 Digit Activity
    1 10 - Manufacture of food products 1071 - Bakery 10712 - Cakes Manufacturing
    DATE OF UDYAM REGISTRATION: 29/12/2024
    """
    doc_res = _make_doc_res(real_sample_text)
    fields, _ = extract_udyam(doc_res)

    assert fields.get("udyam_registration_number") == "UDYAM-MH-26-0804117"
    assert fields.get("enterprise_name") == "ROYAL CAKE HOUSE"
    assert fields.get("enterprise_type") == "Micro"
    assert fields.get("major_activity") == "Trading"

    # End-to-end verification with language coverage
    e2e_fields, _ = extract_document_fields("udyam", doc_res)
    assert e2e_fields["major_activity"] == "Trading"
    assert "devanagari" in e2e_fields["detected_languages"]
    assert e2e_fields["partial_language_coverage"] is False
    assert e2e_fields["language_review_required"] is False


def test_udyam_major_activity_manufacturing_legitimate():
    """
    Verify that when declared Major Activity is legitimately Manufacturing,
    the field correctly reflects Manufacturing and does not confuse with other words.
    """
    mfg_text = """
    UDYAM REGISTRATION CERTIFICATE
    UDYAM REGISTRATION NUMBER: UDYAM-DL-05-0099881
    NAME OF ENTERPRISE: PRECISION GEARS PVT LTD
    TYPE OF ENTERPRISE: Small
    MAJOR ACTIVITY: MANUFACTURING
    NATIONAL INDUSTRY CLASSIFICATION CODE(S)
    Activity: Services
    """
    doc_res = _make_doc_res(mfg_text)
    fields, _ = extract_udyam(doc_res)

    assert fields.get("major_activity") == "Manufacturing"
    assert fields.get("enterprise_type") == "Small"


def test_udyam_major_activity_hindi_devanagari_variants():
    """
    Verify Udyam major activity extraction from Hindi labels (व्यापार, विनिर्माण, सेवाएं).
    """
    # 1. Trading in Hindi
    text_trade = """
    उद्यम नोंदणी क्रमांक: UDYAM-MH-01-0045678
    उद्यमाचे नाव: श्रीराम ट्रेडर्स
    उद्यमाचा प्रकार: लघु
    मुख्य कार्यकलाप: व्यापार
    """
    fields_trade, _ = extract_udyam(_make_doc_res(text_trade))
    assert fields_trade.get("major_activity") == "Trading"
    assert fields_trade.get("enterprise_type") == "Small"

    # 2. Services in Hindi
    text_svc = """
    उद्यम पंजीकरण संख्या: UDYAM-UP-12-0033445
    उद्यम का नाम: आकाश आईटी सॉल्यूशंस
    उद्यम का प्रकार: सूक्ष्म
    मुख्य कार्यकलाप: सेवाएं
    """
    fields_svc, _ = extract_udyam(_make_doc_res(text_svc))
    assert fields_svc.get("major_activity") == "Services"
    assert fields_svc.get("enterprise_type") == "Micro"


def test_devanagari_header_with_english_body_detection():
    """
    Regression test mirroring hybrid Udyam documents with Devanagari header
    ("भारत सरकार", "सूक्ष्म, लघु एवं मध्यम उद्यम मंत्रालय") and English body fields.
    Confirms language detection identifies Devanagari content without false negative.
    """
    hybrid_text = """
    भारत सरकार
    सूक्ष्म, लघु एवं मध्यम उद्यम मंत्रालय
    UDYAM REGISTRATION CERTIFICATE
    UDYAM REGISTRATION NUMBER: UDYAM-MH-26-0804117
    NAME OF ENTERPRISE: ROYAL CAKE HOUSE
    TYPE OF ENTERPRISE: Micro
    MAJOR ACTIVITY: TRADING
    DATE OF UDYAM REGISTRATION: 29/12/2024
    """
    doc_res = _make_doc_res(hybrid_text)
    fields, _ = extract_document_fields("udyam", doc_res)

    assert fields["major_activity"] == "Trading"
    assert fields["enterprise_name"] == "ROYAL CAKE HOUSE"
    assert "devanagari" in fields["detected_languages"]
    assert "en" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is False
    assert fields["language_review_required"] is False
    assert "Devanagari script text detected" in fields["language_coverage_notes"]


def test_engine_info_contains_sidebar_display_fields():
    """
    Verify get_ocr_engine_info returns display_name, engine, and device
    expected by the frontend sidebar and decision badge.
    """
    info = get_ocr_engine_info()
    assert "display_name" in info
    assert info["display_name"] in ("RapidOCR", "PaddleOCR", "None")
    assert "device" in info
    assert info["device"] == "CPU"
    assert "engine" in info
    assert "active_engine" in info
    assert info["status"] in ("ready", "unavailable")


# ==============================================================================
# 8. Utility Bill Devanagari & Guardrail Tests
# ==============================================================================

def test_extract_utility_bill_marathi():
    """
    Verify Marathi utility bill extraction (e.g. Mahavitaran / MSEDCL electricity bill)
    with Marathi field labels: ग्राहक क्रमांक, देयक दिनांक, देय दिनांक, देयक रक्कम.
    """
    sample_text = """
    महाराष्ट्र राज्य विद्युत वितरण कंपनी मर्यादित
    महावितरण वीज देयक
    ग्राहक क्रमांक : 177453132860
    देयक दिनांक : 18/10/2024
    देय दिनांक : 07/11/2024
    देयक रक्कम : Rs. 1200.00
    """
    doc_res = _make_doc_res(sample_text)
    fields, _ = extract_utility_bill(doc_res)

    assert fields.get("utility_provider") == "Mahavitaran (MSEDCL)"
    assert fields.get("consumer_number") == "177453132860"
    assert fields.get("bill_date") == "18/10/2024"
    assert fields.get("due_date") == "07/11/2024"
    assert fields.get("bill_amount") == "1200.00"

    # End-to-end extraction with language coverage
    e2e_fields, _ = extract_document_fields("utility_bill", doc_res)
    assert e2e_fields["utility_provider"] == "Mahavitaran (MSEDCL)"
    assert e2e_fields["consumer_number"] == "177453132860"
    assert e2e_fields["bill_amount"] == "1200.00"
    assert "devanagari" in e2e_fields["detected_languages"]
    assert e2e_fields["partial_language_coverage"] is False
    assert e2e_fields["language_review_required"] is False


def test_utility_bill_devanagari_numbers_normalization():
    """
    Verify conversion of Devanagari numerals (०१२३४५६७८९) in consumer number,
    dates, and amount.
    """
    dev_num_text = """
    महावितरण वीज देयक
    ग्राहक क्रमांक : १७७४५३१३२८६०
    देयक दिनांक : १८-१०-२०२४
    देय दिनांक : ०७-११-२०२४
    देयक रक्कम : रु. १,२००.५०
    """
    doc_res = _make_doc_res(dev_num_text)
    fields, _ = extract_utility_bill(doc_res)

    assert fields.get("consumer_number") == "177453132860"
    assert fields.get("bill_date") == "18/10/2024"
    assert fields.get("due_date") == "07/11/2024"
    assert fields.get("bill_amount") == "1200.50"


def test_utility_bill_language_coverage_guardrail_true_positive():
    """
    True-positive test case for utility_bill:
    When a utility bill contains significant Devanagari content (>= 10 characters),
    but core fields (consumer_number and/or bill_amount) fail to extract,
    partial_language_coverage and language_review_required must be set to True.
    """
    incomplete_bill_text = """
    महाराष्ट्र राज्य विद्युत वितरण कंपनी मर्यादित
    विद्युत नियम व शर्ती संबंधी जाहीर सूचना
    सर्व ग्राहकांनी सौर ऊर्जा प्रकल्पाचा लाभ घ्यावा.
    अधिक माहितीसाठी अधिकृत संकेतस्थळास भेट द्या.
    """
    doc_res = _make_doc_res(incomplete_bill_text)
    fields, _ = extract_document_fields("utility_bill", doc_res)

    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is True
    assert fields["language_review_required"] is True
    assert "Manual review recommended" in fields["language_coverage_notes"]
    assert "consumer_number" in fields["language_coverage_notes"]
    assert "bill_amount" in fields["language_coverage_notes"]


def test_utility_bill_language_coverage_guardrail_true_negative():
    """
    True-negative test case for utility_bill:
    When a utility bill has significant Devanagari content (>= 10 characters),
    and all mandatory core fields (consumer_number and bill_amount) extract cleanly,
    the guardrail must stay False.
    """
    successful_bill_text = """
    महावितरण वीज देयक
    ग्राहक क्रमांक: 177453132860
    देयक रक्कम: Rs. 1200.00
    अंतिम तारीख: 07-11-2024
    """
    doc_res = _make_doc_res(successful_bill_text)
    fields, _ = extract_document_fields("utility_bill", doc_res)

    assert fields.get("consumer_number") == "177453132860"
    assert fields.get("bill_amount") == "1200.00"
    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is False
    assert fields["language_review_required"] is False
    assert "all core fields extracted successfully" in fields["language_coverage_notes"]


# ==============================================================================
# 9. Salary Slip & Bank Passbook Devanagari Tests
# ==============================================================================

def test_extract_salary_slip_marathi():
    """
    Verify Marathi salary slip extraction (Zilla Parishad / State Government)
    with field labels: कार्यालयाचे नाव / संस्था, कर्मचाऱ्याचे नाव, निव्वळ वेतन, माहे.
    Verify strict PII masking on employee name.
    """
    sample_text = """
    महाराष्ट्र शासन
    जिल्हा परिषद पुणे
    वेतन पावती
    कर्मचाऱ्याचे नाव: रमेश पवार
    माहे: ऑगस्ट २०२४
    निव्वळ वेतन: रु. ४५,०००.००
    """
    doc_res = _make_doc_res(sample_text)
    fields, _ = extract_salary_slip(doc_res)

    assert fields.get("employer_name") == "महाराष्ट्र शासन"
    assert fields.get("employee_name_masked") == "रमेश प***"
    assert fields.get("net_pay") == "45000.00"
    assert fields.get("pay_period") == "ऑगस्ट 2024"

    # End-to-end extraction through extract_document_fields (with PII minimisation)
    e2e_fields, _ = extract_document_fields("salary_slip", doc_res)
    assert e2e_fields["employer_name"] == "महाराष्ट्र शासन"
    assert e2e_fields["employee_name_masked"] == "रमेश प***"
    assert e2e_fields["net_pay"] == "45000.00"
    assert e2e_fields["pay_period"] == "ऑगस्ट 2024"
    assert "raw_employee_name" not in e2e_fields
    assert "employee_name" not in e2e_fields
    assert "devanagari" in e2e_fields["detected_languages"]
    assert e2e_fields["partial_language_coverage"] is False
    assert e2e_fields["language_review_required"] is False
    assert "all core fields extracted successfully" in e2e_fields["language_coverage_notes"]


def test_salary_slip_name_masking_parity():
    """
    Verify that Marathi employee names are masked with the exact same
    rules as English names (first token preserved, second token initial + asterisks).
    """
    assert mask_person_name("रमेश पवार") == "रमेश प***"
    assert mask_person_name("सुरेश विष्णू सावंत") == "सुरेश स****"
    assert mask_person_name("JOHN DOE") == "JOHN D**"

    # Ensure raw name is stripped in e2e
    sample_text = """
    कार्यालयाचे नाव: महाराष्ट्र राज्य परिवहन महामंडळ
    कर्मचाऱ्याचे नाव: सुरेश विष्णू सावंत
    निव्वळ वेतन: रु. ३२,५००.००
    वेतन महिना: ०८/२०२४
    """
    doc_res = _make_doc_res(sample_text)
    e2e_fields, _ = extract_document_fields("salary_slip", doc_res)
    assert e2e_fields["employee_name_masked"] == "सुरेश स****"
    assert "raw_employee_name" not in e2e_fields
    assert "सुरेश विष्णू सावंत" not in str(e2e_fields)


def test_salary_slip_language_coverage_guardrail_true_positive():
    """
    True-positive test case for salary_slip:
    When a salary slip contains significant Devanagari text (>= 10 chars),
    but core fields (employer_name and/or net_pay) fail to extract,
    partial_language_coverage and language_review_required must be set to True.
    """
    incomplete_salary_text = """
    महाराष्ट्र शासन परिपत्रक
    सर्व कर्मचाऱ्यांना सूचित करण्यात येते की नवीन नियमावली लागू करण्यात येत आहे.
    अधिक माहितीसाठी संबंधित विभागाशी संपर्क साधावा.
    """
    doc_res = _make_doc_res(incomplete_salary_text)
    fields, _ = extract_document_fields("salary_slip", doc_res)

    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is True
    assert fields["language_review_required"] is True
    assert "Manual review recommended" in fields["language_coverage_notes"]
    assert "net_pay" in fields["language_coverage_notes"]


def test_salary_slip_language_coverage_guardrail_true_negative():
    """
    True-negative test case for salary_slip:
    When a salary slip contains significant Devanagari text (>= 10 chars),
    and all core fields (employer_name and net_pay) extract cleanly,
    the guardrail must stay False.
    """
    sample_text = """
    कार्यालयाचे नाव: पुणे महानगरपालिका
    कर्मचारी नाव: अनिल कांबळे
    निव्वळ वेतन: रु. ५०,०००
    माहे: जुलै २०२४
    """
    doc_res = _make_doc_res(sample_text)
    fields, _ = extract_document_fields("salary_slip", doc_res)

    assert fields.get("employer_name") == "पुणे महानगरपालिका"
    assert fields.get("net_pay") == "50000"
    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is False
    assert fields["language_review_required"] is False
    assert "all core fields extracted successfully" in fields["language_coverage_notes"]


def test_extract_bank_passbook_marathi():
    """
    Verify Marathi bank passbook extraction (e.g. Urban Co-operative Bank, RRB)
    with field labels: बँकेचे नाव, बचत खाते पासबुक, शाखा, खाते क्रमांक, खातेदाराचे नाव, आयएफएससी.
    Verify strict PII masking on account number and holder name.
    """
    sample_text = """
    पुणे जिल्हा मध्यवर्ती सहकारी बँक मर्यादित
    बचत खाते पासबुक
    शाखा: शिवाजीनगर
    खाते क्रमांक: १२३४५६७८९०१२
    खातेदाराचे नाव: रमेश पवार
    आयएफएससी: PDCB0000123
    """
    doc_res = _make_doc_res(sample_text)
    fields, _ = extract_bank_passbook(doc_res)

    assert fields.get("bank_name") == "पुणे जिल्हा मध्यवर्ती सहकारी बँक मर्यादित"
    assert fields.get("branch") == "शिवाजीनगर"
    assert fields.get("account_number_masked") == "XXXXXXXX9012"
    assert fields.get("account_holder_name_masked") == "रमेश प***"
    assert fields.get("ifsc") == "PDCB0000123"

    # End-to-end extraction through extract_document_fields (with PII minimisation)
    e2e_fields, _ = extract_document_fields("bank_passbook", doc_res)
    assert e2e_fields["bank_name"] == "पुणे जिल्हा मध्यवर्ती सहकारी बँक मर्यादित"
    assert e2e_fields["branch"] == "शिवाजीनगर"
    assert e2e_fields["account_number_masked"] == "XXXXXXXX9012"
    assert e2e_fields["account_holder_name_masked"] == "रमेश प***"
    assert e2e_fields["ifsc"] == "PDCB0000123"
    assert "raw_account_number" not in e2e_fields
    assert "raw_account_holder_name" not in e2e_fields
    assert "123456789012" not in str(e2e_fields)
    assert "devanagari" in e2e_fields["detected_languages"]
    assert e2e_fields["partial_language_coverage"] is False
    assert e2e_fields["language_review_required"] is False
    assert "all core fields extracted successfully" in e2e_fields["language_coverage_notes"]


def test_bank_passbook_account_and_name_masking_parity():
    """
    Verify masking parity for Devanagari account number and account holder name.
    Raw account number and name must NEVER leak into e2e extraction result.
    """
    assert mask_account_number("123456789012") == "XXXXXXXX9012"
    assert mask_account_number(normalize_devanagari_numbers("९८७६५४३२१०९८")) == "XXXXXXXX1098"

    sample_text = """
    बँकेचे नाव: महाराष्ट्र ग्रामीण बँक
    शाखा: औरंगाबाद
    खाते क्रमांक: ९८७६५४३२१०९८
    खातेदाराचे नाव: सुनील महादेव शिंदे
    IFSC: MAHG0004567
    """
    doc_res = _make_doc_res(sample_text)
    e2e_fields, _ = extract_document_fields("bank_passbook", doc_res)

    assert e2e_fields["account_number_masked"] == "XXXXXXXX1098"
    assert e2e_fields["account_holder_name_masked"] == "सुनील श****"
    assert "raw_account_number" not in e2e_fields
    assert "raw_account_holder_name" not in e2e_fields
    assert "987654321098" not in str(e2e_fields)
    assert "सुनील महादेव शिंदे" not in str(e2e_fields)


def test_bank_passbook_language_coverage_guardrail_true_positive():
    """
    True-positive test case for bank_passbook:
    When a bank passbook contains significant Devanagari text (>= 10 chars),
    but core fields (account_number_masked and/or bank_name) fail to extract,
    partial_language_coverage and language_review_required must be set to True.
    """
    incomplete_passbook_text = """
    महाराष्ट्र राज्य सहकारी बँक नियमावली
    ग्राहकांसाठी महत्त्वाची सूचना: केवायसी कागदपत्रे वेळेवर सादर करावीत.
    शाखा कार्यालय वेळ सकाळी १० ते दुपारी ४ पर्यंत.
    """
    doc_res = _make_doc_res(incomplete_passbook_text)
    fields, _ = extract_document_fields("bank_passbook", doc_res)

    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is True
    assert fields["language_review_required"] is True
    assert "Manual review recommended" in fields["language_coverage_notes"]
    assert "account_number_masked" in fields["language_coverage_notes"]


def test_bank_passbook_language_coverage_guardrail_true_negative():
    """
    True-negative test case for bank_passbook:
    When a bank passbook contains significant Devanagari text (>= 10 chars),
    and all core fields (account_number_masked and bank_name) extract cleanly,
    the guardrail must stay False.
    """
    sample_text = """
    ठाणे जनता सहकारी बँक
    बचत खाते पासबुक
    शाखा: ठाणे पश्चिम
    खाते क्रमांक: 554433221100
    खातेदाराचे नाव: अमित जोशी
    आयएफएससी: TJSB0000002
    """
    doc_res = _make_doc_res(sample_text)
    fields, _ = extract_document_fields("bank_passbook", doc_res)

    assert fields.get("bank_name") == "ठाणे जनता सहकारी बँक"
    assert fields.get("account_number_masked") == "XXXXXXXX1100"
    assert "devanagari" in fields["detected_languages"]
    assert fields["partial_language_coverage"] is False
    assert fields["language_review_required"] is False
    assert "all core fields extracted successfully" in fields["language_coverage_notes"]



