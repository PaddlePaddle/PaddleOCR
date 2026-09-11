"""
tests/test_verification.py
Tests for Verhoeff checksum, PAN/IFSC validation, cross-checking,
document-type mismatch, and security/verification bug fixes.
"""

import pytest
import verifier
from verifier import (
    VALID_BANK_CODES,
    check_doc_type_mismatch,
    generate_verhoeff_checksum_digit,
    perform_cross_check,
    validate_document_checksums,
    validate_ifsc_code,
    validate_pan_format,
    validate_verhoeff_checksum,
)
from extractors import sanitize_extracted_fields


def test_verhoeff_checksum_validation():
    base_11 = "23456789012"
    checksum_digit = generate_verhoeff_checksum_digit(base_11)
    valid_aadhaar = base_11 + checksum_digit

    # Valid checksum should evaluate to True
    assert validate_verhoeff_checksum(valid_aadhaar) is True

    # Altering any single digit must fail checksum
    tampered = valid_aadhaar[:-1] + str((int(checksum_digit) + 1) % 10)
    assert validate_verhoeff_checksum(tampered) is False

    # Aadhaar starting with 0 or 1 is invalid
    assert validate_verhoeff_checksum("012345678901") is False
    assert validate_verhoeff_checksum("112345678901") is False


def test_pan_format_validation():
    # Valid individual PAN (4th char 'P')
    valid, err = validate_pan_format("ABCPE1234F")
    assert valid is True
    assert err is None

    # Valid company PAN (4th char 'C')
    valid, err = validate_pan_format("AAACC9999Z")
    assert valid is True

    # Invalid entity character (e.g. 'Z')
    valid, err = validate_pan_format("ABCDZ1234F")
    assert valid is False
    assert "invalid_pan_entity_type" in err

    # Malformed regex
    valid, err = validate_pan_format("12345ABCDE")
    assert valid is False
    assert err == "invalid_pan_pattern"


def test_ifsc_code_validation():
    # Valid SBI code
    valid, err = validate_ifsc_code("SBIN0001234")
    assert valid is True
    assert err is None

    # Valid HDFC code
    valid, err = validate_ifsc_code("HDFC0000123")
    assert valid is True

    # Invalid 5th character (must be 0)
    valid, err = validate_ifsc_code("SBIN1001234")
    assert valid is False
    assert err == "invalid_ifsc_format"

    # Unrecognized bank prefix outside registry: failure mode is 'ifsc_needs_review' (not invalid)
    valid, err = validate_ifsc_code("ZZZZ0001234")
    assert valid is False
    assert err == "ifsc_needs_review"


def test_demo_bank_code_rejected_in_production():
    """
    Fix 3: Assert 'DEMO' is not in production VALID_BANK_CODES,
    and DEMO0001234-style codes are rejected (or flagged for review) by default.
    """
    assert "DEMO" not in VALID_BANK_CODES
    valid, err = validate_ifsc_code("DEMO0001234")
    assert valid is False
    assert err == "ifsc_needs_review"


def test_demo_bank_code_accepted_when_monkeypatched(monkeypatch):
    """
    Fix 3: Inject DEMO bank code via monkeypatch for test fixture scopes only.
    """
    monkeypatch.setattr(verifier, "VALID_BANK_CODES", verifier.VALID_BANK_CODES | {"DEMO"})
    valid, err = validate_ifsc_code("DEMO0001234")
    assert valid is True
    assert err is None


def test_legitimate_bank_code_outside_old_short_whitelist():
    """
    Fix 4: Legitimate banks outside the old short list (e.g. AIRP for Airtel Payments Bank,
    KANG for Kangra Central Co-op Bank) must be recognized in the expanded registry.
    """
    valid, err = validate_ifsc_code("AIRP0000001")
    assert valid is True
    assert err is None

    valid, err = validate_ifsc_code("KANG0000001")
    assert valid is True
    assert err is None


def test_doc_type_mismatch_detection():
    dl_text = """
    UNION OF INDIA DRIVING LICENCE
    TRANSPORT DEPARTMENT MAHARASHTRA
    Licence No: MH12 20260000001
    Name: AMIT VERMA
    """

    # If requested doc_type is aadhaar, should flag mismatch
    is_mismatch, detected = check_doc_type_mismatch("aadhaar", dl_text)
    assert is_mismatch is True
    assert detected == "driving_licence"

    # If requested doc_type is driving_licence, should NOT flag mismatch
    is_mismatch, detected = check_doc_type_mismatch("driving_licence", dl_text)
    assert is_mismatch is False


def test_cross_check_matching():
    extracted = {
        "name": "Dr. Rohit Sharma",
        "dob": "24/04/1987",
        "pan_number": "ABCDE1234F",
    }
    expected = {
        "name": "Rohit Sharma",
        "dob": "1987-04-24",
        "pan_number": "ABCDE1234F",
    }
    results = perform_cross_check(extracted, expected)

    assert results["name"]["matched"] is True
    assert results["name"]["score"] > 0.8
    assert results["dob"]["matched"] is True
    assert results["dob"]["score"] == 1.0
    assert results["pan_number"]["matched"] is True


def test_cross_check_mismatch():
    extracted = {
        "name": "John Doe",
        "dob": "01/01/1990",
    }
    expected = {
        "name": "Alice Wonderland",
        "dob": "15/08/1995",
    }
    results = perform_cross_check(extracted, expected)

    assert results["name"]["matched"] is False
    assert results["dob"]["matched"] is False


def test_salary_slip_cross_check_with_unmasked_name():
    """
    Fix 5: Salary slip cross-check against unmasked expected name returns matched: True,
    and final response payload contains only employee_name_masked.
    """
    raw_fields = {
        "employee_name": "DEMO CUSTOMER",
        "raw_employee_name": "DEMO CUSTOMER",
        "employee_name_masked": "DEMO C*******",
        "employer_name": "DEMO ENTERPRISES LIMITED",
        "net_pay": "49500",
        "pay_period": "August 2026",
    }
    expected = {"name": "DEMO CUSTOMER"}

    # 1. Cross-check against raw_fields
    cross_results = perform_cross_check(raw_fields, expected)
    assert cross_results["name"]["matched"] is True
    assert cross_results["name"]["score"] == 1.0

    # 2. Sanitize fields for response
    sanitized, _ = sanitize_extracted_fields("salary_slip", raw_fields, {})
    assert "employee_name" not in sanitized
    assert "raw_employee_name" not in sanitized
    assert sanitized["employee_name_masked"] == "DEMO C*******"
    assert sanitized["employer_name"] == "DEMO ENTERPRISES LIMITED"
    assert sanitized["net_pay"] == "49500"


def test_aadhaar_with_invalid_checksum_rejected():
    """
    Fix 6: A document with an invalid Aadhaar Verhoeff checksum is flagged
    as 'invalid_aadhaar_checksum', not silently passed.
    """
    base_11 = "23456789012"
    correct_digit = generate_verhoeff_checksum_digit(base_11)
    # Corrupted 12th digit
    corrupted_digit = str((int(correct_digit) + 1) % 10)
    invalid_aadhaar = base_11 + corrupted_digit

    raw_fields = {
        "raw_aadhaar": invalid_aadhaar,
        "aadhaar_number": "XXXXXXXX" + invalid_aadhaar[-4:],
        "name": "TEST CITIZEN",
        "dob": "01/01/1990",
    }

    valid, reason = validate_document_checksums("aadhaar", raw_fields)
    assert valid is False
    assert reason == "invalid_aadhaar_checksum"


def test_qr_disagreement_resolution():
    from qr_decoder import reconcile_ocr_and_qr

    ocr_fields = {
        "name": "JOHN DOE",
        "aadhaar_number": "XXXXXXXX1234",
    }
    qr_fields = {
        "name": "JONATHAN DOE",  # Differs from OCR
        "aadhaar_number": "XXXXXXXX1234",
    }
    merged, disagreements = reconcile_ocr_and_qr(ocr_fields, qr_fields)

    # QR takes priority
    assert merged["name"] == "JONATHAN DOE"
    assert len(disagreements) == 1
    assert disagreements[0]["field"] == "name"
    assert disagreements[0]["ocr_value"] == "JOHN DOE"
    assert disagreements[0]["qr_value"] == "JONATHAN DOE"
    assert disagreements[0]["resolution"] == "qr_preferred"


def test_micr_line_parsing():
    from micr_reader import parse_micr_string

    micr_line = "⑆000123⑆ 400002001⑈ 123456⑇ 10"
    parsed, conf = parse_micr_string(micr_line)

    assert parsed["cheque_number"] == "000123"
    assert parsed["micr_code"] == "400002001"
    assert parsed["tran_code"] == "10"
    assert conf == "high"

    uncertain_line = "some broken text without micr symbols"
    parsed_unc, conf_unc = parse_micr_string(uncertain_line)
    assert conf_unc == "low"
    assert parsed_unc["micr_confidence"] == "low"


def test_bank_statement_uploaded_as_cheque_mismatch():
    """
    Task 1: Bank statement containing transaction text with 'PAYMENT' and 'RUPEES'
    must be correctly flagged as doc_type_mismatch when cancelled_cheque is requested.
    """
    bank_statement_text = """
    HDFC BANK LIMITED
    ACCOUNT STATEMENT FOR PERIOD 01/01/2026 TO 31/01/2026
    STATEMENT OF ACCOUNT
    CLOSING BALANCE: INR 1,25,000.00
    TRANSACTION DETAILS:
    05/01/2026 PAYMENT TO VENDOR RUPEES 5000 DR
    10/01/2026 SALARY CREDIT RUPEES 50000 CR
    """
    is_mismatch, detected = check_doc_type_mismatch("cancelled_cheque", bank_statement_text)
    assert is_mismatch is True
    assert detected == "bank_statement"


def test_legitimate_cheque_with_word_pay_not_mismatched():
    """
    Task 1: Legitimate cancelled cheque containing 'PAY' or 'BEARER'
    must NOT be flagged as mismatch when cancelled_cheque is requested.
    """
    cheque_text = """
    STATE BANK OF INDIA
    CANCELLED
    PAY TO JOHN DOE OR BEARER
    A/C NO: 123456789012
    IFS CODE: SBIN0001234
    000123 400002001 000123
    """
    is_mismatch, detected = check_doc_type_mismatch("cancelled_cheque", cheque_text)
    assert is_mismatch is False
    assert detected is None or detected == "cancelled_cheque"


def test_valid_bank_codes_file_not_configured_uses_default(monkeypatch):
    """
    Task 4: When VALID_BANK_CODES_FILE is not configured, load_valid_bank_codes
    explicitly falls back to the bundled default set.
    """
    monkeypatch.delenv("VALID_BANK_CODES_FILE", raising=False)
    monkeypatch.delenv("IFSC_BANK_CODES_FILE", raising=False)
    from verifier import load_valid_bank_codes
    codes = load_valid_bank_codes()
    assert codes == VALID_BANK_CODES
    assert "SBIN" in codes
    assert "HDFC" in codes


def test_valid_bank_codes_file_configured_and_valid(monkeypatch, tmp_path):
    """
    Task 4: When VALID_BANK_CODES_FILE is configured with a valid JSON file,
    the external bank codes are properly loaded.
    """
    import json
    codes_file = tmp_path / "bank_codes.json"
    codes_file.write_text(json.dumps(["CUSTOMBANK", "TESTBANK"]))

    monkeypatch.setenv("VALID_BANK_CODES_FILE", str(codes_file))
    from verifier import load_valid_bank_codes
    codes = load_valid_bank_codes()
    assert codes == {"CUSTOMBANK", "TESTBANK"}


def test_valid_bank_codes_file_configured_and_missing_raises_error(monkeypatch):
    """
    Task 4: When VALID_BANK_CODES_FILE is configured but missing,
    load_valid_bank_codes fails fast with a RuntimeError.
    """
    monkeypatch.setenv("VALID_BANK_CODES_FILE", "/non/existent/path/codes.json")
    from verifier import load_valid_bank_codes
    with pytest.raises(RuntimeError) as exc_info:
        load_valid_bank_codes()
    assert "does not exist" in str(exc_info.value)


def test_valid_bank_codes_file_configured_and_empty_or_invalid_raises_error(monkeypatch, tmp_path):
    """
    Task 4: When VALID_BANK_CODES_FILE is empty or contains malformed JSON,
    load_valid_bank_codes fails fast with a RuntimeError.
    """
    # Malformed JSON
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("NOT_JSON")
    monkeypatch.setenv("VALID_BANK_CODES_FILE", str(bad_file))
    from verifier import load_valid_bank_codes
    with pytest.raises(RuntimeError) as exc_info:
        load_valid_bank_codes()
    assert "invalid or unreadable" in str(exc_info.value)

    # Empty list
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    monkeypatch.setenv("VALID_BANK_CODES_FILE", str(empty_file))
    with pytest.raises(RuntimeError) as exc_info:
        load_valid_bank_codes()
    assert "non-empty list" in str(exc_info.value)


# ==============================================================================
# New Document Types: Mismatch & Format Validation Tests
# ==============================================================================

def test_gst_certificate_vs_udyam_mismatch():
    """GST certificate uploaded when udyam requested must trigger doc_type_mismatch."""
    gst_text = """
    GOVERNMENT OF INDIA
    FORM GST REG-06
    REGISTRATION CERTIFICATE
    GOODS AND SERVICES TAX
    GSTIN: 27AABCU9603R1ZN
    Legal Name: ACME ENTERPRISES PRIVATE LIMITED
    Constitution of Business: Private Limited Company
    """
    is_mismatch, detected = check_doc_type_mismatch("udyam", gst_text)
    assert is_mismatch is True
    assert detected == "gst_certificate"


def test_bank_passbook_vs_bank_statement_mismatch():
    """Bank passbook uploaded when bank_statement requested must trigger doc_type_mismatch."""
    passbook_text = """
    STATE BANK OF INDIA
    SAVINGS BANK PASS BOOK
    ACCOUNT HOLDER NAME: PRIYA SHARMA
    ACCOUNT NUMBER: 20123456789
    IFSC: SBIN0004567
    CUSTOMER ID: CIF89012345
    """
    is_mismatch, detected = check_doc_type_mismatch("bank_statement", passbook_text)
    assert is_mismatch is True
    assert detected == "bank_passbook"


def test_rent_agreement_vs_shop_establishment_mismatch():
    """Rent agreement uploaded when shop_establishment requested must trigger doc_type_mismatch."""
    rent_text = """
    RENT AGREEMENT
    LEASE AGREEMENT
    LESSOR AND LESSEE
    MONTHLY RENT: RS. 30,000
    REFUNDABLE SECURITY DEPOSIT: RS. 90,000
    """
    is_mismatch, detected = check_doc_type_mismatch("shop_establishment", rent_text)
    assert is_mismatch is True
    assert detected == "rent_agreement"


def test_form_16_vs_salary_slip_mismatch():
    """Form 16 uploaded when salary_slip requested must trigger doc_type_mismatch."""
    form_16_text = """
    FORM NO. 16
    CERTIFICATE UNDER SECTION 203
    TAX DEDUCTED AT SOURCE
    CENTRAL BOARD OF DIRECT TAXES
    EMPLOYEE PAN: ABCPS1234E
    """
    is_mismatch, detected = check_doc_type_mismatch("salary_slip", form_16_text)
    assert is_mismatch is True
    assert detected == "form_16"


def test_new_document_format_validations():
    """Test format validators for GSTIN, CIN, and passbook IFSC."""
    from verifier import validate_gstin_format, validate_cin_format, validate_document_checksums

    # GSTIN validation: valid examples with true Luhn mod-36 checksum
    valid1, err1 = validate_gstin_format("27AABCU9603R1ZN")
    assert valid1 is True
    assert err1 is None

    valid2, err2 = validate_gstin_format("29AAACG0517B1Z8")
    assert valid2 is True
    assert err2 is None

    # Invalid GSTIN: wrong length
    val_len, err_len = validate_gstin_format("27AABCU9603R1Z")
    assert val_len is False
    assert err_len == "invalid_gstin_format"

    # Invalid GSTIN: wrong embedded PAN entity type
    val_pan, err_pan = validate_gstin_format("27AABZU9603R1ZN")
    assert val_pan is False
    assert "pan" in err_pan

    # Invalid GSTIN: wrong pattern
    val_pat, err_pat = validate_gstin_format("2712345678901ZN")
    assert val_pat is False
    assert err_pat == "invalid_gstin_format"

    # Invalid GSTIN: wrong check digit (M instead of N)
    val_chk, err_chk = validate_gstin_format("27AABCU9603R1ZM")
    assert val_chk is False
    assert err_chk == "invalid_gstin_checksum"

    # CIN validation: valid Unlisted and Listed structural examples
    valid_cin1, err_cin1 = validate_cin_format("U72900MH2025PTC412345")
    assert valid_cin1 is True
    assert err_cin1 is None

    valid_cin2, err_cin2 = validate_cin_format("L17110MH1973PLC019786")
    assert valid_cin2 is True
    assert err_cin2 is None

    # Invalid CIN: invalid listing prefix (must start with U or L)
    val_cin_pre, err_cin_pre = validate_cin_format("X72900MH2025PTC412345")
    assert val_cin_pre is False
    assert err_cin_pre == "invalid_cin_format"

    # Invalid CIN: wrong length / truncated
    val_cin_trunc, err_cin_trunc = validate_cin_format("U1234")
    assert val_cin_trunc is False
    assert err_cin_trunc == "invalid_cin_format"

    # Invalid CIN: invalid year segment (3 digits instead of 4)
    val_cin_yr, err_cin_yr = validate_cin_format("U72900MH202PTC412345")
    assert val_cin_yr is False
    assert err_cin_yr == "invalid_cin_format"

    # validate_document_checksums routing
    val_gst, _ = validate_document_checksums("gst_certificate", {"gstin": "27AABCU9603R1ZN"})
    assert val_gst is True

    val_cin, _ = validate_document_checksums("certificate_of_incorporation", {"cin": "U72900MH2025PTC412345"})
    assert val_cin is True

    val_pb, _ = validate_document_checksums("bank_passbook", {"ifsc": "SBIN0001234"})
    assert val_pb is True

    val_f16, _ = validate_document_checksums("form_16", {"pan_number": "ABCPE1234F"})
    assert val_f16 is True


def test_cross_check_list_valued_field():
    from verifier import perform_cross_check

    extracted = {
        "firm_name": "EXCELSIOR ASSOCIATES",
        "partner_names": ["VIKRAM MALHOTRA", "ROHAN DESHMUKH"],
        "date_of_deed": "01/04/2024",
    }

    # 1. Expected applicant name matches one partner in the list
    res1 = perform_cross_check(extracted, {"name": "VIKRAM MALHOTRA"})
    assert "name" in res1
    assert res1["name"]["matched"] is True
    assert res1["name"]["score"] >= 0.95
    assert res1["name"]["extracted"] == "VIKRAM MALHOTRA"

    # 2. Token-reversed name matches partner
    res2 = perform_cross_check(extracted, {"applicant_name": "DESHMUKH ROHAN"})
    assert "applicant_name" in res2
    assert res2["applicant_name"]["matched"] is True
    assert res2["applicant_name"]["score"] >= 0.95

    # 3. Explicit partner_name alias matches
    res3 = perform_cross_check(extracted, {"partner_name": "ROHAN DESHMUKH"})
    assert res3["partner_name"]["matched"] is True
    assert res3["partner_name"]["score"] >= 0.95

    # 4. List expectation matching list extracted
    res4 = perform_cross_check(extracted, {"partner_names": ["VIKRAM MALHOTRA", "ROHAN DESHMUKH"]})
    assert res4["partner_names"]["matched"] is True
    assert res4["partner_names"]["score"] >= 0.95

    # 5. Non-matching name returns false without crashing
    res5 = perform_cross_check(extracted, {"name": "SURESH RAINA"})
    assert res5["name"]["matched"] is False
    assert res5["name"]["score"] < 0.50

    # 6. Empty list returns false without crashing
    res6 = perform_cross_check({"partner_names": []}, {"partner_names": ["VIKRAM MALHOTRA"]})
    assert res6["partner_names"]["matched"] is False

    # 7. Verify cross-check works with raw_partner_names (standard post-extractor raw state)
    raw_state = {
        "firm_name": "MALHOTRA & DESHMUKH ENTERPRISES",
        "raw_partner_names": ["VIKRAM MALHOTRA", "ROHAN DESHMUKH"],
        "partner_names_masked": ["VIKRAM M*******", "ROHAN D*******"],
    }
    res7 = perform_cross_check(raw_state, {"name": "VIKRAM MALHOTRA"})
    assert res7["name"]["matched"] is True
    assert res7["name"]["score"] >= 0.95




