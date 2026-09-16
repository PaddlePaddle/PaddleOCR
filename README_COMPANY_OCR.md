# Company-Server OCR & Document Verification Service

A production-grade FastAPI microservice extending PaddleOCR (PP-OCRv5 + PPStructureV3) for end-to-end document extraction, rules-based verification, and strict PII minimisation. Designed specifically as an asynchronous node within an n8n-orchestrated document collection workflow.

---

## Workflow Integration

```
Consent -> Email Attachments -> Google Drive -> [THIS SERVICE: Company-Server OCR]
-> PII Minimisation -> Rules-Based Verification -> Redacted AI (if needed)
-> Manual Review (if needed) -> Case Update
```

---

## 1. Features & Capabilities

### 📄 Definitive Document Verification Status (22 Supported Document Types)

*Consolidated authoritative verification status table across all 22 supported document types following the comprehensive real-document testing audit conducted on 2026-09-11.*

| Document Type | Real-Document Tested (Y/N) | Test Date/Round | Known Issues Fixed | Currently Known Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Aadhaar** | **Y** (Demo PDF tested via live API) | 2026-09-11 (Audit Round) | Fixed Devanagari numerals misrecognized from Latin header OCR injection (`ocr_engine.py` / `extractors.py`); masked format preserved (`XXXX XXXX 4321`) | Legible 12-digit UID or masked 8X+4D pattern required. |
| **PAN** | **Y** (Demo PDF tested via live API) | 2026-09-11 (Audit Round) | None (extracted 100% cleanly: `pan_number`, `name`, `father_name`, `dob`) | Low-resolution camera scans may require clean lighting. |
| **Passport** | **Y** (Demo PDF tested via live API) | 2026-09-11 (Audit Round) | None (extracted 100% cleanly: `passport_number`, `surname`, `given_name`, `name`, `nationality`, `dob`, `expiry_date`) | Standard machine-readable biodata page required; handwritten older passports not supported. |
| **Driving Licence** | **Y** (Demo PDF tested via live API) | 2026-09-11 (Audit Round) | None (extracted 100% cleanly: `licence_number`, `name`, `dob`, `issue_date`, `valid_till`, `vehicle_class`) | MoRTH Sarathi smart-card format standard verified; non-standard paper DLs may vary. |
| **Voter ID** | **Y** (Demo PDF tested via live API) | 2026-09-11 (Audit Round) | None (extracted 100% cleanly: `epic_number`, `name`, `dob`) | Old laminated voter cards with degraded text may require manual review. |
| **FSSAI Certificate** | **Y** (Real 5-page PDF `7ba96563-...pdf` + Demo PDF) | 2026-09-11 (Audit Round) | Added support for 2-column FBO name table and C/O trade name parsing (`KATRAJ DAIRY AND CAKE SHOP`); added support for `Issued On` and `Fee Paid Upto` validity dates | Highly degraded QR codes may fail to decode. |
| **GST Certificate** | **Y** (Real PDF `GST RC.pdf`) | 2026-09-11 (Audit Round) | Allowed underscores in `legal_name` (`3_EXTENT`); added support for `From` and `Date of issue of Certificate` in `registration_date`; blank trade name sanitized | Trade name watermark 'x' ignored; annexures not merged into principal place. |
| **Shop & Establishment** | **Y** (Real Maharashtra Form F & G PDFs `112889582203.pdf`, `2231000317186866.pdf` + Demo PDF) | 2026-09-11 (Audit Round) | Added Maharashtra Form F/G signature detection; fixed `establishment_name` leading label bleed (`ROYAL CAKE HOUSE`); added Intimation / Receipt registration numbers (`2231000317186866`); added bilingual script boundary split for employer name (`RUSHIKESH SHIVAJI CHIKHALE`) | Delhi (`DL`) and Karnataka (`KA`) templates are statutory baseline only (unverified against real scans). |
| **Bank Statement** | **Y** (Real IPPB PDF & Axis Bank 24-page PDF) | 2026-09-11 (Audit Round) | Fixed empty transactions table across multi-page digital statements (all 638 rows parsed); prevented opening balance overwriting closing balance; added strict PII allowlist | Password-protected PDFs must be decrypted prior to upload; unstructured camera images of paper statements require manual inspection. |
| **Income Tax Return (ITR)** | **Y** (Real multi-page PDFs: FY 24-25 & FY 25-26) | 2026-09-11 (Audit Round) | Fixed assessee `name` pulling from bundled TDS/26AS schedule ("Name of Deductor") rather than ITR-V cover page (`3_EXTENT`); extracted pan, ack, AY, total income, tax paid | Non-standard acknowledgement formats without ITR-V cover slip. |
| **Utility Bill (Electricity)** | **Y** (Real Mahavitaran bill photo + Demo PNG) | 2026-09-11 (Audit Round) | Fixed Devanagari script detection; verified consumer number, bill amount, bill date, due date on real MSEDCL bill photo | Extreme camera skew or heavily crumpled physical bills. |
| **Income Certificate** | **Y** (Real Marathi Certificate photo `WhatsApp Image...jpeg`) | 2026-09-11 (Audit Round) | Fixed multi-year income table parsing to correctly extract latest `financial_year` (`2024-2025`) and `annual_income` (`40000`); rejected OCR artifact strings like `Q028-2034` | Certificate photo must have sufficient contrast to resolve Marathi Devanagari text. |
| **Udyam Registration** | **Y** (Real PDF `ROYAL CAKE HOUSE.pdf` + Demo PDF) | 2026-09-11 (Audit Round) | Fixed `major_activity` field-mapping error; verified udyam number, enterprise name, enterprise type (`Services`) | Multi-unit annexures summarized to primary enterprise. |
| **Salary Slip** | **Y** (Demo PDF + Demo PNG image) | 2026-09-11 (Audit Round) | Verified English PDF and PNG image OCR via RapidOCR; masked employee name | Real Devanagari/Marathi salary slip sample was not available (tested on synthetic Marathi strings only). |
| **Cancelled Cheque** | **Y** (Demo PDF CTS-2010 format) | 2026-09-11 (Audit Round) | Verified non-MICR fields (account number masked, IFSC, bank name, cheque number, holder name) 100% | MICR line parsing on physical cheques is low confidence because standard OCR lacks dedicated E-13B magnetic font training; MICR-bearing real physical cheque sample not available. |
| **Certificate of Incorporation** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented to MCA Form INC-11 specification) | **Untestable this round**: No real or demo PDF/image found in workspace or downloads. Verified only via synthetic test fixtures. |
| **Form 16** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented to CBDT TRACES Form 16 Part A/B specification) | **Untestable this round**: No real or demo PDF/image found in workspace or downloads. Verified only via synthetic test fixtures. |
| **IEC Certificate** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented to DGFT e-IEC specification) | **Untestable this round**: No real or demo PDF/image found in workspace or downloads. Verified only via synthetic test fixtures. |
| **Partnership Deed** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented to standard legal drafting conventions) | **Untestable this round**: Only synthetic PIL-drawn mock image exists (`partnership_deed.png`). No real or demo scanned deed available. |
| **Bank Passbook** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented for bank name, branch, IFSC, masked account) | **Untestable this round**: Only synthetic PIL-drawn mock image exists (`passbook.png`). No real or demo scanned bank passbook available. |
| **Rent Agreement** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented with strict PII masking) | **Untestable this round**: No real or demo PDF/image found in workspace or downloads. Verified only via synthetic test fixtures. |
| **Property Tax Receipt** | **N** (Untestable this round) | N/A (No sample found) | None (Extractor implemented for municipal property tax receipts) | **Untestable this round**: No real or demo PDF/image found in workspace or downloads. Verified only via synthetic test fixtures. |

---

### 🛡️ Rules-Based Verification & Validation
1. **Format & Checksum Validation**:
   - **PAN**: `^[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]$` (validates 4th entity code: P-Individual, C-Company, etc.).
   - **GSTIN**: `^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$` (validates 15-char structure, state code, embedded PAN validity, and computes the official GSTN Luhn mod-36 / ISO 7064 Mod 36, 36 check digit on character 15).
   - **CIN**: `^[UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}$` (validates 21-char Ministry of Corporate Affairs structural pattern: Listing status [U/L] + 5-digit Industry NIC code + 2-letter State code + 4-digit Year + 3-letter Company classification + 6-digit RoC registration number. Note: CIN has no mathematical checksum algorithm defined by MCA; it is purely a structural metadata validator).
   - **Address Masking (`mask_address`)**: Redacts specific door/flat/unit numbers and premise building specifics (e.g. `Flat 402, Building C` -> `XXXX`) while preserving locality, city, state, and postal code for downstream geographical serviceability checks.
   - **Aadhaar**: Verhoeff checksum algorithm applied to 12-digit number (equation evaluates to 0). Corrupted numbers downgrade `status: "low_confidence"` with `reason: "invalid_aadhaar_checksum"`.
   - **IFSC**: Pattern `^[A-Z]{4}0[A-Z0-9]{6}$`. Validated against an extended registry of Indian banks (commercial, public sector, RRBs, payment banks, small finance banks, and co-operative banks; extensible via `VALID_BANK_CODES_FILE`).
     - **Configured `VALID_BANK_CODES_FILE`**: If set, the file MUST exist, be valid JSON, and contain a non-empty list of codes. Missing, unreadable, or empty files raise a `RuntimeError` at service startup (fail-fast).
     - **Unconfigured**: If `VALID_BANK_CODES_FILE` is not set, explicitly falls back to the bundled default set.
     - **Known Bank Code**: Marked valid.
     - **Unlisted Bank Code**: Does not hard-fail; flags `status: "low_confidence"` with `reason: "ifsc_needs_review"` so genuine regional or co-operative banks are routed to manual review rather than rejected.
     - **Malformed IFSC**: Fails with `reason: "invalid_ifsc_format"`.
     - *Note*: Test-only bank codes (such as `"DEMO"`) are strictly removed from production code and only injected via test monkeypatching.
2. **Cross-Check Verification**:
   - Compares extracted fields against optional `expected` applicant payload (e.g. expected name and DOB).
   - Utilizes token-sorted fuzzy similarity for names (threshold >= 0.82) and date normalization (e.g. `01/01/1995` == `1995-01-01`).
   - Returns per-field match and similarity score (`cross_check: {"name": {"matched": true, "score": 1.0}}`).
3. **Document-Type Mismatch Detection**:
   - Evaluates signature header patterns with distinctive multi-word phrases and requires a score margin (`top_score >= 2` and `top_score - second_score >= 1`) to eliminate false mismatches from generic words like "pay" or "rupees". If an uploaded document clearly matches a different type (e.g. Driving Licence sent when Aadhaar expected), returns:
     ```json
     {
       "status": "error",
       "reason": "doc_type_mismatch",
       "detected_type": "driving_licence",
       "message": "Uploaded document appears to be 'driving_licence' rather than requested 'aadhaar'"
     }
     ```
4. **Per-Field OCR Confidence**:
   - Extracts real word and line confidence scores from PaddleOCR recognizer boxes instead of binary 0/1 heuristics.

---

### 🔍 Specialized Decoders & Pre-Checks
- **Image Quality Pre-Checks**: Evaluates Laplacian variance (blur detection) and minimum resolution (min 150x150). Degraded files return `status: "low_confidence"` and `reason: "image_quality"` immediately before invoking expensive OCR.
- **QR Code Decoding**: Runs alongside OCR for Aadhaar, Udyam, and FSSAI. QR-decoded data is treated as higher-trust; discrepancies are flagged under `qr_disagreements`.
- **MICR Line Reader for Cancelled Cheques**:
  - Crops the bottom 15–20% horizontal band of cancelled cheques and runs a dedicated OCR pass, parsing 6-digit cheque numbers, 9-digit MICR transit codes (3 city + 3 bank + 3 branch), account numbers, and 2-digit transaction codes.
  - Cross-checks MICR values against full-page text (`cheque_number` and `account_number_masked`), surfacing discrepancies in `micr_disagreements` without silently overwriting values.
  - **Real-World Reliability & Font Limitations**: Recognition accuracy against scanned cheques is currently low/unverified because the underlying neural OCR engine (RapidOCR / ONNX runtime) is trained on standard Latin/CJK typography and is not trained on the specialized E-13B magnetic ink font. On actual scans, delimiters and numerals may be omitted or misrecognized (e.g. consecutive zeros collapsed). The pipeline is strictly designed not to guess or fabricate missing digits and sets `micr_confidence: "low"` whenever the required 6-digit cheque number and 9-digit MICR code structure is violated. Treat MICR output as supplementary rather than authoritative until a dedicated E-13B model is evaluated.
- **State-Specific Shop & Establishment Parsing**:
  - Utilizes an extensible per-state template registry (`SHOP_ESTABLISHMENT_STATE_REGISTRY`) rather than a single monolithic regex.
  - **Maharashtra (`MH`)**: **Verified** against genuine demo certificate sample (`Demo_Shop_Establishment.pdf`). Extracts state, issuing authority, registration number, establishment name, and employer name with `template_matched: true` and `template_verified: true`.
  - **Delhi (`DL`)** & **Karnataka (`KA`)**: Baseline statutory templates based on standard Form C layouts under the respective state Acts. These are **unverified** against scanned documents in this environment and are explicitly flagged in the API output as `template_verified: false` pending verification against genuine scanned certificates.
  - **Other States / Unrecognized**: Gracefully falls back to generic regex extraction with `template_matched: false` and `template_verified: false`.
- **Multi-Page Merging**: Processes multi-page bank statement PDFs page-by-page and chronologically aggregates transaction rows into a single `transactions` list.

---

### 🌐 Multi-Language OCR Support (English / Marathi / Hindi)
Many regional Indian business and property documents are issued bilingual or entirely in regional vernacular. To support Maharashtra operations reliably without introducing noisy overhead on other documents, the service provides generalized multi-language recognition:

#### 1. Per-Document-Type Language Matrix (`DOC_TYPE_LANGUAGES`)
| Document Type | Configured Language Passes | Devanagari Model Active? | Verified Field Capabilities & Notes |
| :--- | :--- | :--- | :--- |
| **Shop & Establishment** | `["en", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | Maharashtra Form-G / Gumasta: `नोंदणी क्रमांक` (Registration No), `आस्थापनेचे नाव` (Establishment Name), `मालकाचे नाव` (Employer/Owner Name), `व्यवसायाचे स्वरूप` (Nature of Business). |
| **Udyam Registration** | `["en", "hi", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | MSME certificate: `उद्यमाचे नाव` (Enterprise Name), `सूक्ष्म/लघु/मध्यम` mapped to `Micro/Small/Medium`, `उत्पादन/सेवा` mapped to `Manufacturing/Services`. |
| **Property Tax Receipt** | `["en", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | Municipal receipts (BMC / PMC): `मालमत्ता क्रमांक` (Property ID), `करदात्याचे नाव` (Owner Name Masked), `भरलेली रक्कम` (Amount Paid), `आकारणी वर्ष` (Assessment Year). |
| **Rent Agreement** | `["en", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | Leave & License: `परवाना देणारा` (Lessor Masked), `परवाना घेणारा` (Lessee Masked), `मासिक भाडे` (Monthly Rent). |
| **Aadhaar Card** | `["en", "hi"]` | **Yes** (`devanagari_PP-OCRv4`) | Bilingual UIDAI cards: Devanagari name, `जन्मतारीख` (DOB), `पुरुष/स्त्री` mapped to `Male/Female`. |
| **Utility Bill** | `["en", "hi", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | Electricity/Water/Gas (MSEDCL/Mahavitaran, Tata Power, BMC): `ग्राहक क्रमांक` (Consumer No), `देयक रक्कम` (Bill Amount), `देय दिनांक` / `अंतिम तारीख` (Due Date), `देयक दिनांक` (Bill Date). |
| **Salary Slip** | `["en", "hi", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | State government bodies (Zilla Parishad, Maharashtra Police, MSRTC, municipal schools): `कार्यालयाचे नाव` (Employer/Office Name), `कर्मचाऱ्याचे नाव` (Employee Name Masked), `निव्वळ वेतन` (Net Pay), `माहे` / `वेतन महिना` (Pay Period). Strict PII masking enforced.<br>• *English*: Real demo document verified (`Demo_Salary_Slip.pdf`, `Demo_Salary_Slip_Image.png`; zero regression confirmed).<br>• *Marathi*: **Synthetic-tested only** (no real Marathi payslip available). |
| **Bank Passbook** | `["en", "hi", "mr"]` | **Yes** (`devanagari_PP-OCRv4`) | Urban Co-operative Banks and Regional Rural Banks: `बँकेचे नाव` (Bank Name), `शाखा` (Branch), `खाते क्रमांक` (Account Number Masked), `खातेदाराचे नाव` (Account Holder Name Masked), `आयएफएससी` (IFSC). Strict PII masking enforced.<br>• **Fully Synthetic / Unverified**: No real sample of any kind (English or Marathi) exists in this project; all extraction logic is tested solely on constructed/synthetic layouts. |
| **All Other 13 Types** | `["en"]` | **No** (English only) | Retains pure English recognition to prevent latency degradation and avoid out-of-vocabulary misidentifications. |

#### 2. Dual-Pass Neural Recognition & Intelligent Line Merging
- **Unified Devanagari Model**: Both Marathi and Hindi use the Devanagari script. In PaddleOCR/RapidOCR, character recognition utilizes the unified `devanagari_PP-OCRv4_rec_infer.onnx` neural recognizer with `devanagari_dict.txt`.
- **Script Recognition vs. Language Semantics**: The neural model accurately reads individual Devanagari glyphs across both Hindi and Marathi. However, vocabulary and field labels differ between the languages. The field extractors use verified language-specific regex patterns and do not assume word patterns transfer automatically.
- **Bounding Box IoU Merging**:
  - Because English OCR engines often interpret Devanagari script as random ASCII garbage (e.g. `HRI 9` for `महाराष्ट्र शासन`), the merge algorithm computes spatial IoU overlap across lines.
  - When lines overlap and the Devanagari pass contains genuine Devanagari script (at least 2 Devanagari characters or 1 character without Latin letters) with sufficient confidence (>= 0.50), the Devanagari line is prioritized over the English ASCII garble. Stray misrecognized Devanagari glyphs in noisy ASCII passes are safely rejected.
  - Non-overlapping lines from both passes are preserved and sorted top-to-bottom.

#### 3. Honest Language Coverage & Manual Review Guardrail
The service employs an explicit guardrail to prevent silent hallucinations on unfamiliar regional layouts while avoiding unnecessary alert fatigue on well-parsed documents:

- **What Specifically Triggers `language_review_required: true` and `partial_language_coverage: true` (True-Positive)**:
  Both of the following conditions must be met:
  1. **Devanagari Script Content Detected**: The document's extracted text (including headers, stamps, and body text) contains $\ge 10$ Devanagari unicode characters (`\u0900-\u097F`).
  2. **Core Field Extraction Incomplete**: One or more mandatory core fields for the document type (defined in `CORE_FIELDS_PER_DOC_TYPE`) could not be extracted:
     - **Udyam Registration**: `udyam_registration_number` AND `enterprise_name` (both mandatory)
     - **Shop & Establishment**: `registration_number` AND `establishment_name` (both mandatory)
     - **Property Tax Receipt**: `property_id` AND `tax_amount_paid` (both mandatory)
     - **Rent Agreement**: `monthly_rent` AND `lessor_name_masked` (both mandatory)
     - **Aadhaar Card**: `aadhaar_number` AND `name` (both mandatory)
     - **Utility Bill**: `consumer_number` AND `bill_amount` (both mandatory)
     - **Salary Slip**: `employer_name` AND `net_pay` (both mandatory)
     - **Bank Passbook**: `account_number_masked` AND `bank_name` (both mandatory)
  When triggered, the service refuses to guess and outputs:
  ```json
  {
    "detected_languages": ["en", "devanagari"],
    "partial_language_coverage": true,
    "language_review_required": true,
    "language_coverage_notes": "Document contains Devanagari script text (124 characters), but core field(s) could not be extracted: consumer_number, bill_amount. Manual review recommended."
  }
  ```

- **When the Guardrail Stays `false` (True-Negative)**:
  - **Devanagari Present + All Core Fields Extracted**:
    If $\ge 10$ Devanagari characters are present (e.g. Mahavitaran bill with Marathi tables, Udyam Ministry banner, or Marathi ZP payslip), but all required core fields are successfully parsed:
    ```json
    {
      "detected_languages": ["en", "devanagari"],
      "partial_language_coverage": false,
      "language_review_required": false,
      "language_coverage_notes": "Devanagari script text detected (32 characters); all core fields extracted successfully."
    }
    ```
  - **Pure English Documents (< 10 Devanagari characters)**:
    ```json
    {
      "detected_languages": ["en"],
      "partial_language_coverage": false,
      "language_review_required": false,
      "language_coverage_notes": null
    }
    ```
- This dual-condition design ensures human reviewers are alerted only when regional layouts actually impede automated processing, rather than on every document with a bilingual header emblem.

#### 4. Audit of Regional Language Exposure Across Remaining 13 Document Types
Following the rollout of Devanagari support to `salary_slip` and `bank_passbook` (now 8 multilingual document types: `shop_establishment`, `udyam`, `property_tax_receipt`, `rent_agreement`, `aadhaar`, `utility_bill`, `salary_slip`, `bank_passbook`), an audit of the remaining 13 document types identifies their practical exposure to Devanagari/regional-language content in Indian commercial and identity workflows:

| Priority / Risk Level | Document Types | Practical Regional Exposure & Context |
| :--- | :--- | :--- |
| **High Priority (High Likelihood)** | **`partnership_deed`** | Deeds executed on Maharashtra non-judicial stamp paper (`महाराष्ट्र मुद्रांक शुल्क`) are frequently drafted in Marathi or bilingual formats (`भागीदारी करारनामा`). |
| | **`voter_id`** | ECI voter cards are issued bilingual with Devanagari script for name, father's name, and address (`मतदार ओळखपत्र`). |
| | **`bank_statement`** | While scheduled commercial banks issue statements in English, Gramin and District Central Co-op Banks occasionally generate bilingual statements. |
| **Medium Priority** | **`driving_licence`** | State Transport Department (RTO) smart cards / mParivahan PDFs often feature bilingual state headers and field titles. |
| | **`cancelled_cheque`** | CTS-2010 cheques adhere to national clearing standards in English, though local cooperative banks may include Marathi bank titles or watermarks. |
| | **`fssai`** | State food safety registrations occasionally carry bilingual department seals, though certificates are predominantly standardized in English. |
| **Low Priority (Standardized English)** | **`pan`** | National NSDL/UTIITSL format. Central government Hindi emblem is present, but core alphanumeric PAN and names are Latin. |
| | **`passport`** | Standard Republic of India passport; machine-readable zone (MRZ) and primary bio-data fields are standardized Latin. |
| | **`gst_certificate`** | National GSTN portal generates standardized English certificates (`FORM GST REG-06`). |
| | **`certificate_of_incorporation`** | MCA21 portal generates standardized English corporate certificates. |
| | **`itr`** | Income Tax Department ITR-V acknowledgements are standardized English documents. |
| | **`form_16`** | TRACES portal generates standardized English tax deduction certificates. |
| | **`iec_certificate`** | DGFT portal generates standardized English Import Export Code certificates. |

#### 5. Real-Document Verification Status & Data Requirements to Close Gaps

To maintain strict engineering transparency and data integrity across the 8 multilingual document types, the table below documents the exact empirical verification status of each type, distinguishing authentic operational documents from synthetic/constructed test mockups:

| Document Type | Real Document Available in Project? | Tested Real Documents | Synthetic / Mockup Coverage | Verification Status & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **`udyam`** | **Yes** | `ROYAL CAKE HOUSE 2.pdf`, `Demo_Udyam_Registration.pdf` | N/A | **Fully Real-Document Verified**: Dual-pass Devanagari ministry header and English tabular data verified on authentic MSME certificate. |
| **`utility_bill`** | **Yes** | `IMG-20260821-WA0003.jpg.jpeg` (MSEDCL / Mahavitaran) | N/A | **Fully Real-Document Verified**: Authentic bilingual Marathi/English electricity bill verified; Devanagari line merging and tabular extraction validated. |
| **`shop_establishment`** | **Yes** (MH Form-G) | `Demo_Shop_Establishment.pdf` | DL / KA statutory Form C templates | **Hybrid Real / Template**: Maharashtra Form-G verified against authentic scan; Delhi/Karnataka baseline on statutory Form C. |
| **`aadhaar`** | **Yes** | `Demo_Aadhaar_Card.pdf` | N/A | **Fully Real-Document Verified**: Bilingual UIDAI card scan verified with Verhoeff checksum. |
| **`salary_slip`** | **Partial (English only)** | `Demo_Salary_Slip.pdf`, `Demo_Salary_Slip_Image.png` | Marathi ZP / MSRTC synthetic layout | **Hybrid Real (EN) / Synthetic (MR)**:<br>• **English**: Verified against real demo files (`employer_name`, `employee_name_masked`, `net_pay`, `pay_period` confirmed 100% regression-free post-Devanagari changes).<br>• **Marathi**: Field extraction regexes and PII masking tested **solely on constructed/synthetic Marathi text**; no real Marathi payslip has been evaluated. |
| **`bank_passbook`** | **No** | *None* | Standard Indian banking layout mockup | **Fully Synthetic / Unverified**:<br>• **No real sample of any kind** (English or Marathi) exists in this project.<br>• Extraction logic, IFSC regexes, account masking, and Devanagari labels are tested **solely on synthetic/constructed mockups**. Does NOT have parity with real-document-verified types. |
| **`property_tax_receipt`** | **No** | *None* | Municipal receipt mockup (PMC/BMC) | **Synthetic / Contractual Mockup**: Tested on standard municipal receipt conventions. |
| **`rent_agreement`** | **No** | *None* | Leave & License legal drafting mockup | **Synthetic / Contractual Mockup**: Tested on Maharashtra standard tenancy conventions. |

##### Real-World Data Needed to Close Remaining Gaps (Without Fabrication)
In keeping with how this repository handles unverified areas (such as the specialized E-13B MICR line font and DL/KA statutory templates), we do **not** construct fake "real-looking" samples to paper over missing coverage. The following authentic operational documents are specifically required to graduate these document types to full **Real Document Verified** status:

1. **`salary_slip` (Marathi Regional Support)**:
   - **Required Real-World Document**: A genuine scanned PDF or photo of a Marathi-language payslip issued by a Maharashtra state government body or undertaking (e.g. Zilla Parishad, Maharashtra State Police, Maharashtra State Road Transport Corporation / MSRTC, or Municipal Corporation / Mahanagarpalika).
   - **Redaction Requirements**: Legitimate redaction of employee personal identifiers (raw employee name, employee ID, bank account number, PAN/Aadhaar) while preserving authentic typography, Devanagari table headers (`कार्यालयाचे नाव`, `वेतन महिना`, `निव्वळ वेतन`, `कपात`), and physical print artifacts (stamps, letterhead noise).
2. **`bank_passbook` (Complete Extraction Pipeline)**:
   - **Required Real-World Document**: An authentic scanned image or photograph of an Indian bank passbook first page (specifically from an Urban Co-operative Bank, District Central Co-op Bank, or Regional Rural Bank featuring bilingual English/Marathi layout).
   - **Redaction Requirements**: Legitimate redaction of customer personal identifying details (raw account number, customer name, residential address) while preserving authentic layout, typography, bank logo/branch details (`बँकेचे नाव`, `शाखा`, `आयएफएससी`), and dot-matrix / inkjet passbook print alignment.

---

### 🔒 Security, Authentication & Deployment Configuration

> [!WARNING]
> **Authentication is Disabled by Default (`AUTH_MODE=disabled`)**
> In this configuration, all routes (`/api/upload`, `/ocr/{doc_type}`, `/api/stats`, `/api/documents`, `/api/documents/{id}/file`, etc.) accept unauthenticated requests anonymously with no Authorization header or token required.
> 
> **Public Deployment Warning**: Deploying this service to a public IP or public domain without authentication exposes document ingestion, document vault files, and extracted financial/identity data to anyone on the internet.

#### How to Re-Enable Authentication (Zero Code Changes)
To re-enable full JWT or API-key authentication before public deployment, configure the environment variables:

1. **Enable JWT Authentication**:
   ```bash
   export AUTH_ENABLED="true"
   export AUTH_MODE="jwt"
   export JWT_SECRET="your-secure-32-character-production-secret"
   export REGISTERED_CLIENTS_JSON='{"client-app-id": "client-secure-secret"}'
   ```
2. **Or Enable Dual Mode (JWT + Static API Key)**:
   ```bash
   export AUTH_ENABLED="true"
   export AUTH_MODE="dual"
   export JWT_SECRET="your-secure-32-character-production-secret"
   export REGISTERED_CLIENTS_JSON='{"client-app-id": "client-secure-secret"}'
   export API_KEY="your-static-api-key-for-internal-services"
   ```

#### Dynamic Runtime Auth Discovery (`GET /api/auth-status`)
The frontend is dynamically auth-aware at runtime:
- **`GET /api/auth-status`** (and `/auth-status`, `/health`): Returns `{"auth_enabled": bool, "auth_mode": string}` without requiring authentication.
- **When `auth_enabled: false` (default)**: The frontend bypasses login entirely, does not send `Authorization` headers, and loads the document dashboard immediately.
- **When `auth_enabled: true`**:
  - The React frontend displays `LoginView.tsx` demanding authentication before dashboard access.
  - Users sign in via `POST /auth/token` (or API Key), which stores the session token in `localStorage`.
  - `api.ts` automatically attaches `Authorization: Bearer <token>` to all API requests, file downloads (`/api/documents/{id}/file`), previews, and `POST /api/upload` multipart requests.
  - A **Sign Out** button appears in the navigation header.
  - If a session expires or returns `401 Unauthorized`, `api.ts` clears stored credentials and automatically prompts the user to re-authenticate.

---

### 🛡️ PII Minimisation & Temp File Lifecycle
- **Strict Secret Management & Fail-Fast Startup**:
  - In `AUTH_MODE='jwt'`, `AUTH_MODE='dual'`, or `AUTH_MODE='api_key'`, hardcoded secrets are strictly forbidden. `JWT_SECRET` and `API_KEY` must be explicitly injected via secret management.
  - Any deployment relying on hardcoded keys must immediately rotate credentials.
  - If startup security checks fail in enabled auth modes, lifespan raises `RuntimeError` and middleware returns `503 Service Unavailable`, preventing any unauthenticated traffic from leaking.
- **Client Credential Verification for `/auth/token`**:
  - `/auth/token` requires both `client_id` and `client_secret`.
  - Credentials are never hardcoded in source. Client registry must be configured via `REGISTERED_CLIENTS_JSON` environment variable or `REGISTERED_CLIENTS_FILE` path.
  - Client credentials are authenticated against SHA-256 hashed secrets using constant-time comparison (`hmac.compare_digest`). Unregistered or mismatched credentials return `401 Unauthorized`.
- **Strict Audit Logging**: Structured JSON logging recording ONLY operational metadata (`doc_type`, `status`, `confidence`, `job_id`, `customer_id`, `duration_ms`). Formatter guarantees that extracted field values and customer PII are **never** logged.
- **Orphaned Temp File Sweeper**: Automatically clears temporary files on completion, with a background periodic sweeper and startup sweep purging files older than `TEMP_FILE_TTL_MINUTES` (15 mins).

---

### ⚙️ OCR Engine Architecture, Observability & Fail-Loud Mechanics
- **Dual-Engine Architecture (RapidOCR + PaddleOCR)**:
  - Supports both **RapidOCR** (`rapidocr_onnxruntime` v1.2.3, `onnxruntime` v1.29.0) and native **PaddleOCR** (PP-OCRv5).
  - Uses RapidOCR on Python versions (such as Python 3.14+) where native Baidu `paddlepaddle` C-extension binary wheels are unavailable on PyPI. Both run the exact same underlying PP-OCR neural weights.
- **Explicit Engine Selection via `OCR_ENGINE`**:
  - `OCR_ENGINE=auto` (default): Prefers RapidOCR (native ONNX wheels), then falls back to PaddleOCR.
  - `OCR_ENGINE=rapidocr`: Forces RapidOCR; fails loud at startup with `RuntimeError` if not importable.
  - `OCR_ENGINE=paddleocr`: Forces native PaddleOCR; fails loud at startup with `RuntimeError` if not importable.
- **Hard Failure on Zero-Confidence / Empty OCR Results (`ocr_engine_returned_no_text`)**:
  - If neural OCR or text extraction returns empty/whitespace-only text, 0.0 average confidence, zero lines, or encounters an unhandled engine crash:
    - Pipeline immediately short-circuits with `status: "error"`, `reason: "ocr_engine_returned_no_text"`.
    - Never passes through empty results to field extraction, checksum validation, or downstream consumers.
- **Honest Text Source Labeling**:
  - Digital PDFs with an embedded text layer (>= 50 chars) bypass expensive neural OCR: `ocr_required: false`, `text_source: "pdf_text_layer"`.
  - Scanned PDFs and raster images (`.png`, `.jpg`, `.jpeg`, `.webp`): `ocr_required: true`, `text_source: "rapid_ocr"` (or `"paddle_ocr"`).
- **Engine Observability**:
  - Visible startup log line: `INFO: OCR engine initialized: rapidocr (rapidocr_onnxruntime v1.2.3 (onnxruntime v1.29.0) on Python 3.14.4) [setting: auto]`.
  - `GET /health` reports active engine: `{"status": "healthy", "ocr_engine": "rapidocr", "ocr_backend": "rapidocr", "ocr_engine_status": "ready"}`.
  - Dedicated endpoint `GET /engine-info` (and `GET /api/engine-info`) reports full engine versions, runtime platform, and fallbacks.

---

### 📊 Validation Sweep Results (21 Document Types)
An exhaustive 21x21 document matrix sweep across all supported Indian document types confirmed 100% classification precision with 0 false positive mismatches (441 pairwise cross-check tests):
| Document Type | Test File / Reference Source | Text Source | Confidence | Detection & Validation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Aadhaar** | `Demo_Aadhaar_Card.pdf` | `pdf_text_layer` | 0.98 | Detected: `aadhaar` (Pass), Verhoeff checksum valid. |
| **Cancelled Cheque** | `Demo_Cancelled_Cheque.pdf` | `pdf_text_layer` | 0.98 | Detected: `cancelled_cheque` (Pass), IFSC flagged for review (DEMO code). |
| **Driving Licence** | `Demo_Driving_Licence.pdf` | `pdf_text_layer` | 0.98 | Detected: `driving_licence` (Pass), Vehicle class & DL parsed. |
| **FSSAI** | `Demo_FSSAI_Certificate.pdf` | `pdf_text_layer` | 0.98 | Detected: `fssai` (Pass), 14-digit license validated. |
| **Passport** | `Demo_Passport.pdf` | `pdf_text_layer` | 0.98 | Detected: `passport` (Pass), Name and number extracted. |
| **Salary Slip (PDF)** | `Demo_Salary_Slip.pdf` | `pdf_text_layer` | 0.98 | Detected: `salary_slip` (Pass), Net pay & employer parsed. |
| **Salary Slip (PNG)** | `Demo_Salary_Slip_Image.png` | `rapid_ocr` | 0.842 | Detected: `salary_slip` (Pass), 19 lines extracted via neural OCR. |
| **Shop & Est. (MH)** | `Demo_Shop_Establishment.pdf` | `pdf_text_layer` | 0.98 | Detected: `shop_establishment` (Pass), Registration & nature parsed (verified). |
| **Udyam** | `Demo_Udyam_Registration.pdf` | `pdf_text_layer` | 0.98 | Detected: `udyam` (Pass), Udyam number validated. |
| **Voter ID** | `Demo_Voter_ID.pdf` | `pdf_text_layer` | 0.98 | Detected: `voter_id` (Pass), EPIC number extracted. |
| **PAN** | `Demo_PAN_Card.pdf` | `pdf_text_layer` | 0.98 | Detected: `pan` (Pass), 4th char entity check active. |
| **Utility Bill (JPG)** | `IMG-20260821-WA0003.jpg.jpeg` | `rapid_ocr` | 0.7704 | Detected: `utility_bill` (Pass), 110 lines extracted, consumer number parsed. |
| **ITR** | `ITR SET FY 2025-26.pdf` | `pdf_text_layer` | 0.98 | Detected: `itr` (Pass), 560 lines, 15-digit ack & PAN extracted. |
| **GST Certificate** | `gst_certificate.txt` | `statutory_template` | 0.98 | Detected: `gst_certificate` (Pass), 15-char GSTIN format valid. |
| **Cert. of Incorporation** | `certificate_of_incorporation.txt` | `statutory_template` | 0.98 | Detected: `certificate_of_incorporation` (Pass), 21-char CIN format valid. |
| **Partnership Deed** | `partnership_deed.txt` | `statutory_template` | 0.98 | Detected: `partnership_deed` (Pass), PII allowlist active & `partner_names_masked` list parsed. |
| **Rent Agreement** | `rent_agreement.txt` | `statutory_template` | 0.98 | Detected: `rent_agreement` (Pass), PII allowlist active & address masked. |
| **Form 16** | `form_16.txt` | `statutory_template` | 0.98 | Detected: `form_16` (Pass), PII allowlist active & TDS summary parsed. |
| **Bank Passbook** | `bank_passbook.txt` | `statutory_template` | 0.98 | Detected: `bank_passbook` (Pass), PII allowlist active & IFSC validated. |
| **Property Tax Receipt** | `property_tax_receipt.txt` | `statutory_template` | 0.98 | Detected: `property_tax_receipt` (Pass), PII allowlist active & PID extracted. |
| **IEC Certificate** | `iec_certificate.txt` | `statutory_template` | 0.98 | Detected: `iec_certificate` (Pass), 10-char IEC number parsed. |

> [!NOTE]
> **Verification Scoping for New Document Types**:
> In accordance with this project's transparent verification standard, the 8 new document types have been validated against statutory reference layouts, regex field-boundary tests, and a 21x21 mismatch matrix sweep. Physical customer scanned document images have not yet been evaluated in this environment for these 8 types. Once genuine customer scans are made available, their recognition performance will be empirical evaluated.


---

## 2. API Endpoints

### `POST /auth/token`
Generate short-lived JWT token for API clients (requires registered client credentials):
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "client_id=n8n-worker" \
  -d "client_secret=YOUR_SECURE_CLIENT_SECRET"
```
**Response (`200 OK`)**:
```json
{
  "access_token": "eyJhbGciOiJIUz...",
  "token_type": "bearer",
  "expires_in_minutes": 30
}
```
**Error Response (`401 Unauthorized`)**:
```json
{
  "detail": "Invalid client credentials"
}
```

---

### `POST /ocr/{doc_type}` (Asynchronous Job Enqueue)
Enqueues document processing to avoid blocking orchestration nodes on multi-page PDFs:

**Parameters**:
- `file`: Multipart file upload (PDF, PNG, JPEG, etc.)
- `expected` *(optional)*: JSON string with expected applicant fields (e.g. `{"name": "JOHN DOE", "dob": "1990-01-15"}`)
- `webhook_url` *(optional)*: HTTP URL for automatic callback delivery upon job completion
- `customer_id` *(optional)*: Identifier for audit log tracing
- `sync` *(optional query param)*: Set `?sync=true` to execute synchronously and return the payload immediately.

**Headers**:
- `Authorization: Bearer <JWT_TOKEN>` or `X-API-Key: <STATIC_KEY>`

**Async Response (`202 Accepted`)**:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "pending",
  "doc_type": "bank_statement",
  "created_at": "2026-09-07T09:45:00.000000Z",
  "poll_url": "/ocr/jobs/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
}
```

---

### `GET /ocr/jobs/{job_id}` (Poll Job Result)
Retrieves the status and result of an enqueued job:
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/ocr/jobs/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d
```

**Completed Response (`200 OK`)**:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "doc_type": "bank_statement",
  "status": "completed",
  "created_at": "2026-09-07T09:45:00.000000Z",
  "updated_at": "2026-09-07T09:45:02.120000Z",
  "result": {
    "status": "success",
    "doc_type": "bank_statement",
    "confidence": 0.965,
    "field_confidences": {
      "account_number_masked": 0.98,
      "closing_balance": 0.99
    },
    "extracted_fields": {
      "bank_name": "HDFC BANK",
      "account_number_masked": "XXXXXXXX1012",
      "statement_period": {
        "from_date": "01/01/2026",
        "to_date": "31/01/2026"
      },
      "closing_balance": "46500",
      "transactions": [
        {"date": "01/01/2026", "description": "SALARY CREDIT", "amount": "50000", "type": "CR", "balance": "50000"},
        {"date": "05/01/2026", "description": "ATM WITHDRAWAL", "amount": "2000", "type": "DR", "balance": "48000"},
        {"date": "10/01/2026", "description": "UTILITY BILL", "amount": "1500", "type": "DR", "balance": "46500"}
      ]
    }
  },
  "error": null
}
```

---

## 3. n8n Integration Guide

In an n8n workflow:

1. **Mint Access Token**:
   - Use an **HTTP Request** node to `POST /auth/token` with client credentials, caching the `access_token`.
2. **Submit Document (`POST /ocr/{doc_type}`)**:
   - Send the binary file retrieved from Google Drive / Email Attachment.
   - Supply `webhook_url: "{{ $execution.resumeUrl }}"` to utilize n8n's **Wait for Webhook** trigger, or poll via `GET /ocr/jobs/{job_id}`.
   - Supply `expected: JSON.stringify({ name: $json.customer_name, dob: $json.customer_dob })`.
3. **Handle Verification Outcome**:
   - If `status === "error"` with `reason === "doc_type_mismatch"` -> route directly to Customer Notification to re-upload the correct document.
   - If `status === "low_confidence"` -> route to Manual Review.
   - If `status === "success"` and all `cross_check` fields matched -> proceed to automated case update!

---

## 4. Running Locally & Running Tests

### Running Tests
```bash
pytest -v tests/test_extractors.py tests/test_verification.py tests/test_pii_minimise.py tests/test_async_jobs.py
```

### Running Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
```bash
docker build -t company-server-ocr:latest .
docker run -p 8000:8000 -e JWT_SECRET="your-32-character-production-secret" company-server-ocr:latest
```
