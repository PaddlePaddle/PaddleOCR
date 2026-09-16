"""
ocr_engine.py
PaddleOCR (PP-OCRv5 + PPStructureV3) engine wrapper:
- Runs text detection & recognition with per-line confidence scores
- Multi-page PDF handling (extracts and runs OCR per page)
- Extracts tabular structures for bank statements
- Seamless fallback for PDF text extraction when PaddlePaddle binaries are absent
"""

import glob
import importlib.metadata
import io
import logging
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

logger = logging.getLogger("company_ocr.engine")

# Global engine state
HAS_PADDLEOCR = False
_paddleocr_instance = None
_ocr_backend: Optional[str] = None
_ocr_engine_details: str = "uninitialized"


def _detect_installed_versions() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Detect installed versions of rapidocr-onnxruntime, onnxruntime, and paddleocr."""
    rapid_ver = None
    onnx_ver = None
    paddle_ver = None
    try:
        rapid_ver = importlib.metadata.version("rapidocr-onnxruntime")
    except Exception:
        pass
    try:
        onnx_ver = importlib.metadata.version("onnxruntime")
    except Exception:
        pass
    try:
        paddle_ver = importlib.metadata.version("paddleocr")
    except Exception:
        pass
    return rapid_ver, onnx_ver, paddle_ver


def init_ocr_engine(force_engine: Optional[str] = None):
    """
    Initialize neural OCR engine according to OCR_ENGINE setting or force_engine parameter.
    Supported values: 'auto', 'rapidocr', 'paddleocr'.
    Fails loud if a specific engine is explicitly requested but cannot be loaded.
    """
    global HAS_PADDLEOCR, _paddleocr_instance, _ocr_backend, _ocr_engine_details

    setting = (force_engine or os.getenv("OCR_ENGINE", "auto")).lower().strip()
    if setting not in ("auto", "rapidocr", "paddleocr"):
        raise ValueError(
            f"Invalid OCR_ENGINE setting '{setting}'. Allowed values are 'auto', 'rapidocr', 'paddleocr'."
        )

    rapid_ver, onnx_ver, paddle_ver = _detect_installed_versions()
    py_ver = platform.python_version()

    _paddleocr_instance = None
    _ocr_backend = None
    HAS_PADDLEOCR = False

    if setting == "paddleocr":
        try:
            from paddleocr import PaddleOCR
            _paddleocr_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            HAS_PADDLEOCR = True
            _ocr_backend = "paddleocr"
            _ocr_engine_details = f"paddleocr v{paddle_ver or 'unknown'} on Python {py_ver}"
        except Exception as ex:
            _ocr_engine_details = f"paddleocr initialization failed: {str(ex)}"
            raise RuntimeError(
                f"Configured OCR_ENGINE 'paddleocr' could not be initialized: {str(ex)}. "
                f"Please install PaddlePaddle wheels for your Python version, or switch to "
                f"'OCR_ENGINE=rapidocr' or 'OCR_ENGINE=auto' to use the ONNX-backed engine."
            ) from ex

    elif setting == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR
            _paddleocr_instance = RapidOCR()
            HAS_PADDLEOCR = True
            _ocr_backend = "rapidocr"
            _ocr_engine_details = (
                f"rapidocr_onnxruntime v{rapid_ver or '1.2.3'} (onnxruntime v{onnx_ver or 'unknown'}) on Python {py_ver}"
            )
        except Exception as ex:
            _ocr_engine_details = f"rapidocr initialization failed: {str(ex)}"
            raise RuntimeError(
                f"Configured OCR_ENGINE 'rapidocr' could not be initialized: {str(ex)}"
            ) from ex

    else:  # 'auto'
        # First preference: RapidOCR (cross-platform, native ONNX runtime wheels for Python 3.14+)
        try:
            from rapidocr_onnxruntime import RapidOCR
            _paddleocr_instance = RapidOCR()
            HAS_PADDLEOCR = True
            _ocr_backend = "rapidocr"
            _ocr_engine_details = (
                f"rapidocr_onnxruntime v{rapid_ver or '1.2.3'} (onnxruntime v{onnx_ver or 'unknown'}) on Python {py_ver}"
            )
        except Exception as rapid_ex:
            # Second preference: native PaddleOCR
            try:
                from paddleocr import PaddleOCR
                _paddleocr_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                HAS_PADDLEOCR = True
                _ocr_backend = "paddleocr"
                _ocr_engine_details = f"paddleocr v{paddle_ver or 'unknown'} on Python {py_ver}"
            except Exception as paddle_ex:
                HAS_PADDLEOCR = False
                _ocr_backend = None
                _ocr_engine_details = (
                    f"no neural OCR engine available (rapidocr error: {str(rapid_ex)}; paddleocr error: {str(paddle_ex)})"
                )
                logger.warning(
                    "Startup OCR check: No neural OCR engine available (%s). Only digital PDFs with embedded text layers will be processed.",
                    _ocr_engine_details,
                )

    if HAS_PADDLEOCR:
        logger.info(
            "OCR engine initialized: %s (%s) [setting: %s]",
            _ocr_backend,
            _ocr_engine_details,
            setting,
        )


# Run initial engine detection at import time
try:
    init_ocr_engine()
except Exception as _init_err:
    logger.warning("Initial OCR engine startup notice: %s", _init_err)


def validate_ocr_engine_configuration():
    """
    Called during application startup to enforce fail-loud behavior if configured OCR_ENGINE cannot run.
    """
    setting = os.getenv("OCR_ENGINE", "auto").lower().strip()
    if setting not in ("auto", "rapidocr", "paddleocr"):
        raise ValueError(
            f"Invalid OCR_ENGINE '{setting}'. Allowed values: 'auto', 'rapidocr', 'paddleocr'."
        )
    if not HAS_PADDLEOCR:
        if setting in ("paddleocr", "rapidocr"):
            raise RuntimeError(
                f"Configured OCR_ENGINE '{setting}' is not available: {_ocr_engine_details}"
            )
    logger.info("Active OCR Engine: %s | Details: %s", _ocr_backend or "none", _ocr_engine_details)


def get_active_ocr_engine() -> Optional[str]:
    """Returns identifier of the active neural OCR engine ('rapidocr', 'paddleocr', or None)."""
    return _ocr_backend


def get_ocr_engine_info() -> Dict[str, Any]:
    """Returns structured engine metadata for /health and /engine-info endpoints."""
    rapid_ver, onnx_ver, paddle_ver = _detect_installed_versions()
    active = _ocr_backend or "none"
    display = (
        "RapidOCR" if active == "rapidocr" else ("PaddleOCR" if active == "paddleocr" else "None")
    )
    version = rapid_ver if active == "rapidocr" else (paddle_ver or "1.0")
    return {
        "engine": active,
        "active_engine": active,
        "display_name": display,
        "device": "CPU",
        "backend": active,
        "version": version,
        "configured_engine": os.getenv("OCR_ENGINE", "auto").lower().strip(),
        "status": "ready" if HAS_PADDLEOCR else "unavailable",
        "details": _ocr_engine_details,
        "rapidocr_version": rapid_ver,
        "onnxruntime_version": onnx_ver,
        "paddleocr_version": paddle_ver,
        "python_version": platform.python_version(),
        "pdf_fallback": "pdftotext",
        "devanagari_model_available": bool(
            HAS_DEVANAGARI_MODEL
            or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "devanagari_PP-OCRv4_rec_infer.onnx"))
            or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "devanagari_PP-OCRv3_rec_infer.onnx"))
        ),
    }


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: Optional[List[List[float]]] = None


@dataclass
class OCRPageResult:
    page_num: int
    full_text: str
    lines: List[OCRLine]
    average_confidence: float
    tables: List[List[List[str]]] = field(default_factory=list)
    image: Optional[Image.Image] = None
    engine_error: Optional[str] = None


@dataclass
class OCRDocumentResult:
    pages: List[OCRPageResult]
    full_text: str
    average_confidence: float
    field_confidences: Dict[str, float] = field(default_factory=dict)
    ocr_required: bool = False
    text_source: str = "pdf_text_layer"
    engine_error: Optional[str] = None


def get_paddleocr_instance():
    """Lazily initialize and reuse PaddleOCR / RapidOCR instance."""
    global _paddleocr_instance
    if _paddleocr_instance is None and os.getenv("OCR_ENGINE", "auto").lower().strip() != "none":
        try:
            init_ocr_engine()
        except Exception:
            pass
    return _paddleocr_instance


# ------------------------------------------------------------------------------
# Multi-Language OCR Configuration & Devanagari Engine
# ------------------------------------------------------------------------------

_devanagari_ocr_instance = None
HAS_DEVANAGARI_MODEL = False

# Mapping of document type to required language passes
# Maharashtra-originating documents and national IDs enable Devanagari alongside English.
DOC_TYPE_LANGUAGES: Dict[str, List[str]] = {
    "shop_establishment": ["en", "mr"],
    "udyam": ["en", "hi", "mr"],
    "property_tax_receipt": ["en", "mr"],
    "rent_agreement": ["en", "mr"],
    "aadhaar": ["en", "hi"],
    "utility_bill": ["en", "hi", "mr"],
    "salary_slip": ["en", "hi", "mr"],
    "bank_passbook": ["en", "hi", "mr"],
    "income_certificate": ["en", "hi", "mr"],
}



def get_languages_for_doc_type(doc_type: Optional[str]) -> List[str]:
    """Return list of language codes to execute for a given document type."""
    if not doc_type:
        return ["en"]
    return DOC_TYPE_LANGUAGES.get(doc_type.lower().strip(), ["en"])


def get_devanagari_ocr_instance():
    """
    Lazily initialize and reuse Devanagari OCR instance for Hindi/Marathi text recognition.
    PaddleOCR: maps 'mr' and 'hi' to unified 'devanagari' model family.
    RapidOCR: loads devanagari ONNX recognition model with devanagari_dict.txt.
    """
    global _devanagari_ocr_instance, HAS_DEVANAGARI_MODEL
    if _devanagari_ocr_instance is not None:
        return _devanagari_ocr_instance

    if _ocr_backend == "paddleocr":
        try:
            from paddleocr import PaddleOCR
            _devanagari_ocr_instance = PaddleOCR(use_angle_cls=True, lang="devanagari", show_log=False)
            HAS_DEVANAGARI_MODEL = True
            logger.info("PaddleOCR Devanagari engine initialized successfully")
            return _devanagari_ocr_instance
        except Exception as e:
            logger.warning("PaddleOCR Devanagari model initialization failed: %s", e)
            return None

    # RapidOCR backend (ONNX runtime)
    try:
        from rapidocr_onnxruntime.utils import UpdateParameters
        orig_update_rec = UpdateParameters.update_rec_params

        def custom_update_rec(self, config, rec_dict):
            if rec_dict:
                need_remove_prefix = ["rec_model_path", "rec_keys_path"]
                new_rec_dict = {}
                for k, v in rec_dict.items():
                    if k in need_remove_prefix:
                        k = k.split("rec_")[1]
                    new_rec_dict[k] = v
                if "model_path" not in new_rec_dict:
                    new_rec_dict["model_path"] = config.get("model_path", "")
                config.update(new_rec_dict)
            return config

        UpdateParameters.update_rec_params = custom_update_rec

        from rapidocr_onnxruntime import RapidOCR

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.join(base_dir, "models"),
            "/home/vighnesh/PaddleOCR/models",
            "/opt/models",
        ]
        model_path = None
        dict_path = None

        for cdir in candidate_dirs:
            v4 = os.path.join(cdir, "devanagari_PP-OCRv4_rec_infer.onnx")
            v3 = os.path.join(cdir, "devanagari_PP-OCRv3_rec_infer.onnx")
            d = os.path.join(cdir, "devanagari_dict.txt")
            if not model_path:
                if os.path.exists(v4):
                    model_path = v4
                elif os.path.exists(v3):
                    model_path = v3
            if not dict_path and os.path.exists(d):
                dict_path = d

        if not dict_path:
            dict_path = os.path.join(base_dir, "ppocr", "utils", "dict", "devanagari_dict.txt")

        if model_path and os.path.exists(dict_path):
            _devanagari_ocr_instance = RapidOCR(
                rec_model_path=model_path,
                rec_keys_path=dict_path,
            )
            HAS_DEVANAGARI_MODEL = True
            logger.info("RapidOCR Devanagari engine initialized (model=%s, dict=%s)", model_path, dict_path)
            return _devanagari_ocr_instance
        else:
            logger.warning(
                "Devanagari OCR model or dict not found on disk (model=%s, dict=%s)",
                model_path,
                dict_path,
            )
            return None
    except Exception as ex:
        logger.warning("RapidOCR Devanagari initialization failed: %s", ex)
        return None


def _calc_bbox_overlap(b1: Optional[List[List[float]]], b2: Optional[List[List[float]]]) -> Tuple[float, float]:
    """Compute IoU and intersection-over-min-area between two 4-point bounding boxes."""
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return 0.0, 0.0
    x1_min, y1_min = min(p[0] for p in b1), min(p[1] for p in b1)
    x1_max, y1_max = max(p[0] for p in b1), max(p[1] for p in b1)

    x2_min, y2_min = min(p[0] for p in b2), min(p[1] for p in b2)
    x2_max, y2_max = max(p[0] for p in b2), max(p[1] for p in b2)

    iw = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    ih = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    iarea = iw * ih
    a1 = max(0.0, (x1_max - x1_min) * (y1_max - y1_min))
    a2 = max(0.0, (x2_max - x2_min) * (y2_max - y2_min))
    uarea = a1 + a2 - iarea
    if uarea <= 0.0 or a1 <= 0.0 or a2 <= 0.0:
        return 0.0, 0.0
    return iarea / uarea, iarea / min(a1, a2)


def merge_ocr_lines(lines_en: List[OCRLine], lines_dev: List[OCRLine]) -> List[OCRLine]:
    """
    Intelligently merge OCR lines from English and Devanagari passes.
    Matches bounding boxes across passes via spatial overlap:
    - If Devanagari candidate contains genuine Devanagari characters (\\u0900-\\u097F)
      and has adequate confidence, choose Devanagari (since English model outputs ASCII garbage for Devanagari).
    - If neither or both contain Devanagari, select line with higher confidence.
    - Unmatched boxes from either pass are retained.
    Lines are sorted top-to-bottom in reading order.
    """
    if not lines_dev:
        return lines_en
    if not lines_en:
        return lines_dev

    has_devanagari_chars = re.compile(r"[\u0900-\u097F]")
    matched_dev_indices = set()
    merged: List[OCRLine] = []

    for line_en in lines_en:
        best_dev_idx = None
        best_overlap = 0.0

        for idx, line_dev in enumerate(lines_dev):
            if idx in matched_dev_indices:
                continue
            iou, io_min = _calc_bbox_overlap(line_en.bbox, line_dev.bbox)
            if (iou >= 0.35 or io_min >= 0.50) and io_min > best_overlap:
                best_overlap = io_min
                best_dev_idx = idx

        if best_dev_idx is not None:
            matched_dev_indices.add(best_dev_idx)
            line_dev = lines_dev[best_dev_idx]

            # Require genuine Devanagari script (at least 2 characters, or 1 character with no Latin letters)
            # to prevent noisy ASCII lines with a single stray Devanagari glyph from overriding clear English OCR.
            dev_chars_dev = len(has_devanagari_chars.findall(line_dev.text))
            lat_chars_dev = len(re.findall(r"[A-Za-z]", line_dev.text))
            dev_has_script = (dev_chars_dev >= 2) or (dev_chars_dev >= 1 and lat_chars_dev == 0)

            dev_chars_en = len(has_devanagari_chars.findall(line_en.text))
            lat_chars_en = len(re.findall(r"[A-Za-z]", line_en.text))
            en_has_script = (dev_chars_en >= 2) or (dev_chars_en >= 1 and lat_chars_en == 0)

            if dev_has_script and not en_has_script:
                # Devanagari model recognized native script; English model produced ASCII substitute
                if line_dev.confidence >= 0.50 or line_dev.confidence >= (line_en.confidence - 0.20):
                    merged.append(line_dev)
                elif line_en.confidence > line_dev.confidence:
                    merged.append(line_en)
                else:
                    merged.append(line_dev)
            elif en_has_script and not dev_has_script:
                merged.append(line_en)
            else:
                # Both or neither have script: choose higher confidence
                if line_dev.confidence > line_en.confidence:
                    merged.append(line_dev)
                else:
                    merged.append(line_en)
        else:
            merged.append(line_en)

    # Add unmatched lines from Devanagari pass
    for idx, line_dev in enumerate(lines_dev):
        if idx not in matched_dev_indices:
            merged.append(line_dev)

    # Sort lines in natural top-to-bottom, left-to-right reading order
    def _sort_key(line: OCRLine):
        if line.bbox and len(line.bbox) >= 4:
            min_y = min(p[1] for p in line.bbox)
            min_x = min(p[0] for p in line.bbox)
            return (round(min_y / 15.0) * 15.0, min_x)
        return (0.0, 0.0)

    merged.sort(key=_sort_key)
    return merged



def extract_text_from_pdf_pdftotext(pdf_path: str) -> List[str]:
    """Extract text page-by-page using pdftotext utility if available."""
    pages_text = []
    try:
        # Check number of pages or extract with form feed separator
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        raw_output = result.stdout
        # Form feed (\x0c) separates pages in pdftotext
        pages = raw_output.split("\x0c")
        pages_text = [p.strip() for p in pages if p.strip()]
        if not pages_text and raw_output.strip():
            pages_text = [raw_output.strip()]
    except Exception:
        pass
    return pages_text


def parse_text_into_ocr_lines(text: str, default_conf: float = 0.96) -> List[OCRLine]:
    """Convert raw page text into structured OCRLine objects with baseline confidence."""
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line:
            # Synthetic / digital text extraction has near-perfect baseline confidence
            lines.append(OCRLine(text=line, confidence=default_conf))
    return lines


def check_pdf_text_layer(pdf_path_or_bytes: Union[str, bytes], min_char_threshold: int = 50) -> Tuple[bool, str, List[str]]:
    """
    Determines if a PDF has an existing text layer.
    Extracts text using pdftotext.
    Returns:
        (has_text_layer: bool, full_text: str, pages_text: List[str])
    If len(full_text.strip()) >= min_char_threshold (default 50 chars),
    has_text_layer is True (meaning OCR is NOT required).
    """
    tmp_file = None
    try:
        if isinstance(pdf_path_or_bytes, bytes):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_path_or_bytes)
                tmp_file = f.name
            path_to_read = tmp_file
        else:
            path_to_read = pdf_path_or_bytes

        pages_text = extract_text_from_pdf_pdftotext(path_to_read)
        full_text = "\n\n".join(pages_text).strip()
        has_text_layer = len(full_text) >= min_char_threshold
        return has_text_layer, full_text, pages_text
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


def render_pdf_pages_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """
    Render all pages of a PDF to PIL Images using Poppler's pdftoppm.
    Falls back to PyMuPDF (fitz) if available.
    """
    images = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_prefix = os.path.join(tmp_dir, "page")
        cmd = ["pdftoppm", "-png", "-r", str(dpi), pdf_path, out_prefix]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            png_files = sorted(
                glob.glob(os.path.join(tmp_dir, "page-*.png")),
                key=lambda p: int(re.findall(r"page-(\d+)\.png", p)[0]) if re.findall(r"page-(\d+)\.png", p) else 0
            )
            for p in png_files:
                with Image.open(p) as img:
                    images.append(img.convert("RGB").copy())
        except Exception:
            pass

    if not images:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        except Exception:
            pass

    return images


def render_thumbnail(file_path_or_bytes: Union[str, bytes], is_pdf: bool = False, max_size: Tuple[int, int] = (400, 400)) -> Optional[bytes]:
    """
    Generates PNG thumbnail bytes for a document (PDF or image).
    """
    try:
        if is_pdf:
            tmp_pdf = None
            try:
                if isinstance(file_path_or_bytes, bytes):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                        f.write(file_path_or_bytes)
                        tmp_pdf = f.name
                    path_to_read = tmp_pdf
                else:
                    path_to_read = file_path_or_bytes

                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_prefix = os.path.join(tmp_dir, "thumb")
                    cmd = ["pdftoppm", "-png", "-r", "100", "-f", "1", "-l", "1", path_to_read, out_prefix]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    pngs = glob.glob(os.path.join(tmp_dir, "thumb-*.png"))
                    if pngs:
                        with Image.open(pngs[0]) as img:
                            img.thumbnail(max_size, Image.Resampling.LANCZOS)
                            buf = io.BytesIO()
                            img.convert("RGB").save(buf, format="PNG", optimize=True)
                            return buf.getvalue()
            finally:
                if tmp_pdf and os.path.exists(tmp_pdf):
                    try:
                        os.remove(tmp_pdf)
                    except Exception:
                        pass
        else:
            if isinstance(file_path_or_bytes, bytes):
                img = Image.open(io.BytesIO(file_path_or_bytes))
            else:
                img = Image.open(file_path_or_bytes)
            with img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG", optimize=True)
                return buf.getvalue()
    except Exception:
        pass
    return None


class OCREngine:
    """Production OCR Engine utilizing PaddleOCR with multi-page handling."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.paddle = get_paddleocr_instance()

    @staticmethod
    def get_languages_for_doc_type(doc_type: Optional[str]) -> List[str]:
        return get_languages_for_doc_type(doc_type)

    def _run_single_engine_ocr(self, engine, arr) -> List[OCRLine]:
        """Run a single OCR engine instance on a numpy RGB image array."""
        lines: List[OCRLine] = []
        if _ocr_backend == "rapidocr":
            results, elapse = engine(arr)
            if results:
                for item in results:
                    bbox = item[0]
                    txt = item[1]
                    score = float(item[2])
                    lines.append(OCRLine(text=txt, confidence=score, bbox=bbox))
        else:
            results = engine.ocr(arr, cls=True)
            if results and results[0]:
                for item in results[0]:
                    bbox = item[0]
                    txt, score = item[1]
                    lines.append(OCRLine(text=txt, confidence=float(score), bbox=bbox))
        return lines

    def process_image(
        self,
        img: Image.Image,
        page_num: int = 1,
        languages: Optional[List[str]] = None,
    ) -> OCRPageResult:
        """
        Run OCR on a single PIL Image.
        Optionally executes bilingual pass (English + Devanagari) if Marathi/Hindi languages are configured.
        """
        engine = get_paddleocr_instance()
        if engine is not None:
            try:
                import numpy as np
                arr = np.array(img.convert("RGB"))
                lines_en = self._run_single_engine_ocr(engine, arr)

                # Determine if a Devanagari pass is requested
                needs_devanagari = False
                if languages:
                    lang_set = {l.lower().strip() for l in languages}
                    if any(l in lang_set for l in ("hi", "mr", "devanagari")):
                        needs_devanagari = True

                dev_engine = get_devanagari_ocr_instance() if needs_devanagari else None
                if dev_engine is not None:
                    lines_dev = self._run_single_engine_ocr(dev_engine, arr)
                    lines = merge_ocr_lines(lines_en, lines_dev)
                else:
                    lines = lines_en


                total_conf = sum(l.confidence for l in lines)
                avg_conf = (total_conf / len(lines)) if lines else 0.0
                full_text = "\n".join([line.text for line in lines])
                return OCRPageResult(
                    page_num=page_num,
                    full_text=full_text,
                    lines=lines,
                    average_confidence=round(avg_conf, 4),
                    image=img,
                    engine_error=None,
                )
            except Exception as ex:
                logger.error("OCR inference error on page %s: %s", page_num, ex, exc_info=True)
                return OCRPageResult(
                    page_num=page_num,
                    full_text="",
                    lines=[],
                    average_confidence=0.0,
                    image=img,
                    engine_error=f"ocr_inference_failed: {str(ex)}",
                )

        # Fallback for image when neural OCR engine is not loaded
        return OCRPageResult(
            page_num=page_num,
            full_text="",
            lines=[],
            average_confidence=0.0,
            image=img,
            engine_error="neural_ocr_engine_not_available",
        )

    def process_pdf(
        self,
        pdf_path: str,
        languages: Optional[List[str]] = None,
    ) -> OCRDocumentResult:
        """
        Process a multi-page PDF document.
        First checks if the PDF has an embedded text layer (>= 50 chars).
        If text layer exists:
            OCR is not required. Extracts embedded text directly.
            ocr_required=False, text_source="pdf_text_layer"
        If text layer does NOT exist (scanned / image-only):
            OCR is required. Renders pages to images and runs neural OCR with configured languages.
            ocr_required=True, text_source="rapid_ocr" / "paddle_ocr"
        """
        has_text_layer, full_text, pages_text = check_pdf_text_layer(pdf_path, min_char_threshold=50)

        if has_text_layer:
            # Check if requested languages specify non-English (Devanagari) script that is missing
            # from the digital text layer (e.g. raster image emblems/headers in official PDF certificates)
            dev_in_text = len(re.findall(r"[\u0900-\u097F]", full_text))
            needs_devanagari = False
            if languages:
                lang_set = {l.lower().strip() for l in languages}
                if any(l in lang_set for l in ("hi", "mr", "devanagari")) and dev_in_text < 10:
                    needs_devanagari = True

            extra_lines: List[OCRLine] = []
            first_rendered_img: Optional[Image.Image] = None
            if needs_devanagari:
                try:
                    # Scan page 1 image for visual Devanagari headers/logos
                    rendered_imgs = render_pdf_pages_to_images(pdf_path)
                    if rendered_imgs:
                        first_rendered_img = rendered_imgs[0]
                        ocr_p1 = self.process_image(first_rendered_img, page_num=1, languages=languages)
                        extra_lines = [l for l in ocr_p1.lines if re.search(r"[\u0905-\u0939]{2,}", l.text)]
                except Exception as ex:
                    logger.warning("Error running header OCR scan on digital PDF: %s", ex)

            page_results: List[OCRPageResult] = []
            for idx, p_text in enumerate(pages_text, start=1):
                lines = parse_text_into_ocr_lines(p_text, default_conf=0.98)
                page_img = first_rendered_img if idx == 1 else None
                if idx == 1 and extra_lines:
                    lines = extra_lines + lines
                    p_text = "\n".join([l.text for l in extra_lines]) + "\n" + p_text
                avg_conf = sum(l.confidence for l in lines) / len(lines) if lines else 0.98
                page_results.append(
                    OCRPageResult(
                        page_num=idx,
                        full_text=p_text,
                        lines=lines,
                        average_confidence=round(avg_conf, 4),
                        image=page_img,
                        engine_error=None,
                    )
                )
            if extra_lines:
                full_text = "\n".join([l.text for l in extra_lines]) + "\n\n" + full_text

            overall_conf = (
                sum(p.average_confidence for p in page_results) / len(page_results)
                if page_results
                else 0.98
            )
            return OCRDocumentResult(
                pages=page_results,
                full_text=full_text,
                average_confidence=round(overall_conf, 4),
                ocr_required=False,
                text_source="pdf_text_layer",
                engine_error=None,
            )

        # Scanned PDF or insufficient text layer (< 50 chars) -> OCR required
        rendered_images = render_pdf_pages_to_images(pdf_path)
        page_results: List[OCRPageResult] = []

        if rendered_images:
            for idx, img in enumerate(rendered_images, start=1):
                page_res = self.process_image(img, page_num=idx, languages=languages)
                page_results.append(page_res)
        else:
            # Fallback if rendering completely failed
            if pages_text:
                for idx, p_text in enumerate(pages_text, start=1):
                    lines = parse_text_into_ocr_lines(p_text, default_conf=0.70)
                    page_results.append(
                        OCRPageResult(
                            page_num=idx,
                            full_text=p_text,
                            lines=lines,
                            average_confidence=0.70,
                            engine_error="pdf_rendering_fallback",
                        )
                    )

        full_doc_text = "\n\n".join([p.full_text for p in page_results])
        overall_conf = (
            sum(p.average_confidence for p in page_results) / len(page_results)
            if page_results
            else 0.0
        )
        source_name = "rapid_ocr" if _ocr_backend == "rapidocr" else "paddle_ocr"
        page_errors = [p.engine_error for p in page_results if p.engine_error]
        combined_error = "; ".join(page_errors) if page_errors else None

        return OCRDocumentResult(
            pages=page_results,
            full_text=full_doc_text,
            average_confidence=round(overall_conf, 4),
            ocr_required=True,
            text_source=source_name,
            engine_error=combined_error,
        )

    def process_file(
        self,
        file_path: str,
        languages: Optional[List[str]] = None,
    ) -> OCRDocumentResult:
        """Unified entry point to process either an image or a PDF with language pass support."""
        lower = file_path.lower()
        if lower.endswith(".pdf"):
            return self.process_pdf(file_path, languages=languages)
        else:
            with Image.open(file_path) as img:
                page_res = self.process_image(img.copy(), page_num=1, languages=languages)
                source_name = "rapid_ocr" if _ocr_backend == "rapidocr" else "paddle_ocr"
                return OCRDocumentResult(
                    pages=[page_res],
                    full_text=page_res.full_text,
                    average_confidence=page_res.average_confidence,
                    ocr_required=True,
                    text_source=source_name,
                    engine_error=page_res.engine_error,
                )

