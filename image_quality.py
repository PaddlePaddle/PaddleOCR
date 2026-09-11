"""
image_quality.py
Performs pre-OCR image quality checks:
- Blur detection (Laplacian variance)
- Resolution validation
- Returns status and issues list
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class QualityCheckResult:
    is_acceptable: bool
    status: str
    reason: Optional[str]
    issues: List[str]
    blur_score: float
    width: int
    height: int


def calculate_blur_score(pil_img: Image.Image) -> float:
    """Calculate variance of Laplacian to evaluate sharpness/blur."""
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float64)

    if HAS_CV2:
        # Standard OpenCV Laplacian variance
        lap = cv2.Laplacian(arr.astype(np.uint8), cv2.CV_64F)
        return float(lap.var())
    else:
        # High-performance NumPy discrete Laplacian kernel approximation:
        # [[0,  1, 0],
        #  [1, -4, 1],
        #  [0,  1, 0]]
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return 0.0
        lap = (
            arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
            - 4 * arr[1:-1, 1:-1]
        )
        return float(np.var(lap))


def evaluate_image_quality(
    pil_img: Image.Image,
    min_width: int = 150,
    min_height: int = 150,
    blur_threshold: float = 15.0,
) -> QualityCheckResult:
    """
    Evaluate if an image meets the minimum quality threshold for OCR.
    Flags extremely blurry or low-resolution images.
    """
    width, height = pil_img.size
    issues: List[str] = []

    # 1. Resolution Check
    if width < min_width or height < min_height:
        issues.append(f"low_resolution (got {width}x{height}, min {min_width}x{min_height})")

    # 2. Blur Check
    blur_score = calculate_blur_score(pil_img)
    # Documents with text usually have Laplacian variance > 30-50; completely blurry < 15
    if blur_score < blur_threshold:
        issues.append(f"excessive_blur (score {round(blur_score, 2)} < threshold {blur_threshold})")

    if issues:
        return QualityCheckResult(
            is_acceptable=False,
            status="low_confidence",
            reason="image_quality",
            issues=issues,
            blur_score=round(blur_score, 2),
            width=width,
            height=height,
        )

    return QualityCheckResult(
        is_acceptable=True,
        status="acceptable",
        reason=None,
        issues=[],
        blur_score=round(blur_score, 2),
        width=width,
        height=height,
    )
