# OCR Fundamentals

This document teaches OCR (Optical Character Recognition) from scratch. If you
already understand text detection, text recognition, and CTC decoding, skip
ahead to [Architecture](architecture.md).

## What Is OCR?

OCR converts images of text into machine-readable character sequences. It is
the technology behind:

- Digitizing scanned documents and books
- Reading license plates, street signs, and receipts
- Extracting data from invoices, forms, and ID cards
- Making PDFs searchable and accessible

OCR sounds simple — "just read the text" — but in practice it is hard because
real-world images contain curved text, uneven lighting, blur, rotation,
multiple languages, complex layouts (tables, formulas, seals), and overlapping
elements.

## The OCR Pipeline

Modern OCR systems break the problem into stages. The most common pipeline has
three core steps:

```
┌─────────────┐    ┌────────────────┐    ┌──────────────────┐
│             │    │                │    │                  │
│  Input      │───>│  Text          │───>│  Text            │───> Output
│  Image      │    │  Detection     │    │  Recognition     │     Text
│             │    │                │    │                  │
└─────────────┘    └────────────────┘    └──────────────────┘
                     Finds WHERE           Reads WHAT
                     text is               text says

                   ┌────────────────┐
                   │  (Optional)    │
                   │  Text Angle    │
                   │  Classification│
                   └────────────────┘
                     Corrects rotation
                     before recognition
```

1. **Text Detection** — Locates text regions in the image and outputs their
   coordinates (bounding boxes or polygons).
2. **Text Angle Classification** (optional) — Determines if a text line is
   rotated (0, 90, 180, or 270 degrees) and corrects it.
3. **Text Recognition** — Takes each cropped text region and produces the
   character sequence.

The detection model finds *where* text is. The recognition model reads *what*
it says. Splitting the problem this way allows each model to specialize.

## Text Detection

### What It Does

A text detection model takes an image and outputs a set of regions where text
appears. Each region is described as either:

- A **bounding box**: four coordinates (x_min, y_min, x_max, y_max)
- A **polygon**: a sequence of points that tightly follows the text shape
  (needed for curved or rotated text)

### The DB Algorithm

PaddleOCR's default detection algorithm is **DB (Differentiable
Binarization)**. Here is how it works at a high level:

```
                    ┌───────────┐
    Input           │           │     Feature
    Image ─────────>│  Backbone │────> Maps
    (H x W x 3)    │ (e.g.     │     (multi-scale)
                    │ MobileNet)│
                    └───────────┘
                          │
                          v
                    ┌───────────┐
                    │           │     Fused
                    │   FPN     │────> Feature
                    │  (Neck)   │      Map
                    └───────────┘
                          │
                    ┌─────┴──────┐
                    │            │
                    v            v
              ┌──────────┐ ┌──────────┐
              │Probability│ │Threshold │
              │   Map     │ │   Map    │
              └──────────┘ └──────────┘
                    │            │
                    v            v
              ┌──────────────────────┐
              │  Differentiable      │
              │  Binarization        │
              │  P > T ? text : bg   │
              └──────────────────────┘
                         │
                         v
              ┌──────────────────────┐
              │  Contour Extraction  │
              │  + Polygon Fitting   │──────> Text Polygons
              │  + Unclip Expansion  │
              └──────────────────────┘
```

Key ideas:
- The **backbone** (e.g., MobileNetV3, ResNet) extracts visual features at
  multiple scales.
- The **FPN (Feature Pyramid Network)** neck fuses features from different
  scales so the model can detect both large and small text.
- The model produces a **probability map** (where is text?) and a **threshold
  map** (adaptive per-pixel threshold). The innovation of DB is that the
  binarization step (probability > threshold) is made *differentiable*, so the
  whole pipeline can be trained end-to-end.
- Post-processing extracts contours from the binary map, fits polygons, and
  expands them slightly (unclip) to recover the full text region.

In PaddleOCR, this is implemented in:
- Detection heads: `ppocr/modeling/heads/det_db_head.py`
- Post-processing: `ppocr/postprocess/db_postprocess.py`

Other detection algorithms in PaddleOCR include **EAST**, **SAST**, **PSE**,
**FCE** (for irregular text), **CT**, and **DRRG**.

## Text Recognition

### What It Does

A text recognition model takes a cropped image of a single text line and
outputs a character sequence with a confidence score. For example:

```
Input:  [image of "Hello World"]
Output: ("Hello World", 0.97)
```

### CTC vs Attention Decoding

There are two main approaches to converting a model's output into text:

```
CTC (Connectionist Temporal Classification)     Attention-Based
─────────────────────────────────────────        ─────────────────────────

Image ──> Backbone ──> Sequence of             Image ──> Backbone ──> Encoder
          features     frame predictions                              │
                       [H, e, -, l, l, -, o]                          v
                            │                                     Decoder
                            v                                   (autoregressive)
                       CTC Decode:                                    │
                       collapse repeats,                              v
                       remove blanks                            H ─> e ─> l ─> ...
                            │                                   (one char at a time,
                            v                                    attending to image)
                       "Hello"                                        │
                                                                      v
                                                                 "Hello"

Pros:                                          Pros:
  - Fast (non-autoregressive)                    - Handles variable-length
  - Simple training                                output well
  - Good for regular text                        - Better for irregular text

Cons:                                          Cons:
  - Assumes monotonic alignment                  - Slower (sequential decoding)
  - Struggles with very long text                - More complex to train
```

**CTC** works by predicting a character (or blank) at every position in the
feature sequence, then collapsing repeated characters and removing blanks. It
is fast because all positions are predicted in parallel.

**Attention-based** decoders generate characters one at a time, using an
attention mechanism to "look at" relevant parts of the image for each
character. They handle irregular and variable-length text better but are
slower.

PaddleOCR supports both approaches:
- CTC: `ppocr/modeling/heads/rec_ctc_head.py`, decoded by
  `ppocr/postprocess/rec_postprocess.py:CTCLabelDecode`
- Attention: `ppocr/modeling/heads/rec_att_head.py`, decoded by
  `ppocr/postprocess/rec_postprocess.py:AttnLabelDecode`
- And many more: SAR, NRTR, SVTR, ParseQ, ABINet, etc.

### Character Dictionaries

Recognition models output indices into a **character dictionary** — a file
listing every character the model can recognize. PaddleOCR ships dictionaries
for 80+ languages in `ppocr/utils/dict/`. The size of the dictionary
determines the output dimension of the recognition head.

## Text Angle Classification

Some documents contain text that is rotated 90 or 180 degrees (e.g., text
printed vertically on a spine, or an upside-down scan). The **text angle
classifier** predicts the rotation angle (0, 90, 180, or 270 degrees) so the
image can be corrected before recognition.

This is an optional step in the pipeline — only needed when input images may
contain rotated text.

In PaddleOCR: `ppocr/modeling/heads/cls_head.py`

## Beyond Basic OCR

PaddleOCR goes well beyond simple text detection + recognition:

### Table Structure Recognition

Detects table boundaries, rows, columns, and cells, then extracts content into
structured formats (HTML, JSON). Uses specialized models like **SLANet** and
**TableMaster**.

### Layout Analysis

Identifies document regions: titles, paragraphs, images, tables, headers,
footers, page numbers, formulas, and more. The **PP-StructureV3** pipeline
combines layout detection with OCR, table recognition, formula recognition,
and chart parsing to produce complete Markdown output from complex documents.

### Formula Recognition

Converts mathematical formula images to LaTeX markup. Uses **PP-FormulaNet**
with specialized backbones and heads.

### Key Information Extraction (KIE)

Extracts structured key-value pairs from documents (e.g., "Name: John Smith"
from a form). Uses models like **LayoutXLM** that combine visual, textual, and
layout features.

### Document Vision-Language Models (VLM)

**PaddleOCR-VL** is a 0.9B-parameter Vision-Language Model that combines a
NaViT-style visual encoder with ERNIE-4.5-0.3B language model. Unlike the
traditional pipeline approach (separate detection + recognition), it performs
end-to-end document parsing as a single model, handling text, tables, formulas,
charts, and seals simultaneously. It supports 109 languages and achieves 94.5%
accuracy on OmniDocBench v1.5.

## Key Metrics

Understanding evaluation metrics is essential for training and comparing
models.

### Detection Metrics

| Metric        | What it measures                                               |
|---------------|----------------------------------------------------------------|
| **Precision** | Of all regions the model detected, what fraction are real text? |
| **Recall**    | Of all real text regions, what fraction did the model find?    |
| **H-mean**    | Harmonic mean of precision and recall (the primary metric)     |

A detection is considered correct if its IoU (Intersection over Union) with a
ground-truth region exceeds a threshold (typically 0.5).

In PaddleOCR: `ppocr/metrics/det_metric.py:DetMetric` with
`main_indicator: hmean`

### Recognition Metrics

| Metric        | What it measures                                               |
|---------------|----------------------------------------------------------------|
| **Accuracy**  | Fraction of text lines recognized exactly correctly            |
| **NED**       | Normalized Edit Distance — how close the prediction is         |

In PaddleOCR: `ppocr/metrics/rec_metric.py:RecMetric` with
`main_indicator: acc`

### The `main_indicator` Field

In PaddleOCR config files, the `Metric` section specifies a `main_indicator`
— the metric used to decide which checkpoint is "best" during training. For
detection, this is `hmean`. For recognition, this is `acc`.

```yaml
Metric:
  name: DetMetric
  main_indicator: hmean
```

## What's Next?

Now that you understand the core OCR concepts, proceed to
[Architecture](architecture.md) to learn how PaddleOCR implements all of this
in code.
