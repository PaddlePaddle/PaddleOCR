# Testing and Debugging

Guide for running the PaddleOCR test suite, understanding test patterns, and
debugging common training and inference issues.

## Test Suite Overview

PaddleOCR uses **pytest** with tests organized into two categories:

```
tests/
├── testing_utils.py         # Shared test helpers
├── test_files/              # Sample images used by tests
├── models/                  # Unit tests for individual models (Layer 1)
│   ├── test_text_detection.py
│   ├── test_text_recognition.py
│   ├── test_layout_detection.py
│   ├── test_table_structure_recognition.py
│   ├── test_table_cells_detection.py
│   ├── test_table_classifcation.py
│   ├── test_formula_recognition.py
│   ├── test_seal_text_detection.py
│   ├── test_doc_img_orientation_classifcation.py
│   ├── test_textline_orientation_classifcation.py
│   ├── test_text_image_unwarping.py
│   ├── test_doc_vlm.py
│   └── conftest.py
└── pipelines/               # Integration tests for pipelines (Layer 1)
    ├── test_ocr.py
    ├── test_pp_structurev3.py
    ├── test_seal_rec.py
    ├── test_formula_recognition.py
    ├── test_table_recognition_v2.py
    ├── test_doc_preprocessor.py
    ├── test_doc_understanding.py
    ├── test_pp_chatocrv4_doc.py
    ├── test_pp_doctranslation.py
    ├── test_patch_layout_parsing.py
    └── conftest.py
```

There are also additional unit tests at the top level of `tests/`:
- `test_rec_postprocess.py` — Recognition post-processing
- `test_cls_postprocess.py` — Classification post-processing
- `test_formula_model.py` — Formula model specifics
- `test_french_accents.py` — French accent handling
- `test_iaa_augment.py` — Data augmentation

## Running Tests

### Full Suite

```bash
pytest tests/
```

### By Category

```bash
# Layer 1 model tests only
pytest tests/models/

# Layer 1 pipeline tests only
pytest tests/pipelines/
```

### Individual Test Files

```bash
pytest tests/pipelines/test_ocr.py
pytest tests/models/test_text_detection.py
```

### With Verbose Output

```bash
pytest tests/pipelines/test_ocr.py -v
```

## Test Patterns

### Model Tests (tests/models/)

Model tests verify that Layer 1 model wrappers produce valid results:

```python
# Typical model test pattern
def test_text_detection(text_detection_model):
    result = text_detection_model.predict(TEST_DATA_DIR / "sample.jpg")
    check_simple_inference_result(result, expected_length=1)
```

- A **fixture** (in `conftest.py`) creates the model instance
- The test calls `predict()` with a sample image from `test_files/`
- `check_simple_inference_result()` verifies the result is a non-empty list
  of dicts

### Pipeline Tests (tests/pipelines/)

Pipeline tests verify end-to-end behavior and parameter forwarding:

```python
# Inference test
def test_ocr_predict(ocr_pipeline):
    result = ocr_pipeline.predict(TEST_DATA_DIR / "sample.jpg")
    check_simple_inference_result(result)

# Parameter forwarding test
def test_ocr_param_forwarding(monkeypatch, ocr_pipeline):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        ocr_pipeline,
        "paddlex_pipeline",
        TEST_DATA_DIR / "sample.jpg",
        {"use_textline_orientation": True, "text_det_thresh": 0.5},
    )
```

The parameter forwarding test uses `monkeypatch` to replace the underlying
PaddleX pipeline's `predict` method with a dummy that captures arguments,
then verifies all parameters are correctly forwarded.

### Testing Utilities (tests/testing_utils.py)

```python
TEST_DATA_DIR = Path(__file__).parent / "test_files"

def check_simple_inference_result(result, *, expected_length=1):
    """Verify result is a non-empty list of dicts."""
    assert isinstance(result, list)
    assert len(result) == expected_length
    for res in result:
        assert isinstance(res, dict)

def check_wrapper_simple_inference_param_forwarding(
    monkeypatch, wrapper, wrapped_obj_attr_name, input, params
):
    """Verify all params are forwarded to the underlying predictor."""
    # Patches predict, calls wrapper.predict, checks params arrived
```

---

## Debugging Training Issues

### Config Validation Errors

**Symptom**: `AssertionError: backbone only support [...]`

**Cause**: The `name` in your YAML config does not match any class in the
`support_dict` for the given `model_type`.

**Fix**:
1. Check the class name matches exactly (case-sensitive)
2. Check you registered in the correct `model_type` branch of
   `build_backbone`
3. Check the import is present in `__init__.py`

---

**Symptom**: `TypeError: __init__() got an unexpected keyword argument`

**Cause**: A YAML config key does not match a constructor parameter. Remember
that after `name` is popped from the config dict, all remaining keys are
passed as `**kwargs`.

**Fix**: Check the constructor signature of your component class and ensure
YAML keys match parameter names exactly.

### Data Pipeline Issues

**Symptom**: Training hangs or crashes at the data loading step.

**Debug**: Use the `test_reader` function in `tools/train.py` (line 248).
Uncomment the call at the bottom of the file to test data loading without
running training:

```python
# At bottom of tools/train.py, uncomment:
test_reader(config, device, logger)
```

Or run directly:
```bash
python -c "
import tools.program as program
config, device, logger, _ = program.preprocess(is_train=True)
from tools.train import test_reader
test_reader(config, device, logger)
"
```

**Common data issues**:
- Wrong `data_dir` path (images not found)
- Malformed label file (wrong delimiter, bad JSON)
- Label file references images that don't exist
- `num_workers` too high for your system

### Loss NaN or Explosion

**Symptom**: Loss becomes NaN or increases rapidly.

**Common causes and fixes**:
- **Learning rate too high**: Reduce `Optimizer.lr.learning_rate`
- **AMP level too aggressive**: Try `amp_level: O1` instead of `O2`
- **Missing gradient clipping**: Some algorithms require it
- **Data issue**: Bad images or labels causing extreme values
- **Batch size too small**: Increase `batch_size_per_card`

### Checkpoint Loading Errors

**Symptom**: `KeyError` or shape mismatch when loading a checkpoint.

**Cause**: The model architecture in the config does not match the saved
checkpoint. This happens when:
- You changed the model architecture after training started
- You are loading a pretrained model with a different head size
- The character dictionary changed (different number of characters)

**Fix**: Ensure the config matches the one used to create the checkpoint.
For fine-tuning with a different character set, use `pretrained_model`
(which ignores mismatched keys) instead of `checkpoints`.

---

## Debugging Inference Issues

### Layer 1 Issues

**Symptom**: `RuntimeError: A dependency error occurred during predictor creation`

**Cause**: Missing PaddleX dependencies.

**Fix**: Install required extras:
```bash
pip install "paddleocr[doc-parser]"   # For VLM features
pip install "paddleocr[all]"          # All optional dependencies
```

---

**Symptom**: Model download fails or times out.

**Cause**: Network issues or disk space.

**Fix**:
- Check internet connectivity
- Manually download the model and use `model_dir` parameter
- Check available disk space

### Layer 2 Issues

**Symptom**: `InputSpec` error during model export.

**Cause**: The export script (`ppocr/utils/export_model.py`) defines
input shapes per algorithm. If your algorithm is not handled, export fails.

**Fix**: Add an `InputSpec` branch for your algorithm in
`ppocr/utils/export_model.py:dynamic_to_static()`.

---

**Symptom**: Shape mismatch during inference but training works.

**Cause**: Dynamic shapes during training vs. fixed shapes during inference.
Common with models that use different input sizes.

**Fix**: Check the `InputSpec` used during export and ensure your inference
input matches the expected shape.

---

## Logging

### Layer 2 Logging (ppocr/utils/logging.py)

The training scripts use PaddleOCR's custom logger:

```python
from ppocr.utils.logging import get_logger
logger = get_logger()
logger.info("Training started")
```

Log verbosity is controlled by `Global.log_smooth_window` and
`Global.print_batch_step` in the config.

### Layer 1 Logging (paddleocr/_utils/logging.py)

The pipeline API uses a separate logger:

```python
from paddleocr import logger
logger.setLevel("DEBUG")  # For verbose output
```

---

## Common Errors Quick Reference

| Error                                      | Likely cause                                     | Fix                                                    |
|--------------------------------------------|--------------------------------------------------|--------------------------------------------------------|
| `backbone only support [...]`              | Class name not in support_dict                   | Register in correct `__init__.py` branch               |
| `head only support [...]`                  | Head class not registered                        | Add to `ppocr/modeling/heads/__init__.py`               |
| `loss only support [...]`                  | Loss class not registered                        | Add to `ppocr/losses/__init__.py`                      |
| `unexpected keyword argument 'xxx'`        | YAML key doesn't match constructor param         | Align YAML keys with `__init__` parameter names        |
| Shape mismatch in forward pass             | `out_channels` wrong in backbone or neck         | Verify `self.out_channels` is set correctly             |
| `No Images in train dataset`               | Empty dataset or wrong `data_dir`                | Check paths in config, verify images exist              |
| Loss is NaN                                | LR too high, AMP issues, bad data                | Reduce LR, try `amp_level: O1`, check data             |
| Checkpoint load `KeyError`                 | Architecture mismatch                            | Use matching config, or `pretrained_model` for fine-tune|
| `DependencyError` in predictor creation    | Missing PaddleX extras                           | `pip install "paddleocr[all]"`                         |
| Model only gets image tensor (no labels)   | Algorithm not in `extra_input_models`            | Add to list in `tools/program.py`                      |
