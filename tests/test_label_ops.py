import os
import sys
import pytest
import numpy as np
import json

# Import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from ppocr.data.imaug.label_ops import (
    ClsLabelEncode,
    DetLabelEncode,
    CTCLabelEncode,
    AttnLabelEncode,
)

# Data generator function
def generate_character_dict(tmp_path, characters):
    character_dict = tmp_path / "char_dict.txt"
    character_dict.write_text("\n".join(characters) + "\n")
    return str(character_dict)

# Fixture: ClsLabelEncode
@pytest.fixture
def setup_cls_label_encode():
    return ClsLabelEncode(label_list=["label1", "label2", "label3"])

# Fixture: CTCLabelEncode
@pytest.fixture
def setup_ctc_label_encode(tmp_path):
    character_dict_path = generate_character_dict(tmp_path, ["a", "b", "c"])
    return CTCLabelEncode(max_text_length=10, character_dict_path=character_dict_path)

@pytest.fixture
def setup_ctc_label_encode_chinese(tmp_path):
    character_dict_path = generate_character_dict(tmp_path, ["你", "好", "世", "界"])
    return CTCLabelEncode(max_text_length=10, character_dict_path=character_dict_path)

@pytest.fixture
def setup_ctc_label_encode_tibetan(tmp_path):
    character_dict_path = generate_character_dict(tmp_path, ["ཀ", "ཁ", "ག", "ང", "ཀྵ", "ཀྪོ", "ཀྩོ", "ཀྤྲེ", "ཀླཱ", "གྒྲ"])
    print(f"Character dictionary path: {character_dict_path}")
    with open(character_dict_path, 'r', encoding='utf-8') as f:
        print(f"Character dictionary content:\n{f.read()}")    
    return CTCLabelEncode(max_text_length=25, character_dict_path=character_dict_path)

# Fixture: AttnLabelEncode
@pytest.fixture
def setup_attn_label_encode(tmp_path):
    character_dict_path = generate_character_dict(tmp_path, ["a", "b", "c"])
    return AttnLabelEncode(max_text_length=10, character_dict_path=character_dict_path)

@pytest.fixture
def setup_attn_label_encode_chinese(tmp_path):
    character_dict_path = generate_character_dict(tmp_path, ["你", "好", "世", "界"])
    return AttnLabelEncode(max_text_length=10, character_dict_path=character_dict_path)

# Fixture: DetLabelEncode
@pytest.fixture
def setup_det_label_encode():
    return DetLabelEncode()

# Test functions
@pytest.mark.parametrize("label, expected", [
    ("label1", 0),
    ("unknown_label", None),
    ("", None),
])
def test_cls_label_encode_call(setup_cls_label_encode, label, expected):
    encoder = setup_cls_label_encode
    data = {"label": label}
    encoded_data = encoder(data)
    
    if expected is not None:
        assert encoded_data["label"] == expected, f"Expected {expected} for label: {label}, but got {encoded_data['label']}"
    else:
        assert encoded_data is None, f"Expected None for label: {label}, but got {encoded_data}"

@pytest.mark.parametrize("label, expected", [
    ("abc", np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0])),
    ("unknown", None),
    ("", None),
    ("a" * 20, None),
])
def test_ctc_label_encode_call(setup_ctc_label_encode, label, expected):
    encoder = setup_ctc_label_encode
    data = {"label": label}
    encoded_data = encoder(data)
    
    if expected is not None:
        assert np.array_equal(encoded_data["label"], expected), f"Failed for label: {label}, expected {expected} but got {encoded_data['label']}"
        assert encoded_data["length"] == len(label), f"Expected length {len(label)} but got {encoded_data['length']}"
    else:
        assert encoded_data is None, f"Expected None for label: {label}, but got {encoded_data}"

@pytest.mark.parametrize("label, expected_result", [
    ("你好世界", np.array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0])),
])
def test_ctc_label_encode_call_valid_text_chinese(setup_ctc_label_encode_chinese, label, expected_result):
    encoder = setup_ctc_label_encode_chinese
    data = {"label": label}
    encoded_data = encoder(data)
    
    assert np.array_equal(encoded_data["label"], expected_result), f"Failed for Chinese text: {label}"
    assert encoded_data["length"] == len(label), f"Expected length {len(label)} but got {encoded_data['length']}"

@pytest.mark.parametrize("label, expected_result", [
    ("ཀཁགངཀྪོཀྩོཀྤྲེཀླཱགྒྲགྒྲ", np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("ཀྵཁགངཀྩོཀྪོ", np.array([5, 2, 3, 4, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
])
def test_ctc_label_encode_call_valid_text_tibetan(setup_ctc_label_encode_tibetan, label, expected_result):
    encoder = setup_ctc_label_encode_tibetan
    data = {"label": label}
    encoded_data = encoder(data)
    print(f"Encoded data for label '{label}': {encoded_data}")        
    assert np.array_equal(encoded_data["label"], expected_result), f"Failed for Tibetan text: {label}"
    assert encoded_data["length"] == len(expected_result[expected_result != 0]), f"Expected length {len(expected_result[expected_result != 0])} but got {encoded_data['length']}"

@pytest.mark.parametrize("label, expected_shape, expected_length", [
    ("abc", (10,), 3),
    ("unknown", None, None),
    ("", None, None),
    ("a" * 20, None, None),
])
def test_attn_label_encode_call(setup_attn_label_encode, label, expected_shape, expected_length):
    encoder = setup_attn_label_encode
    data = {"label": label}
    encoded_data = encoder(data)
    
    if expected_shape is not None:
        assert encoded_data["label"].shape == expected_shape, f"Expected shape {expected_shape} but got {encoded_data['label'].shape}"
        assert encoded_data["label"][0] == 0, f"Expected SOS token at start but got {encoded_data['label'][0]}"
        assert encoded_data["label"][expected_length + 1] == len(encoder.character) - 1, f"Expected EOS token at position {expected_length + 1} but got {encoded_data['label'][expected_length + 1]}"
        assert encoded_data["length"] == expected_length, f"Expected length {expected_length} but got {encoded_data['length']}"
    else:
        assert encoded_data is None, f"Expected None for label: {label}, but got {encoded_data}"

@pytest.mark.parametrize("label, expected_shape, expected_length", [
    ("你好世界", (10,), 4),
])
def test_attn_label_encode_call_valid_text_chinese(setup_attn_label_encode_chinese, label, expected_shape, expected_length):
    encoder = setup_attn_label_encode_chinese
    data = {"label": label}
    encoded_data = encoder(data)
    
    assert encoded_data["label"].shape == expected_shape, f"Expected shape {expected_shape} but got {encoded_data['label'].shape}"
    assert encoded_data["label"][0] == 0, f"Expected SOS token at start but got {encoded_data['label'][0]}"
    assert encoded_data["label"][expected_length + 1] == len(encoder.character) - 1, f"Expected EOS token at position {expected_length + 1} but got {encoded_data['label'][expected_length + 1]}"
    assert encoded_data["length"] == expected_length, f"Expected length {expected_length} but got {encoded_data['length']}"

@pytest.mark.parametrize("label, expected_texts", [
    ('[{"points": [[0,0],[1,0],[1,1],[0,1]], "transcription": "text"}]', ["text"]),
    ("[]", None),
    ("", pytest.raises(json.JSONDecodeError)),
    ('[{"points": [0,0],[1,0],[1,1],[0,1]], "transcription": "text"}]', pytest.raises(json.JSONDecodeError)),
])
def test_det_label_encode_call(setup_det_label_encode, label, expected_texts):
    encoder = setup_det_label_encode
    data = {"label": label}
    
    if isinstance(expected_texts, list):
        encoded_data = encoder(data)
        assert "polys" in encoded_data, "Missing polys in encoded data"
        assert "texts" in encoded_data, "Missing texts in encoded data"
        assert "ignore_tags" in encoded_data, "Missing ignore_tags in encoded data"
        assert encoded_data["texts"] == expected_texts, f"Expected texts {expected_texts} but got {encoded_data['texts']}"
    elif isinstance(expected_texts, type(pytest.raises(Exception))):
        with expected_texts:
            encoder(data)
    else:
        encoded_data = encoder(data)
        assert encoded_data is None, f"Expected None for label: {label}, but got {encoded_data}"
