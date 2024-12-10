import sys
import os
from pathlib import Path
from typing import Any

import paddle
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from ppocr.modeling.backbones.rec_donut_swin import DonutSwinModel, DonutSwinModelOutput
from ppocr.modeling.heads.rec_unimernet_head import UniMERNetHead


@pytest.fixture
def sample_image():
    return paddle.randn([1, 1, 192, 672])


@pytest.fixture
def encoder_feat():
    encoded_feat = paddle.randn([1, 126, 1024])
    return DonutSwinModelOutput(
        last_hidden_state=encoded_feat,
    )


def test_unimernet_backbone(sample_image):
    """
    Test UniMERNet backbone.

    Args:
        sample_image: sample image to be processed.
    """
    backbone = DonutSwinModel(
        hidden_size=1024,
        num_layers=4,
        num_heads=[4, 8, 16, 32],
        add_pooling_layer=True,
        use_mask_token=False,
    )
    backbone.eval()
    with paddle.no_grad():
        result = backbone(sample_image)
        encoder_feat = result[0]
        assert encoder_feat.shape == [1, 126, 1024]


def test_unimernet_head(encoder_feat):
    """
    Test UniMERNet head.

    Args:
        encoder_feat: encoder feature from unimernet backbone.
    """
    head = UniMERNetHead(
        max_new_tokens=5,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        encoder_hidden_size=1024,
        is_export=False,
        length_aware=True,
    )

    head.eval()
    with paddle.no_grad():
        result = head(encoder_feat)
        assert result.shape == [1, 6]
