import sys
import os
from pathlib import Path
from typing import Any

import paddle
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from ppocr.modeling.backbones.rec_donut_swin import DonutSwinModel, DonutSwinModelOutput
from ppocr.modeling.backbones.rec_pphgnetv2 import PPHGNetV2_B4
from ppocr.modeling.backbones.rec_vary_vit import Vary_VIT_B_Formula
from ppocr.modeling.heads.rec_unimernet_head import UniMERNetHead
from ppocr.modeling.heads.rec_ppformulanet_head import PPFormulaNet_Head


@pytest.fixture
def sample_image():
    return paddle.randn([1, 1, 192, 672])


@pytest.fixture
def sample_image_ppformulanet_s():
    return paddle.randn([1, 1, 384, 384])


@pytest.fixture
def sample_image_ppformulanet_l():
    return paddle.randn([1, 1, 768, 768])


@pytest.fixture
def encoder_feat():
    encoded_feat = paddle.randn([1, 126, 1024])
    return DonutSwinModelOutput(
        last_hidden_state=encoded_feat,
    )


@pytest.fixture
def encoder_feat_ppformulanet_s():
    encoded_feat = paddle.randn([1, 144, 2048])
    return DonutSwinModelOutput(
        last_hidden_state=encoded_feat,
    )


@pytest.fixture
def encoder_feat_ppformulanet_l():
    encoded_feat = paddle.randn([1, 144, 1024])
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


def test_ppformulanet_s_backbone(sample_image_ppformulanet_s):
    """
    Test PP-FormulaNet-S backbone.

    Args:
        sample_image_ppformulanet_s: sample image to be processed.
    """
    backbone = PPHGNetV2_B4(
        class_num=1024,
    )
    backbone.eval()
    with paddle.no_grad():
        result = backbone(sample_image_ppformulanet_s)
        encoder_feat = result[0]
        assert encoder_feat.shape == [1, 144, 2048]


def test_ppformulanet_s_head(encoder_feat_ppformulanet_s):
    """
    Test PP-FormulaNet-S head.

    Args:
        encoder_feat_ppformulanet_s: encoder feature from PP-FormulaNet-S backbone.
    """
    head = PPFormulaNet_Head(
        max_new_tokens=6,
        decoder_start_token_id=0,
        decoder_ffn_dim=1536,
        decoder_hidden_size=384,
        decoder_layers=2,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        encoder_hidden_size=2048,
        is_export=False,
        length_aware=True,
        use_parallel=True,
        parallel_step=3,
    )

    head.eval()
    with paddle.no_grad():
        result = head(encoder_feat_ppformulanet_s)
        assert result.shape == [1, 9]


def test_ppformulanet_l_backbone(sample_image_ppformulanet_l):
    """
    Test PP-FormulaNet-L backbone.

    Args:
        sample_image_ppformulanet_l: sample image to be processed.
    """
    backbone = Vary_VIT_B_Formula(
        image_size=768,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )
    backbone.eval()
    with paddle.no_grad():
        result = backbone(sample_image_ppformulanet_l)
        encoder_feat = result[0]
        assert encoder_feat.shape == [1, 144, 1024]


def test_ppformulanet_l_head(encoder_feat_ppformulanet_l):
    """
    Test PP-FormulaNet-L head.

    Args:
        encoder_feat_ppformulanet_l: encoder feature from PP-FormulaNet-L Head.
    """
    head = PPFormulaNet_Head(
        max_new_tokens=6,
        decoder_start_token_id=0,
        decoder_ffn_dim=2048,
        decoder_hidden_size=512,
        decoder_layers=8,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        encoder_hidden_size=1024,
        is_export=False,
        length_aware=False,
        use_parallel=False,
        parallel_step=0,
    )

    head.eval()
    with paddle.no_grad():
        result = head(encoder_feat_ppformulanet_l)
        assert result.shape == [1, 7]
