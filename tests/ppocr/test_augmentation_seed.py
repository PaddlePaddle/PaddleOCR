# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.py38_incompatible
if sys.version_info < (3, 9):
    pytest.skip(
        "Requires Albumentations with set_random_state", allow_module_level=True
    )

from ppocr.data.imaug.unimernet_aug import Fog, Frost, Rain, Shadow
from ppocr.utils.utility import seed_albumentations
import albumentations as A


@pytest.fixture
def fixed_seed(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda high: 123)


@pytest.mark.parametrize("transform", [Fog, Frost, Rain, Shadow])
def test_weather_augmentation_reproducible(transform, fixed_seed):
    image = np.full((64, 64, 3), 235, dtype=np.uint8)
    image[15:31, 9:21] = 10

    def augment_sequence():
        pipeline = A.Compose([transform(p=1)])
        augmented = []
        for _ in range(3):
            seed_albumentations(pipeline)
            augmented.append(pipeline(image=image)["image"])
        return np.stack(augmented)

    first_run = augment_sequence()
    np.testing.assert_array_equal(first_run, augment_sequence())
    assert not np.array_equal(first_run[0], first_run[1])


def test_sibling_augmentations_can_apply_separately(fixed_seed):
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    pipeline = A.Compose(
        [
            A.OneOf([A.HorizontalFlip(p=1)], p=0.2),
            A.OneOf([A.VerticalFlip(p=1)], p=0.2),
        ]
    )
    seed_albumentations(pipeline)
    outcomes = {pipeline(image=image)["image"].tobytes() for _ in range(128)}

    assert image[:, ::-1].tobytes() in outcomes
    assert image[::-1, :].tobytes() in outcomes
