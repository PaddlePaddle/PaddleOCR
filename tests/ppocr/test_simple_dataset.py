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

import logging

import cv2
import numpy as np
import pytest

from ppocr.data.multi_scale_sampler import MultiScaleSampler
from ppocr.data.simple_dataset import MultiScaleDataSet


def _create_dataset_file(tmp_path, prefix, num_samples, width, height):
    label_file = tmp_path / f"{prefix}.txt"

    with open(label_file, "w", encoding="utf-8") as f:
        for idx in range(num_samples):
            image_name = f"{prefix}_{idx}.jpg"
            image_path = tmp_path / image_name

            image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            f.write(f"{image_name}\t{prefix}_{idx}\t{width}\t{height}\n")

    return str(label_file)


def _build_config(
    tmp_path,
    label_file_list,
    ratio_list,
    ds_width,
    transforms,
    shuffle,
):
    return {
        "Global": {},
        "Train": {
            "dataset": {
                "name": "MultiScaleDataSet",
                "data_dir": str(tmp_path),
                "label_file_list": label_file_list,
                "ratio_list": ratio_list,
                "ds_width": ds_width,
                "transforms": transforms,
            },
            "loader": {
                "shuffle": shuffle,
            },
        },
    }


@pytest.mark.parametrize("ds_width", [False, True])
def test_multiscale_dataset_respects_ratio_sampling(tmp_path, ds_width):
    # Dataset A has a smaller aspect ratio than dataset B. When ds_width
    # is enabled, sorting the full dataset instead of the sampled subset
    # therefore exposes the indexing bug deterministically.
    a_file = _create_dataset_file(
        tmp_path,
        prefix="A",
        num_samples=2,
        width=10,
        height=10,
    )
    b_file = _create_dataset_file(
        tmp_path,
        prefix="B",
        num_samples=2,
        width=20,
        height=10,
    )

    config = _build_config(
        tmp_path=tmp_path,
        label_file_list=[a_file, b_file],
        ratio_list=[0.5, 0.5],
        ds_width=ds_width,
        transforms=[
            {
                "DecodeImage": {
                    "img_mode": "BGR",
                    "channel_first": False,
                }
            },
            {
                "KeepKeys": {
                    "keep_keys": [
                        "image",
                        "label",
                    ]
                }
            },
        ],
        shuffle=True,
    )

    dataset = MultiScaleDataSet(
        config=config,
        mode="Train",
        logger=logging.getLogger(__name__),
        seed=0,
    )

    assert len(dataset) == 2

    # ratio_list should select one sample from each label file.
    sampled_indices = dataset._index_map
    assert sum(index < 2 for index in sampled_indices) == 1
    assert sum(index >= 2 for index in sampled_indices) == 1

    samples = [dataset.__getitem__([320, 48, idx, 1.0]) for idx in range(len(dataset))]
    labels = [sample[1] for sample in samples]

    assert sum(label.startswith("A_") for label in labels) == 1
    assert sum(label.startswith("B_") for label in labels) == 1


def test_multiscale_sampler_refreshes_ratio_after_dataset_reset(tmp_path):
    label_file = tmp_path / "data.txt"

    sizes = [
        (10, 10),
        (20, 10),
        (30, 10),
        (40, 10),
    ]

    with open(label_file, "w", encoding="utf-8") as f:
        for idx, (width, height) in enumerate(sizes):
            f.write(f"image_{idx}.jpg\tlabel_{idx}\t{width}\t{height}\n")

    config = _build_config(
        tmp_path=tmp_path,
        label_file_list=[str(label_file)],
        ratio_list=[0.5],
        ds_width=True,
        transforms=[],
        shuffle=False,
    )

    dataset = MultiScaleDataSet(
        config=config,
        mode="Train",
        logger=logging.getLogger(__name__),
        seed=0,
    )

    sampler = MultiScaleSampler(
        data_source=dataset,
        scales=[[320, 48]],
        first_bs=2,
        fix_bs=True,
        is_training=True,
        seed=0,
    )

    first_expected_ratio = np.mean(dataset.wh_ratio[dataset.wh_ratio_sort])

    first_epoch_batch = list(iter(sampler))[0]
    first_epoch_ratio = first_epoch_batch[0][3]

    assert np.isclose(
        first_epoch_ratio,
        first_expected_ratio,
    )

    # Find another sampling seed that changes the expected aspect ratio.
    for seed in range(1, 100):
        dataset.reset_data_lines(seed=seed, epoch=seed)

        second_expected_ratio = np.mean(dataset.wh_ratio[dataset.wh_ratio_sort])

        if not np.isclose(
            first_expected_ratio,
            second_expected_ratio,
        ):
            break
    else:
        pytest.fail("Could not generate a different sampled subset")

    # program.py calls set_epoch() after reset_data_lines().
    sampler.set_epoch(0)

    second_epoch_batch = list(iter(sampler))[0]
    second_epoch_ratio = second_epoch_batch[0][3]

    assert np.isclose(
        second_epoch_ratio,
        second_expected_ratio,
    )
