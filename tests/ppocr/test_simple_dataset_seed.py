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
import random

import pytest

from ppocr.data.simple_dataset import SimpleDataSet


@pytest.mark.parametrize("ratio_list", [[1.0, 1.0], [0.5, 0.75]])
def test_simple_dataset_sampling_preserves_python_rng(tmp_path, ratio_list):
    label_files = []
    for source in range(2):
        label_file = tmp_path / f"labels-{source}.txt"
        label_file.write_text(
            "".join(f"{source}-{sample}.png\ttext\n" for sample in range(4)),
            encoding="utf-8",
        )
        label_files.append(str(label_file))

    config = {
        "Global": {},
        "Train": {
            "dataset": {
                "data_dir": str(tmp_path),
                "label_file_list": label_files,
                "ratio_list": ratio_list,
                "transforms": [],
            },
            "loader": {"shuffle": True},
        },
    }
    state = random.getstate()
    dataset = SimpleDataSet(config, "Train", logging.getLogger(__name__))
    assert random.getstate() == state

    dataset.reset_data_lines(seed=7, epoch=7)
    assert random.getstate() == state
