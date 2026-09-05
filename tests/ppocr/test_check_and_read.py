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

import pytest

from ppocr.utils.utility import check_and_read


@pytest.fixture
def corrupted_gif(tmp_path):
    """A file with a .gif name whose contents cv2 cannot decode."""
    path = tmp_path / "broken.gif"
    path.write_bytes(b"GIF89a" + b"\x00\xff not a real gif")
    return str(path)


def test_corrupted_gif_returns_three_values(corrupted_gif):
    """The unreadable-gif branch must return the same arity as every other branch.

    Every caller unpacks three values, e.g. ``tools/infer/predict_system.py``:

        img, flag_gif, flag_pdf = check_and_read(image_file)

    The branch used to return only ``(None, False)``, so a corrupted gif raised
    ``ValueError: not enough values to unpack (expected 3, got 2)`` from the
    caller instead of being skipped -- an error that names neither the gif nor
    the file that produced it, even though this branch already logs and returns
    a sentinel precisely so the caller can move on.
    """
    img, flag_gif, flag_pdf = check_and_read(corrupted_gif)

    assert img is None
    assert flag_gif is False
    assert flag_pdf is False


def test_corrupted_gif_logs_the_path(corrupted_gif, caplog):
    """The log line exists to say which file failed, so it must interpolate it.

    The message was passed to ``logger.info`` with an unformatted ``{}``, so the
    output read literally ``Cannot read {}.`` and lost the only piece of
    information it was there to carry.
    """
    with caplog.at_level(logging.INFO, logger="ppocr"):
        check_and_read(corrupted_gif)

    assert corrupted_gif in caplog.text
    assert "Cannot read {}" not in caplog.text


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".txt", ""])
def test_non_gif_non_pdf_paths_are_unchanged(tmp_path, suffix):
    """Paths this helper does not handle keep returning the plain sentinel."""
    path = tmp_path / f"sample{suffix}"
    path.write_bytes(b"whatever")

    assert check_and_read(str(path)) == (None, False, False)
