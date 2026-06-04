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

import argparse
from pathlib import Path

from paddleocr._api_client import cli
from paddleocr._api_client.results import OCRPage, OCRResult


def test_cli_save_resources_uses_result_specific_helper(tmp_path, monkeypatch, capsys):
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, **kwargs):
            return OCRResult(
                job_id="job-1",
                pages=[
                    OCRPage(
                        pruned_result={}, ocr_image_url="https://example.test/a.png"
                    )
                ],
            )

        def save_ocr_result_resources(self, result, destination, overwrite=False):
            assert result.job_id == "job-1"
            assert destination == str(tmp_path)
            assert overwrite is True
            return [str(Path(destination) / "ocr-page-1.png")]

        def close(self):
            return None

    monkeypatch.setattr(cli, "PaddleOCRClient", FakeClient)

    args = argparse.Namespace(
        token="token",
        client_platform=None,
        base_url=None,
        request_timeout=300.0,
        poll_timeout=600.0,
        model=None,
        model_type="ocr",
        file_url="https://example.test/input.pdf",
        file_path=None,
        output=None,
        save_resources=str(tmp_path),
        overwrite_resources=True,
        page_ranges=None,
        batch_id=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        use_chart_recognition=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        use_formula_recognition=None,
        use_layout_detection=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_rec_score_thresh=None,
        prettify_markdown=None,
        visualize=None,
    )

    cli._execute_api(args)

    captured = capsys.readouterr()
    assert '"jobId": "job-1"' in captured.out
    assert "Resources saved to:" in captured.err
