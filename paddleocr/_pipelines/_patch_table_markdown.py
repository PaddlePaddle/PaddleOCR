# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Patches for PP-StructureV3 table markdown escaping.

Fixes HTML-sensitive OCR text being injected into table cells without escaping,
which breaks `save_to_markdown()` output for content like `<recv .../>`.

See: https://github.com/PaddlePaddle/PaddleOCR/issues/16096
"""

import html
import logging

logger = logging.getLogger(__name__)

_patched = False


def _split_bold_wrapper(content):
    """Return whether content is wrapped by a single outer `<b>...</b>` pair."""
    if content.startswith("<b>") and content.endswith("</b>"):
        return True, content[3:-4]
    return False, content


def _escape_cell_content(content):
    """Escape OCR cell text while preserving a single outer bold wrapper."""
    is_bold, inner_content = _split_bold_wrapper(content)
    escaped_content = html.escape(inner_content, quote=True)
    if is_bold:
        return f"<b>{escaped_content}</b>"
    return escaped_content


def _fixed_get_html_result(matched_index, ocr_contents, pred_structures):
    """Generate HTML table content with HTML-sensitive OCR text escaped."""
    pred_html = []
    td_index = 0
    head_structure = pred_structures[0:3]
    html_text = "".join(head_structure)
    table_structure = pred_structures[3:-3]

    for tag in table_structure:
        if "</td>" not in tag:
            pred_html.append(tag)
            continue

        if tag == "<td></td>":
            pred_html.append("<td>")

        if td_index in matched_index:
            matched_cell_indexes = matched_index[td_index]
            use_shared_bold_wrapper = (
                len(matched_cell_indexes) > 1
                and "<b>" in ocr_contents[matched_cell_indexes[0]]
            )
            if use_shared_bold_wrapper:
                pred_html.append("<b>")

            for content_index, ocr_index in enumerate(matched_cell_indexes):
                content = ocr_contents[ocr_index]
                if len(matched_cell_indexes) > 1:
                    if len(content) == 0:
                        continue
                    if content[0] == " ":
                        content = content[1:]
                    if content.startswith("<b>"):
                        content = content[3:]
                    if content.endswith("</b>"):
                        content = content[:-4]
                    if len(content) == 0:
                        continue
                    if (
                        content_index != len(matched_cell_indexes) - 1
                        and content[-1] != " "
                    ):
                        content += " "
                    pred_html.append(html.escape(content, quote=True))
                else:
                    pred_html.append(_escape_cell_content(content))

            if use_shared_bold_wrapper:
                pred_html.append("</b>")

        if tag == "<td></td>":
            pred_html.append("</td>")
        else:
            pred_html.append(tag)
        td_index += 1

    html_text += "".join(pred_html)
    html_text += "".join(pred_structures[-3:])
    return html_text


def apply_patches():
    """Monkey-patch PaddleX table post-processing helpers."""
    global _patched
    if _patched:
        return

    try:
        import paddlex.inference.pipelines.table_recognition.table_recognition_post_processing as post_processing
        import paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2 as post_processing_v2
    except ImportError:
        logger.debug(
            "paddlex table recognition modules not available; skipping patches"
        )
        return

    post_processing.get_html_result = _fixed_get_html_result
    post_processing_v2.get_html_result = _fixed_get_html_result

    _patched = True
    logger.debug("Applied table markdown escaping patches for issue #16096")
