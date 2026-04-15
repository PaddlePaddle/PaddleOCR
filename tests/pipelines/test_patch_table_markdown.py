"""Regression tests for table markdown escaping patches (issue #16096).

These tests import the patch module directly so they can verify the patched
HTML assembly logic without importing the full paddleocr package.
"""

import importlib.util
import sys
import types

def _import_patch_module():
    """Import _patch_table_markdown without triggering paddleocr.__init__."""
    spec = importlib.util.spec_from_file_location(
        "paddleocr._pipelines._patch_table_markdown",
        "paddleocr/_pipelines/_patch_table_markdown.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_patch_mod = _import_patch_module()
_fixed_get_html_result = _patch_mod._fixed_get_html_result


def _minimal_table_structure():
    return [
        "<html>",
        "<body>",
        "<table>",
        "<tr>",
        "<td></td>",
        "</tr>",
        "</table>",
        "</body>",
        "</html>",
    ]


def _buggy_get_html_result(matched_index, ocr_contents, pred_structures):
    """Current upstream behavior used to build fake paddlex modules."""
    pred_html = []
    td_index = 0
    head_structure = pred_structures[0:3]
    html = "".join(head_structure)
    table_structure = pred_structures[3:-3]
    for tag in table_structure:
        if "</td>" in tag:
            if "<td></td>" == tag:
                pred_html.extend("<td>")
            if td_index in matched_index.keys():
                b_with = False
                if (
                    "<b>" in ocr_contents[matched_index[td_index][0]]
                    and len(matched_index[td_index]) > 1
                ):
                    b_with = True
                    pred_html.extend("<b>")
                for i, td_index_index in enumerate(matched_index[td_index]):
                    content = ocr_contents[td_index_index]
                    if len(matched_index[td_index]) > 1:
                        if len(content) == 0:
                            continue
                        if content[0] == " ":
                            content = content[1:]
                        if "<b>" in content:
                            content = content[3:]
                        if "</b>" in content:
                            content = content[:-4]
                        if len(content) == 0:
                            continue
                        if i != len(matched_index[td_index]) - 1 and " " != content[-1]:
                            content += " "
                    pred_html.extend(content)
                if b_with:
                    pred_html.extend("</b>")
            if "<td></td>" == tag:
                pred_html.append("</td>")
            else:
                pred_html.append(tag)
            td_index += 1
        else:
            pred_html.append(tag)
    html += "".join(pred_html)
    html += "".join(pred_structures[-3:])
    return html


def _install_fake_paddlex_modules():
    """Install a minimal paddlex module tree for apply_patches tests."""
    module_names = [
        "paddlex",
        "paddlex.inference",
        "paddlex.inference.pipelines",
        "paddlex.inference.pipelines.table_recognition",
        "paddlex.inference.pipelines.table_recognition.table_recognition_post_processing",
        "paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2",
    ]
    originals = {name: sys.modules.get(name) for name in module_names}

    paddlex_mod = types.ModuleType("paddlex")
    inference_mod = types.ModuleType("paddlex.inference")
    pipelines_mod = types.ModuleType("paddlex.inference.pipelines")
    table_rec_mod = types.ModuleType("paddlex.inference.pipelines.table_recognition")
    post_mod = types.ModuleType(
        "paddlex.inference.pipelines.table_recognition.table_recognition_post_processing"
    )
    post_v2_mod = types.ModuleType(
        "paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2"
    )

    post_mod.get_html_result = _buggy_get_html_result
    post_v2_mod.get_html_result = _buggy_get_html_result

    paddlex_mod.inference = inference_mod
    inference_mod.pipelines = pipelines_mod
    pipelines_mod.table_recognition = table_rec_mod
    table_rec_mod.table_recognition_post_processing = post_mod
    table_rec_mod.table_recognition_post_processing_v2 = post_v2_mod

    sys.modules["paddlex"] = paddlex_mod
    sys.modules["paddlex.inference"] = inference_mod
    sys.modules["paddlex.inference.pipelines"] = pipelines_mod
    sys.modules["paddlex.inference.pipelines.table_recognition"] = table_rec_mod
    sys.modules[
        "paddlex.inference.pipelines.table_recognition.table_recognition_post_processing"
    ] = post_mod
    sys.modules[
        "paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2"
    ] = post_v2_mod

    return originals, post_mod, post_v2_mod


def _restore_modules(originals):
    for name, module in originals.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestFixedGetHtmlResult:
    def test_escapes_html_sensitive_cell_text(self):
        result = _fixed_get_html_result(
            {0: [0]},
            ['<recv response="200" response_txn="invite" />'],
            _minimal_table_structure(),
        )
        assert (
            "<td>&lt;recv response=&quot;200&quot; response_txn=&quot;invite&quot; /&gt;</td>"
            in result
        )
        assert '<td><recv response="200" response_txn="invite" /></td>' not in result

    def test_preserves_bold_markup_while_escaping_text(self):
        result = _fixed_get_html_result(
            {0: [0]},
            ['<b><pause milliseconds="5000"/></b>'],
            _minimal_table_structure(),
        )
        assert (
            "<td><b>&lt;pause milliseconds=&quot;5000&quot;/&gt;</b></td>"
            in result
        )


class TestApplyPatches:
    def test_apply_patches_monkey_patches_both_post_processing_modules(self):
        originals, post_mod, post_v2_mod = _install_fake_paddlex_modules()
        try:
            _patch_mod._patched = False
            _patch_mod.apply_patches()

            assert post_mod.get_html_result is _fixed_get_html_result
            assert post_v2_mod.get_html_result is _fixed_get_html_result

            _patch_mod.apply_patches()
            assert post_mod.get_html_result is _fixed_get_html_result
            assert post_v2_mod.get_html_result is _fixed_get_html_result
        finally:
            _patch_mod._patched = False
            _restore_modules(originals)
