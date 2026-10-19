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

"""Tests for PaddleOCR-VL CLI arguments, configuration merging, and argument forwarding (issue #18371).

These unit tests verify:
1. --markdown_ignore_labels CLI argument parsing behavior.
2. PaddleOCR-VL pipeline configuration merging semantics.
3. End-to-end CLI argument forwarding into pipeline creation.
4. Backward compatibility with existing CLI invocations.
5. Parameter and method forwarding through the PaddleOCRVL wrapper using repository test utilities.

Tests are isolated: any lightweight mocks used for optional heavy dependencies
(e.g., paddlex) are cleanly registered and restored in tearDownModule to prevent
process-level test contamination.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from tests.testing_utils import check_wrapper_simple_inference_param_forwarding

REPO_ROOT = Path(__file__).resolve().parents[2]

_ORIG_MODULES = {}
_MOCKED_KEYS = set()


def _setup_isolated_modules():
    """Load paddleocr pipeline submodules with scoped mocks for heavy dependencies."""
    mock_needed = [
        "paddlex",
        "paddlex.inference",
        "paddlex.utils",
        "paddlex.utils.config",
        "paddlex.utils.deps",
        "paddlex.utils.device",
    ]
    for mod in mock_needed:
        if mod in sys.modules:
            _ORIG_MODULES[mod] = sys.modules[mod]
        else:
            m = MagicMock()
            if mod == "paddlex.utils.device":
                m.parse_device.return_value = ("cpu", None)
            sys.modules[mod] = m
            _MOCKED_KEYS.add(mod)

    for pkg_name, pkg_rel in [
        ("paddleocr", "paddleocr"),
        ("paddleocr._pipelines", "paddleocr/_pipelines"),
        ("paddleocr._utils", "paddleocr/_utils"),
    ]:
        if pkg_name in sys.modules:
            _ORIG_MODULES[pkg_name] = sys.modules[pkg_name]
        else:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(REPO_ROOT / pkg_rel)]
            sys.modules[pkg_name] = pkg
            _MOCKED_KEYS.add(pkg_name)

    submodules = [
        ("paddleocr._abstract", "paddleocr/_abstract.py"),
        ("paddleocr._common_args", "paddleocr/_common_args.py"),
        ("paddleocr._utils.cli", "paddleocr/_utils/cli.py"),
        ("paddleocr._pipelines.utils", "paddleocr/_pipelines/utils.py"),
        ("paddleocr._pipelines.base", "paddleocr/_pipelines/base.py"),
        ("paddleocr._pipelines.paddleocr_vl", "paddleocr/_pipelines/paddleocr_vl.py"),
    ]
    for mod_name, file_rel in submodules:
        if mod_name in sys.modules:
            _ORIG_MODULES[mod_name] = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(
                mod_name, REPO_ROOT / file_rel
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            _MOCKED_KEYS.add(mod_name)
            spec.loader.exec_module(mod)

    return (
        sys.modules["paddleocr._pipelines.paddleocr_vl"].PaddleOCRVL,
        sys.modules[
            "paddleocr._pipelines.paddleocr_vl"
        ].PaddleOCRVLCLISubcommandExecutor,
        sys.modules["paddleocr._pipelines.base"],
    )


PaddleOCRVL, PaddleOCRVLCLISubcommandExecutor, base_mod = (
    _setup_isolated_modules()
)


def tearDownModule():
    """Restore sys.modules to prevent cross-test contamination."""
    for k in list(_MOCKED_KEYS):
        sys.modules.pop(k, None)
    for k, v in _ORIG_MODULES.items():
        sys.modules[k] = v


setup_module = lambda: None
teardown_module = tearDownModule


class TestPaddleOCRVLCLI(unittest.TestCase):
    """Tests for the --markdown_ignore_labels CLI argument, config merging, and forwarding."""

    def setUp(self):
        self.executor = PaddleOCRVLCLISubcommandExecutor()
        self.parser = argparse.ArgumentParser()
        subparsers = self.parser.add_subparsers(dest="subcommand")
        self.subparser = self.executor.add_subparser(subparsers)
        self.subparser.set_defaults(executor=self.executor.execute_with_args)

        self.default_yaml_cfg = {
            "markdown_ignore_labels": [
                "number",
                "footnote",
                "header",
                "header_image",
                "footer",
                "footer_image",
                "aside_text",
            ]
        }

    # 1. Option is optional
    def test_option_is_optional(self):
        """Verify omitting --markdown_ignore_labels returns None."""
        args = self.parser.parse_args(["doc_parser", "-i", "dummy.jpg"])
        self.assertIsNone(args.markdown_ignore_labels)

    # 2. Empty argument list is accepted
    def test_empty_argument_list_accepted(self):
        """Verify passing --markdown_ignore_labels without values produces an empty list."""
        args = self.parser.parse_args(
            ["doc_parser", "-i", "dummy.jpg", "--markdown_ignore_labels"]
        )
        self.assertEqual(args.markdown_ignore_labels, [])

    # 3. One explicit label is parsed correctly
    def test_one_explicit_label(self):
        """Verify passing a single label produces a single-item list."""
        args = self.parser.parse_args(
            [
                "doc_parser",
                "-i",
                "dummy.jpg",
                "--markdown_ignore_labels",
                "footer",
            ]
        )
        self.assertEqual(args.markdown_ignore_labels, ["footer"])

    # 4. Multiple explicit labels are parsed correctly
    def test_multiple_explicit_labels(self):
        """Verify passing multiple labels produces the full list of labels."""
        args = self.parser.parse_args(
            [
                "doc_parser",
                "-i",
                "dummy.jpg",
                "--markdown_ignore_labels",
                "header",
                "footer",
            ]
        )
        self.assertEqual(args.markdown_ignore_labels, ["header", "footer"])

    # 5. Omission preserves existing defaults through config merging
    def test_omission_preserves_defaults(self):
        """Verify None preserves the default YAML ignore labels in the merged configuration."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline", return_value=MagicMock()
        ):
            pipe = PaddleOCRVL(
                paddlex_config=self.default_yaml_cfg.copy(),
                markdown_ignore_labels=None,
            )
            self.assertEqual(
                pipe._merged_paddlex_config["markdown_ignore_labels"],
                self.default_yaml_cfg["markdown_ignore_labels"],
            )

    # 6. Empty list disables filtering (clears ignore list in config)
    def test_empty_list_clears_ignored_labels(self):
        """Verify [] explicitly clears the ignore label list in the merged configuration."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline", return_value=MagicMock()
        ):
            pipe = PaddleOCRVL(
                paddlex_config=self.default_yaml_cfg.copy(),
                markdown_ignore_labels=[],
            )
            self.assertEqual(
                pipe._merged_paddlex_config["markdown_ignore_labels"], []
            )

    # 7. Explicit labels replace default list
    def test_explicit_labels_replace_defaults(self):
        """Verify a non-empty list overrides the default ignore labels in the merged configuration."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline", return_value=MagicMock()
        ):
            pipe = PaddleOCRVL(
                paddlex_config=self.default_yaml_cfg.copy(),
                markdown_ignore_labels=["footer"],
            )
            self.assertEqual(
                pipe._merged_paddlex_config["markdown_ignore_labels"],
                ["footer"],
            )

    # 8. Existing CLI invocations still parse correctly (backward compatibility)
    def test_existing_cli_invocations_preserved(self):
        """Verify standard CLI invocations and flag position variations remain backward-compatible."""
        args1 = self.parser.parse_args(
            [
                "doc_parser",
                "-i",
                "demo.png",
                "--device",
                "cpu",
                "--pipeline_version",
                "v1.6",
            ]
        )
        self.assertEqual(args1.input, "demo.png")
        self.assertEqual(args1.device, "cpu")
        self.assertEqual(args1.pipeline_version, "v1.6")
        self.assertIsNone(args1.markdown_ignore_labels)

        # Flag placed before -i
        args2 = self.parser.parse_args(
            ["doc_parser", "--markdown_ignore_labels", "-i", "demo.png"]
        )
        self.assertEqual(args2.input, "demo.png")
        self.assertEqual(args2.markdown_ignore_labels, [])

        # Flag with values placed before -i
        args3 = self.parser.parse_args(
            [
                "doc_parser",
                "--markdown_ignore_labels",
                "footer",
                "-i",
                "demo.png",
            ]
        )
        self.assertEqual(args3.input, "demo.png")
        self.assertEqual(args3.markdown_ignore_labels, ["footer"])

    # 9. Real CLI execution and argument forwarding path
    def test_cli_execution_forwarding(self):
        """Verify arguments flow through execute_with_args into pipeline creation config."""
        fake_pipeline = MagicMock()
        dummy_res = MagicMock()
        fake_pipeline.predict.return_value = [dummy_res]

        with patch.object(
            base_mod, "create_pipeline", return_value=fake_pipeline
        ) as mock_create, patch.object(
            base_mod,
            "load_pipeline_config",
            return_value=self.default_yaml_cfg.copy(),
        ), patch.object(
            PaddleOCRVL, "predict_iter", return_value=[dummy_res]
        ):
            # Case A: --markdown_ignore_labels (empty) forwards []
            args_empty = self.parser.parse_args(
                ["doc_parser", "-i", "dummy.jpg", "--markdown_ignore_labels"]
            )
            args_empty.executor(args_empty)
            called_config_empty = mock_create.call_args.kwargs["config"]
            self.assertEqual(called_config_empty["markdown_ignore_labels"], [])

            # Case B: Option omitted forwards YAML defaults
            args_omitted = self.parser.parse_args(
                ["doc_parser", "-i", "dummy.jpg"]
            )
            args_omitted.executor(args_omitted)
            called_config_omitted = mock_create.call_args.kwargs["config"]
            self.assertEqual(
                called_config_omitted["markdown_ignore_labels"],
                self.default_yaml_cfg["markdown_ignore_labels"],
            )

            # Case C: Explicit labels forward specified list
            args_specific = self.parser.parse_args(
                [
                    "doc_parser",
                    "-i",
                    "dummy.jpg",
                    "--markdown_ignore_labels",
                    "footer",
                ]
            )
            args_specific.executor(args_specific)
            called_config_specific = mock_create.call_args.kwargs["config"]
            self.assertEqual(
                called_config_specific["markdown_ignore_labels"], ["footer"]
            )


class _SimpleMonkeyPatch:
    """Lightweight test helper matching pytest monkeypatch.setattr interface."""

    def setattr(self, target, name, value):
        setattr(target, name, value)


class TestPaddleOCRVLWrapperForwarding(unittest.TestCase):
    """Tests parameter and method forwarding on the PaddleOCRVL wrapper class.

    Uses the established repository test utility check_wrapper_simple_inference_param_forwarding
    from tests.testing_utils to verify prediction parameter delegation.
    """

    def setUp(self):
        self.mp = _SimpleMonkeyPatch()

    def test_predict_param_forwarding_empty_list(self):
        """Verify markdown_ignore_labels=[] is forwarded from predict to paddlex_pipeline."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline", return_value=MagicMock()
        ):
            pipe = PaddleOCRVL(
                paddlex_config={"markdown_ignore_labels": ["header", "footer"]}
            )
            check_wrapper_simple_inference_param_forwarding(
                self.mp,
                pipe,
                "paddlex_pipeline",
                "dummy.jpg",
                {"markdown_ignore_labels": []},
            )

    def test_predict_param_forwarding_explicit_labels(self):
        """Verify markdown_ignore_labels=['footer'] is forwarded from predict to paddlex_pipeline."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline", return_value=MagicMock()
        ):
            pipe = PaddleOCRVL(
                paddlex_config={"markdown_ignore_labels": ["header", "footer"]}
            )
            check_wrapper_simple_inference_param_forwarding(
                self.mp,
                pipe,
                "paddlex_pipeline",
                "dummy.jpg",
                {"markdown_ignore_labels": ["footer"]},
            )

    def test_predict_param_forwarding_none(self):
        """Verify markdown_ignore_labels=None is forwarded from predict to paddlex_pipeline."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline", return_value=MagicMock()
        ):
            pipe = PaddleOCRVL(
                paddlex_config={"markdown_ignore_labels": ["header", "footer"]}
            )
            check_wrapper_simple_inference_param_forwarding(
                self.mp,
                pipe,
                "paddlex_pipeline",
                "dummy.jpg",
                {"markdown_ignore_labels": None},
            )

    def test_concatenate_markdown_pages_forwarding(self):
        """Verify concatenate_markdown_pages delegates to paddlex_pipeline."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline"
        ) as mock_create_pipe:
            mock_pipeline = MagicMock()
            mock_pipeline.concatenate_markdown_pages.return_value = (
                "# Page 1\n\n# Page 2"
            )
            mock_create_pipe.return_value = mock_pipeline
            pipe = PaddleOCRVL(paddlex_config={"markdown_ignore_labels": []})
            result = pipe.concatenate_markdown_pages(["# Page 1", "# Page 2"])
            mock_pipeline.concatenate_markdown_pages.assert_called_once_with(
                ["# Page 1", "# Page 2"]
            )
            self.assertEqual(result, "# Page 1\n\n# Page 2")

    def test_restructure_pages_forwarding(self):
        """Verify restructure_pages delegates to paddlex_pipeline with options."""
        with patch.object(
            PaddleOCRVL, "_create_paddlex_pipeline"
        ) as mock_create_pipe:
            mock_pipeline = MagicMock()
            mock_pipeline.restructure_pages.return_value = [
                "restructured_page"
            ]
            mock_create_pipe.return_value = mock_pipeline
            pipe = PaddleOCRVL(paddlex_config={"markdown_ignore_labels": []})
            result = pipe.restructure_pages(["raw_page"])
            mock_pipeline.restructure_pages.assert_called_once_with(
                ["raw_page"],
                merge_tables=True,
                relevel_titles=True,
                concatenate_pages=False,
            )
            self.assertEqual(result, ["restructured_page"])


if __name__ == "__main__":
    unittest.main()
