#!/usr/bin/env python3
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

"""Extract named XCTAttachments from an .xcresult bundle into files.

Uses `xcrun xcresulttool` only. Prerequisites are probed against the **currently
selected** Xcode (`xcode-select`): see `_check_xcresulttool_capability`.

**Discovery (documented CLI behavior)**

- ``xcresulttool get test-results tests`` — “Get all tests from test report.”
  (``xcrun xcresulttool help get test-results tests``). We walk Test Case nodes
  and read each ``nodeIdentifierURL`` as ``--test-id``.

- ``xcresulttool get test-results activities`` — “Get the activity trees for the
  specified test.” (``help get test-results activities``). Attachment records
  (``name``, ``payloadId``) are taken from activity trees under ``testRuns``.

- ``xcresulttool export attachments`` — “An additional manifest.json file will be
  generated… contains a list of attachment details for each test.”
  (``help export attachments``). We resolve on-disk filenames via
  ``exportedFileName`` / ``suggestedHumanReadableName`` in that manifest when
  present.

**Logical filename matching (repo heuristic, not from Apple help)**

We match ``--name`` to each activities record’s ``name`` by **exact string**, or
by **stem + underscore-suffixed** basename before the extension. Adjust
``_matches_wanted_attachment`` if your tests use different attachment titles.

If ``activities`` yields no attachments, we fall back to walking the ``tests``
JSON for ``attachments`` on nodes (empty after the activities pass in some
bundles).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

AttachmentRow = Tuple[str, str, str]  # stored_name, test_id_url, payload_id


def _die(what: str, where: str, next_step: str, code: int = 1) -> int:
    sys.stderr.write(
        f"[extract_xcresult_attachments] FAIL: {what}\n  Where: {where}\n  Next:  {next_step}\n"
    )
    return code


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, check=False)


def _check_xcresulttool_capability() -> None:
    """Require `xcrun xcresulttool help get` to list `test-results` (command exists)."""
    r = _run(["xcrun", "xcresulttool", "help", "get"])
    text = (r.stdout + r.stderr).decode(errors="replace")
    if "test-results" not in text:
        raise SystemExit(
            _die(
                "`xcrun xcresulttool help get` has no `test-results` subcommand in this Xcode.",
                "xcrun xcresulttool help get",
                "Install a newer Xcode or `xcode-select` a toolchain that ships this subcommand.",
            )
        )


def _load_tests_json(result_path: Path) -> dict:
    r = _run(
        [
            "xcrun",
            "xcresulttool",
            "get",
            "test-results",
            "tests",
            "--path",
            str(result_path),
            "--format",
            "json",
        ]
    )
    if r.returncode != 0:
        raise SystemExit(
            _die(
                f"xcresulttool get test-results failed with exit {r.returncode}.",
                str(result_path),
                "Confirm the .xcresult was produced by a completed `xcodebuild test`.",
            )
        )
    try:
        return json.loads(r.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise SystemExit(
            _die(
                f"Could not parse xcresulttool JSON: {e}.",
                str(result_path),
                "Re-run xcodebuild test to regenerate the .xcresult.",
            )
        )


def _collect_test_case_urls(node: dict) -> List[str]:
    urls: List[str] = []
    if node.get("nodeType") == "Test Case":
        u = node.get("nodeIdentifierURL")
        if isinstance(u, str) and u.startswith("test://"):
            urls.append(u)
    for ch in node.get("children") or []:
        urls.extend(_collect_test_case_urls(ch))
    return urls


def _walk_activity_tree(act: dict, acc: List[AttachmentRow], test_url: str) -> None:
    for att in act.get("attachments") or []:
        name = att.get("name")
        pid = att.get("payloadId")
        if isinstance(name, str) and isinstance(pid, str):
            acc.append((name, test_url, pid))
    for ch in act.get("childActivities") or []:
        _walk_activity_tree(ch, acc, test_url)


def _attachments_from_activities(
    result_path: Path, test_url: str
) -> List[AttachmentRow]:
    r = _run(
        [
            "xcrun",
            "xcresulttool",
            "get",
            "test-results",
            "activities",
            "--path",
            str(result_path),
            "--test-id",
            test_url,
            "--format",
            "json",
        ]
    )
    if r.returncode != 0:
        return []
    try:
        parsed = json.loads(r.stdout or b"{}")
    except json.JSONDecodeError:
        return []
    acc: List[AttachmentRow] = []
    for tr in parsed.get("testRuns") or []:
        for act in tr.get("activities") or []:
            _walk_activity_tree(act, acc, test_url)
    return acc


def _walk_attachments_legacy(node: dict, acc: List[AttachmentRow]) -> None:
    """Fallback: walk ``tests`` JSON for ``attachments`` on nodes (used if activities path is empty)."""
    identifier = node.get("identifier") or node.get("nodeIdentifierURL") or ""
    for att in node.get("attachments") or []:
        name = att.get("name")
        pid = att.get("payloadId")
        if isinstance(name, str) and isinstance(pid, str):
            acc.append((name, str(identifier), pid))
    for child in node.get("children") or []:
        _walk_attachments_legacy(child, acc)


def _enumerate(result_path: Path) -> List[AttachmentRow]:
    parsed = _load_tests_json(result_path)
    acc: List[AttachmentRow] = []
    urls: List[str] = []
    for top in parsed.get("testNodes") or []:
        urls.extend(_collect_test_case_urls(top))
    for url in urls:
        acc.extend(_attachments_from_activities(result_path, url))
    if acc:
        return acc
    for top in parsed.get("testNodes") or []:
        _walk_attachments_legacy(top, acc)
    return acc


def _matches_wanted_attachment(stored: str, wanted: str) -> bool:
    if stored == wanted:
        return True
    stem = Path(wanted).stem
    suffix = Path(wanted).suffix if Path(wanted).suffix else ".json"
    return stored.startswith(stem + "_") and stored.endswith(suffix)


def _pick_exported_file(
    td_path: Path, wanted_out_name: str, stored_name: str
) -> Optional[Path]:
    """Pick the exported file using ``manifest.json`` when `xcresulttool export` writes it (see ``help export``)."""
    manifest_path = td_path / "manifest.json"
    if manifest_path.is_file():
        try:
            raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = []
        blocks: List[dict] = raw if isinstance(raw, list) else []
        for block in blocks:
            for att in block.get("attachments") or []:
                if not isinstance(att, dict):
                    continue
                sug = att.get("suggestedHumanReadableName") or ""
                fn = att.get("exportedFileName")
                if not isinstance(fn, str):
                    continue
                candidate = td_path / fn
                if not candidate.is_file():
                    continue
                if sug == stored_name or _matches_wanted_attachment(
                    sug, wanted_out_name
                ):
                    return candidate

    direct = td_path / stored_name
    if direct.is_file():
        return direct

    for p in sorted(td_path.iterdir()):
        if not p.is_file() or p.name == "manifest.json":
            continue
        if _matches_wanted_attachment(p.name, wanted_out_name):
            return p

    orphans = [
        p
        for p in td_path.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name != "manifest.json"
    ]
    if len(orphans) == 1:
        return orphans[0]

    return None


def _export_one(
    result_path: Path,
    test_identifier: str,
    wanted_out_name: str,
    stored_name: str,
    out_path: Path,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        r = _run(
            [
                "xcrun",
                "xcresulttool",
                "export",
                "attachments",
                "--path",
                str(result_path),
                "--test-id",
                test_identifier,
                "--output-path",
                td,
            ]
        )
        if r.returncode != 0:
            raise SystemExit(
                _die(
                    f"xcresulttool export attachments failed with exit {r.returncode}.",
                    f"test-id={test_identifier}",
                    "Inspect stderr above.",
                )
            )
        td_path = Path(td)
        src = _pick_exported_file(td_path, wanted_out_name, stored_name)
        if src is None or not src.is_file():
            raise SystemExit(
                _die(
                    f"Could not resolve exported file (wanted {wanted_out_name!r}, stored {stored_name!r}).",
                    f"export dir: {td}",
                    "Check `manifest.json` and `xcresulttool export attachments` stderr.",
                )
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(out_path))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", required=True, type=Path, help="Path to .xcresult bundle."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write exported attachments into.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Attachment name to extract (may be repeated). Fails the run if missing.",
    )
    parser.add_argument(
        "--optional-name",
        action="append",
        default=[],
        dest="optional_name",
        help="Like --name, but a missing attachment is skipped with a message (exit still 0).",
    )
    args = parser.parse_args(argv)
    if not args.name and not args.optional_name:
        parser.error("at least one of --name or --optional-name is required")

    _check_xcresulttool_capability()

    if not args.result.exists():
        return _die(
            ".xcresult bundle does not exist.",
            str(args.result),
            "Run `xcodebuild test -resultBundlePath <path>` first.",
        )

    attachments = _enumerate(args.result)

    errors = 0

    def _extract_wanted(wanted: str, optional: bool) -> None:
        nonlocal errors
        hits = [
            (tid, sn)
            for sn, tid, _pid in attachments
            if _matches_wanted_attachment(sn, wanted)
        ]
        uniq: Set[Tuple[str, str]] = set(hits)
        if len(uniq) == 0:
            if optional:
                sys.stderr.write(
                    f"[extract_xcresult_attachments] optional attachment not found, skipping: {wanted!r}\n"
                )
                return
            errors += _die(
                f"Attachment not found: {wanted}.",
                str(args.result),
                "Confirm the test run attached JSON and inspect `get test-results activities` / `export attachments` for this bundle.",
            )
            return
        if len(uniq) > 1:
            errors += _die(
                f"Attachment name collision: {wanted} matches {len(uniq)} exports.",
                str(args.result),
                "Ensure attachment base names are unique across tests.",
            )
            return
        test_id, stored = next(iter(uniq))
        _export_one(args.result, test_id, wanted, stored, args.out_dir / wanted)

    for wanted in args.name:
        _extract_wanted(wanted, optional=False)
    for wanted in args.optional_name:
        _extract_wanted(wanted, optional=True)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
