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

Requires Xcode 16+ (uses the non-legacy `xcresulttool` subcommands).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def _die(what: str, where: str, next_step: str, code: int = 1) -> int:
    sys.stderr.write(
        f"[extract_xcresult_attachments] FAIL: {what}\n  Where: {where}\n  Next:  {next_step}\n"
    )
    return code


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, check=False)


def _check_xcresulttool_capability() -> None:
    """Capability probe: `xcrun xcresulttool help get` must mention `test-results`.

    Using a capability check instead of a version number; `xcresulttool version`
    is a build number in Xcode 15/16 and doesn't parse as semver.
    """
    r = _run(["xcrun", "xcresulttool", "help", "get"])
    text = (r.stdout + r.stderr).decode(errors="replace")
    if "test-results" not in text:
        raise SystemExit(
            _die(
                "`xcrun xcresulttool get test-results` not available; Xcode 16+ required.",
                "xcrun xcresulttool help get",
                "Install Xcode 16+ or select it with `xcode-select -s /Applications/Xcode.app`.",
            )
        )


def _walk_attachments(node: dict, acc: List[Tuple[str, str, str]]) -> None:
    """Append (name, testIdentifier, payloadId) for every attachment in `node` and its children."""
    identifier = node.get("identifier") or ""
    for att in node.get("attachments") or []:
        name = att.get("name")
        pid = att.get("payloadId")
        if isinstance(name, str) and isinstance(pid, str):
            acc.append((name, identifier, pid))
    for child in node.get("children") or []:
        _walk_attachments(child, acc)


def _enumerate(result_path: Path) -> List[Tuple[str, str, str]]:
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
        parsed = json.loads(r.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise SystemExit(
            _die(
                f"Could not parse xcresulttool JSON: {e}.",
                str(result_path),
                "Re-run xcodebuild test to regenerate the .xcresult.",
            )
        )
    acc: List[Tuple[str, str, str]] = []
    for top in parsed.get("testNodes") or []:
        _walk_attachments(top, acc)
    return acc


def _export_one(
    result_path: Path, test_identifier: str, wanted_name: str, out_path: Path
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
        src = Path(td) / wanted_name
        if not src.is_file():
            # Export may use alternate filenames; try first matching file by suffix.
            matches = [p for p in Path(td).iterdir() if p.name == wanted_name]
            if not matches:
                raise SystemExit(
                    _die(
                        f"Exported attachment missing: {wanted_name}.",
                        f"export dir: {td}",
                        "Upgrade Xcode, or rename attachment in test code to match.",
                    )
                )
            src = matches[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(out_path))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", required=True, type=Path, help="Path to .xcresult bundle."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write exported attachments into.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        required=True,
        help="Attachment name to extract (may be repeated).",
    )
    args = parser.parse_args(argv)

    _check_xcresulttool_capability()

    if not args.result.exists():
        return _die(
            ".xcresult bundle does not exist.",
            str(args.result),
            "Run `xcodebuild test -resultBundlePath <path>` first.",
        )

    attachments = _enumerate(args.result)

    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for name, test_id, pid in attachments:
        grouped.setdefault(name, []).append((test_id, pid))

    errors = 0
    for wanted in args.name:
        hits = grouped.get(wanted, [])
        if len(hits) == 0:
            _die(
                f"Attachment not found: {wanted}.",
                str(args.result),
                "Test likely failed to attach; inspect xcodebuild output.",
            )
            errors += 1
            continue
        if len(hits) > 1:
            _die(
                f"Attachment name collision: {wanted} appears {len(hits)} times.",
                str(args.result),
                "Ensure ValidationArtifact names are unique across tests.",
            )
            errors += 1
            continue
        test_id, _pid = hits[0]
        _export_one(args.result, test_id, wanted, args.out_dir / wanted)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
