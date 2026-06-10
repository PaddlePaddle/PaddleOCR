#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    path: Path
    line_number: int
    ref: str
    url: str


def _compile_link_pattern(repo_slug):
    escaped_repo = re.escape(repo_slug)
    return re.compile(
        rf"https://github\.com/{escaped_repo}/(?:blob|tree)/(?P<ref>[^/\s)\]\"'<>]+)"
        rf"(?P<rest>/[^\s)\]\"'<>]*)?"
    )


def find_forbidden_links(root, repo_slug, forbidden_refs):
    root = Path(root)
    forbidden_refs = set(forbidden_refs)
    link_pattern = _compile_link_pattern(repo_slug)
    violations = []

    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            for match in link_pattern.finditer(line):
                ref = match.group("ref")
                if ref in forbidden_refs:
                    violations.append(
                        Violation(
                            path=path,
                            line_number=line_number,
                            ref=ref,
                            url=match.group(0),
                        )
                    )
    return violations


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reject docs links that point to moving GitHub source refs."
    )
    parser.add_argument("--root", default="docs", help="Directory to scan.")
    parser.add_argument(
        "--repo-slug", required=True, help="Example: PaddlePaddle/PaddleOCR"
    )
    parser.add_argument(
        "--forbidden-ref",
        action="append",
        required=True,
        help="Moving source ref to reject. Can be passed multiple times.",
    )
    args = parser.parse_args(argv)

    violations = find_forbidden_links(
        args.root,
        repo_slug=args.repo_slug,
        forbidden_refs=set(args.forbidden_ref),
    )
    if violations:
        for violation in violations:
            print(
                f"{violation.path}:{violation.line_number}: "
                f"forbidden GitHub ref '{violation.ref}' in {violation.url}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
