#!/usr/bin/env python3
import argparse
from pathlib import Path


TEXT_SUFFIXES = {".md", ".yml", ".yaml"}


def resolve_placeholders(root, placeholder, source_ref):
    if (
        not source_ref
        or source_ref.strip() != source_ref
        or any(c.isspace() for c in source_ref)
    ):
        raise ValueError("source_ref must be a non-empty ref without whitespace")

    root = Path(root)
    changed = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        if placeholder not in content:
            continue
        path.write_text(content.replace(placeholder, source_ref), encoding="utf-8")
        changed.append(path)
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Resolve docs GitHub source-ref placeholders before building docs."
    )
    parser.add_argument("--root", default="docs", help="Directory to rewrite.")
    parser.add_argument("--placeholder", required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args(argv)

    changed = resolve_placeholders(
        args.root,
        placeholder=args.placeholder,
        source_ref=args.source_ref,
    )
    print(f"Resolved {len(changed)} file(s) under {args.root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
