#!/usr/bin/env python
import argparse
import json
import sys
from itertools import chain

from . import convert
from .pdf import PDF


def parse_page_spec(p_str):
    if "-" in p_str:
        start, end = map(int, p_str.split("-"))
        return range(start, end + 1)
    else:
        return [int(p_str)]


def parse_args(args_raw):
    parser = argparse.ArgumentParser("pdfplumber")

    parser.add_argument(
        "infile", nargs="?", type=argparse.FileType("rb"), default=sys.stdin.buffer
    )

    parser.add_argument("--format", choices=["csv", "json"], default="csv")

    parser.add_argument("--types", nargs="+")

    parser.add_argument("--laparams", type=json.loads)

    parser.add_argument("--precision", type=int)

    parser.add_argument("--pages", nargs="+", type=parse_page_spec)

    parser.add_argument(
        "--indent", type=int, help="Indent level for JSON pretty-printing."
    )

    args = parser.parse_args(args_raw)
    if args.pages is not None:
        args.pages = list(chain(*args.pages))
    return args


def main(args_raw=sys.argv[1:]):
    args = parse_args(args_raw)
    converter = {"csv": convert.to_csv, "json": convert.to_json}[args.format]
    kwargs = {
        "csv": {"precision": args.precision},
        "json": {"precision": args.precision, "indent": args.indent},
    }[args.format]

    with PDF.open(args.infile, pages=args.pages, laparams=args.laparams) as pdf:
        converter(pdf, sys.stdout, args.types, **kwargs)


if __name__ == "__main__":
    main()
