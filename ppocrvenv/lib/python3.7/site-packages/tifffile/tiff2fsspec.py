#!/usr/bin/env python3
# tifffile/tiff2fsspec.py

"""Write fsspec ReferenceFileSystem for TIFF file.

positional arguments:
  tifffile              path to the local TIFF input file
  url                   remote URL of TIFF file without file name

optional arguments:
  -h, --help            show this help message and exit
  --out OUT             path to the JSON output file
  --series SERIES       index of series in file
  --level LEVEL         index of level in series
  --key KEY             index of page in file or series
  --chunkmode CHUNKMODE
                        mode used for chunking {None, pages}

For example:

    ``tiff2fsspec ./test.ome.tif https://server.com/path/``

"""

import argparse

try:
    from .tifffile import tiff2fsspec
except ImportError:
    try:
        from tifffile.tifffile import tiff2fsspec
    except ImportError:
        from tifffile import tiff2fsspec


def main():
    """Tiff2fsspec command line usage main function."""
    parser = argparse.ArgumentParser(
        'tiff2fsspec',
        description='Write fsspec ReferenceFileSystem for TIFF file.',
    )
    parser.add_argument(
        'tifffile', type=str, help='path to the local TIFF input file'
    )
    parser.add_argument(
        'url', type=str, help='remote URL of TIFF file without file name'
    )
    parser.add_argument(
        '--out', type=str, default=None, help='path to the JSON output file'
    )
    parser.add_argument(
        '--series', type=int, default=None, help='index of series in file'
    )
    parser.add_argument(
        '--level', type=int, default=None, help='index of level in series'
    )
    parser.add_argument(
        '--key', type=int, default=None, help='index of page in file or series'
    )
    parser.add_argument(
        '--chunkmode',
        type=int,
        default=None,
        help='mode used for chunking {None, pages}',
    )
    parser.add_argument(
        '--ver', type=int, default=None, help='version of ReferenceFileSystem'
    )
    args = parser.parse_args()

    tiff2fsspec(
        args.tifffile,
        args.url,
        out=args.out,
        key=args.key,
        series=args.series,
        level=args.level,
        chunkmode=args.chunkmode,
        version=args.ver,
    )


if __name__ == '__main__':
    import sys

    sys.exit(main())
