import sys
from typing import Optional
from typing import Sequence

from pre_commit import output


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    for arg in argv:
        output.write_line(arg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
