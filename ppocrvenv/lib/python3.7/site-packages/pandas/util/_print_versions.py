from __future__ import annotations

import codecs
import json
import locale
import os
import platform
import struct
import sys

from pandas._typing import JSONSerializable
from pandas.compat._optional import (
    VERSIONS,
    get_version,
    import_optional_dependency,
)


def _get_commit_hash() -> str | None:
    """
    Use vendored versioneer code to get git hash, which handles
    git worktree correctly.
    """
    from pandas._version import get_versions

    versions = get_versions()
    return versions["full-revisionid"]


def _get_sys_info() -> dict[str, JSONSerializable]:
    """
    Returns system information as a JSON serializable dictionary.
    """
    uname_result = platform.uname()
    language_code, encoding = locale.getlocale()
    return {
        "commit": _get_commit_hash(),
        "python": ".".join(str(i) for i in sys.version_info),
        "python-bits": struct.calcsize("P") * 8,
        "OS": uname_result.system,
        "OS-release": uname_result.release,
        "Version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": {"language-code": language_code, "encoding": encoding},
    }


def _get_dependency_info() -> dict[str, JSONSerializable]:
    """
    Returns dependency information as a JSON serializable dictionary.
    """
    deps = [
        "pandas",
        # required
        "numpy",
        "pytz",
        "dateutil",
        # install / build,
        "pip",
        "setuptools",
        "Cython",
        # test
        "pytest",
        "hypothesis",
        # docs
        "sphinx",
        # Other, need a min version
        "blosc",
        "feather",
        "xlsxwriter",
        "lxml.etree",
        "html5lib",
        "pymysql",
        "psycopg2",
        "jinja2",
        # Other, not imported.
        "IPython",
        "pandas_datareader",
    ]
    deps.extend(list(VERSIONS))

    result: dict[str, JSONSerializable] = {}
    for modname in deps:
        mod = import_optional_dependency(modname, errors="ignore")
        result[modname] = get_version(mod) if mod else None
    return result


def show_versions(as_json: str | bool = False) -> None:
    """
    Provide useful information, important for bug reports.

    It comprises info about hosting operation system, pandas version,
    and versions of other installed relative packages.

    Parameters
    ----------
    as_json : str or bool, default False
        * If False, outputs info in a human readable form to the console.
        * If str, it will be considered as a path to a file.
          Info will be written to that file in JSON format.
        * If True, outputs info in JSON format to the console.
    """
    sys_info = _get_sys_info()
    deps = _get_dependency_info()

    if as_json:
        j = {"system": sys_info, "dependencies": deps}

        if as_json is True:
            sys.stdout.writelines(json.dumps(j, indent=2))
        else:
            assert isinstance(as_json, str)  # needed for mypy
            with codecs.open(as_json, "wb", encoding="utf8") as f:
                json.dump(j, f, indent=2)

    else:
        assert isinstance(sys_info["LOCALE"], dict)  # needed for mypy
        language_code = sys_info["LOCALE"]["language-code"]
        encoding = sys_info["LOCALE"]["encoding"]
        sys_info["LOCALE"] = f"{language_code}.{encoding}"

        maxlen = max(len(x) for x in deps)
        print("\nINSTALLED VERSIONS")
        print("------------------")
        for k, v in sys_info.items():
            print(f"{k:<{maxlen}}: {v}")
        print("")
        for k, v in deps.items():
            print(f"{k:<{maxlen}}: {v}")


def main() -> int:
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option(
        "-j",
        "--json",
        metavar="FILE",
        nargs=1,
        help="Save output as JSON into file, pass in '-' to output to stdout",
    )

    (options, args) = parser.parse_args()

    if options.json == "-":
        options.json = True

    show_versions(as_json=options.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
