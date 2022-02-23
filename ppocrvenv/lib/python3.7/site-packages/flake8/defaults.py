"""Constants that define defaults."""
import re

EXCLUDE = (
    ".svn",
    "CVS",
    ".bzr",
    ".hg",
    ".git",
    "__pycache__",
    ".tox",
    ".eggs",
    "*.egg",
)
IGNORE = ("E121", "E123", "E126", "E226", "E24", "E704", "W503", "W504")
SELECT = ("E", "F", "W", "C90")
MAX_LINE_LENGTH = 79
INDENT_SIZE = 4

TRUTHY_VALUES = {"true", "1", "t"}

# Other constants
WHITESPACE = frozenset(" \t")

STATISTIC_NAMES = ("logical lines", "physical lines", "tokens")

NOQA_INLINE_REGEXP = re.compile(
    # We're looking for items that look like this:
    # ``# noqa``
    # ``# noqa: E123``
    # ``# noqa: E123,W451,F921``
    # ``# noqa:E123,W451,F921``
    # ``# NoQA: E123,W451,F921``
    # ``# NOQA: E123,W451,F921``
    # ``# NOQA:E123,W451,F921``
    # We do not want to capture the ``: `` that follows ``noqa``
    # We do not care about the casing of ``noqa``
    # We want a comma-separated list of errors
    # https://regex101.com/r/4XUuax/2 full explanation of the regex
    r"# noqa(?::[\s]?(?P<codes>([A-Z]+[0-9]+(?:[,\s]+)?)+))?",
    re.IGNORECASE,
)

NOQA_FILE = re.compile(r"\s*# flake8[:=]\s*noqa", re.I)
