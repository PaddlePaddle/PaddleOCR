"""
Generates a dictionary of ANSI escape codes.

http://en.wikipedia.org/wiki/ANSI_escape_code

Uses colorama as an optional dependency to support color on Windows
"""
import sys

try:
    import colorama
except ImportError:
    pass
else:
    if sys.platform == "win32":
        colorama.init(strip=False)

__all__ = ("escape_codes", "parse_colors")


# Returns escape codes from format codes
def esc(*codes: int) -> str:
    return "\033[" + ";".join(str(code) for code in codes) + "m"


escape_codes = {
    "reset": esc(0),
    "bold": esc(1),
    "thin": esc(2),
}

escape_codes_foreground = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "purple": 35,
    "cyan": 36,
    "white": 37,
    "light_black": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_purple": 95,
    "light_cyan": 96,
    "light_white": 97,
}

escape_codes_background = {
    "black": 40,
    "red": 41,
    "green": 42,
    "yellow": 43,
    "blue": 44,
    "purple": 45,
    "cyan": 46,
    "white": 47,
    "light_black": 100,
    "light_red": 101,
    "light_green": 102,
    "light_yellow": 103,
    "light_blue": 104,
    "light_purple": 105,
    "light_cyan": 106,
    "light_white": 107,
    # Bold background colors don't exist,
    # but we used to provide these names.
    "bold_black": 100,
    "bold_red": 101,
    "bold_green": 102,
    "bold_yellow": 103,
    "bold_blue": 104,
    "bold_purple": 105,
    "bold_cyan": 106,
    "bold_white": 107,
}

# Foreground without prefix
for name, code in escape_codes_foreground.items():
    escape_codes["%s" % name] = esc(code)
    escape_codes["bold_%s" % name] = esc(1, code)
    escape_codes["thin_%s" % name] = esc(2, code)

# Foreground with fg_ prefix
for name, code in escape_codes_foreground.items():
    escape_codes["fg_%s" % name] = esc(code)
    escape_codes["fg_bold_%s" % name] = esc(1, code)
    escape_codes["fg_thin_%s" % name] = esc(2, code)

# Background with bg_ prefix
for name, code in escape_codes_background.items():
    escape_codes["bg_%s" % name] = esc(code)

# 256 colour support
for code in range(256):
    escape_codes["fg_%d" % code] = esc(38, 5, code)
    escape_codes["bg_%d" % code] = esc(48, 5, code)


def parse_colors(string: str) -> str:
    """Return escape codes from a color sequence string."""
    return "".join(escape_codes[n] for n in string.split(",") if n)
