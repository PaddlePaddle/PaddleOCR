"""Built-in converters - importing this package triggers registration of all converters."""

import importlib

_converter_modules = [
    "docx",
    "pptx",
    "xlsx",
]

for _mod in _converter_modules:
    try:
        importlib.import_module(f".{_mod}", package=__name__)
    except Exception:
        # Silently skip missing optional dependencies; users will get a clear error on actual use
        pass
