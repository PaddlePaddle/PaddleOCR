PYTHON_EXTENSIONS_PATHS = [
    LOADER_DIR
] + PYTHON_EXTENSIONS_PATHS

ci_and_not_headless = False

try:
    from .version import ci_build, headless

    ci_and_not_headless = ci_build and not headless
except:
    pass

# the Qt plugin is included currently only in the pre-built wheels
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qt", "plugins"
    )

# Qt will throw warning on Linux if fonts are not found
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ["QT_QPA_FONTDIR"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qt", "fonts"
    )
