from skimage._shared.testing import setup_test, teardown_test

# List of files that pytest should ignore
collect_ignore = ["io/_plugins",]
try:
    import visvis
except ImportError:
    collect_ignore.append("measure/mc_meta/visual_test.py")


def pytest_runtest_setup(item):
    setup_test()


def pytest_runtest_teardown(item):
    teardown_test()
