# Do not collect any tests in externals. This is more robust than using
# --ignore because --ignore needs a path and it is not convenient to pass in
# the externals path (very long install-dependent path in site-packages) when
# using --pyargs
def pytest_ignore_collect(path, config):
    return True

