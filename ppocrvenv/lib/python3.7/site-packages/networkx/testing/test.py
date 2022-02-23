import warnings


def run(verbosity=1, doctest=False):
    """Run NetworkX tests.

    Parameters
    ----------
    verbosity: integer, optional
      Level of detail in test reports.  Higher numbers provide more detail.

    doctest: bool, optional
      True to run doctests in code modules
    """
    warnings.warn(
        (
            "`run` is deprecated and will be removed in version 3.0.\n"
            "Call `pytest` directly from the commandline instead.\n"
        ),
        DeprecationWarning,
    )

    import pytest

    pytest_args = ["-l"]

    if verbosity and int(verbosity) > 1:
        pytest_args += ["-" + "v" * (int(verbosity) - 1)]

    if doctest:
        pytest_args += ["--doctest-modules"]

    pytest_args += ["--pyargs", "networkx"]

    try:
        code = pytest.main(pytest_args)
    except SystemExit as exc:
        code = exc.code

    return code == 0


if __name__ == "__main__":
    run()
