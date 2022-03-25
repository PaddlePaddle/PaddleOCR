"""Run testsuites written for nose."""
from _pytest.config import hookimpl
from _pytest.fixtures import getfixturemarker
from _pytest.nodes import Item
from _pytest.python import Function
from _pytest.unittest import TestCaseFunction


@hookimpl(trylast=True)
def pytest_runtest_setup(item: Item) -> None:
    if not isinstance(item, Function):
        return
    # Don't do nose style setup/teardown on direct unittest style classes.
    if isinstance(item, TestCaseFunction):
        return

    # Capture the narrowed type of item for the teardown closure,
    # see https://github.com/python/mypy/issues/2608
    func = item

    call_optional(func.obj, "setup")
    func.addfinalizer(lambda: call_optional(func.obj, "teardown"))

    # NOTE: Module- and class-level fixtures are handled in python.py
    # with `pluginmanager.has_plugin("nose")` checks.
    # It would have been nicer to implement them outside of core, but
    # it's not straightforward.


def call_optional(obj: object, name: str) -> bool:
    method = getattr(obj, name, None)
    if method is None:
        return False
    is_fixture = getfixturemarker(method) is not None
    if is_fixture:
        return False
    if not callable(method):
        return False
    # If there are any problems allow the exception to raise rather than
    # silently ignoring it.
    method()
    return True
