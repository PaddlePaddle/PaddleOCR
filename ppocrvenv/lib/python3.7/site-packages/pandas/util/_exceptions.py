from __future__ import annotations

import contextlib
import inspect
import os


@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str):
    """
    Rewrite the message of an exception.
    """
    try:
        yield
    except Exception as err:
        if not err.args:
            raise
        msg = str(err.args[0])
        msg = msg.replace(old_name, new_name)
        args: tuple[str, ...] = (msg,)
        if len(err.args) > 1:
            args = args + err.args[1:]
        err.args = args
        raise


def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """
    stack = inspect.stack()

    import pandas as pd

    pkg_dir = os.path.dirname(pd.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    for n in range(len(stack)):
        fname = stack[n].filename
        if fname.startswith(pkg_dir) and not fname.startswith(test_dir):
            continue
        else:
            break
    return n
