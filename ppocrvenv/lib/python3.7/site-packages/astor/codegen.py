import warnings

from .code_gen import *  # NOQA


warnings.warn(
    'astor.codegen module is deprecated. Please import '
    'astor.code_gen module instead.',
    DeprecationWarning,
    stacklevel=2
)
