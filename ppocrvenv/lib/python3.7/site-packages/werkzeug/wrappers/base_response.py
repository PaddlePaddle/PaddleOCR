import typing as t
import warnings

from .response import Response


class _FakeSubclassCheck(type):
    def __subclasscheck__(cls, subclass: t.Type) -> bool:
        warnings.warn(
            "'BaseResponse' is deprecated and will be removed in"
            " Werkzeug 2.1. Use 'issubclass(cls, Response)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return issubclass(subclass, Response)

    def __instancecheck__(cls, instance: t.Any) -> bool:
        warnings.warn(
            "'BaseResponse' is deprecated and will be removed in"
            " Werkzeug 2.1. Use 'isinstance(obj, Response)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return isinstance(instance, Response)


class BaseResponse(Response, metaclass=_FakeSubclassCheck):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        warnings.warn(
            "'BaseResponse' is deprecated and will be removed in"
            " Werkzeug 2.1. 'Response' now includes the functionality"
            " directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
