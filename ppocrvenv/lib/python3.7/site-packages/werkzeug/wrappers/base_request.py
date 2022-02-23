import typing as t
import warnings

from .request import Request


class _FakeSubclassCheck(type):
    def __subclasscheck__(cls, subclass: t.Type) -> bool:
        warnings.warn(
            "'BaseRequest' is deprecated and will be removed in"
            " Werkzeug 2.1. Use 'issubclass(cls, Request)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return issubclass(subclass, Request)

    def __instancecheck__(cls, instance: t.Any) -> bool:
        warnings.warn(
            "'BaseRequest' is deprecated and will be removed in"
            " Werkzeug 2.1. Use 'isinstance(obj, Request)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return isinstance(instance, Request)


class BaseRequest(Request, metaclass=_FakeSubclassCheck):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        warnings.warn(
            "'BaseRequest' is deprecated and will be removed in"
            " Werkzeug 2.1. 'Request' now includes the functionality"
            " directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
