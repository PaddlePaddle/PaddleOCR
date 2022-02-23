import typing as t
import warnings


class AuthorizationMixin:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        warnings.warn(
            "'AuthorizationMixin' is deprecated and will be removed in"
            " Werkzeug 2.1. 'Request' now includes the functionality"
            " directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)  # type: ignore


class WWWAuthenticateMixin:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        warnings.warn(
            "'WWWAuthenticateMixin' is deprecated and will be removed"
            " in Werkzeug 2.1. 'Response' now includes the"
            " functionality directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)  # type: ignore
