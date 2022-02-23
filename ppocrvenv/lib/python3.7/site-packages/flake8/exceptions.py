"""Exception classes for all of Flake8."""
from typing import Dict


class Flake8Exception(Exception):
    """Plain Flake8 exception."""


class EarlyQuit(Flake8Exception):
    """Except raised when encountering a KeyboardInterrupt."""


class ExecutionError(Flake8Exception):
    """Exception raised during execution of Flake8."""


class FailedToLoadPlugin(Flake8Exception):
    """Exception raised when a plugin fails to load."""

    FORMAT = 'Flake8 failed to load plugin "%(name)s" due to %(exc)s.'

    def __init__(self, plugin_name: str, exception: Exception) -> None:
        """Initialize our FailedToLoadPlugin exception."""
        self.plugin_name = plugin_name
        self.original_exception = exception
        super().__init__(plugin_name, exception)

    def __str__(self) -> str:
        """Format our exception message."""
        return self.FORMAT % {
            "name": self.plugin_name,
            "exc": self.original_exception,
        }


class PluginRequestedUnknownParameters(Flake8Exception):
    """The plugin requested unknown parameters."""

    FORMAT = '"%(name)s" requested unknown parameters causing %(exc)s'

    def __init__(self, plugin: Dict[str, str], exception: Exception) -> None:
        """Pop certain keyword arguments for initialization."""
        self.plugin = plugin
        self.original_exception = exception
        super().__init__(plugin, exception)

    def __str__(self) -> str:
        """Format our exception message."""
        return self.FORMAT % {
            "name": self.plugin["plugin_name"],
            "exc": self.original_exception,
        }


class PluginExecutionFailed(Flake8Exception):
    """The plugin failed during execution."""

    FORMAT = '"%(name)s" failed during execution due to "%(exc)s"'

    def __init__(self, plugin: Dict[str, str], exception: Exception) -> None:
        """Utilize keyword arguments for message generation."""
        self.plugin = plugin
        self.original_exception = exception
        super().__init__(plugin, exception)

    def __str__(self) -> str:
        """Format our exception message."""
        return self.FORMAT % {
            "name": self.plugin["plugin_name"],
            "exc": self.original_exception,
        }
