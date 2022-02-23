"""Module containing the logic for our debugging logic."""
import argparse
import json
import platform
from typing import Dict
from typing import List


class DebugAction(argparse.Action):
    """argparse action to print debug information."""

    def __init__(self, *args, **kwargs):
        """Initialize the action.

        This takes an extra `option_manager` keyword argument which will be
        used to delay response.
        """
        self._option_manager = kwargs.pop("option_manager")
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Perform the argparse action for printing debug information."""
        # NOTE(sigmavirus24): Flake8 parses options twice. The first time, we
        # will not have any registered plugins. We can skip this one and only
        # take action on the second time we're called.
        if not self._option_manager.registered_plugins:
            return
        print(
            json.dumps(
                information(self._option_manager), indent=2, sort_keys=True
            )
        )
        raise SystemExit(0)


def information(option_manager):
    """Generate the information to be printed for the bug report."""
    return {
        "version": option_manager.version,
        "plugins": plugins_from(option_manager),
        "dependencies": dependencies(),
        "platform": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "system": platform.system(),
        },
    }


def plugins_from(option_manager):
    """Generate the list of plugins installed."""
    return [
        {
            "plugin": plugin.name,
            "version": plugin.version,
            "is_local": plugin.local,
        }
        for plugin in sorted(option_manager.registered_plugins)
    ]


def dependencies() -> List[Dict[str, str]]:
    """Generate the list of dependencies we care about."""
    return []
