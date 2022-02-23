"""Aggregation function for CLI specified options and config file options.

This holds the logic that uses the collected and merged config files and
applies the user-specified command-line configuration on top of it.
"""
import argparse
import logging
from typing import List
from typing import Tuple

from flake8.options import config
from flake8.options.manager import OptionManager

LOG = logging.getLogger(__name__)


def aggregate_options(
    manager: OptionManager,
    config_finder: config.ConfigFileFinder,
    argv: List[str],
) -> Tuple[argparse.Namespace, List[str]]:
    """Aggregate and merge CLI and config file options.

    :param flake8.options.manager.OptionManager manager:
        The instance of the OptionManager that we're presently using.
    :param flake8.options.config.ConfigFileFinder config_finder:
        The config file finder to use.
    :param list argv:
        The list of remaining command-line arguments that were unknown during
        preliminary option parsing to pass to ``manager.parse_args``.
    :returns:
        Tuple of the parsed options and extra arguments returned by
        ``manager.parse_args``.
    :rtype:
        tuple(argparse.Namespace, list)
    """
    # Get defaults from the option parser
    default_values, _ = manager.parse_args([])

    # Make our new configuration file mergerator
    config_parser = config.ConfigParser(
        option_manager=manager, config_finder=config_finder
    )

    # Get the parsed config
    parsed_config = config_parser.parse()

    # Extend the default ignore value with the extended default ignore list,
    # registered by plugins.
    extended_default_ignore = manager.extended_default_ignore.copy()
    # Let's store our extended default ignore for use by the decision engine
    default_values.extended_default_ignore = (
        manager.extended_default_ignore.copy()
    )
    LOG.debug(
        "Extended default ignore list: %s", list(extended_default_ignore)
    )
    extended_default_ignore.update(default_values.ignore)
    default_values.ignore = list(extended_default_ignore)
    LOG.debug("Merged default ignore list: %s", default_values.ignore)

    extended_default_select = manager.extended_default_select.copy()
    LOG.debug(
        "Extended default select list: %s", list(extended_default_select)
    )
    default_values.extended_default_select = extended_default_select

    # Merge values parsed from config onto the default values returned
    for config_name, value in parsed_config.items():
        dest_name = config_name
        # If the config name is somehow different from the destination name,
        # fetch the destination name from our Option
        if not hasattr(default_values, config_name):
            dest_name = config_parser.config_options[config_name].dest

        LOG.debug(
            'Overriding default value of (%s) for "%s" with (%s)',
            getattr(default_values, dest_name, None),
            dest_name,
            value,
        )
        # Override the default values with the config values
        setattr(default_values, dest_name, value)

    # Finally parse the command-line options
    return manager.parse_args(argv, default_values)
