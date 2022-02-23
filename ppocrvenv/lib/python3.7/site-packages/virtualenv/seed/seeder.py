from __future__ import absolute_import, unicode_literals

from abc import ABCMeta, abstractmethod

from six import add_metaclass


@add_metaclass(ABCMeta)
class Seeder(object):
    """A seeder will install some seed packages into a virtual environment."""

    # noinspection PyUnusedLocal
    def __init__(self, options, enabled):
        """

        :param options: the parsed options as defined within :meth:`add_parser_arguments`
        :param enabled: a flag weather the seeder is enabled or not
        """
        self.enabled = enabled
        self.env = options.env

    @classmethod
    def add_parser_arguments(cls, parser, interpreter, app_data):
        """
        Add CLI arguments for this seed mechanisms.

        :param parser: the CLI parser
        :param app_data: the CLI parser
        :param interpreter: the interpreter this virtual environment is based of
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, creator):
        """Perform the seed operation.

        :param creator: the creator (based of :class:`virtualenv.create.creator.Creator`) we used to create this \
        virtual environment
        """
        raise NotImplementedError
