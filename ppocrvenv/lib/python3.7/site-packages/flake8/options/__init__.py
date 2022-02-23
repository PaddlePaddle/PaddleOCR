"""Package containing the option manager and config management logic.

- :mod:`flake8.options.config` contains the logic for finding, parsing, and
  merging configuration files.

- :mod:`flake8.options.manager` contains the logic for managing customized
  Flake8 command-line and configuration options.

- :mod:`flake8.options.aggregator` uses objects from both of the above modules
  to aggregate configuration into one object used by plugins and Flake8.

"""
