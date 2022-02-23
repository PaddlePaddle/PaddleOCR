import pytest


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "slow: Tests that are slow.")
