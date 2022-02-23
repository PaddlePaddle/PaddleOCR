"""Autouse fixtures for doctests."""

import pytest

from .linestring import LineString


@pytest.fixture(autouse=True)
def add_linestring(doctest_namespace):
        doctest_namespace["LineString"] = LineString
