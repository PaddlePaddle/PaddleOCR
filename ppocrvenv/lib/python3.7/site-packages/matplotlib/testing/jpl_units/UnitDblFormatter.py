"""UnitDblFormatter module containing class UnitDblFormatter."""

import matplotlib.ticker as ticker

__all__ = ['UnitDblFormatter']


class UnitDblFormatter(ticker.ScalarFormatter):
    """
    The formatter for UnitDbl data types.

    This allows for formatting with the unit string.
    """

    def __call__(self, x, pos=None):
        # docstring inherited
        if len(self.locs) == 0:
            return ''
        else:
            return '{:.12}'.format(x)

    def format_data_short(self, value):
        # docstring inherited
        return '{:.12}'.format(value)

    def format_data(self, value):
        # docstring inherited
        return '{:.12}'.format(value)
