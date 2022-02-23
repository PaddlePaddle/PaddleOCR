# Copyright (c) 2010-2021 openpyxl

from itertools import accumulate
import operator

from openpyxl.compat.product import prod


def dataframe_to_rows(df, index=True, header=True):
    """
    Convert a Pandas dataframe into something suitable for passing into a worksheet.
    If index is True then the index will be included, starting one row below the header.
    If header is True then column headers will be included starting one column to the right.
    Formatting should be done by client code.
    """
    import numpy
    from pandas import Timestamp
    blocks = df._data.blocks
    ncols = sum(b.shape[0] for b in blocks)
    data = [None] * ncols

    for b in blocks:
        values = b.values

        if b.dtype.type == numpy.datetime64:
            values = numpy.array([Timestamp(v) for v in values.ravel()])
            values = values.reshape(b.shape)

        result = values.tolist()

        for col_loc, col in zip(b.mgr_locs, result):
            data[col_loc] = col

    if header:
        if df.columns.nlevels > 1:
            rows = expand_index(df.columns, header)
        else:
            rows = [list(df.columns.values)]
        for row in rows:
            n = []
            for v in row:
                if isinstance(v, numpy.datetime64):
                    v = Timestamp(v)
                n.append(v)
            row = n
            if index:
                row = [None]*df.index.nlevels + row
            yield row

    if index:
        yield df.index.names

    expanded = ([v] for v in df.index)
    if df.index.nlevels > 1:
        expanded = expand_index(df.index)

    for idx, v in enumerate(expanded):
        row = [data[j][idx] for j in range(ncols)]
        if index:
            row = v + row
        yield row


def expand_index(index, header=False):
    """
    Expand axis or column Multiindex
    For columns use header = True
    For axes use header = False (default)
    """

    shape = index.levshape
    depth = prod(shape)
    row = [None] * index.nlevels
    lengths = [depth / size for size in accumulate(shape, operator.mul)] # child index lengths
    columns = [ [] for l in index.names] # avoid copied list gotchas

    for idx, entry in enumerate(index):
        row = [None] * index.nlevels
        for level, v in enumerate(entry):
            length = lengths[level]
            if idx % length:
                v = None
            row[level] = v
            if header:
                columns[level].append(v)

        if not header:
            yield row

    if header:
        for row in columns:
            yield row
