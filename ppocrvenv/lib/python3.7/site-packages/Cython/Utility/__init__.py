
def pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
    """
    Generate an unrolled shift-then-or loop over the first 'count' digits.
    Assumes that they fit into 'join_type'.

    (((d[2] << n) | d[1]) << n) | d[0]
    """
    return ('(' * (count * 2) + ' | '.join(
        "(%s)%s[%d])%s)" % (join_type, digits_ptr, _i, " << PyLong_SHIFT" if _i else '')
        for _i in range(count-1, -1, -1)))


# although it could potentially make use of data independence,
# this implementation is a bit slower than the simpler one above
def _pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
    """
    Generate an or-ed series of shifts for the first 'count' digits.
    Assumes that they fit into 'join_type'.

    (d[2] << 2*n) | (d[1] << 1*n) | d[0]
    """
    def shift(n):
        # avoid compiler warnings for overly large shifts that will be discarded anyway
        return " << (%d * PyLong_SHIFT < 8 * sizeof(%s) ? %d * PyLong_SHIFT : 0)" % (n, join_type, n) if n else ''

    return '(%s)' % ' | '.join(
        "(((%s)%s[%d])%s)" % (join_type, digits_ptr, i, shift(i))
        for i in range(count-1, -1, -1))
