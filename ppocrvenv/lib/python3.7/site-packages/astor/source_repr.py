# -*- coding: utf-8 -*-
"""
Part of the astor library for Python AST manipulation.

License: 3-clause BSD

Copyright (c) 2015 Patrick Maupin

Pretty-print source -- post-process for the decompiler

The goals of the initial cut of this engine are:

1) Do a passable, if not PEP8, job of line-wrapping.

2) Serve as an example of an interface to the decompiler
   for anybody who wants to do a better job. :)
"""


def pretty_source(source):
    """ Prettify the source.
    """

    return ''.join(split_lines(source))


def split_lines(source, maxline=79):
    """Split inputs according to lines.
       If a line is short enough, just yield it.
       Otherwise, fix it.
    """
    result = []
    extend = result.extend
    append = result.append
    line = []
    multiline = False
    count = 0
    for item in source:
        newline = type(item)('\n')
        index = item.find(newline)
        if index:
            line.append(item)
            multiline = index > 0
            count += len(item)
        else:
            if line:
                if count <= maxline or multiline:
                    extend(line)
                else:
                    wrap_line(line, maxline, result)
                count = 0
                multiline = False
                line = []
            append(item)
    return result


def count(group, slen=str.__len__):
    return sum([slen(x) for x in group])


def wrap_line(line, maxline=79, result=[], count=count):
    """ We have a line that is too long,
        so we're going to try to wrap it.
    """

    # Extract the indentation

    append = result.append
    extend = result.extend

    indentation = line[0]
    lenfirst = len(indentation)
    indent = lenfirst - len(indentation.lstrip())
    assert indent in (0, lenfirst)
    indentation = line.pop(0) if indent else ''

    # Get splittable/non-splittable groups

    dgroups = list(delimiter_groups(line))
    unsplittable = dgroups[::2]
    splittable = dgroups[1::2]

    # If the largest non-splittable group won't fit
    # on a line, try to add parentheses to the line.

    if max(count(x) for x in unsplittable) > maxline - indent:
        line = add_parens(line, maxline, indent)
        dgroups = list(delimiter_groups(line))
        unsplittable = dgroups[::2]
        splittable = dgroups[1::2]

    # Deal with the first (always unsplittable) group, and
    # then set up to deal with the remainder in pairs.

    first = unsplittable[0]
    append(indentation)
    extend(first)
    if not splittable:
        return result
    pos = indent + count(first)
    indentation += '    '
    indent += 4
    if indent >= maxline / 2:
        maxline = maxline / 2 + indent

    for sg, nsg in zip(splittable, unsplittable[1:]):

        if sg:
            # If we already have stuff on the line and even
            # the very first item won't fit, start a new line
            if pos > indent and pos + len(sg[0]) > maxline:
                append('\n')
                append(indentation)
                pos = indent

            # Dump lines out of the splittable group
            # until the entire thing fits
            csg = count(sg)
            while pos + csg > maxline:
                ready, sg = split_group(sg, pos, maxline)
                if ready[-1].endswith(' '):
                    ready[-1] = ready[-1][:-1]
                extend(ready)
                append('\n')
                append(indentation)
                pos = indent
                csg = count(sg)

            # Dump the remainder of the splittable group
            if sg:
                extend(sg)
                pos += csg

        # Dump the unsplittable group, optionally
        # preceded by a linefeed.
        cnsg = count(nsg)
        if pos > indent and pos + cnsg > maxline:
            append('\n')
            append(indentation)
            pos = indent
        extend(nsg)
        pos += cnsg


def split_group(source, pos, maxline):
    """ Split a group into two subgroups.  The
        first will be appended to the current
        line, the second will start the new line.

        Note that the first group must always
        contain at least one item.

        The original group may be destroyed.
    """
    first = []
    source.reverse()
    while source:
        tok = source.pop()
        first.append(tok)
        pos += len(tok)
        if source:
            tok = source[-1]
            allowed = (maxline + 1) if tok.endswith(' ') else (maxline - 4)
            if pos + len(tok) > allowed:
                break

    source.reverse()
    return first, source


begin_delim = set('([{')
end_delim = set(')]}')
end_delim.add('):')


def delimiter_groups(line, begin_delim=begin_delim,
                     end_delim=end_delim):
    """Split a line into alternating groups.
       The first group cannot have a line feed inserted,
       the next one can, etc.
    """
    text = []
    line = iter(line)
    while True:
        # First build and yield an unsplittable group
        for item in line:
            text.append(item)
            if item in begin_delim:
                break
        if not text:
            break
        yield text

        # Now build and yield a splittable group
        level = 0
        text = []
        for item in line:
            if item in begin_delim:
                level += 1
            elif item in end_delim:
                level -= 1
                if level < 0:
                    yield text
                    text = [item]
                    break
            text.append(item)
        else:
            assert not text, text
            break


statements = set(['del ', 'return', 'yield ', 'if ', 'while '])


def add_parens(line, maxline, indent, statements=statements, count=count):
    """Attempt to add parentheses around the line
       in order to make it splittable.
    """

    if line[0] in statements:
        index = 1
        if not line[0].endswith(' '):
            index = 2
            assert line[1] == ' '
        line.insert(index, '(')
        if line[-1] == ':':
            line.insert(-1, ')')
        else:
            line.append(')')

    # That was the easy stuff.  Now for assignments.
    groups = list(get_assign_groups(line))
    if len(groups) == 1:
        # So sad, too bad
        return line

    counts = list(count(x) for x in groups)
    didwrap = False

    # If the LHS is large, wrap it first
    if sum(counts[:-1]) >= maxline - indent - 4:
        for group in groups[:-1]:
            didwrap = False  # Only want to know about last group
            if len(group) > 1:
                group.insert(0, '(')
                group.insert(-1, ')')
                didwrap = True

    # Might not need to wrap the RHS if wrapped the LHS
    if not didwrap or counts[-1] > maxline - indent - 10:
        groups[-1].insert(0, '(')
        groups[-1].append(')')

    return [item for group in groups for item in group]


# Assignment operators
ops = list('|^&+-*/%@~') + '<< >> // **'.split() + ['']
ops = set(' %s= ' % x for x in ops)


def get_assign_groups(line, ops=ops):
    """ Split a line into groups by assignment (including
        augmented assignment)
    """
    group = []
    for item in line:
        group.append(item)
        if item in ops:
            yield group
            group = []
    yield group
