''' Utilities to allow inserting docstring fragments for common
parameters into function and method docstrings'''

import sys

__all__ = ['docformat', 'inherit_docstring_from', 'indentcount_lines',
           'filldoc', 'unindent_dict', 'unindent_string', 'doc_replace']


def docformat(docstring, docdict=None):
    ''' Fill a function docstring from variables in dictionary

    Adapt the indent of the inserted docs

    Parameters
    ----------
    docstring : string
        docstring from function, possibly with dict formatting strings
    docdict : dict, optional
        dictionary with keys that match the dict formatting strings
        and values that are docstring fragments to be inserted. The
        indentation of the inserted docstrings is set to match the
        minimum indentation of the ``docstring`` by adding this
        indentation to all lines of the inserted string, except the
        first.

    Returns
    -------
    outstring : string
        string with requested ``docdict`` strings inserted

    Examples
    --------
    >>> docformat(' Test string with %(value)s', {'value':'inserted value'})
    ' Test string with inserted value'
    >>> docstring = 'First line\\n    Second line\\n    %(value)s'
    >>> inserted_string = "indented\\nstring"
    >>> docdict = {'value': inserted_string}
    >>> docformat(docstring, docdict)
    'First line\\n    Second line\\n    indented\\n    string'
    '''
    if not docstring:
        return docstring
    if docdict is None:
        docdict = {}
    if not docdict:
        return docstring
    lines = docstring.expandtabs().splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    indent = ' ' * icount
    # Insert this indent to dictionary docstrings
    indented = {}
    for name, dstr in docdict.items():
        lines = dstr.expandtabs().splitlines()
        try:
            newlines = [lines[0]]
            for line in lines[1:]:
                newlines.append(indent+line)
            indented[name] = '\n'.join(newlines)
        except IndexError:
            indented[name] = dstr
    return docstring % indented


def inherit_docstring_from(cls):
    """
    This decorator modifies the decorated function's docstring by
    replacing occurrences of '%(super)s' with the docstring of the
    method of the same name from the class `cls`.

    If the decorated method has no docstring, it is simply given the
    docstring of `cls`s method.

    Parameters
    ----------
    cls : Python class or instance
        A class with a method with the same name as the decorated method.
        The docstring of the method in this class replaces '%(super)s' in the
        docstring of the decorated method.

    Returns
    -------
    f : function
        The decorator function that modifies the __doc__ attribute
        of its argument.

    Examples
    --------
    In the following, the docstring for Bar.func created using the
    docstring of `Foo.func`.

    >>> class Foo:
    ...     def func(self):
    ...         '''Do something useful.'''
    ...         return
    ...
    >>> class Bar(Foo):
    ...     @inherit_docstring_from(Foo)
    ...     def func(self):
    ...         '''%(super)s
    ...         Do it fast.
    ...         '''
    ...         return
    ...
    >>> b = Bar()
    >>> b.func.__doc__
    'Do something useful.\n        Do it fast.\n        '

    """
    def _doc(func):
        cls_docstring = getattr(cls, func.__name__).__doc__
        func_docstring = func.__doc__
        if func_docstring is None:
            func.__doc__ = cls_docstring
        else:
            new_docstring = func_docstring % dict(super=cls_docstring)
            func.__doc__ = new_docstring
        return func
    return _doc


def extend_notes_in_docstring(cls, notes):
    """
    This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It extends the 'Notes' section of that docstring to include
    the given `notes`.
    """
    def _doc(func):
        cls_docstring = getattr(cls, func.__name__).__doc__
        # If python is called with -OO option,
        # there is no docstring
        if cls_docstring is None:
            return func
        end_of_notes = cls_docstring.find('        References\n')
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find('        Examples\n')
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        func.__doc__ = (cls_docstring[:end_of_notes] + notes +
                        cls_docstring[end_of_notes:])
        return func
    return _doc


def replace_notes_in_docstring(cls, notes):
    """
    This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It replaces the 'Notes' section of that docstring with
    the given `notes`.
    """
    def _doc(func):
        cls_docstring = getattr(cls, func.__name__).__doc__
        notes_header = '        Notes\n        -----\n'
        # If python is called with -OO option,
        # there is no docstring
        if cls_docstring is None:
            return func
        start_of_notes = cls_docstring.find(notes_header)
        end_of_notes = cls_docstring.find('        References\n')
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find('        Examples\n')
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        func.__doc__ = (cls_docstring[:start_of_notes + len(notes_header)] +
                        notes +
                        cls_docstring[end_of_notes:])
        return func
    return _doc


def indentcount_lines(lines):
    ''' Minimum indent for all lines in line list

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    '''
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def filldoc(docdict, unindent_params=True):
    ''' Return docstring decorator using docdict variable dictionary

    Parameters
    ----------
    docdict : dictionary
        dictionary containing name, docstring fragment pairs
    unindent_params : {False, True}, boolean, optional
        If True, strip common indentation from all parameters in
        docdict

    Returns
    -------
    decfunc : function
        decorator that applies dictionary to input function docstring

    '''
    if unindent_params:
        docdict = unindent_dict(docdict)

    def decorate(f):
        f.__doc__ = docformat(f.__doc__, docdict)
        return f
    return decorate


def unindent_dict(docdict):
    ''' Unindent all strings in a docdict '''
    can_dict = {}
    for name, dstr in docdict.items():
        can_dict[name] = unindent_string(dstr)
    return can_dict


def unindent_string(docstring):
    ''' Set docstring to minimum indent for all lines, including first

    >>> unindent_string(' two')
    'two'
    >>> unindent_string('  two\\n   three')
    'two\\n three'
    '''
    lines = docstring.expandtabs().splitlines()
    icount = indentcount_lines(lines)
    if icount == 0:
        return docstring
    return '\n'.join([line[icount:] for line in lines])


def doc_replace(obj, oldval, newval):
    """Decorator to take the docstring from obj, with oldval replaced by newval

    Equivalent to ``func.__doc__ = obj.__doc__.replace(oldval, newval)``

    Parameters
    ----------
    obj : object
        The object to take the docstring from.
    oldval : string
        The string to replace from the original docstring.
    newval : string
        The string to replace ``oldval`` with.
    """
    # __doc__ may be None for optimized Python (-OO)
    doc = (obj.__doc__ or '').replace(oldval, newval)

    def inner(func):
        func.__doc__ = doc
        return func

    return inner
