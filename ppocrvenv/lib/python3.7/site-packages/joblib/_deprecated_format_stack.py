"""
Represent an exception with a lot of information.

Provides 2 useful functions:

format_exc: format an exception into a complete traceback, with full
            debugging instruction.

format_outer_frames: format the current position in the stack call.

Adapted from IPython's VerboseTB.
"""
# Authors: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
#          Nathaniel Gray <n8gray@caltech.edu>
#          Fernando Perez <fperez@colorado.edu>
# Copyright: 2010, Gael Varoquaux
#            2001-2004, Fernando Perez
#            2001 Nathaniel Gray
# License: BSD 3 clause
# flake8: noqa


import inspect
import keyword
import linecache
import os
import pydoc
import sys
import time
import tokenize
import traceback

INDENT = ' ' * 8


###############################################################################
# some internal-use functions
def safe_repr(value):
    """Hopefully pretty robust repr equivalent."""
    # this is pretty horrible but should always return *something*
    try:
        return pydoc.text.repr(value)
    except KeyboardInterrupt:
        raise
    except:
        try:
            return repr(value)
        except KeyboardInterrupt:
            raise
        except:
            try:
                # all still in an except block so we catch
                # getattr raising
                name = getattr(value, '__name__', None)
                if name:
                    # ick, recursion
                    return safe_repr(name)
                klass = getattr(value, '__class__', None)
                if klass:
                    return '%s instance' % safe_repr(klass)
            except KeyboardInterrupt:
                raise
            except:
                return 'UNRECOVERABLE REPR FAILURE'


def eq_repr(value, repr=safe_repr):
    return '=%s' % repr(value)


###############################################################################
def uniq_stable(elems):
    """uniq_stable(elems) -> list

    Return from an iterable, a list of all the unique elements in the input,
    but maintaining the order in which they first appear.

    A naive solution to this problem which just makes a dictionary with the
    elements as keys fails to respect the stability condition, since
    dictionaries are unsorted by nature.

    Note: All elements in the input must be hashable.
    """
    unique = []
    unique_set = set()
    for nn in elems:
        if nn not in unique_set:
            unique.append(nn)
            unique_set.add(nn)
    return unique


###############################################################################
def fix_frame_records_filenames(records):
    """Try to fix the filenames in each record from inspect.getinnerframes().

    Particularly, modules loaded from within zip files have useless filenames
    attached to their code object, and inspect.getinnerframes() just uses it.
    """
    fixed_records = []
    for frame, filename, line_no, func_name, lines, index in records:
        # Look inside the frame's globals dictionary for __file__, which should
        # be better.
        better_fn = frame.f_globals.get('__file__', None)
        if isinstance(better_fn, str):
            # Check the type just in case someone did something weird with
            # __file__. It might also be None if the error occurred during
            # import.
            filename = better_fn
        fixed_records.append((frame, filename, line_no, func_name, lines,
                              index))
    return fixed_records


def _fixed_getframes(etb, context=1, tb_offset=0):
    LNUM_POS, LINES_POS, INDEX_POS = 2, 4, 5

    records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))

    # If the error is at the console, don't build any context, since it would
    # otherwise produce 5 blank lines printed out (there is no file at the
    # console)
    rec_check = records[tb_offset:]
    try:
        rname = rec_check[0][1]
        if rname == '<ipython console>' or rname.endswith('<string>'):
            return rec_check
    except IndexError:
        pass

    aux = traceback.extract_tb(etb)
    assert len(records) == len(aux)
    for i, (file, lnum, _, _) in enumerate(aux):
        maybe_start = lnum - 1 - context // 2
        start = max(maybe_start, 0)
        end = start + context
        lines = linecache.getlines(file)[start:end]
        buf = list(records[i])
        buf[LNUM_POS] = lnum
        buf[INDEX_POS] = lnum - 1 - start
        buf[LINES_POS] = lines
        records[i] = tuple(buf)
    return records[tb_offset:]


def _format_traceback_lines(lnum, index, lines, lvals=None):
    numbers_width = 7
    res = []
    i = lnum - index

    for line in lines:
        if i == lnum:
            # This is the line with the error
            pad = numbers_width - len(str(i))
            if pad >= 3:
                marker = '-' * (pad - 3) + '-> '
            elif pad == 2:
                marker = '> '
            elif pad == 1:
                marker = '>'
            else:
                marker = ''
            num = marker + str(i)
        else:
            num = '%*s' % (numbers_width, i)
        line = '%s %s' % (num, line)

        res.append(line)
        if lvals and i == lnum:
            res.append(lvals + '\n')
        i = i + 1
    return res


def format_records(records):   # , print_globals=False):
    # Loop over all records printing context and info
    frames = []
    abspath = os.path.abspath
    for frame, file, lnum, func, lines, index in records:
        try:
            file = file and abspath(file) or '?'
        except OSError:
            # if file is '<console>' or something not in the filesystem,
            # the abspath call will throw an OSError.  Just ignore it and
            # keep the original file string.
            pass

        if file.endswith('.pyc'):
            file = file[:-4] + '.py'

        link = file

        args, varargs, varkw, locals = inspect.getargvalues(frame)

        if func == '?':
            call = ''
        else:
            # Decide whether to include variable details or not
            try:
                call = 'in %s%s' % (func, inspect.formatargvalues(args,
                                            varargs, varkw, locals,
                                            formatvalue=eq_repr))
            except KeyError:
                # Very odd crash from inspect.formatargvalues().  The
                # scenario under which it appeared was a call to
                # view(array,scale) in NumTut.view.view(), where scale had
                # been defined as a scalar (it should be a tuple). Somehow
                # inspect messes up resolving the argument list of view()
                # and barfs out. At some point I should dig into this one
                # and file a bug report about it.
                print("\nJoblib's exception reporting continues...\n")
                call = 'in %s(***failed resolving arguments***)' % func

        # Initialize a list of names on the current line, which the
        # tokenizer below will populate.
        names = []

        def tokeneater(token_type, token, start, end, line):
            """Stateful tokeneater which builds dotted names.

            The list of names it appends to (from the enclosing scope) can
            contain repeated composite names.  This is unavoidable, since
            there is no way to disambiguate partial dotted structures until
            the full list is known.  The caller is responsible for pruning
            the final list of duplicates before using it."""

            # build composite names
            if token == '.':
                try:
                    names[-1] += '.'
                    # store state so the next token is added for x.y.z names
                    tokeneater.name_cont = True
                    return
                except IndexError:
                    pass
            if token_type == tokenize.NAME and token not in keyword.kwlist:
                if tokeneater.name_cont:
                    # Dotted names
                    names[-1] += token
                    tokeneater.name_cont = False
                else:
                    # Regular new names.  We append everything, the caller
                    # will be responsible for pruning the list later.  It's
                    # very tricky to try to prune as we go, b/c composite
                    # names can fool us.  The pruning at the end is easy
                    # to do (or the caller can print a list with repeated
                    # names if so desired.
                    names.append(token)
            elif token_type == tokenize.NEWLINE:
                raise IndexError
        # we need to store a bit of state in the tokenizer to build
        # dotted names
        tokeneater.name_cont = False

        def linereader(file=file, lnum=[lnum], getline=linecache.getline):
            line = getline(file, lnum[0])
            lnum[0] += 1
            return line

        # Build the list of names on this line of code where the exception
        # occurred.
        try:
            # This builds the names list in-place by capturing it from the
            # enclosing scope.
            for token in tokenize.generate_tokens(linereader):
                tokeneater(*token)
        except (IndexError, UnicodeDecodeError, SyntaxError):
            # signals exit of tokenizer
            # SyntaxError can happen when trying to tokenize
            # a compiled (e.g. .so or .pyd) extension
            pass
        except tokenize.TokenError as msg:
            _m = ("An unexpected error occurred while tokenizing input file %s\n"
                  "The following traceback may be corrupted or invalid\n"
                  "The error message is: %s\n" % (file, msg))
            print(_m)

        # prune names list of duplicates, but keep the right order
        unique_names = uniq_stable(names)

        # Start loop over vars
        lvals = []
        for name_full in unique_names:
            name_base = name_full.split('.', 1)[0]
            if name_base in frame.f_code.co_varnames:
                if name_base in locals.keys():
                    try:
                        value = safe_repr(eval(name_full, locals))
                    except:
                        value = "undefined"
                else:
                    value = "undefined"
                name = name_full
                lvals.append('%s = %s' % (name, value))
            #elif print_globals:
            #    if frame.f_globals.has_key(name_base):
            #        try:
            #            value = safe_repr(eval(name_full,frame.f_globals))
            #        except:
            #            value = "undefined"
            #    else:
            #        value = "undefined"
            #    name = 'global %s' % name_full
            #    lvals.append('%s = %s' % (name,value))
        if lvals:
            lvals = '%s%s' % (INDENT, ('\n%s' % INDENT).join(lvals))
        else:
            lvals = ''

        level = '%s\n%s %s\n' % (75 * '.', link, call)

        if index is None:
            frames.append(level)
        else:
            frames.append('%s%s' % (level, ''.join(
                _format_traceback_lines(lnum, index, lines, lvals))))

    return frames


###############################################################################
def format_exc(etype, evalue, etb, context=5, tb_offset=0):
    """ Return a nice text document describing the traceback.

        Parameters
        -----------
        etype, evalue, etb: as returned by sys.exc_info
        context: number of lines of the source file to plot
        tb_offset: the number of stack frame not to use (0 = use all)

    """
    # some locals
    try:
        etype = etype.__name__
    except AttributeError:
        pass

    # Header with the exception type, python version, and date
    pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
    date = time.ctime(time.time())
    pid = 'PID: %i' % os.getpid()

    head = '%s%s%s\n%s%s%s' % (
        etype, ' ' * (75 - len(str(etype)) - len(date)),
        date, pid, ' ' * (75 - len(str(pid)) - len(pyver)),
        pyver)

    # Drop topmost frames if requested
    records = _fixed_getframes(etb, context, tb_offset)

    # Get (safely) a string form of the exception info
    try:
        etype_str, evalue_str = map(str, (etype, evalue))
    except BaseException:
        # User exception is improperly defined.
        etype, evalue = str, sys.exc_info()[:2]
        etype_str, evalue_str = map(str, (etype, evalue))
    # ... and format it
    exception = ['%s: %s' % (etype_str, evalue_str)]
    frames = format_records(records)
    return '%s\n%s\n%s' % (head, '\n'.join(frames), ''.join(exception[0]))


###############################################################################
def format_outer_frames(context=5, stack_start=None, stack_end=None,
                        ignore_ipython=True):
    LNUM_POS, LINES_POS, INDEX_POS = 2, 4, 5
    records = inspect.getouterframes(inspect.currentframe())
    output = list()

    for i, (frame, filename, line_no, func_name, lines, index) \
                                                in enumerate(records):
        # Look inside the frame's globals dictionary for __file__, which should
        # be better.
        better_fn = frame.f_globals.get('__file__', None)
        if isinstance(better_fn, str):
            # Check the type just in case someone did something weird with
            # __file__. It might also be None if the error occurred during
            # import.
            filename = better_fn
            if filename.endswith('.pyc'):
                filename = filename[:-4] + '.py'
        if ignore_ipython:
            # Hack to avoid printing the internals of IPython
            if (os.path.basename(filename) in ('iplib.py', 'py3compat.py')
                        and func_name in ('execfile', 'safe_execfile', 'runcode')):
                break
        maybe_start = line_no - 1 - context // 2
        start = max(maybe_start, 0)
        end = start + context
        lines = linecache.getlines(filename)[start:end]
        buf = list(records[i])
        buf[LNUM_POS] = line_no
        buf[INDEX_POS] = line_no - 1 - start
        buf[LINES_POS] = lines
        output.append(tuple(buf))
    return '\n'.join(format_records(output[stack_end:stack_start:-1]))
