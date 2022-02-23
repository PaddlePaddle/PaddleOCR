"""
This module is designed to be used as follows::

    from past.builtins.noniterators import filter, map, range, reduce, zip

And then, for example::

    assert isinstance(range(5), list)

The list-producing functions this brings in are::

- ``filter``
- ``map``
- ``range``
- ``reduce``
- ``zip``

"""

from __future__ import division, absolute_import, print_function

from itertools import chain, starmap
import itertools       # since zip_longest doesn't exist on Py2
from past.types import basestring
from past.utils import PY3


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


if PY3:
    import builtins

    # list-producing versions of the major Python iterating functions
    def oldfilter(*args):
        """
        filter(function or None, sequence) -> list, tuple, or string

        Return those items of sequence for which function(item) is true.
        If function is None, return the items that are true.  If sequence
        is a tuple or string, return the same type, else return a list.
        """
        mytype = type(args[1])
        if isinstance(args[1], basestring):
            return mytype().join(builtins.filter(*args))
        elif isinstance(args[1], (tuple, list)):
            return mytype(builtins.filter(*args))
        else:
            # Fall back to list. Is this the right thing to do?
            return list(builtins.filter(*args))

    # This is surprisingly difficult to get right. For example, the
    # solutions here fail with the test cases in the docstring below:
    # http://stackoverflow.com/questions/8072755/
    def oldmap(func, *iterables):
        """
        map(function, sequence[, sequence, ...]) -> list

        Return a list of the results of applying the function to the
        items of the argument sequence(s).  If more than one sequence is
        given, the function is called with an argument list consisting of
        the corresponding item of each sequence, substituting None for
        missing values when not all sequences have the same length.  If
        the function is None, return a list of the items of the sequence
        (or a list of tuples if more than one sequence).

        Test cases:
        >>> oldmap(None, 'hello world')
        ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

        >>> oldmap(None, range(4))
        [0, 1, 2, 3]

        More test cases are in test_past.test_builtins.
        """
        zipped = itertools.zip_longest(*iterables)
        l = list(zipped)
        if len(l) == 0:
            return []
        if func is None:
            result = l
        else:
            result = list(starmap(func, l))

        # Inspect to see whether it's a simple sequence of tuples
        try:
            if max([len(item) for item in result]) == 1:
                return list(chain.from_iterable(result))
            # return list(flatmap(func, result))
        except TypeError as e:
            # Simple objects like ints have no len()
            pass
        return result

        ############################
        ### For reference, the source code for Py2.7 map function:
        # static PyObject *
        # builtin_map(PyObject *self, PyObject *args)
        # {
        #     typedef struct {
        #         PyObject *it;           /* the iterator object */
        #         int saw_StopIteration;  /* bool:  did the iterator end? */
        #     } sequence;
        #
        #     PyObject *func, *result;
        #     sequence *seqs = NULL, *sqp;
        #     Py_ssize_t n, len;
        #     register int i, j;
        #
        #     n = PyTuple_Size(args);
        #     if (n < 2) {
        #         PyErr_SetString(PyExc_TypeError,
        #                         "map() requires at least two args");
        #         return NULL;
        #     }
        #
        #     func = PyTuple_GetItem(args, 0);
        #     n--;
        #
        #     if (func == Py_None) {
        #         if (PyErr_WarnPy3k("map(None, ...) not supported in 3.x; "
        #                            "use list(...)", 1) < 0)
        #             return NULL;
        #         if (n == 1) {
        #             /* map(None, S) is the same as list(S). */
        #             return PySequence_List(PyTuple_GetItem(args, 1));
        #         }
        #     }
        #
        #     /* Get space for sequence descriptors.  Must NULL out the iterator
        #      * pointers so that jumping to Fail_2 later doesn't see trash.
        #      */
        #     if ((seqs = PyMem_NEW(sequence, n)) == NULL) {
        #         PyErr_NoMemory();
        #         return NULL;
        #     }
        #     for (i = 0; i < n; ++i) {
        #         seqs[i].it = (PyObject*)NULL;
        #         seqs[i].saw_StopIteration = 0;
        #     }
        #
        #     /* Do a first pass to obtain iterators for the arguments, and set len
        #      * to the largest of their lengths.
        #      */
        #     len = 0;
        #     for (i = 0, sqp = seqs; i < n; ++i, ++sqp) {
        #         PyObject *curseq;
        #         Py_ssize_t curlen;
        #
        #         /* Get iterator. */
        #         curseq = PyTuple_GetItem(args, i+1);
        #         sqp->it = PyObject_GetIter(curseq);
        #         if (sqp->it == NULL) {
        #             static char errmsg[] =
        #                 "argument %d to map() must support iteration";
        #             char errbuf[sizeof(errmsg) + 25];
        #             PyOS_snprintf(errbuf, sizeof(errbuf), errmsg, i+2);
        #             PyErr_SetString(PyExc_TypeError, errbuf);
        #             goto Fail_2;
        #         }
        #
        #         /* Update len. */
        #         curlen = _PyObject_LengthHint(curseq, 8);
        #         if (curlen > len)
        #             len = curlen;
        #     }
        #
        #     /* Get space for the result list. */
        #     if ((result = (PyObject *) PyList_New(len)) == NULL)
        #         goto Fail_2;
        #
        #     /* Iterate over the sequences until all have stopped. */
        #     for (i = 0; ; ++i) {
        #         PyObject *alist, *item=NULL, *value;
        #         int numactive = 0;
        #
        #         if (func == Py_None && n == 1)
        #             alist = NULL;
        #         else if ((alist = PyTuple_New(n)) == NULL)
        #             goto Fail_1;
        #
        #         for (j = 0, sqp = seqs; j < n; ++j, ++sqp) {
        #             if (sqp->saw_StopIteration) {
        #                 Py_INCREF(Py_None);
        #                 item = Py_None;
        #             }
        #             else {
        #                 item = PyIter_Next(sqp->it);
        #                 if (item)
        #                     ++numactive;
        #                 else {
        #                     if (PyErr_Occurred()) {
        #                         Py_XDECREF(alist);
        #                         goto Fail_1;
        #                     }
        #                     Py_INCREF(Py_None);
        #                     item = Py_None;
        #                     sqp->saw_StopIteration = 1;
        #                 }
        #             }
        #             if (alist)
        #                 PyTuple_SET_ITEM(alist, j, item);
        #             else
        #                 break;
        #         }
        #
        #         if (!alist)
        #             alist = item;
        #
        #         if (numactive == 0) {
        #             Py_DECREF(alist);
        #             break;
        #         }
        #
        #         if (func == Py_None)
        #             value = alist;
        #         else {
        #             value = PyEval_CallObject(func, alist);
        #             Py_DECREF(alist);
        #             if (value == NULL)
        #                 goto Fail_1;
        #         }
        #         if (i >= len) {
        #             int status = PyList_Append(result, value);
        #             Py_DECREF(value);
        #             if (status < 0)
        #                 goto Fail_1;
        #         }
        #         else if (PyList_SetItem(result, i, value) < 0)
        #             goto Fail_1;
        #     }
        #
        #     if (i < len && PyList_SetSlice(result, i, len, NULL) < 0)
        #         goto Fail_1;
        #
        #     goto Succeed;
        #
        # Fail_1:
        #     Py_DECREF(result);
        # Fail_2:
        #     result = NULL;
        # Succeed:
        #     assert(seqs);
        #     for (i = 0; i < n; ++i)
        #         Py_XDECREF(seqs[i].it);
        #     PyMem_DEL(seqs);
        #     return result;
        # }

    def oldrange(*args, **kwargs):
        return list(builtins.range(*args, **kwargs))

    def oldzip(*args, **kwargs):
        return list(builtins.zip(*args, **kwargs))

    filter = oldfilter
    map = oldmap
    range = oldrange
    from functools import reduce
    zip = oldzip
    __all__ = ['filter', 'map', 'range', 'reduce', 'zip']

else:
    import __builtin__
    # Python 2-builtin ranges produce lists
    filter = __builtin__.filter
    map = __builtin__.map
    range = __builtin__.range
    reduce = __builtin__.reduce
    zip = __builtin__.zip
    __all__ = []
