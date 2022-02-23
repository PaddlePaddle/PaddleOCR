from __future__ import absolute_import

import cython

from ..Plex.Scanners cimport Scanner

cdef unicode any_string_prefix, IDENT

cdef get_lexicon()
cdef initial_compile_time_env()

cdef class Method:
    cdef object name
    cdef dict kwargs
    cdef readonly object __name__  # for tracing the scanner

## methods commented with '##' out are used by Parsing.py when compiled.

@cython.final
cdef class CompileTimeScope:
    cdef public dict entries
    cdef public CompileTimeScope outer
    ##cdef declare(self, name, value)
    ##cdef lookup_here(self, name)
    ##cpdef lookup(self, name)

@cython.final
cdef class PyrexScanner(Scanner):
    cdef public context
    cdef public list included_files
    cdef public CompileTimeScope compile_time_env
    cdef public bint compile_time_eval
    cdef public bint compile_time_expr
    cdef public bint parse_comments
    cdef public bint in_python_file
    cdef public source_encoding
    cdef set keywords
    cdef public list indentation_stack
    cdef public indentation_char
    cdef public int bracket_nesting_level
    cdef readonly bint async_enabled
    cdef public sy
    cdef public systring

    cdef long current_level(self)
    #cpdef commentline(self, text)
    #cpdef open_bracket_action(self, text)
    #cpdef close_bracket_action(self, text)
    #cpdef newline_action(self, text)
    #cpdef begin_string_action(self, text)
    #cpdef end_string_action(self, text)
    #cpdef unclosed_string_action(self, text)
    @cython.locals(current_level=cython.long, new_level=cython.long)
    cpdef indentation_action(self, text)
    #cpdef eof_action(self, text)
    ##cdef next(self)
    ##cdef peek(self)
    #cpdef put_back(self, sy, systring)
    #cdef unread(self, token, value)
    ##cdef bint expect(self, what, message = *) except -2
    ##cdef expect_keyword(self, what, message = *)
    ##cdef expected(self, what, message = *)
    ##cdef expect_indent(self)
    ##cdef expect_dedent(self)
    ##cdef expect_newline(self, message=*, bint ignore_semicolon=*)
    ##cdef int enter_async(self) except -1
    ##cdef int exit_async(self) except -1
