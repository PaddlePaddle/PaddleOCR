"""
Provide the class Message and its subclasses.
"""


class Message(object):
    message = ''
    message_args = ()

    def __init__(self, filename, loc):
        self.filename = filename
        self.lineno = loc.lineno
        self.col = getattr(loc, 'col_offset', 0)

    def __str__(self):
        return '%s:%s:%s %s' % (self.filename, self.lineno, self.col+1,
                                self.message % self.message_args)


class UnusedImport(Message):
    message = '%r imported but unused'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class RedefinedWhileUnused(Message):
    message = 'redefinition of unused %r from line %r'

    def __init__(self, filename, loc, name, orig_loc):
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class RedefinedInListComp(Message):
    message = 'list comprehension redefines %r from line %r'

    def __init__(self, filename, loc, name, orig_loc):
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportShadowedByLoopVar(Message):
    message = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename, loc, name, orig_loc):
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportStarNotPermitted(Message):
    message = "'from %s import *' only allowed at module level"

    def __init__(self, filename, loc, modname):
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)


class ImportStarUsed(Message):
    message = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename, loc, modname):
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)


class ImportStarUsage(Message):
    message = "%r may be undefined, or defined from star imports: %s"

    def __init__(self, filename, loc, name, from_list):
        Message.__init__(self, filename, loc)
        self.message_args = (name, from_list)


class UndefinedName(Message):
    message = 'undefined name %r'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class DoctestSyntaxError(Message):
    message = 'syntax error in doctest'

    def __init__(self, filename, loc, position=None):
        Message.__init__(self, filename, loc)
        if position:
            (self.lineno, self.col) = position
        self.message_args = ()


class UndefinedExport(Message):
    message = 'undefined name %r in __all__'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class UndefinedLocal(Message):
    message = 'local variable %r {0} referenced before assignment'

    default = 'defined in enclosing scope on line %r'
    builtin = 'defined as a builtin'

    def __init__(self, filename, loc, name, orig_loc):
        Message.__init__(self, filename, loc)
        if orig_loc is None:
            self.message = self.message.format(self.builtin)
            self.message_args = name
        else:
            self.message = self.message.format(self.default)
            self.message_args = (name, orig_loc.lineno)


class DuplicateArgument(Message):
    message = 'duplicate argument %r in function definition'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class MultiValueRepeatedKeyLiteral(Message):
    message = 'dictionary key %r repeated with different values'

    def __init__(self, filename, loc, key):
        Message.__init__(self, filename, loc)
        self.message_args = (key,)


class MultiValueRepeatedKeyVariable(Message):
    message = 'dictionary key variable %s repeated with different values'

    def __init__(self, filename, loc, key):
        Message.__init__(self, filename, loc)
        self.message_args = (key,)


class LateFutureImport(Message):
    message = 'from __future__ imports must occur at the beginning of the file'

    def __init__(self, filename, loc, names):
        Message.__init__(self, filename, loc)
        self.message_args = ()


class FutureFeatureNotDefined(Message):
    """An undefined __future__ feature name was imported."""
    message = 'future feature %s is not defined'

    def __init__(self, filename, loc, name):
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class UnusedVariable(Message):
    """
    Indicates that a variable has been explicitly assigned to but not actually
    used.
    """
    message = 'local variable %r is assigned to but never used'

    def __init__(self, filename, loc, names):
        Message.__init__(self, filename, loc)
        self.message_args = (names,)


class ReturnWithArgsInsideGenerator(Message):
    """
    Indicates a return statement with arguments inside a generator.
    """
    message = '\'return\' with argument inside generator'


class ReturnOutsideFunction(Message):
    """
    Indicates a return statement outside of a function/method.
    """
    message = '\'return\' outside function'


class YieldOutsideFunction(Message):
    """
    Indicates a yield or yield from statement outside of a function/method.
    """
    message = '\'yield\' outside function'


# For whatever reason, Python gives different error messages for these two. We
# match the Python error message exactly.
class ContinueOutsideLoop(Message):
    """
    Indicates a continue statement outside of a while or for loop.
    """
    message = '\'continue\' not properly in loop'


class BreakOutsideLoop(Message):
    """
    Indicates a break statement outside of a while or for loop.
    """
    message = '\'break\' outside loop'


class ContinueInFinally(Message):
    """
    Indicates a continue statement in a finally block in a while or for loop.
    """
    message = '\'continue\' not supported inside \'finally\' clause'


class DefaultExceptNotLast(Message):
    """
    Indicates an except: block as not the last exception handler.
    """
    message = 'default \'except:\' must be last'


class TwoStarredExpressions(Message):
    """
    Two or more starred expressions in an assignment (a, *b, *c = d).
    """
    message = 'two starred expressions in assignment'


class TooManyExpressionsInStarredAssignment(Message):
    """
    Too many expressions in an assignment with star-unpacking
    """
    message = 'too many expressions in star-unpacking assignment'


class IfTuple(Message):
    """
    Conditional test is a non-empty tuple literal, which are always True.
    """
    message = '\'if tuple literal\' is always true, perhaps remove accidental comma?'


class AssertTuple(Message):
    """
    Assertion test is a non-empty tuple literal, which are always True.
    """
    message = 'assertion is always true, perhaps remove parentheses?'


class ForwardAnnotationSyntaxError(Message):
    message = 'syntax error in forward annotation %r'

    def __init__(self, filename, loc, annotation):
        Message.__init__(self, filename, loc)
        self.message_args = (annotation,)


class CommentAnnotationSyntaxError(Message):
    message = 'syntax error in type comment %r'

    def __init__(self, filename, loc, annotation):
        Message.__init__(self, filename, loc)
        self.message_args = (annotation,)


class RaiseNotImplemented(Message):
    message = "'raise NotImplemented' should be 'raise NotImplementedError'"


class InvalidPrintSyntax(Message):
    message = 'use of >> is invalid with print function'


class IsLiteral(Message):
    message = 'use ==/!= to compare constant literals (str, bytes, int, float, tuple)'


class FStringMissingPlaceholders(Message):
    message = 'f-string is missing placeholders'


class StringDotFormatExtraPositionalArguments(Message):
    message = "'...'.format(...) has unused arguments at position(s): %s"

    def __init__(self, filename, loc, extra_positions):
        Message.__init__(self, filename, loc)
        self.message_args = (extra_positions,)


class StringDotFormatExtraNamedArguments(Message):
    message = "'...'.format(...) has unused named argument(s): %s"

    def __init__(self, filename, loc, extra_keywords):
        Message.__init__(self, filename, loc)
        self.message_args = (extra_keywords,)


class StringDotFormatMissingArgument(Message):
    message = "'...'.format(...) is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename, loc, missing_arguments):
        Message.__init__(self, filename, loc)
        self.message_args = (missing_arguments,)


class StringDotFormatMixingAutomatic(Message):
    message = "'...'.format(...) mixes automatic and manual numbering"


class StringDotFormatInvalidFormat(Message):
    message = "'...'.format(...) has invalid format string: %s"

    def __init__(self, filename, loc, error):
        Message.__init__(self, filename, loc)
        self.message_args = (error,)


class PercentFormatInvalidFormat(Message):
    message = "'...' %% ... has invalid format string: %s"

    def __init__(self, filename, loc, error):
        Message.__init__(self, filename, loc)
        self.message_args = (error,)


class PercentFormatMixedPositionalAndNamed(Message):
    message = "'...' %% ... has mixed positional and named placeholders"


class PercentFormatUnsupportedFormatCharacter(Message):
    message = "'...' %% ... has unsupported format character %r"

    def __init__(self, filename, loc, c):
        Message.__init__(self, filename, loc)
        self.message_args = (c,)


class PercentFormatPositionalCountMismatch(Message):
    message = "'...' %% ... has %d placeholder(s) but %d substitution(s)"

    def __init__(self, filename, loc, n_placeholders, n_substitutions):
        Message.__init__(self, filename, loc)
        self.message_args = (n_placeholders, n_substitutions)


class PercentFormatExtraNamedArguments(Message):
    message = "'...' %% ... has unused named argument(s): %s"

    def __init__(self, filename, loc, extra_keywords):
        Message.__init__(self, filename, loc)
        self.message_args = (extra_keywords,)


class PercentFormatMissingArgument(Message):
    message = "'...' %% ... is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename, loc, missing_arguments):
        Message.__init__(self, filename, loc)
        self.message_args = (missing_arguments,)


class PercentFormatExpectedMapping(Message):
    message = "'...' %% ... expected mapping but got sequence"


class PercentFormatExpectedSequence(Message):
    message = "'...' %% ... expected sequence but got mapping"


class PercentFormatStarRequiresSequence(Message):
    message = "'...' %% ... `*` specifier requires sequence"
