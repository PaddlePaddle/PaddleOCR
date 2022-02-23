"""
Preliminary module to handle Fortran formats for IO. Does not use this outside
scipy.sparse io for now, until the API is deemed reasonable.

The *Format classes handle conversion between Fortran and Python format, and
FortranFormatParser can create *Format instances from raw Fortran format
strings (e.g. '(3I4)', '(10I3)', etc...)
"""
import re

import numpy as np


__all__ = ["BadFortranFormat", "FortranFormatParser", "IntFormat", "ExpFormat"]


TOKENS = {
    "LPAR": r"\(",
    "RPAR": r"\)",
    "INT_ID": r"I",
    "EXP_ID": r"E",
    "INT": r"\d+",
    "DOT": r"\.",
}


class BadFortranFormat(SyntaxError):
    pass


def number_digits(n):
    return int(np.floor(np.log10(np.abs(n))) + 1)


class IntFormat:
    @classmethod
    def from_number(cls, n, min=None):
        """Given an integer, returns a "reasonable" IntFormat instance to represent
        any number between 0 and n if n > 0, -n and n if n < 0

        Parameters
        ----------
        n : int
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : IntFormat
            IntFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        without losing precision. For example, IntFormat.from_number(1) will
        return an IntFormat instance of width 2, so that any 0 and 1 may be
        represented as 1-character strings without loss of information.
        """
        width = number_digits(n) + 1
        if n < 0:
            width += 1
        repeat = 80 // width
        return cls(width, min, repeat=repeat)

    def __init__(self, width, min=None, repeat=None):
        self.width = width
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = "IntFormat("
        if self.repeat:
            r += "%d" % self.repeat
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r + ")"

    @property
    def fortran_format(self):
        r = "("
        if self.repeat:
            r += "%d" % self.repeat
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r + ")"

    @property
    def python_format(self):
        return "%" + str(self.width) + "d"


class ExpFormat:
    @classmethod
    def from_number(cls, n, min=None):
        """Given a float number, returns a "reasonable" ExpFormat instance to
        represent any number between -n and n.

        Parameters
        ----------
        n : float
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : ExpFormat
            ExpFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        to avoid losing precision.
        """
        # len of one number in exp format: sign + 1|0 + "." +
        # number of digit for fractional part + 'E' + sign of exponent +
        # len of exponent
        finfo = np.finfo(n.dtype)
        # Number of digits for fractional part
        n_prec = finfo.precision + 1
        # Number of digits for exponential part
        n_exp = number_digits(np.max(np.abs([finfo.maxexp, finfo.minexp])))
        width = 1 + 1 + n_prec + 1 + n_exp + 1
        if n < 0:
            width += 1
        repeat = int(np.floor(80 / width))
        return cls(width, n_prec, min, repeat=repeat)

    def __init__(self, width, significand, min=None, repeat=None):
        """\
        Parameters
        ----------
        width : int
            number of characters taken by the string (includes space).
        """
        self.width = width
        self.significand = significand
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = "ExpFormat("
        if self.repeat:
            r += "%d" % self.repeat
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r + ")"

    @property
    def fortran_format(self):
        r = "("
        if self.repeat:
            r += "%d" % self.repeat
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r + ")"

    @property
    def python_format(self):
        return "%" + str(self.width-1) + "." + str(self.significand) + "E"


class Token:
    def __init__(self, type, value, pos):
        self.type = type
        self.value = value
        self.pos = pos

    def __str__(self):
        return """Token('%s', "%s")""" % (self.type, self.value)

    def __repr__(self):
        return self.__str__()


class Tokenizer:
    def __init__(self):
        self.tokens = list(TOKENS.keys())
        self.res = [re.compile(TOKENS[i]) for i in self.tokens]

    def input(self, s):
        self.data = s
        self.curpos = 0
        self.len = len(s)

    def next_token(self):
        curpos = self.curpos

        while curpos < self.len:
            for i, r in enumerate(self.res):
                m = r.match(self.data, curpos)
                if m is None:
                    continue
                else:
                    self.curpos = m.end()
                    return Token(self.tokens[i], m.group(), self.curpos)
            raise SyntaxError("Unknown character at position %d (%s)"
                              % (self.curpos, self.data[curpos]))


# Grammar for fortran format:
# format            : LPAR format_string RPAR
# format_string     : repeated | simple
# repeated          : repeat simple
# simple            : int_fmt | exp_fmt
# int_fmt           : INT_ID width
# exp_fmt           : simple_exp_fmt
# simple_exp_fmt    : EXP_ID width DOT significand
# extended_exp_fmt  : EXP_ID width DOT significand EXP_ID ndigits
# repeat            : INT
# width             : INT
# significand       : INT
# ndigits           : INT

# Naive fortran formatter - parser is hand-made
class FortranFormatParser:
    """Parser for Fortran format strings. The parse method returns a *Format
    instance.

    Notes
    -----
    Only ExpFormat (exponential format for floating values) and IntFormat
    (integer format) for now.
    """
    def __init__(self):
        self.tokenizer = Tokenizer()

    def parse(self, s):
        self.tokenizer.input(s)

        tokens = []

        try:
            while True:
                t = self.tokenizer.next_token()
                if t is None:
                    break
                else:
                    tokens.append(t)
            return self._parse_format(tokens)
        except SyntaxError as e:
            raise BadFortranFormat(str(e)) from e

    def _get_min(self, tokens):
        next = tokens.pop(0)
        if not next.type == "DOT":
            raise SyntaxError()
        next = tokens.pop(0)
        return next.value

    def _expect(self, token, tp):
        if not token.type == tp:
            raise SyntaxError()

    def _parse_format(self, tokens):
        if not tokens[0].type == "LPAR":
            raise SyntaxError("Expected left parenthesis at position "
                              "%d (got '%s')" % (0, tokens[0].value))
        elif not tokens[-1].type == "RPAR":
            raise SyntaxError("Expected right parenthesis at position "
                              "%d (got '%s')" % (len(tokens), tokens[-1].value))

        tokens = tokens[1:-1]
        types = [t.type for t in tokens]
        if types[0] == "INT":
            repeat = int(tokens.pop(0).value)
        else:
            repeat = None

        next = tokens.pop(0)
        if next.type == "INT_ID":
            next = self._next(tokens, "INT")
            width = int(next.value)
            if tokens:
                min = int(self._get_min(tokens))
            else:
                min = None
            return IntFormat(width, min, repeat)
        elif next.type == "EXP_ID":
            next = self._next(tokens, "INT")
            width = int(next.value)

            next = self._next(tokens, "DOT")

            next = self._next(tokens, "INT")
            significand = int(next.value)

            if tokens:
                next = self._next(tokens, "EXP_ID")

                next = self._next(tokens, "INT")
                min = int(next.value)
            else:
                min = None
            return ExpFormat(width, significand, min, repeat)
        else:
            raise SyntaxError("Invalid formater type %s" % next.value)

    def _next(self, tokens, tp):
        if not len(tokens) > 0:
            raise SyntaxError()
        next = tokens.pop(0)
        self._expect(next, tp)
        return next
