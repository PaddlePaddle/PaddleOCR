"""Productions parser used by css and stylesheets classes to parse
test into a cssutils.util.Seq and at the same time retrieving
additional specific cssutils.util.Item objects for later use.

TODO:
    - ProdsParser
        - handle EOF or STOP?
        - handle unknown @rules
        - handle S: maybe save to Seq? parameterized?
        - store['_raw']: always?

    - Sequence:
        - opt first(), naive impl for now

"""
__all__ = ['ProdParser', 'Sequence', 'Choice', 'Prod', 'PreDef']

from .helper import pushtoken
import cssutils
import itertools
import re
import sys
import types


class ParseError(Exception):
    """Base Exception class for ProdParser (used internally)."""

    pass


class Done(ParseError):
    """Raised if Sequence or Choice is finished and no more Prods left."""

    pass


class Exhausted(ParseError):
    """Raised if Sequence or Choice is finished but token is given."""

    pass


class Missing(ParseError):
    """Raised if Sequence or Choice is not finished but no matching token given."""

    pass


class NoMatch(ParseError):
    """Raised if nothing in Sequence or Choice does match."""

    pass


class Choice(object):
    """A Choice of productions (Sequence or single Prod)."""

    def __init__(self, *prods, **options):
        """
        *prods
            Prod or Sequence objects
        options:
            optional=False
        """
        self._prods = prods

        try:
            self.optional = options['optional']
        except KeyError:
            for p in self._prods:
                if p.optional:
                    self.optional = True
                    break
            else:
                self.optional = False

        self.reset()

    def reset(self):
        """Start Choice from zero"""
        self._exhausted = False

    def matches(self, token):
        """Check if token matches"""
        for prod in self._prods:
            if prod.matches(token):
                return True
        return False

    def nextProd(self, token):
        """
        Return:

        - next matching Prod or Sequence
        - ``None`` if any Prod or Sequence is optional and no token matched
        - raise ParseError if nothing matches and all are mandatory
        - raise Exhausted if choice already done

        ``token`` may be None but this occurs when no tokens left."""
        # print u'TEST for %s in %s' % (token, self)
        if not self._exhausted:
            optional = False
            for p in self._prods:
                if p.matches(token):
                    self._exhausted = True
                    p.reset()
                    # print u'FOUND for %s: %s' % (token, p);#print
                    return p
                elif p.optional:
                    optional = True
            else:
                if not optional:
                    # None matched but also None is optional
                    raise NoMatch('No match for %s in %s' % (token, self))
                    # raise ParseError(u'No match in %s for %s' % (self, token))
        elif token:
            raise Exhausted('Extra token')

    def __repr__(self):
        return "<cssutils.prodsparser.%s object sequence=%r optional=%r at 0x%x>" % (
            self.__class__.__name__,
            self.__str__(),
            self.optional,
            id(self),
        )

    def __str__(self):
        return 'Choice(%s)' % ', '.join([str(x) for x in self._prods])


class Sequence(object):
    """A Sequence of productions (Choice or single Prod)."""

    def __init__(self, *prods, **options):
        """
        *prods
            Prod or Choice or Sequence objects
        **options:
            minmax = lambda: (1, 1)
                callback returning number of times this sequence may run
        """
        self._prods = prods
        try:
            minmax = options['minmax']
        except KeyError:
            minmax = lambda: (1, 1)  # noqa: E731

        self._min, self._max = minmax()
        if self._max is None:
            # unlimited
            try:
                # py2.6/3
                self._max = sys.maxsize
            except AttributeError:
                # py<2.6
                self._max = sys.maxsize

        self._prodcount = len(self._prods)
        self.reset()

    def matches(self, token):
        """Called by Choice to try to find if Sequence matches."""
        for prod in self._prods:
            if prod.matches(token):
                return True
            try:
                if not prod.optional:
                    break
            except AttributeError:
                pass
        return False

    def reset(self):
        """Reset this Sequence if it is nested."""
        self._roundstarted = False
        self._i = 0
        self._round = 0

    def _currentName(self):
        """Return current element of Sequence, used by name"""
        # TODO: current impl first only if 1st if an prod!
        for prod in self._prods[self._i :]:
            if not prod.optional:
                return str(prod)
        else:
            return 'Sequence'

    optional = property(lambda self: self._min == 0)

    def nextProd(self, token):
        """Return

        - next matching Prod or Choice
        - raises ParseError if nothing matches
        - raises Exhausted if sequence already done
        """
        # print u'TEST for %s in %s' % (token, self)
        while self._round < self._max:

            # for this round
            i = self._i
            round = self._round
            p = self._prods[i]
            if i == 0:
                self._roundstarted = False

            # for next round
            self._i += 1
            if self._i == self._prodcount:
                self._round += 1
                self._i = 0

            if p.matches(token):
                self._roundstarted = True
                # reset nested Choice or Prod to use from start
                p.reset()
                # print u'FOUND for %s: %s' % (token, p);#print
                return p

            elif p.optional:
                continue

            elif (
                round < self._min or self._roundstarted
            ):  # or (round == 0 and self._min == 0):
                raise Missing('Missing token for production %s' % p)

            elif not token:
                if self._roundstarted:
                    raise Missing('Missing token for production %s' % p)
                else:
                    raise Done()

            else:
                raise NoMatch('No match for %s in %s' % (token, self))

        if token:
            raise Exhausted('Extra token')

    def __repr__(self):
        return "<cssutils.prodsparser.%s object sequence=%r optional=%r at 0x%x>" % (
            self.__class__.__name__,
            self.__str__(),
            self.optional,
            id(self),
        )

    def __str__(self):
        return 'Sequence(%s)' % ', '.join([str(x) for x in self._prods])


class Prod(object):
    """Single Prod in Sequence or Choice."""

    def __init__(
        self,
        name,
        match,
        optional=False,
        toSeq=None,
        toStore=None,
        stop=False,
        stopAndKeep=False,
        stopIfNoMoreMatch=False,
        nextSor=False,
        mayEnd=False,
        storeToken=None,
        exception=None,
    ):
        """
        name
            name used for error reporting
        match callback
            function called with parameters tokentype and tokenvalue
            returning True, False or raising ParseError
        toSeq callback (optional) or False
            calling toSeq(token, tokens) returns (type_, val) == (token[0], token[1])
            to be appended to seq else simply unaltered (type_, val)

            if False nothing is added

        toStore (optional)
            key to save util.Item to store or callback(store, util.Item)
        optional = False
            whether Prod is optional or not
        stop = False
            if True stop parsing of tokens here
        stopAndKeep
            if True stop parsing of tokens here but return stopping
            token in unused tokens
        stopIfNoMoreMatch = False
            stop even if more tokens available, similar to stop and keep but with
            condition no more matches
        nextSor=False
            next is S or other like , or / (CSSValue)
        mayEnd = False
            no token must follow even defined by Sequence.
            Used for operator ',/ ' currently only

        storeToken = None
            if True toStore saves simple token tuple and not and Item object
            to store. Old style processing, TODO: resolve

        exception = None
            exception to be raised in case of error, normaly SyntaxErr
        """
        self._name = name
        self.match = match
        self.optional = optional
        self.stop = stop
        self.stopAndKeep = stopAndKeep
        self.stopIfNoMoreMatch = stopIfNoMoreMatch
        self.nextSor = nextSor
        self.mayEnd = mayEnd
        self.storeToken = storeToken
        self.exception = exception

        def makeToStore(key):
            "Return a function used by toStore."

            def toStore(store, item):
                "Set or append store item."
                if key in store:
                    _v = store[key]
                    if not isinstance(_v, list):
                        store[key] = [_v]
                    store[key].append(item)
                else:
                    store[key] = item

            return toStore

        if toSeq or toSeq is False:
            # called: seq.append(toSeq(value))
            self.toSeq = toSeq
        else:
            self.toSeq = lambda t, tokens: (t[0], t[1])

        if hasattr(toStore, '__call__'):
            self.toStore = toStore
        elif toStore:
            self.toStore = makeToStore(toStore)
        else:
            # always set!
            self.toStore = None

    def matches(self, token):
        """Return if token matches."""
        if not token:
            return False
        type_, val, line, col = token
        return self.match(type_, val)

    def reset(self):
        pass

    def __str__(self):
        return self._name

    def __repr__(self):
        return "<cssutils.prodsparser.%s object name=%r at 0x%x>" % (
            self.__class__.__name__,
            self._name,
            id(self),
        )


# global tokenizer as there is only one!
tokenizer = cssutils.tokenize2.Tokenizer()

# global: saved from subProds
savedTokens = []


class ProdParser(object):
    """Productions parser."""

    def __init__(self, clear=True):
        self.types = cssutils.cssproductions.CSSProductions
        self._log = cssutils.log
        if clear:
            tokenizer.clear()

    def _texttotokens(self, text):
        """Build a generator which is the only thing that is parsed!
        old classes may use lists etc
        """
        if isinstance(text, str):
            # DEFAULT, to tokenize strip space
            return tokenizer.tokenize(text.strip())

        elif isinstance(text, types.GeneratorType):
            # DEFAULT, already tokenized, should be generator
            return text

        elif isinstance(text, tuple):
            # OLD: (token, tokens) or a single token
            if len(text) == 2:
                # (token, tokens)
                # chain([token], tokens)
                raise NotImplementedError()
            else:
                # single token
                return iter([text])

        elif isinstance(text, list):
            # OLD: generator from list
            return iter(text)

        else:
            # ?
            return text

    def _SorTokens(self, tokens, until=',/'):
        """New tokens generator which has S tokens removed,
        if followed by anything in ``until``, normally a ``,``."""
        for token in tokens:
            if token[0] == self.types.S:
                try:
                    next_ = next(tokens)
                except StopIteration:
                    yield token
                else:
                    if next_[1] in until:
                        # omit S as e.g. ``,`` has been found
                        yield next_
                    elif next_[0] == self.types.COMMENT:
                        # pass COMMENT
                        yield next_
                    else:
                        yield token
                        yield next_

            elif token[0] == self.types.COMMENT:
                # pass COMMENT
                yield token
            else:
                yield token
                break
        # normal mode again
        for token in tokens:
            yield token

    def parse(  # noqa: C901
        self,
        text,
        name,
        productions,
        keepS=False,
        checkS=False,
        store=None,
        emptyOk=False,
        debug=False,
    ):
        """
        text (or token generator)
            to parse, will be tokenized if not a generator yet

            may be:
            - a string to be tokenized
            - a single token, a tuple
            - a tuple of (token, tokensGenerator)
            - already tokenized so a tokens generator

        name
            used for logging
        productions
            used to parse tokens
        keepS
            if WS should be added to Seq or just be ignored
        store  UPDATED
            If a Prod defines ``toStore`` the key defined there
            is a key in store to be set or if store[key] is a list
            the next Item is appended here.

            TODO: NEEDED? :
            Key ``raw`` is always added and holds all unprocessed
            values found
        emptyOk
            if True text may be empty, hard to test before as may be generator

        returns
            :wellformed: True or False
            :seq: a filled cssutils.util.Seq object which is NOT readonly yet
            :store: filled keys defined by Prod.toStore
            :unusedtokens: token generator containing tokens not used yet
        """
        tokens = self._texttotokens(text)

        if not tokens:
            self._log.error('No content to parse.')
            return False, [], None, None

        seq = cssutils.util.Seq(readonly=False)
        if not store:  # store for specific values
            store = {}
        prods = [productions]  # stack of productions
        wellformed = True
        # while no real token is found any S are ignored
        started = False
        stopall = False
        prod = None
        # flag if default S handling should be done
        defaultS = True

        stopIfNoMoreMatch = False

        while True:
            # get from savedTokens or normal tokens
            try:
                # print debug, "SAVED", savedTokens
                token = savedTokens.pop()
            except IndexError:
                try:
                    token = next(tokens)
                except StopIteration:
                    break

            # print debug, token, stopIfNoMoreMatch

            type_, val, line, col = token

            # default productions
            if type_ == self.types.COMMENT:
                # always append COMMENT
                seq.append(
                    cssutils.css.CSSComment(val), cssutils.css.CSSComment, line, col
                )

            elif defaultS and type_ == self.types.S and not checkS:
                # append S (but ignore starting ones)
                if not keepS or not started:
                    continue
                else:
                    seq.append(val, type_, line, col)

            #            elif type_ == self.types.ATKEYWORD:
            #                # @rule
            #                r = cssutils.css.CSSUnknownRule(cssText=val)
            #                seq.append(r, type(r), line, col)
            elif type_ == self.types.INVALID:
                # invalidate parse
                wellformed = False
                self._log.error('Invalid token: %r' % (token,))
                break

            elif type_ == 'EOF':
                # do nothing? (self.types.EOF == True!)
                stopall = True

            else:
                started = True  # check S now

                try:
                    while True:
                        # find next matching production
                        try:
                            prod = prods[-1].nextProd(token)
                        except (Exhausted, NoMatch):
                            # try next
                            prod = None

                        if isinstance(prod, Prod):
                            # found actual Prod, not a Choice or Sequence
                            break
                        elif prod:
                            # nested Sequence, Choice
                            prods.append(prod)
                        else:
                            # nested exhausted, try in parent
                            if len(prods) > 1:
                                prods.pop()
                            else:
                                raise NoMatch('No match')

                except NoMatch as e:
                    if stopIfNoMoreMatch:  # and token:
                        # print "\t1stopIfNoMoreMatch", e, token, prod, 'PUSHING'
                        # tokenizer.push(token)
                        savedTokens.append(token)
                        stopall = True

                    else:
                        wellformed = False
                        self._log.error('%s: %s: %r' % (name, e, token))
                    break

                except ParseError as e:
                    # needed???
                    if stopIfNoMoreMatch:  # and token:
                        # print "\t2stopIfNoMoreMatch", e, token, prod
                        tokenizer.push(token)
                        stopall = True

                    else:
                        wellformed = False
                        self._log.error('%s: %s: %r' % (name, e, token))
                    break

                else:
                    # print '\t1', debug, 'PROD', prod

                    # may stop next time, once set stays
                    stopIfNoMoreMatch = prod.stopIfNoMoreMatch or stopIfNoMoreMatch

                    # process prod
                    if prod.toSeq and not prod.stopAndKeep:
                        type_, val = prod.toSeq(token, tokens)
                        if val is not None:
                            seq.append(val, type_, line, col)
                            if prod.toStore:
                                if not prod.storeToken:
                                    prod.toStore(store, seq[-1])
                                else:
                                    # workaround for now for old style token
                                    # parsing!
                                    # TODO: remove when all new style
                                    prod.toStore(store, token)

                    if prod.stop:
                        # stop here and ignore following tokens
                        # EOF? or end of e.g. func ")"
                        break

                    if prod.stopAndKeep:  # e.g. ;
                        # stop here and ignore following tokens
                        # but keep this token for next run

                        # TODO: CHECK!!!!
                        tokenizer.push(token)
                        tokens = itertools.chain(token, tokens)

                        stopall = True
                        break

                    if prod.nextSor:
                        # following is S or other token (e.g. ",")?
                        # remove S if
                        tokens = self._SorTokens(tokens, ',/')
                        defaultS = False
                    else:
                        defaultS = True

        lastprod = prod
        # print debug, 'parse done', token, stopall, '\n'
        if not stopall:
            # stop immediately

            while True:
                # all productions exhausted?
                try:
                    prod = prods[-1].nextProd(token=None)
                except Done:
                    # ok
                    prod = None

                except Missing as e:
                    prod = None
                    # last was a S operator which may End a Sequence, then ok
                    if hasattr(lastprod, 'mayEnd') and not lastprod.mayEnd:
                        wellformed = False
                        self._log.error('%s: %s' % (name, e))

                except ParseError as e:
                    prod = None
                    wellformed = False
                    self._log.error('%s: %s' % (name, e))

                else:
                    if prods[-1].optional:
                        prod = None
                    elif prod and prod.optional:
                        # ignore optional
                        continue

                if prod and not prod.optional:
                    wellformed = False
                    self._log.error(
                        '%s: Missing token for production %r' % (name, str(prod))
                    )
                    break
                elif len(prods) > 1:
                    # nested exhausted, next in parent
                    prods.pop()
                else:
                    break

            if not emptyOk and not len(seq):
                self._log.error('No content to parse.')
                return False, [], None, None

        # trim S from end
        seq.rstrip()
        return wellformed, seq, store, tokens


class PreDef(object):
    """Predefined Prod definition for use in productions definition
    for ProdParser instances.
    """

    types = cssutils.cssproductions.CSSProductions
    reHexcolor = re.compile(r'^\#(?:[0-9abcdefABCDEF]{3}|[0-9abcdefABCDEF]{6})$')

    @staticmethod
    def calc(toSeq=None, nextSor=False):
        return Prod(
            name='calcfunction',
            match=lambda t, v: 'calc(' == cssutils.helper.normalize(v),
            toSeq=toSeq,
            nextSor=nextSor,
        )

    @staticmethod
    def char(
        name='char',
        char=',',
        toSeq=None,
        stop=False,
        stopAndKeep=False,
        mayEnd=False,
        stopIfNoMoreMatch=False,
        optional=False,  # WAS: optional=True,
        nextSor=False,
    ):
        "any CHAR"
        return Prod(
            name=name,
            match=lambda t, v: v == char,
            toSeq=toSeq,
            stop=stop,
            stopAndKeep=stopAndKeep,
            mayEnd=mayEnd,
            stopIfNoMoreMatch=stopIfNoMoreMatch,
            optional=optional,
            nextSor=nextSor,
        )

    @staticmethod
    def comma(optional=False, toSeq=None):
        return PreDef.char('comma', ',', optional=optional, toSeq=toSeq)

    @staticmethod
    def comment(parent=None):
        return Prod(
            name='comment',
            match=lambda t, v: t == 'COMMENT',
            toSeq=lambda t, tokens: (
                t[0],
                cssutils.css.CSSComment([1], parentRule=parent),
            ),
            optional=True,
        )

    @staticmethod
    def dimension(nextSor=False, stop=False):
        return Prod(
            name='dimension',
            match=lambda t, v: t == PreDef.types.DIMENSION,
            toSeq=lambda t, tokens: (t[0], cssutils.helper.normalize(t[1])),
            stop=stop,
            nextSor=nextSor,
        )

    @staticmethod
    def function(toSeq=None, nextSor=False, toStore=None):
        return Prod(
            name='function',
            match=lambda t, v: t == PreDef.types.FUNCTION,
            toStore=toStore,
            toSeq=toSeq,
            nextSor=nextSor,
        )

    @staticmethod
    def funcEnd(stop=False, mayEnd=False):
        ")"
        return PreDef.char('end FUNC ")"', ')', stop=stop, mayEnd=mayEnd)

    @staticmethod
    def hexcolor(stop=False, nextSor=False):
        "#123 or #123456"
        return Prod(
            name='HEX color',
            match=lambda t, v: (t == PreDef.types.HASH and PreDef.reHexcolor.match(v)),
            stop=stop,
            nextSor=nextSor,
        )

    @staticmethod
    def ident(stop=False, toStore=None, nextSor=False):
        return Prod(
            name='ident',
            match=lambda t, v: t == PreDef.types.IDENT,
            stop=stop,
            toStore=toStore,
            nextSor=nextSor,
        )

    @staticmethod
    def number(stop=False, toSeq=None, nextSor=False):
        return Prod(
            name='number',
            match=lambda t, v: t == PreDef.types.NUMBER,
            stop=stop,
            toSeq=toSeq,
            nextSor=nextSor,
        )

    @staticmethod
    def percentage(stop=False, toSeq=None, nextSor=False):
        return Prod(
            name='percentage',
            match=lambda t, v: t == PreDef.types.PERCENTAGE,
            stop=stop,
            toSeq=toSeq,
            nextSor=nextSor,
        )

    @staticmethod
    def string(stop=False, nextSor=False):
        "string delimiters are removed by default"
        return Prod(
            name='string',
            match=lambda t, v: t == PreDef.types.STRING,
            toSeq=lambda t, tokens: (t[0], cssutils.helper.stringvalue(t[1])),
            stop=stop,
            nextSor=nextSor,
        )

    @staticmethod
    def S(name='whitespace', toSeq=None, optional=False):
        return Prod(
            name=name,
            match=lambda t, v: t == PreDef.types.S,
            toSeq=toSeq,
            optional=optional,
            mayEnd=True,
        )

    @staticmethod
    def unary(stop=False, toSeq=None, nextSor=False):
        "+ or -"
        return Prod(
            name='unary +-',
            match=lambda t, v: v in ('+', '-'),
            optional=True,
            stop=stop,
            toSeq=toSeq,
            nextSor=nextSor,
        )

    @staticmethod
    def uri(stop=False, nextSor=False):
        "'url(' and ')' are removed and URI is stripped"
        return Prod(
            name='URI',
            match=lambda t, v: t == PreDef.types.URI,
            toSeq=lambda t, tokens: (t[0], cssutils.helper.urivalue(t[1])),
            stop=stop,
            nextSor=nextSor,
        )

    @staticmethod
    def unicode_range(stop=False, nextSor=False):
        "u+123456-abc normalized to lower `u`"
        return Prod(
            name='unicode-range',
            match=lambda t, v: t == PreDef.types.UNICODE_RANGE,
            toSeq=lambda t, tokens: (t[0], t[1].lower()),
            stop=stop,
            nextSor=nextSor,
        )

    @staticmethod
    def variable(toSeq=None, stop=False, nextSor=False, toStore=None):
        return Prod(
            name='variable',
            match=lambda t, v: 'var(' == cssutils.helper.normalize(v),
            toSeq=toSeq,
            toStore=toStore,
            stop=stop,
            nextSor=nextSor,
        )

    # used for MarginRule for now:
    @staticmethod
    def unknownrule(name='@', toStore=None):
        """@rule dummy (matches ATKEYWORD to remove unknown rule tokens from
        stream::

            @x;
            @x {...}

        no nested yet!
        """

        def rule(tokens):
            saved = []
            for t in tokens:
                saved.append(t)
                if t[1] == '}' or t[1] == ';':
                    return cssutils.css.CSSUnknownRule(saved)

        return Prod(
            name=name,
            match=lambda t, v: t == 'ATKEYWORD',
            toSeq=lambda t, tokens: ('CSSUnknownRule', rule(pushtoken(t, tokens))),
            toStore=toStore,
        )
