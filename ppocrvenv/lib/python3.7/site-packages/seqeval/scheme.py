import enum
from itertools import chain
from typing import List, Set, Tuple, Type


class Entity:

    def __init__(self, sent_id: int, start: int, end: int, tag: str):
        self.sent_id = sent_id
        self.start = start
        self.end = end
        self.tag = tag

    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.sent_id, self.tag, self.start, self.end)

    def __eq__(self, other: 'Entity'):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        return self.sent_id, self.tag, self.start, self.end


class Prefix(enum.Flag):
    I = enum.auto()
    O = enum.auto()
    B = enum.auto()
    E = enum.auto()
    S = enum.auto()
    U = enum.auto()
    L = enum.auto()
    ANY = I | O | B | E | S | U | L


Prefixes = dict(Prefix.__members__)


class Tag(enum.Flag):
    SAME = enum.auto()
    DIFF = enum.auto()
    ANY = SAME | DIFF


class Token:
    allowed_prefix = None
    start_patterns = None
    inside_patterns = None
    end_patterns = None

    def __init__(self, token: str, suffix: bool = False, delimiter: str = '-'):
        self.token = token
        self.prefix = Prefixes[token[-1]] if suffix else Prefixes[token[0]]
        tag = token[:-1] if suffix else token[1:]
        self.tag = tag.strip(delimiter) or '_'

    def __repr__(self):
        return self.token

    def is_valid(self):
        """Check whether the prefix is allowed or not."""
        if self.prefix not in self.allowed_prefix:
            allowed_prefixes = str(self.allowed_prefix).replace('Prefix.', '')
            message = 'Invalid token is found: {}. Allowed prefixes are: {}.'
            raise ValueError(message.format(self.token, allowed_prefixes))
        return True

    def is_start(self, prev: 'Token'):
        """Check whether the current token is the start of chunk."""
        return self.check_patterns(prev, self.start_patterns)

    def is_inside(self, prev: 'Token'):
        """Check whether the current token is inside of chunk."""
        return self.check_patterns(prev, self.inside_patterns)

    def is_end(self, prev: 'Token'):
        """Check whether the previous token is the end of chunk."""
        return self.check_patterns(prev, self.end_patterns)

    def check_tag(self, prev: 'Token', cond: Tag):
        """Check whether the tag pattern is matched."""
        if cond == Tag.ANY:
            return True
        if prev.tag == self.tag and cond == Tag.SAME:
            return True
        if prev.tag != self.tag and cond == Tag.DIFF:
            return True
        return False

    def check_patterns(self, prev: 'Token', patterns: Set[Tuple[Prefix, Prefix, Tag]]):
        """Check whether the prefix patterns are matched."""
        for prev_prefix, current_prefix, tag_cond in patterns:
            if prev.prefix in prev_prefix and self.prefix in current_prefix and self.check_tag(prev, tag_cond):
                return True
        return False


class IOB1(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.B, Tag.SAME),
        (Prefix.B, Prefix.B, Tag.SAME)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.B, Tag.ANY),
        (Prefix.B, Prefix.O, Tag.ANY),
        (Prefix.B, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.B, Tag.SAME)
    }


class IOE1(Token):
    # Todo: IOE1 hasn't yet been able to handle some cases. See unit testing.
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.E, Prefix.I, Tag.ANY),
        (Prefix.E, Prefix.E, Tag.SAME)
    }
    inside_patterns = {
        (Prefix.I, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.E, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.E, Tag.DIFF),
        (Prefix.E, Prefix.I, Tag.SAME),
        (Prefix.E, Prefix.E, Tag.SAME)
    }


class IOB2(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    start_patterns = {
        (Prefix.ANY, Prefix.B, Tag.ANY)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.B, Tag.ANY),
        (Prefix.B, Prefix.O, Tag.ANY),
        (Prefix.B, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.B, Tag.ANY)
    }


class IOE2(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.O, Prefix.E, Tag.ANY),
        (Prefix.E, Prefix.I, Tag.ANY),
        (Prefix.E, Prefix.E, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.E, Tag.DIFF)
    }
    inside_patterns = {
        (Prefix.I, Prefix.E, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.E, Prefix.ANY, Tag.ANY)
    }


class IOBES(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B | Prefix.E | Prefix.S
    start_patterns = {
        (Prefix.ANY, Prefix.B, Tag.ANY),
        (Prefix.ANY, Prefix.S, Tag.ANY)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.B, Prefix.E, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.E, Tag.SAME)
    }
    end_patterns = {
        (Prefix.S, Prefix.ANY, Tag.ANY),
        (Prefix.E, Prefix.ANY, Tag.ANY)
    }


class BILOU(Token):
    allowed_prefix = Prefix.B | Prefix.I | Prefix.L | Prefix.O | Prefix.U
    start_patterns = {
        (Prefix.ANY, Prefix.B, Tag.ANY),
        (Prefix.ANY, Prefix.U, Tag.ANY)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.B, Prefix.L, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.L, Tag.SAME)
    }
    end_patterns = {
        (Prefix.U, Prefix.ANY, Tag.ANY),
        (Prefix.L, Prefix.ANY, Tag.ANY)
    }


class Tokens:

    def __init__(self, tokens: List[str], scheme: Type[Token],
                 suffix: bool = False, delimiter: str = '-', sent_id: int = None):
        self.outside_token = scheme('O', suffix=suffix, delimiter=delimiter)
        self.tokens = [scheme(token, suffix=suffix, delimiter=delimiter) for token in tokens]
        self.extended_tokens = self.tokens + [self.outside_token]
        self.sent_id = sent_id

    @property
    def entities(self):
        """Extract entities from tokens.

        Returns:
            list: list of Entity.

        Example:
            >>> tokens = Tokens(['B-PER', 'I-PER', 'O', 'B-LOC'], IOB2)
            >>> tokens.entities
            [('PER', 0, 2), ('LOC', 3, 4)]
        """
        i = 0
        entities = []
        prev = self.outside_token
        while i < len(self.extended_tokens):
            token = self.extended_tokens[i]
            token.is_valid()
            if token.is_start(prev):
                end = self._forward(start=i + 1, prev=token)
                if self._is_end(end):
                    entity = Entity(sent_id=self.sent_id, start=i, end=end, tag=token.tag)
                    entities.append(entity)
                i = end
            else:
                i += 1
            prev = self.extended_tokens[i - 1]
        return entities

    def _forward(self, start: int, prev: Token):
        for i, token in enumerate(self.extended_tokens[start:], start):
            if token.is_inside(prev):
                prev = token
            else:
                return i
        return len(self.tokens) - 1

    def _is_end(self, i: int):
        token = self.extended_tokens[i]
        prev = self.extended_tokens[i - 1]
        return token.is_end(prev)


class Entities:

    def __init__(self, sequences: List[List[str]], scheme: Type[Token], suffix: bool = False, delimiter: str = '-'):
        self.entities = [
            Tokens(seq, scheme=scheme, suffix=suffix, delimiter=delimiter, sent_id=sent_id).entities
            for sent_id, seq in enumerate(sequences)
        ]

    def filter(self, tag_name: str):
        entities = {entity for entity in chain(*self.entities) if entity.tag == tag_name}
        return entities

    @property
    def unique_tags(self):
        tags = {
            entity.tag for entity in chain(*self.entities)
        }
        return tags


def auto_detect(sequences: List[List[str]], suffix: bool = False, delimiter: str = '-'):
    """Detects scheme automatically.

    auto_detect supports the following schemes:
    - IOB2
    - IOE2
    - IOBES
    """
    prefixes = set()
    error_message = 'This scheme is not supported: {}'
    for tokens in sequences:
        for token in tokens:
            try:
                token = Token(token, suffix=suffix, delimiter=delimiter)
                prefixes.add(token.prefix)
            except KeyError:
                raise ValueError(error_message.format(token))

    allowed_iob2_prefixes = [
        {Prefix.I, Prefix.O, Prefix.B},
        {Prefix.I, Prefix.B},
        {Prefix.B, Prefix.O},
        {Prefix.B}
    ]
    allowed_ioe2_prefixes = [
        {Prefix.I, Prefix.O, Prefix.E},
        {Prefix.I, Prefix.E},
        {Prefix.E, Prefix.O},
        {Prefix.E}
    ]
    allowed_iobes_prefixes = [
        {Prefix.I, Prefix.O, Prefix.B, Prefix.E, Prefix.S},
        {Prefix.I, Prefix.B, Prefix.E, Prefix.S},
        {Prefix.I, Prefix.O, Prefix.B, Prefix.E},
        {Prefix.O, Prefix.B, Prefix.E, Prefix.S},
        {Prefix.I, Prefix.B, Prefix.E},
        {Prefix.B, Prefix.E, Prefix.S},
        {Prefix.O, Prefix.B, Prefix.E},
        {Prefix.B, Prefix.E},
        {Prefix.S}
    ]
    allowed_bilou_prefixes = [
        {Prefix.I, Prefix.O, Prefix.B, Prefix.L, Prefix.U},
        {Prefix.I, Prefix.B, Prefix.L, Prefix.U},
        {Prefix.I, Prefix.O, Prefix.B, Prefix.L},
        {Prefix.O, Prefix.B, Prefix.L, Prefix.U},
        {Prefix.I, Prefix.B, Prefix.L},
        {Prefix.B, Prefix.L, Prefix.U},
        {Prefix.O, Prefix.B, Prefix.L},
        {Prefix.B, Prefix.L},
        {Prefix.U}
    ]
    if prefixes in allowed_iob2_prefixes:
        return IOB2
    elif prefixes in allowed_ioe2_prefixes:
        return IOE2
    elif prefixes in allowed_iobes_prefixes:
        return IOBES
    elif prefixes in allowed_bilou_prefixes:
        return BILOU
    else:
        raise ValueError(error_message.format(prefixes))
