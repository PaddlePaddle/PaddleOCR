"""Experimental settings for special stuff."""


def set(key, value):
    """Call to enable special settings:

    ('DXImageTransform.Microsoft', True)
        enable support for parsing special MS only filter values

    Clears the tokenizer cache which holds the compiled productions!
    """
    if key == 'DXImageTransform.Microsoft' and value is True:
        from . import cssproductions
        from . import tokenize2

        tokenize2._TOKENIZER_CACHE.clear()
        cssproductions.PRODUCTIONS.insert(1, cssproductions._DXImageTransform)
