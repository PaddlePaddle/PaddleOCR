u"""
Fixer for Python 3 function parameter syntax
This fixer is rather sensitive to incorrect py3k syntax.
"""

# Note: "relevant" parameters are parameters following the first STAR in the list.

from lib2to3 import fixer_base
from lib2to3.fixer_util import token, String, Newline, Comma, Name
from libfuturize.fixer_util import indentation, suitify, DoubleStar

_assign_template = u"%(name)s = %(kwargs)s['%(name)s']; del %(kwargs)s['%(name)s']"
_if_template = u"if '%(name)s' in %(kwargs)s: %(assign)s"
_else_template = u"else: %(name)s = %(default)s"
_kwargs_default_name = u"_3to2kwargs"

def gen_params(raw_params):
    u"""
    Generator that yields tuples of (name, default_value) for each parameter in the list
    If no default is given, then it is default_value is None (not Leaf(token.NAME, 'None'))
    """
    assert raw_params[0].type == token.STAR and len(raw_params) > 2
    curr_idx = 2 # the first place a keyword-only parameter name can be is index 2
    max_idx = len(raw_params)
    while curr_idx < max_idx:
        curr_item = raw_params[curr_idx]
        prev_item = curr_item.prev_sibling
        if curr_item.type != token.NAME:
            curr_idx += 1
            continue
        if prev_item is not None and prev_item.type == token.DOUBLESTAR:
            break
        name = curr_item.value
        nxt = curr_item.next_sibling
        if nxt is not None and nxt.type == token.EQUAL:
            default_value = nxt.next_sibling
            curr_idx += 2
        else:
            default_value = None
        yield (name, default_value)
        curr_idx += 1

def remove_params(raw_params, kwargs_default=_kwargs_default_name):
    u"""
    Removes all keyword-only args from the params list and a bare star, if any.
    Does not add the kwargs dict if needed.
    Returns True if more action is needed, False if not
    (more action is needed if no kwargs dict exists)
    """
    assert raw_params[0].type == token.STAR
    if raw_params[1].type == token.COMMA:
        raw_params[0].remove()
        raw_params[1].remove()
        kw_params = raw_params[2:]
    else:
        kw_params = raw_params[3:]
    for param in kw_params:
        if param.type != token.DOUBLESTAR:
            param.remove()
        else:
            return False
    else:
        return True

def needs_fixing(raw_params, kwargs_default=_kwargs_default_name):
    u"""
    Returns string with the name of the kwargs dict if the params after the first star need fixing
    Otherwise returns empty string
    """
    found_kwargs = False
    needs_fix = False

    for t in raw_params[2:]:
        if t.type == token.COMMA:
            # Commas are irrelevant at this stage.
            continue
        elif t.type == token.NAME and not found_kwargs:
            # Keyword-only argument: definitely need to fix.
            needs_fix = True
        elif t.type == token.NAME and found_kwargs:
            # Return 'foobar' of **foobar, if needed.
            return t.value if needs_fix else u''
        elif t.type == token.DOUBLESTAR:
            # Found either '*' from **foobar.
            found_kwargs = True
    else:
        # Never found **foobar.  Return a synthetic name, if needed.
        return kwargs_default if needs_fix else u''

class FixKwargs(fixer_base.BaseFix):

    run_order = 7 # Run after function annotations are removed

    PATTERN = u"funcdef< 'def' NAME parameters< '(' arglist=typedargslist< params=any* > ')' > ':' suite=any >"

    def transform(self, node, results):
        params_rawlist = results[u"params"]
        for i, item in enumerate(params_rawlist):
            if item.type == token.STAR:
                params_rawlist = params_rawlist[i:]
                break
        else:
            return
        # params is guaranteed to be a list starting with *.
        # if fixing is needed, there will be at least 3 items in this list:
        # [STAR, COMMA, NAME] is the minimum that we need to worry about.
        new_kwargs = needs_fixing(params_rawlist)
        # new_kwargs is the name of the kwargs dictionary.
        if not new_kwargs:
            return
        suitify(node)

        # At this point, params_rawlist is guaranteed to be a list
        # beginning with a star that includes at least one keyword-only param
        # e.g., [STAR, NAME, COMMA, NAME, COMMA, DOUBLESTAR, NAME] or
        # [STAR, COMMA, NAME], or [STAR, COMMA, NAME, COMMA, DOUBLESTAR, NAME]

        # Anatomy of a funcdef: ['def', 'name', parameters, ':', suite]
        # Anatomy of that suite: [NEWLINE, INDENT, first_stmt, all_other_stmts]
        # We need to insert our new stuff before the first_stmt and change the
        # first_stmt's prefix.

        suite = node.children[4]
        first_stmt = suite.children[2]
        ident = indentation(first_stmt)

        for name, default_value in gen_params(params_rawlist):
            if default_value is None:
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_assign_template %{u'name':name, u'kwargs':new_kwargs}, prefix=ident))
            else:
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_else_template %{u'name':name, u'default':default_value}, prefix=ident))
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_if_template %{u'assign':_assign_template %{u'name':name, u'kwargs':new_kwargs}, u'name':name, u'kwargs':new_kwargs}, prefix=ident))
        first_stmt.prefix = ident
        suite.children[2].prefix = u""

        # Now, we need to fix up the list of params.

        must_add_kwargs = remove_params(params_rawlist)
        if must_add_kwargs:
            arglist = results[u'arglist']
            if len(arglist.children) > 0 and arglist.children[-1].type != token.COMMA:
                arglist.append_child(Comma())
            arglist.append_child(DoubleStar(prefix=u" "))
            arglist.append_child(Name(new_kwargs))
