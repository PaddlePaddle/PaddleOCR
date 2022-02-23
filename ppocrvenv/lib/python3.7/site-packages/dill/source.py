#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
#
# inspired by inspect.py from Python-2.7.6
# inspect.py author: 'Ka-Ping Yee <ping@lfw.org>'
# inspect.py merged into original dill.source by Mike McKerns 4/13/14
"""
Extensions to python's 'inspect' module, which can be used
to retrieve information from live python objects. The methods
defined in this module are augmented to facilitate access to 
source code of interactively defined functions and classes,
as well as provide access to source code for objects defined
in a file.
"""

__all__ = ['findsource', 'getsourcelines', 'getsource', 'indent', 'outdent', \
           '_wrap', 'dumpsource', 'getname', '_namespace', 'getimport', \
           '_importable', 'importable','isdynamic', 'isfrommain']

import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
                     isbuiltin, isclass, iscode, isframe, isfunction, ismethod,
                     ismodule, istraceback)
from tokenize import TokenError

from ._dill import PY3


def isfrommain(obj):
    "check if object was built in __main__"
    module = getmodule(obj)
    if module and module.__name__ == '__main__':
        return True
    return False


def isdynamic(obj):
    "check if object was built in the interpreter"
    try: file = getfile(obj)
    except TypeError: file = None 
    if file == '<stdin>' and isfrommain(obj):
        return True
    return False


def _matchlambda(func, line):
    """check if lambda object 'func' matches raw line of code 'line'"""
    from .detect import code as getcode
    from .detect import freevars, globalvars, varnames
    dummy = lambda : '__this_is_a_big_dummy_function__'
    # process the line (removing leading whitespace, etc)
    lhs,rhs = line.split('lambda ',1)[-1].split(":", 1) #FIXME: if !1 inputs
    try: #FIXME: unsafe
        _ = eval("lambda %s : %s" % (lhs,rhs), globals(),locals())
    except: _ = dummy
    # get code objects, for comparison
    _, code = getcode(_).co_code, getcode(func).co_code
    # check if func is in closure
    _f = [line.count(i) for i in freevars(func).keys()]
    if not _f: # not in closure
        # check if code matches
        if _ == code: return True
        return False
    # weak check on freevars
    if not all(_f): return False  #XXX: VERY WEAK
    # weak check on varnames and globalvars
    _f = varnames(func)
    _f = [line.count(i) for i in _f[0]+_f[1]]
    if _f and not all(_f): return False  #XXX: VERY WEAK
    _f = [line.count(i) for i in globalvars(func).keys()]
    if _f and not all(_f): return False  #XXX: VERY WEAK
    # check if func is a double lambda
    if (line.count('lambda ') > 1) and (lhs in freevars(func).keys()):
        _lhs,_rhs = rhs.split('lambda ',1)[-1].split(":",1) #FIXME: if !1 inputs
        try: #FIXME: unsafe
            _f = eval("lambda %s : %s" % (_lhs,_rhs), globals(),locals())
        except: _f = dummy
        # get code objects, for comparison
        _, code = getcode(_f).co_code, getcode(func).co_code
        if len(_) != len(code): return False
        #NOTE: should be same code same order, but except for 't' and '\x88'
        _ = set((i,j) for (i,j) in zip(_,code) if i != j)
        if len(_) != 1: return False #('t','\x88')
        return True
    # check indentsize
    if not indentsize(line): return False #FIXME: is this a good check???
    # check if code 'pattern' matches
    #XXX: or pattern match against dis.dis(code)? (or use uncompyle2?)
    _ = _.split(_[0])  # 't' #XXX: remove matching values if starts the same?
    _f = code.split(code[0])  # '\x88'
    #NOTE: should be same code different order, with different first element
    _ = dict(re.match(r'([\W\D\S])(.*)', _[i]).groups() for i in range(1,len(_)))
    _f = dict(re.match(r'([\W\D\S])(.*)', _f[i]).groups() for i in range(1,len(_f)))
    if (_.keys() == _f.keys()) and (sorted(_.values()) == sorted(_f.values())):
        return True
    return False


def findsource(object):
    """Return the entire source file and starting line number for an object.
    For interactively-defined objects, the 'file' is the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of all the lines
    in the file and the line number indexes a line in that list.  An IOError
    is raised if the source code cannot be retrieved, while a TypeError is
    raised for objects where the source code is unavailable (e.g. builtins)."""

    module = getmodule(object)
    try: file = getfile(module)
    except TypeError: file = None 
    # use readline when working in interpreter (i.e. __main__ and not file)
    if module and module.__name__ == '__main__' and not file:
        try: 
            import readline
            err = ''
        except:
            import sys
            err = sys.exc_info()[1].args[0]
            if sys.platform[:3] == 'win':
                err += ", please install 'pyreadline'"
        if err:
            raise IOError(err)
        lbuf = readline.get_current_history_length()
        lines = [readline.get_history_item(i)+'\n' for i in range(1,lbuf)]
    else:
        try: # special handling for class instances
            if not isclass(object) and isclass(type(object)): # __class__
                file = getfile(module)        
                sourcefile = getsourcefile(module)
            else: # builtins fail with a TypeError
                file = getfile(object)
                sourcefile = getsourcefile(object)
        except (TypeError, AttributeError): # fail with better error
            file = getfile(object)
            sourcefile = getsourcefile(object)
        if not sourcefile and file[:1] + file[-1:] != '<>':
            raise IOError('source code not available')
        file = sourcefile if sourcefile else file

        module = getmodule(object, file)
        if module:
            lines = linecache.getlines(file, module.__dict__)
        else:
            lines = linecache.getlines(file)

    if not lines:
        raise IOError('could not extract source code')

    #FIXME: all below may fail if exec used (i.e. exec('f = lambda x:x') )
    if ismodule(object):
        return lines, 0

    #NOTE: beneficial if search goes from end to start of buffer history
    name = pat1 = obj = ''
    pat2 = r'^(\s*@)'
#   pat1b = r'^(\s*%s\W*=)' % name #FIXME: finds 'f = decorate(f)', not exec
    if ismethod(object):
        name = object.__name__
        if name == '<lambda>': pat1 = r'(.*(?<!\w)lambda(:|\s))'
        else: pat1 = r'^(\s*def\s)'
        if PY3: object = object.__func__
        else: object = object.im_func
    if isfunction(object):
        name = object.__name__
        if name == '<lambda>':
            pat1 = r'(.*(?<!\w)lambda(:|\s))'
            obj = object #XXX: better a copy?
        else: pat1 = r'^(\s*def\s)'
        if PY3: object = object.__code__
        else: object = object.func_code
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        if not hasattr(object, 'co_firstlineno'):
            raise IOError('could not find function definition')
        stdin = object.co_filename == '<stdin>'
        if stdin:
            lnum = len(lines) - 1 # can't get lnum easily, so leverage pat
            if not pat1: pat1 = r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)'
        else:
            lnum = object.co_firstlineno - 1
            pat1 = r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)'
        pat1 = re.compile(pat1); pat2 = re.compile(pat2)
       #XXX: candidate_lnum = [n for n in range(lnum) if pat1.match(lines[n])]
        while lnum > 0: #XXX: won't find decorators in <stdin> ?
            line = lines[lnum]
            if pat1.match(line):
                if not stdin: break # co_firstlineno does the job
                if name == '<lambda>': # hackery needed to confirm a match
                    if _matchlambda(obj, line): break
                else: # not a lambda, just look for the name
                    if name in line: # need to check for decorator...
                        hats = 0
                        for _lnum in range(lnum-1,-1,-1):
                            if pat2.match(lines[_lnum]): hats += 1
                            else: break
                        lnum = lnum - hats
                        break
            lnum = lnum - 1
        return lines, lnum

    try: # turn instances into classes
        if not isclass(object) and isclass(type(object)): # __class__
            object = object.__class__ #XXX: sometimes type(class) is better?
            #XXX: we don't find how the instance was built
    except AttributeError: pass
    if isclass(object):
        name = object.__name__
        pat = re.compile(r'^(\s*)class\s*' + name + r'\b')
        # make some effort to find the best matching class definition:
        # use the one with the least indentation, which is the one
        # that's most probably not inside a function definition.
        candidates = []
        for i in range(len(lines)-1,-1,-1):
            match = pat.match(lines[i])
            if match:
                # if it's at toplevel, it's already the best one
                if lines[i][0] == 'c':
                    return lines, i
                # else add whitespace to candidate list
                candidates.append((match.group(1), i))
        if candidates:
            # this will sort by whitespace, and by line number,
            # less whitespace first  #XXX: should sort high lnum before low
            candidates.sort()
            return lines, candidates[0][1]
        else:
            raise IOError('could not find class definition')
    raise IOError('could not find code object')


def getblocks(object, lstrip=False, enclosing=False, locate=False):
    """Return a list of source lines and starting line number for an object.
    Interactively-defined objects refer to lines in the interpreter's history.

    If enclosing=True, then also return any enclosing code.
    If lstrip=True, ensure there is no indentation in the first line of code.
    If locate=True, then also return the line number for the block of code.

    DEPRECATED: use 'getsourcelines' instead
    """
    lines, lnum = findsource(object)

    if ismodule(object):
        if lstrip: lines = _outdent(lines)
        return ([lines], [0]) if locate is True else [lines]

    #XXX: 'enclosing' means: closures only? or classes and files?
    indent = indentsize(lines[lnum])
    block = getblock(lines[lnum:]) #XXX: catch any TokenError here?

    if not enclosing or not indent:
        if lstrip: block = _outdent(block)
        return ([block], [lnum]) if locate is True else [block]

    pat1 = r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))'; pat1 = re.compile(pat1)
    pat2 = r'^(\s*@)'; pat2 = re.compile(pat2)
   #pat3 = r'^(\s*class\s)'; pat3 = re.compile(pat3) #XXX: enclosing class?
    #FIXME: bound methods need enclosing class (and then instantiation)
    #       *or* somehow apply a partial using the instance

    skip = 0
    line = 0
    blocks = []; _lnum = []
    target = ''.join(block)
    while line <= lnum: #XXX: repeat lnum? or until line < lnum?
        # see if starts with ('def','lambda') and contains our target block
        if pat1.match(lines[line]):
            if not skip:
                try: code = getblock(lines[line:])
                except TokenError: code = [lines[line]]
            if indentsize(lines[line]) > indent: #XXX: should be >= ?
                line += len(code) - skip
            elif target in ''.join(code):
                blocks.append(code) # save code block as the potential winner
                _lnum.append(line - skip) # save the line number for the match
                line += len(code) - skip
            else:
                line += 1
            skip = 0
        # find skip: the number of consecutive decorators
        elif pat2.match(lines[line]):
            try: code = getblock(lines[line:])
            except TokenError: code = [lines[line]]
            skip = 1
            for _line in code[1:]: # skip lines that are decorators
                if not pat2.match(_line): break
                skip += 1
            line += skip
        # no match: reset skip and go to the next line
        else:
            line +=1
            skip = 0

    if not blocks:
        blocks = [block]
        _lnum = [lnum]
    if lstrip: blocks = [_outdent(block) for block in blocks]
    # return last match
    return (blocks, _lnum) if locate is True else blocks


def getsourcelines(object, lstrip=False, enclosing=False):
    """Return a list of source lines and starting line number for an object.
    Interactively-defined objects refer to lines in the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of the lines
    corresponding to the object and the line number indicates where in the
    original source file the first line of code was found.  An IOError is
    raised if the source code cannot be retrieved, while a TypeError is
    raised for objects where the source code is unavailable (e.g. builtins).

    If lstrip=True, ensure there is no indentation in the first line of code.
    If enclosing=True, then also return any enclosing code."""
    code, n = getblocks(object, lstrip=lstrip, enclosing=enclosing, locate=True)
    return code[-1], n[-1]


#NOTE: broke backward compatibility 4/16/14 (was lstrip=True, force=True)
def getsource(object, alias='', lstrip=False, enclosing=False, \
                                              force=False, builtin=False):
    """Return the text of the source code for an object. The source code for
    interactively-defined objects are extracted from the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a single string.  An
    IOError is raised if the source code cannot be retrieved, while a
    TypeError is raised for objects where the source code is unavailable
    (e.g. builtins).

    If alias is provided, then add a line of code that renames the object.
    If lstrip=True, ensure there is no indentation in the first line of code.
    If enclosing=True, then also return any enclosing code.
    If force=True, catch (TypeError,IOError) and try to use import hooks.
    If builtin=True, force an import for any builtins
    """
    # hascode denotes a callable
    hascode = _hascode(object)
    # is a class instance type (and not in builtins)
    instance = _isinstance(object)

    # get source lines; if fail, try to 'force' an import
    try: # fails for builtins, and other assorted object types
        lines, lnum = getsourcelines(object, enclosing=enclosing)
    except (TypeError, IOError): # failed to get source, resort to import hooks
        if not force: # don't try to get types that findsource can't get
            raise
        if not getmodule(object): # get things like 'None' and '1'
            if not instance: return getimport(object, alias, builtin=builtin)
            # special handling (numpy arrays, ...)
            _import = getimport(object, builtin=builtin)
            name = getname(object, force=True)
            _alias = "%s = " % alias if alias else ""
            if alias == name: _alias = ""
            return _import+_alias+"%s\n" % name
        else: #FIXME: could use a good bit of cleanup, since using getimport...
            if not instance: return getimport(object, alias, builtin=builtin)
            # now we are dealing with an instance...
            name = object.__class__.__name__
            module = object.__module__
            if module in ['builtins','__builtin__']:
                return getimport(object, alias, builtin=builtin)
            else: #FIXME: leverage getimport? use 'from module import name'?
                lines, lnum = ["%s = __import__('%s', fromlist=['%s']).%s\n" % (name,module,name,name)], 0
                obj = eval(lines[0].lstrip(name + ' = '))
                lines, lnum = getsourcelines(obj, enclosing=enclosing)

    # strip leading indent (helps ensure can be imported)
    if lstrip or alias:
        lines = _outdent(lines)

    # instantiate, if there's a nice repr  #XXX: BAD IDEA???
    if instance: #and force: #XXX: move into findsource or getsourcelines ?
        if '(' in repr(object): lines.append('%r\n' % object)
       #else: #XXX: better to somehow to leverage __reduce__ ?
       #    reconstructor,args = object.__reduce__()
       #    _ = reconstructor(*args)
        else: # fall back to serialization #XXX: bad idea?
            #XXX: better not duplicate work? #XXX: better new/enclose=True?
            lines = dumpsource(object, alias='', new=force, enclose=False)
            lines, lnum = [line+'\n' for line in lines.split('\n')][:-1], 0
       #else: object.__code__ # raise AttributeError

    # add an alias to the source code
    if alias:
        if hascode:
            skip = 0
            for line in lines: # skip lines that are decorators
                if not line.startswith('@'): break
                skip += 1
            #XXX: use regex from findsource / getsourcelines ?
            if lines[skip].lstrip().startswith('def '): # we have a function
                if alias != object.__name__:
                    lines.append('\n%s = %s\n' % (alias, object.__name__))
            elif 'lambda ' in lines[skip]: # we have a lambda
                if alias != lines[skip].split('=')[0].strip():
                    lines[skip] = '%s = %s' % (alias, lines[skip])
            else: # ...try to use the object's name
                if alias != object.__name__:
                    lines.append('\n%s = %s\n' % (alias, object.__name__))
        else: # class or class instance
            if instance:
                if alias != lines[-1].split('=')[0].strip():
                    lines[-1] = ('%s = ' % alias) + lines[-1]
            else:
                name = getname(object, force=True) or object.__name__
                if alias != name:
                    lines.append('\n%s = %s\n' % (alias, name))
    return ''.join(lines)


def _hascode(object):
    '''True if object has an attribute that stores it's __code__'''
    return getattr(object,'__code__',None) or getattr(object,'func_code',None)

def _isinstance(object):
    '''True if object is a class instance type (and is not a builtin)'''
    if _hascode(object) or isclass(object) or ismodule(object):
        return False
    if istraceback(object) or isframe(object) or iscode(object):
        return False
    # special handling (numpy arrays, ...)
    if not getmodule(object) and getmodule(type(object)).__name__ in ['numpy']:
        return True
#   # check if is instance of a builtin
#   if not getmodule(object) and getmodule(type(object)).__name__ in ['__builtin__','builtins']:
#       return False
    _types = ('<class ',"<type 'instance'>")
    if not repr(type(object)).startswith(_types): #FIXME: weak hack
        return False
    if not getmodule(object) or object.__module__ in ['builtins','__builtin__'] or getname(object, force=True) in ['array']:
        return False
    return True # by process of elimination... it's what we want


def _intypes(object):
    '''check if object is in the 'types' module'''
    import types
    # allow user to pass in object or object.__name__
    if type(object) is not type(''):
        object = getname(object, force=True)
    if object == 'ellipsis': object = 'EllipsisType'
    return True if hasattr(types, object) else False


def _isstring(object): #XXX: isstringlike better?
    '''check if object is a string-like type'''
    if PY3: return isinstance(object, (str, bytes))
    return isinstance(object, basestring)


def indent(code, spaces=4):
    '''indent a block of code with whitespace (default is 4 spaces)'''
    indent = indentsize(code) 
    if type(spaces) is int: spaces = ' '*spaces
    # if '\t' is provided, will indent with a tab
    nspaces = indentsize(spaces)
    # blank lines (etc) need to be ignored
    lines = code.split('\n')
##  stq = "'''"; dtq = '"""'
##  in_stq = in_dtq = False
    for i in range(len(lines)):
        #FIXME: works... but shouldn't indent 2nd+ lines of multiline doc
        _indent = indentsize(lines[i])
        if indent > _indent: continue
        lines[i] = spaces+lines[i]
##      #FIXME: may fail when stq and dtq in same line (depends on ordering)
##      nstq, ndtq = lines[i].count(stq), lines[i].count(dtq)
##      if not in_dtq and not in_stq:
##          lines[i] = spaces+lines[i] # we indent
##          # entering a comment block
##          if nstq%2: in_stq = not in_stq
##          if ndtq%2: in_dtq = not in_dtq
##      # leaving a comment block
##      elif in_dtq and ndtq%2: in_dtq = not in_dtq
##      elif in_stq and nstq%2: in_stq = not in_stq
##      else: pass
    if lines[-1].strip() == '': lines[-1] = ''
    return '\n'.join(lines)


def _outdent(lines, spaces=None, all=True):
    '''outdent lines of code, accounting for docs and line continuations'''
    indent = indentsize(lines[0]) 
    if spaces is None or spaces > indent or spaces < 0: spaces = indent
    for i in range(len(lines) if all else 1):
        #FIXME: works... but shouldn't outdent 2nd+ lines of multiline doc
        _indent = indentsize(lines[i])
        if spaces > _indent: _spaces = _indent
        else: _spaces = spaces
        lines[i] = lines[i][_spaces:]
    return lines

def outdent(code, spaces=None, all=True):
    '''outdent a block of code (default is to strip all leading whitespace)'''
    indent = indentsize(code) 
    if spaces is None or spaces > indent or spaces < 0: spaces = indent
    #XXX: will this delete '\n' in some cases?
    if not all: return code[spaces:]
    return '\n'.join(_outdent(code.split('\n'), spaces=spaces, all=all))


#XXX: not sure what the point of _wrap is...
#exec_ = lambda s, *a: eval(compile(s, '<string>', 'exec'), *a)
__globals__ = globals()
__locals__ = locals()
wrap2 = '''
def _wrap(f):
    """ encapsulate a function and it's __import__ """
    def func(*args, **kwds):
        try:
            # _ = eval(getsource(f, force=True)) #XXX: safer but less robust
            exec getimportable(f, alias='_') in %s, %s
        except:
            raise ImportError('cannot import name ' + f.__name__)
        return _(*args, **kwds)
    func.__name__ = f.__name__
    func.__doc__ = f.__doc__
    return func
''' % ('__globals__', '__locals__')
wrap3 = '''
def _wrap(f):
    """ encapsulate a function and it's __import__ """
    def func(*args, **kwds):
        try:
            # _ = eval(getsource(f, force=True)) #XXX: safer but less robust
            exec(getimportable(f, alias='_'), %s, %s)
        except:
            raise ImportError('cannot import name ' + f.__name__)
        return _(*args, **kwds)
    func.__name__ = f.__name__
    func.__doc__ = f.__doc__
    return func
''' % ('__globals__', '__locals__')
if PY3:
    exec(wrap3)
else:
    exec(wrap2)
del wrap2, wrap3


def _enclose(object, alias=''): #FIXME: needs alias to hold returned object
    """create a function enclosure around the source of some object"""
    #XXX: dummy and stub should append a random string
    dummy = '__this_is_a_big_dummy_enclosing_function__'
    stub = '__this_is_a_stub_variable__'
    code = 'def %s():\n' % dummy
    code += indent(getsource(object, alias=stub, lstrip=True, force=True))
    code += indent('return %s\n' % stub)
    if alias: code += '%s = ' % alias
    code += '%s(); del %s\n' % (dummy, dummy)
   #code += "globals().pop('%s',lambda :None)()\n" % dummy
    return code


def dumpsource(object, alias='', new=False, enclose=True):
    """'dump to source', where the code includes a pickled object.

    If new=True and object is a class instance, then create a new
    instance using the unpacked class source code. If enclose, then
    create the object inside a function enclosure (thus minimizing
    any global namespace pollution).
    """
    from dill import dumps
    pik = repr(dumps(object))
    code = 'import dill\n'
    if enclose:
        stub = '__this_is_a_stub_variable__' #XXX: *must* be same _enclose.stub
        pre = '%s = ' % stub
        new = False #FIXME: new=True doesn't work with enclose=True
    else:
        stub = alias
        pre = '%s = ' % stub if alias else alias
    
    # if a 'new' instance is not needed, then just dump and load
    if not new or not _isinstance(object):
        code += pre + 'dill.loads(%s)\n' % pik
    else: #XXX: other cases where source code is needed???
        code += getsource(object.__class__, alias='', lstrip=True, force=True)
        mod = repr(object.__module__) # should have a module (no builtins here)
        if PY3:
            code += pre + 'dill.loads(%s.replace(b%s,bytes(__name__,"UTF-8")))\n' % (pik,mod)
        else:
            code += pre + 'dill.loads(%s.replace(%s,__name__))\n' % (pik,mod)
       #code += 'del %s' % object.__class__.__name__ #NOTE: kills any existing!

    if enclose:
        # generation of the 'enclosure'
        dummy = '__this_is_a_big_dummy_object__'
        dummy = _enclose(dummy, alias=alias)
        # hack to replace the 'dummy' with the 'real' code
        dummy = dummy.split('\n')
        code = dummy[0]+'\n' + indent(code) + '\n'.join(dummy[-3:])

    return code #XXX: better 'dumpsourcelines', returning list of lines?


def getname(obj, force=False, fqn=False): #XXX: throw(?) to raise error on fail?
    """get the name of the object. for lambdas, get the name of the pointer """
    if fqn: return '.'.join(_namespace(obj))
    module = getmodule(obj)
    if not module: # things like "None" and "1"
        if not force: return None
        return repr(obj)
    try:
        #XXX: 'wrong' for decorators and curried functions ?
        #       if obj.func_closure: ...use logic from getimportable, etc ?
        name = obj.__name__
        if name == '<lambda>':
            return getsource(obj).split('=',1)[0].strip()
        # handle some special cases
        if module.__name__ in ['builtins','__builtin__']:
            if name == 'ellipsis': name = 'EllipsisType'
        return name
    except AttributeError: #XXX: better to just throw AttributeError ?
        if not force: return None
        name = repr(obj)
        if name.startswith('<'): # or name.split('('):
            return None
        return name


def _namespace(obj):
    """_namespace(obj); return namespace hierarchy (as a list of names)
    for the given object.  For an instance, find the class hierarchy.

    For example:

    >>> from functools import partial
    >>> p = partial(int, base=2)
    >>> _namespace(p)
    [\'functools\', \'partial\']
    """
    # mostly for functions and modules and such
    #FIXME: 'wrong' for decorators and curried functions
    try: #XXX: needs some work and testing on different types
        module = qual = str(getmodule(obj)).split()[1].strip('"').strip("'")
        qual = qual.split('.')
        if ismodule(obj):
            return qual
        # get name of a lambda, function, etc
        name = getname(obj) or obj.__name__ # failing, raise AttributeError
        # check special cases (NoneType, ...)
        if module in ['builtins','__builtin__']: # BuiltinFunctionType
            if _intypes(name): return ['types'] + [name]
        return qual + [name] #XXX: can be wrong for some aliased objects
    except: pass
    # special case: numpy.inf and numpy.nan (we don't want them as floats)
    if str(obj) in ['inf','nan','Inf','NaN']: # is more, but are they needed?
        return ['numpy'] + [str(obj)]
    # mostly for classes and class instances and such
    module = getattr(obj.__class__, '__module__', None)
    qual = str(obj.__class__)
    try: qual = qual[qual.index("'")+1:-2]
    except ValueError: pass # str(obj.__class__) made the 'try' unnecessary
    qual = qual.split(".")
    if module in ['builtins','__builtin__']:
        # check special cases (NoneType, Ellipsis, ...)
        if qual[-1] == 'ellipsis': qual[-1] = 'EllipsisType'
        if _intypes(qual[-1]): module = 'types' #XXX: BuiltinFunctionType
        qual = [module] + qual
    return qual


#NOTE: 05/25/14 broke backward compatability: added 'alias' as 3rd argument
def _getimport(head, tail, alias='', verify=True, builtin=False):
    """helper to build a likely import string from head and tail of namespace.
    ('head','tail') are used in the following context: "from head import tail"

    If verify=True, then test the import string before returning it.
    If builtin=True, then force an import for builtins where possible.
    If alias is provided, then rename the object on import.
    """
    # special handling for a few common types
    if tail in ['Ellipsis', 'NotImplemented'] and head in ['types']:
        head = len.__module__
    elif tail in ['None'] and head in ['types']:
        _alias = '%s = ' % alias if alias else ''
        if alias == tail: _alias = ''
        return _alias+'%s\n' % tail
    # we don't need to import from builtins, so return ''
#   elif tail in ['NoneType','int','float','long','complex']: return '' #XXX: ?
    if head in ['builtins','__builtin__']:
        # special cases (NoneType, Ellipsis, ...) #XXX: BuiltinFunctionType
        if tail == 'ellipsis': tail = 'EllipsisType'
        if _intypes(tail): head = 'types'
        elif not builtin:
            _alias = '%s = ' % alias if alias else ''
            if alias == tail: _alias = ''
            return _alias+'%s\n' % tail
        else: pass # handle builtins below
    # get likely import string
    if not head: _str = "import %s" % tail
    else: _str = "from %s import %s" % (head, tail)
    _alias = " as %s\n" % alias if alias else "\n"
    if alias == tail: _alias = "\n"
    _str += _alias
    # FIXME: fails on most decorators, currying, and such...
    #        (could look for magic __wrapped__ or __func__ attr)
    #        (could fix in 'namespace' to check obj for closure)
    if verify and not head.startswith('dill.'):# weird behavior for dill
       #print(_str)
        try: exec(_str) #XXX: check if == obj? (name collision)
        except ImportError: #XXX: better top-down or bottom-up recursion?
            _head = head.rsplit(".",1)[0] #(or get all, then compare == obj?)
            if not _head: raise
            if _head != head:
                _str = _getimport(_head, tail, alias, verify)
    return _str


#XXX: rename builtin to force? vice versa? verify to force? (as in getsource)
#NOTE: 05/25/14 broke backward compatability: added 'alias' as 2nd argument
def getimport(obj, alias='', verify=True, builtin=False, enclosing=False):
    """get the likely import string for the given object

    obj is the object to inspect
    If verify=True, then test the import string before returning it.
    If builtin=True, then force an import for builtins where possible.
    If enclosing=True, get the import for the outermost enclosing callable.
    If alias is provided, then rename the object on import.
    """
    if enclosing:
        from .detect import outermost
        _obj = outermost(obj)
        obj = _obj if _obj else obj
    # get the namespace
    qual = _namespace(obj)
    head = '.'.join(qual[:-1])
    tail = qual[-1]
    # for named things... with a nice repr #XXX: move into _namespace?
    try: # look for '<...>' and be mindful it might be in lists, dicts, etc...
        name = repr(obj).split('<',1)[1].split('>',1)[1]
        name = None # we have a 'object'-style repr
    except: # it's probably something 'importable'
        if head in ['builtins','__builtin__']:
            name = repr(obj) #XXX: catch [1,2], (1,2), set([1,2])... others?
        else:
            name = repr(obj).split('(')[0]
   #if not repr(obj).startswith('<'): name = repr(obj).split('(')[0]
   #else: name = None
    if name: # try using name instead of tail
        try: return _getimport(head, name, alias, verify, builtin)
        except ImportError: pass
        except SyntaxError:
            if head in ['builtins','__builtin__']:
                _alias = '%s = ' % alias if alias else ''
                if alias == name: _alias = ''
                return _alias+'%s\n' % name
            else: pass
    try:
       #if type(obj) is type(abs): _builtin = builtin # BuiltinFunctionType
       #else: _builtin = False
        return _getimport(head, tail, alias, verify, builtin)
    except ImportError:
        raise # could do some checking against obj
    except SyntaxError:
        if head in ['builtins','__builtin__']:
            _alias = '%s = ' % alias if alias else ''
            if alias == tail: _alias = ''
            return _alias+'%s\n' % tail
        raise # could do some checking against obj


def _importable(obj, alias='', source=None, enclosing=False, force=True, \
                                              builtin=True, lstrip=True):
    """get an import string (or the source code) for the given object

    This function will attempt to discover the name of the object, or the repr
    of the object, or the source code for the object. To attempt to force
    discovery of the source code, use source=True, to attempt to force the
    use of an import, use source=False; otherwise an import will be sought
    for objects not defined in __main__. The intent is to build a string
    that can be imported from a python file. obj is the object to inspect.
    If alias is provided, then rename the object with the given alias.

    If source=True, use these options:
      If enclosing=True, then also return any enclosing code.
      If force=True, catch (TypeError,IOError) and try to use import hooks.
      If lstrip=True, ensure there is no indentation in the first line of code.

    If source=False, use these options:
      If enclosing=True, get the import for the outermost enclosing callable.
      If force=True, then don't test the import string before returning it.
      If builtin=True, then force an import for builtins where possible.
    """
    if source is None:
        source = True if isfrommain(obj) else False
    if source: # first try to get the source
        try:
            return getsource(obj, alias, enclosing=enclosing, \
                             force=force, lstrip=lstrip, builtin=builtin)
        except: pass
    try:
        if not _isinstance(obj):
            return getimport(obj, alias, enclosing=enclosing, \
                                  verify=(not force), builtin=builtin)
        # first 'get the import', then 'get the instance'
        _import = getimport(obj, enclosing=enclosing, \
                                 verify=(not force), builtin=builtin)
        name = getname(obj, force=True)
        if not name:
            raise AttributeError("object has no atribute '__name__'")
        _alias = "%s = " % alias if alias else ""
        if alias == name: _alias = ""
        return _import+_alias+"%s\n" % name

    except: pass
    if not source: # try getsource, only if it hasn't been tried yet
        try:
            return getsource(obj, alias, enclosing=enclosing, \
                             force=force, lstrip=lstrip, builtin=builtin)
        except: pass
    # get the name (of functions, lambdas, and classes)
    # or hope that obj can be built from the __repr__
    #XXX: what to do about class instances and such?
    obj = getname(obj, force=force)
    # we either have __repr__ or __name__ (or None)
    if not obj or obj.startswith('<'):
        raise AttributeError("object has no atribute '__name__'")
    _alias = '%s = ' % alias if alias else ''
    if alias == obj: _alias = ''
    return _alias+'%s\n' % obj
    #XXX: possible failsafe... (for example, for instances when source=False)
    #     "import dill; result = dill.loads(<pickled_object>); # repr(<object>)"

def _closuredimport(func, alias='', builtin=False):
    """get import for closured objects; return a dict of 'name' and 'import'"""
    import re
    from .detect import freevars, outermost
    free_vars = freevars(func)
    func_vars = {}
    # split into 'funcs' and 'non-funcs'
    for name,obj in list(free_vars.items()):
        if not isfunction(obj): continue
        # get import for 'funcs'
        fobj = free_vars.pop(name)
        src = getsource(fobj)
        if src.lstrip().startswith('@'): # we have a decorator
            src = getimport(fobj, alias=alias, builtin=builtin)
        else: # we have to "hack" a bit... and maybe be lucky
            encl = outermost(func)
            # pattern: 'func = enclosing(fobj'
            pat = r'.*[\w\s]=\s*'+getname(encl)+r'\('+getname(fobj)
            mod = getname(getmodule(encl))
            #HACK: get file containing 'outer' function; is func there?
            lines,_ = findsource(encl)
            candidate = [line for line in lines if getname(encl) in line and \
                         re.match(pat, line)]
            if not candidate:
                mod = getname(getmodule(fobj))
                #HACK: get file containing 'inner' function; is func there? 
                lines,_ = findsource(fobj)
                candidate = [line for line in lines \
                             if getname(fobj) in line and re.match(pat, line)]
            if not len(candidate): raise TypeError('import could not be found')
            candidate = candidate[-1]
            name = candidate.split('=',1)[0].split()[-1].strip()
            src = _getimport(mod, name, alias=alias, builtin=builtin)
        func_vars[name] = src
    if not func_vars:
        name = outermost(func)
        mod = getname(getmodule(name))
        if not mod or name is func: # then it can be handled by getimport
            name = getname(func, force=True) #XXX: better key?
            src = getimport(func, alias=alias, builtin=builtin)
        else:
            lines,_ = findsource(name)
            # pattern: 'func = enclosing('
            candidate = [line for line in lines if getname(name) in line and \
                         re.match(r'.*[\w\s]=\s*'+getname(name)+r'\(', line)]
            if not len(candidate): raise TypeError('import could not be found')
            candidate = candidate[-1]
            name = candidate.split('=',1)[0].split()[-1].strip()
            src = _getimport(mod, name, alias=alias, builtin=builtin)
        func_vars[name] = src
    return func_vars

#XXX: should be able to use __qualname__
def _closuredsource(func, alias=''):
    """get source code for closured objects; return a dict of 'name'
    and 'code blocks'"""
    #FIXME: this entire function is a messy messy HACK
    #      - pollutes global namespace
    #      - fails if name of freevars are reused
    #      - can unnecessarily duplicate function code
    from .detect import freevars
    free_vars = freevars(func)
    func_vars = {}
    # split into 'funcs' and 'non-funcs'
    for name,obj in list(free_vars.items()):
        if not isfunction(obj):
            # get source for 'non-funcs'
            free_vars[name] = getsource(obj, force=True, alias=name)
            continue
        # get source for 'funcs'
        fobj = free_vars.pop(name)
        src = getsource(fobj, alias) # DO NOT include dependencies
        # if source doesn't start with '@', use name as the alias
        if not src.lstrip().startswith('@'): #FIXME: 'enclose' in dummy;
            src = importable(fobj,alias=name)#        wrong ref 'name'
            org = getsource(func, alias, enclosing=False, lstrip=True)
            src = (src, org) # undecorated first, then target
        else: #NOTE: reproduces the code!
            org = getsource(func, enclosing=True, lstrip=False)
            src = importable(fobj, alias, source=True) # include dependencies
            src = (org, src) # target first, then decorated
        func_vars[name] = src
    src = ''.join(free_vars.values())
    if not func_vars: #FIXME: 'enclose' in dummy; wrong ref 'name'
        org = getsource(func, alias, force=True, enclosing=False, lstrip=True)
        src = (src, org) # variables first, then target
    else:
        src = (src, None) # just variables        (better '' instead of None?)
    func_vars[None] = src
    # FIXME: remove duplicates (however, order is important...)
    return func_vars

def importable(obj, alias='', source=None, builtin=True):
    """get an importable string (i.e. source code or the import string)
    for the given object, including any required objects from the enclosing
    and global scope

    This function will attempt to discover the name of the object, or the repr
    of the object, or the source code for the object. To attempt to force
    discovery of the source code, use source=True, to attempt to force the
    use of an import, use source=False; otherwise an import will be sought
    for objects not defined in __main__. The intent is to build a string
    that can be imported from a python file.

    obj is the object to inspect. If alias is provided, then rename the
    object with the given alias. If builtin=True, then force an import for
    builtins where possible.
    """
    #NOTE: we always 'force', and 'lstrip' as necessary
    #NOTE: for 'enclosing', use importable(outermost(obj))
    if source is None:
        source = True if isfrommain(obj) else False
    elif builtin and isbuiltin(obj):
        source = False
    tried_source = tried_import = False
    while True:
        if not source: # we want an import
            try:
                if _isinstance(obj): # for instances, punt to _importable
                    return _importable(obj, alias, source=False, builtin=builtin)
                src = _closuredimport(obj, alias=alias, builtin=builtin)
                if len(src) == 0:
                    raise NotImplementedError('not implemented')
                if len(src) > 1:
                    raise NotImplementedError('not implemented')
                return list(src.values())[0]
            except:
                if tried_source: raise
                tried_import = True
        # we want the source
        try:
            src = _closuredsource(obj, alias=alias)
            if len(src) == 0:
                raise NotImplementedError('not implemented')
            # groan... an inline code stitcher
            def _code_stitcher(block):
                "stitch together the strings in tuple 'block'"
                if block[0] and block[-1]: block = '\n'.join(block)
                elif block[0]: block = block[0]
                elif block[-1]: block = block[-1]
                else: block = ''
                return block
            # get free_vars first
            _src = _code_stitcher(src.pop(None))
            _src = [_src] if _src else []
            # get func_vars
            for xxx in src.values():
                xxx = _code_stitcher(xxx)
                if xxx: _src.append(xxx)
            # make a single source string
            if not len(_src):
                src = ''
            elif len(_src) == 1:
                src = _src[0]
            else:
                src = '\n'.join(_src)
            # get source code of objects referred to by obj in global scope
            from .detect import globalvars
            obj = globalvars(obj) #XXX: don't worry about alias? recurse? etc?
            obj = list(getsource(_obj,name,force=True) for (name,_obj) in obj.items() if not isbuiltin(_obj))
            obj = '\n'.join(obj) if obj else ''
            # combine all referred-to source (global then enclosing)
            if not obj: return src
            if not src: return obj
            return obj + src
        except:
            if tried_import: raise
            tried_source = True
            source = not source
    # should never get here
    return


# backward compatability
def getimportable(obj, alias='', byname=True, explicit=False):
    return importable(obj,alias,source=(not byname),builtin=explicit)
   #return outdent(_importable(obj,alias,source=(not byname),builtin=explicit))
def likely_import(obj, passive=False, explicit=False):
    return getimport(obj, verify=(not passive), builtin=explicit)
def _likely_import(first, last, passive=False, explicit=True):
    return _getimport(first, last, verify=(not passive), builtin=explicit)
_get_name = getname
getblocks_from_history = getblocks



# EOF
