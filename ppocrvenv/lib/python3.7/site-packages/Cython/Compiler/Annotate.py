# Note: Work in progress

from __future__ import absolute_import

import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO  # does not support writing 'str' in Py2

from . import Version
from .Code import CCodeWriter
from .. import Utils


class AnnotationCCodeWriter(CCodeWriter):

    def __init__(self, create_from=None, buffer=None, copy_formatting=True):
        CCodeWriter.__init__(self, create_from, buffer, copy_formatting=copy_formatting)
        if create_from is None:
            self.annotation_buffer = StringIO()
            self.last_annotated_pos = None
            # annotations[filename][line] -> [(column, AnnotationItem)*]
            self.annotations = defaultdict(partial(defaultdict, list))
            # code[filename][line] -> str
            self.code = defaultdict(partial(defaultdict, str))
            # scopes[filename][line] -> set(scopes)
            self.scopes = defaultdict(partial(defaultdict, set))
        else:
            # When creating an insertion point, keep references to the same database
            self.annotation_buffer = create_from.annotation_buffer
            self.annotations = create_from.annotations
            self.code = create_from.code
            self.scopes = create_from.scopes
            self.last_annotated_pos = create_from.last_annotated_pos

    def create_new(self, create_from, buffer, copy_formatting):
        return AnnotationCCodeWriter(create_from, buffer, copy_formatting)

    def write(self, s):
        CCodeWriter.write(self, s)
        self.annotation_buffer.write(s)

    def mark_pos(self, pos, trace=True):
        if pos is not None:
            CCodeWriter.mark_pos(self, pos, trace)
            if self.funcstate and self.funcstate.scope:
                # lambdas and genexprs can result in multiple scopes per line => keep them in a set
                self.scopes[pos[0].filename][pos[1]].add(self.funcstate.scope)
        if self.last_annotated_pos:
            source_desc, line, _ = self.last_annotated_pos
            pos_code = self.code[source_desc.filename]
            pos_code[line] += self.annotation_buffer.getvalue()
        self.annotation_buffer = StringIO()
        self.last_annotated_pos = pos

    def annotate(self, pos, item):
        self.annotations[pos[0].filename][pos[1]].append((pos[2], item))

    def _css(self):
        """css template will later allow to choose a colormap"""
        css = [self._css_template]
        for i in range(255):
            color = u"FFFF%02x" % int(255/(1+i/10.0))
            css.append('.cython.score-%d {background-color: #%s;}' % (i, color))
        try:
            from pygments.formatters import HtmlFormatter
        except ImportError:
            pass
        else:
            css.append(HtmlFormatter().get_style_defs('.cython'))
        return '\n'.join(css)

    _css_template = textwrap.dedent("""
        body.cython { font-family: courier; font-size: 12; }

        .cython.tag  {  }
        .cython.line { margin: 0em }
        .cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

        .cython.line .run { background-color: #B0FFB0; }
        .cython.line .mis { background-color: #FFB0B0; }
        .cython.code.run  { border-left: 8px solid #B0FFB0; }
        .cython.code.mis  { border-left: 8px solid #FFB0B0; }

        .cython.code .py_c_api  { color: red; }
        .cython.code .py_macro_api  { color: #FF7000; }
        .cython.code .pyx_c_api  { color: #FF3000; }
        .cython.code .pyx_macro_api  { color: #FF7000; }
        .cython.code .refnanny  { color: #FFA000; }
        .cython.code .trace  { color: #FFA000; }
        .cython.code .error_goto  { color: #FFA000; }

        .cython.code .coerce  { color: #008000; border: 1px dotted #008000 }
        .cython.code .py_attr { color: #FF0000; font-weight: bold; }
        .cython.code .c_attr  { color: #0000FF; }
        .cython.code .py_call { color: #FF0000; font-weight: bold; }
        .cython.code .c_call  { color: #0000FF; }
    """)

    # on-click toggle function to show/hide C source code
    _onclick_attr = ' onclick="{0}"'.format((
        "(function(s){"
        "    s.display =  s.display === 'block' ? 'none' : 'block'"
        "})(this.nextElementSibling.style)"
        ).replace(' ', '')  # poor dev's JS minification
    )

    def save_annotation(self, source_filename, target_filename, coverage_xml=None):
        with Utils.open_source_file(source_filename) as f:
            code = f.read()
        generated_code = self.code.get(source_filename, {})
        c_file = Utils.decode_filename(os.path.basename(target_filename))
        html_filename = os.path.splitext(target_filename)[0] + ".html"

        with codecs.open(html_filename, "w", encoding="UTF-8") as out_buffer:
            out_buffer.write(self._save_annotation(code, generated_code, c_file, source_filename, coverage_xml))

    def _save_annotation_header(self, c_file, source_filename, coverage_timestamp=None):
        coverage_info = ''
        if coverage_timestamp:
            coverage_info = u' with coverage data from {timestamp}'.format(
                timestamp=datetime.fromtimestamp(int(coverage_timestamp) // 1000))

        outlist = [
            textwrap.dedent(u'''\
            <!DOCTYPE html>
            <!-- Generated by Cython {watermark} -->
            <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
                <title>Cython: {filename}</title>
                <style type="text/css">
                {css}
                </style>
            </head>
            <body class="cython">
            <p><span style="border-bottom: solid 1px grey;">Generated by Cython {watermark}</span>{more_info}</p>
            <p>
                <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
                Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
            </p>
            ''').format(css=self._css(), watermark=Version.watermark,
                        filename=os.path.basename(source_filename) if source_filename else '',
                        more_info=coverage_info)
        ]
        if c_file:
            outlist.append(u'<p>Raw output: <a href="%s">%s</a></p>\n' % (c_file, c_file))
        return outlist

    def _save_annotation_footer(self):
        return (u'</body></html>\n',)

    def _save_annotation(self, code, generated_code, c_file=None, source_filename=None, coverage_xml=None):
        """
        lines : original cython source code split by lines
        generated_code : generated c code keyed by line number in original file
        target filename : name of the file in which to store the generated html
        c_file : filename in which the c_code has been written
        """
        if coverage_xml is not None and source_filename:
            coverage_timestamp = coverage_xml.get('timestamp', '').strip()
            covered_lines = self._get_line_coverage(coverage_xml, source_filename)
        else:
            coverage_timestamp = covered_lines = None
        annotation_items = dict(self.annotations[source_filename])
        scopes = dict(self.scopes[source_filename])

        outlist = []
        outlist.extend(self._save_annotation_header(c_file, source_filename, coverage_timestamp))
        outlist.extend(self._save_annotation_body(code, generated_code, annotation_items, scopes, covered_lines))
        outlist.extend(self._save_annotation_footer())
        return ''.join(outlist)

    def _get_line_coverage(self, coverage_xml, source_filename):
        coverage_data = None
        for entry in coverage_xml.iterfind('.//class'):
            if not entry.get('filename'):
                continue
            if (entry.get('filename') == source_filename or
                    os.path.abspath(entry.get('filename')) == source_filename):
                coverage_data = entry
                break
            elif source_filename.endswith(entry.get('filename')):
                coverage_data = entry  # but we might still find a better match...
        if coverage_data is None:
            return None
        return dict(
            (int(line.get('number')), int(line.get('hits')))
            for line in coverage_data.iterfind('lines/line')
        )

    def _htmlify_code(self, code):
        try:
            from pygments import highlight
            from pygments.lexers import CythonLexer
            from pygments.formatters import HtmlFormatter
        except ImportError:
            # no Pygments, just escape the code
            return html_escape(code)

        html_code = highlight(
            code, CythonLexer(stripnl=False, stripall=False),
            HtmlFormatter(nowrap=True))
        return html_code

    def _save_annotation_body(self, cython_code, generated_code, annotation_items, scopes, covered_lines=None):
        outlist = [u'<div class="cython">']
        pos_comment_marker = u'/* \N{HORIZONTAL ELLIPSIS} */\n'
        new_calls_map = dict(
            (name, 0) for name in
            'refnanny trace py_macro_api py_c_api pyx_macro_api pyx_c_api error_goto'.split()
        ).copy

        self.mark_pos(None)

        def annotate(match):
            group_name = match.lastgroup
            calls[group_name] += 1
            return u"<span class='%s'>%s</span>" % (
                group_name, match.group(group_name))

        lines = self._htmlify_code(cython_code).splitlines()
        lineno_width = len(str(len(lines)))
        if not covered_lines:
            covered_lines = None

        for k, line in enumerate(lines, 1):
            try:
                c_code = generated_code[k]
            except KeyError:
                c_code = ''
            else:
                c_code = _replace_pos_comment(pos_comment_marker, c_code)
                if c_code.startswith(pos_comment_marker):
                    c_code = c_code[len(pos_comment_marker):]
                c_code = html_escape(c_code)

            calls = new_calls_map()
            c_code = _parse_code(annotate, c_code)
            score = (5 * calls['py_c_api'] + 2 * calls['pyx_c_api'] +
                     calls['py_macro_api'] + calls['pyx_macro_api'])

            if c_code:
                onclick = self._onclick_attr
                expandsymbol = '+'
            else:
                onclick = ''
                expandsymbol = '&#xA0;'

            covered = ''
            if covered_lines is not None and k in covered_lines:
                hits = covered_lines[k]
                if hits is not None:
                    covered = 'run' if hits else 'mis'

            outlist.append(
                u'<pre class="cython line score-{score}"{onclick}>'
                # generate line number with expand symbol in front,
                # and the right  number of digit
                u'{expandsymbol}<span class="{covered}">{line:0{lineno_width}d}</span>: {code}</pre>\n'.format(
                    score=score,
                    expandsymbol=expandsymbol,
                    covered=covered,
                    lineno_width=lineno_width,
                    line=k,
                    code=line.rstrip(),
                    onclick=onclick,
                ))
            if c_code:
                outlist.append(u"<pre class='cython code score-{score} {covered}'>{code}</pre>".format(
                    score=score, covered=covered, code=c_code))
        outlist.append(u"</div>")
        return outlist


_parse_code = re.compile((
    br'(?P<refnanny>__Pyx_X?(?:GOT|GIVE)REF|__Pyx_RefNanny[A-Za-z]+)|'
    br'(?P<trace>__Pyx_Trace[A-Za-z]+)|'
    br'(?:'
    br'(?P<pyx_macro_api>__Pyx_[A-Z][A-Z_]+)|'
    br'(?P<pyx_c_api>(?:__Pyx_[A-Z][a-z_][A-Za-z_]*)|__pyx_convert_[A-Za-z_]*)|'
    br'(?P<py_macro_api>Py[A-Z][a-z]+_[A-Z][A-Z_]+)|'
    br'(?P<py_c_api>Py[A-Z][a-z]+_[A-Z][a-z][A-Za-z_]*)'
    br')(?=\()|'       # look-ahead to exclude subsequent '(' from replacement
    br'(?P<error_goto>(?:(?<=;) *if [^;]* +)?__PYX_ERR\([^)]+\))'
).decode('ascii')).sub


_replace_pos_comment = re.compile(
    # this matches what Cython generates as code line marker comment
    br'^\s*/\*(?:(?:[^*]|\*[^/])*\n)+\s*\*/\s*\n'.decode('ascii'),
    re.M
).sub


class AnnotationItem(object):

    def __init__(self, style, text, tag="", size=0):
        self.style = style
        self.text = text
        self.tag = tag
        self.size = size

    def start(self):
        return u"<span class='cython tag %s' title='%s'>%s" % (self.style, self.text, self.tag)

    def end(self):
        return self.size, u"</span>"
