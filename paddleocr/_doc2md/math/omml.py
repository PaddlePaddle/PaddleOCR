# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Office Math Markup Language (OMML) to LaTeX converter.

Adapted from https://github.com/xiilei/dwml/blob/master/dwml/omml.py
via MinerU 3.0.0 (MIT License)
"""

import logging
import re

import lxml.etree as ET

from .latex_dict import (
    ALN,
    ARR,
    BACKSLASH,
    BLANK,
    BRK,
    CHARS,
    CHR,
    CHR_BO,
    CHR_DEFAULT,
    D_DEFAULT,
    F_DEFAULT,
    FUNC,
    FUNC_PLACE,
    LIM_FUNC,
    LIM_TO,
    LIM_UPP,
    POS,
    POS_DEFAULT,
    RAD,
    RAD_DEFAULT,
    SUB,
    SUP,
    D,
    F,
    M,
    T,
)

logger = logging.getLogger(__name__)

OMML_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

# Mapping from OMML <m:scr> values to LaTeX math font commands.
SCR_TO_LATEX = {
    "script": "\\mathscr{{{0}}}",
    "fraktur": "\\mathfrak{{{0}}}",
    "double-struck": "\\mathbb{{{0}}}",
    "sans-serif": "\\mathsf{{{0}}}",
    "monospace": "\\mathtt{{{0}}}",
}


def escape_latex(strs):
    last = None
    new_chr = []
    strs = strs.replace(r"\\", "\\")
    for c in strs:
        if (c in CHARS) and (last != BACKSLASH):
            new_chr.append(BACKSLASH + c)
        else:
            new_chr.append(c)
        last = c
    return BLANK.join(new_chr)


def get_val(key, default=None, store=CHR):
    if key is not None:
        return key if not store else store.get(key, key)
    else:
        return default


class Tag2Method:
    def call_method(self, elm, stag=None):
        getmethod = self.tag2meth.get
        if stag is None:
            stag = elm.tag.replace(OMML_NS, "")
        method = getmethod(stag)
        if method:
            return method(self, elm)
        else:
            return None

    def process_children_list(self, elm, include=None):
        """
        process children of the elm,return iterable
        """
        for _e in list(elm):
            if OMML_NS not in _e.tag:
                continue
            stag = _e.tag.replace(OMML_NS, "")
            if include and (stag not in include):
                continue
            t = self.call_method(_e, stag=stag)
            if t is None:
                t = self.process_unknow(_e, stag)
                if t is None:
                    continue
            yield (stag, t, _e)

    def process_children_dict(self, elm, include=None):
        """
        process children of the elm,return dict
        """
        latex_chars = dict()
        for stag, t, e in self.process_children_list(elm, include):
            latex_chars[stag] = t
        return latex_chars

    def process_children(self, elm, include=None):
        """
        process children of the elm,return string
        """
        return BLANK.join(
            (
                t if not isinstance(t, Tag2Method) else str(t)
                for stag, t, e in self.process_children_list(elm, include)
            )
        )

    def process_unknow(self, elm, stag):
        return None


class Pr(Tag2Method):
    text = ""

    __val_tags = ("chr", "pos", "begChr", "endChr", "type")

    __innerdict = None  # can't use the __dict__

    """ common properties of element"""

    def __init__(self, elm):
        self.__innerdict = {}
        self.text = self.process_children(elm)

    def __str__(self):
        return self.text

    def __getattr__(self, name):
        return self.__innerdict.get(name, None)

    def do_brk(self, elm):
        self.__innerdict["brk"] = BRK
        return BRK

    def do_common(self, elm):
        stag = elm.tag.replace(OMML_NS, "")
        if stag in self.__val_tags:
            t = elm.get(f"{OMML_NS}val")
            self.__innerdict[stag] = t
        return None

    tag2meth = {
        "brk": do_brk,
        "chr": do_common,
        "pos": do_common,
        "begChr": do_common,
        "endChr": do_common,
        "type": do_common,
    }


class oMath2Latex(Tag2Method):
    """
    Convert oMath element of omml to latex
    """

    _t_dict = T

    __direct_tags = ("box", "sSub", "sSup", "sSubSup", "num", "den", "deg", "e")

    _encoder = None  # cached UnicodeToLatexEncoder instance (class-level)

    def __init__(self, element):
        if oMath2Latex._encoder is None:
            # Delayed import: pylatexenc is an optional dependency
            try:
                from pylatexenc.latexencode import UnicodeToLatexEncoder
            except ImportError:
                raise RuntimeError(
                    "pylatexenc is required for math formula conversion. "
                    "Install it with: pip install pylatexenc"
                )
            oMath2Latex._encoder = UnicodeToLatexEncoder(
                replacement_latex_protection="braces-all",
                unknown_char_policy="keep",
                unknown_char_warning=False,
            )
        self.u = oMath2Latex._encoder
        self._latex = self.process_children(element)

    def __str__(self):
        return self.latex.replace("  ", " ")

    def process_unknow(self, elm, stag):
        if stag in self.__direct_tags:
            return self.process_children(elm)
        elif stag[-2:] == "Pr":
            return Pr(elm)
        else:
            return None

    @property
    def latex(self):
        return self._latex

    def do_acc(self, elm):
        """
        the accent function
        """
        c_dict = self.process_children_dict(elm)
        latex_s = get_val(
            c_dict["accPr"].chr, default=CHR_DEFAULT.get("ACC_VAL"), store=CHR
        )
        return latex_s.format(c_dict["e"])

    def do_bar(self, elm):
        """
        the bar function
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict["barPr"]
        latex_s = get_val(pr.pos, default=POS_DEFAULT.get("BAR_VAL"), store=POS)
        return pr.text + latex_s.format(c_dict["e"])

    def do_d(self, elm):
        """
        the delimiter object
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict["dPr"]
        null = D_DEFAULT.get("null")

        s_val = get_val(pr.begChr, default=D_DEFAULT.get("left"), store=T)
        e_val = get_val(pr.endChr, default=D_DEFAULT.get("right"), store=T)
        delim = pr.text + D.format(
            left=null if not s_val else escape_latex(s_val),
            text=c_dict["e"],
            right=null if not e_val else escape_latex(e_val),
        )
        return delim

    def do_sub(self, elm):
        text = self.process_children(elm)
        return SUB.format(text)

    def do_sup(self, elm):
        text = self.process_children(elm)
        return SUP.format(text)

    def do_f(self, elm):
        """
        the fraction object
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict.get("fPr")
        if pr is None:
            logger.debug("Missing fPr element in fraction, using default formatting")
            latex_s = F_DEFAULT
            return latex_s.format(
                num=c_dict.get("num"),
                den=c_dict.get("den"),
            )
        latex_s = get_val(pr.type, default=F_DEFAULT, store=F)
        return pr.text + latex_s.format(num=c_dict.get("num"), den=c_dict.get("den"))

    def do_func(self, elm):
        """
        the Function-Apply object (Examples:sin cos)
        """
        c_dict = self.process_children_dict(elm)
        func_name = c_dict.get("fName")
        return func_name.replace(FUNC_PLACE, c_dict.get("e"))

    def do_fname(self, elm):
        """
        the func name
        """
        latex_chars = []
        for stag, t, e in self.process_children_list(elm):
            if stag == "r":
                if FUNC.get(t):
                    latex_chars.append(FUNC[t])
                else:
                    logger.warning(
                        "Function not supported, will default to text: %s", t
                    )
                    if isinstance(t, str):
                        latex_chars.append(t)
            elif isinstance(t, str):
                latex_chars.append(t)
        t = BLANK.join(latex_chars)
        return t if FUNC_PLACE in t else t + FUNC_PLACE  # do_func will replace this

    def do_groupchr(self, elm):
        """
        the Group-Character object
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict["groupChrPr"]
        latex_s = get_val(pr.chr)
        return pr.text + latex_s.format(c_dict["e"])

    def do_rad(self, elm):
        """
        the radical object
        """
        c_dict = self.process_children_dict(elm)
        text = c_dict.get("e")
        deg_text = c_dict.get("deg")
        if deg_text:
            return RAD.format(deg=deg_text, text=text)
        else:
            return RAD_DEFAULT.format(text=text)

    def do_eqarr(self, elm):
        """
        the Array object.
        """
        rows = [t for stag, t, e in self.process_children_list(elm, include=("e",))]

        if len(rows) == 1:
            row = rows[0]
            tag_match = re.search(r"\\#\s*\(([^)]*)\)\s*$", row)
            if tag_match:
                formula = row[: tag_match.start()].rstrip()
                tag_content = tag_match.group(1)
                return f"{formula}\\tag{{{tag_content}}}"
            return row

        return ARR.format(text=BRK.join(rows))

    def do_limlow(self, elm):
        """
        the Lower-Limit object
        """
        t_dict = self.process_children_dict(elm, include=("e", "lim"))
        latex_s = LIM_FUNC.get(t_dict["e"])
        if not latex_s:
            raise RuntimeError("Not support lim {}".format(t_dict["e"]))
        else:
            return latex_s.format(lim=t_dict.get("lim"))

    def do_limupp(self, elm):
        """
        the Upper-Limit object
        """
        t_dict = self.process_children_dict(elm, include=("e", "lim"))
        return LIM_UPP.format(lim=t_dict.get("lim"), text=t_dict.get("e"))

    def do_lim(self, elm):
        """
        the lower limit of the limLow object and the upper limit of the limUpp function
        """
        return self.process_children(elm).replace(LIM_TO[0], LIM_TO[1])

    def do_m(self, elm):
        """
        the Matrix object
        """
        rows = []
        for stag, t, e in self.process_children_list(elm):
            if stag == "mPr":
                pass
            elif stag == "mr":
                rows.append(t)
        return M.format(text=BRK.join(rows))

    def do_mr(self, elm):
        """
        a single row of the matrix m
        """
        return ALN.join(
            [t for stag, t, e in self.process_children_list(elm, include=("e",))]
        )

    def do_nary(self, elm):
        """
        the n-ary object
        """
        res = []
        bo = ""
        for stag, t, e in self.process_children_list(elm):
            if stag == "naryPr":
                bo = get_val(t.chr, default="\\int", store=CHR_BO)
            else:
                res.append(t)
        return bo + BLANK.join(res)

    def process_unicode(self, s):
        t_result = self._t_dict.get(s)
        if t_result is not None:
            return t_result

        out_latex_str = self.u.unicode_to_latex(s)

        if (
            s.startswith("{") is False
            and out_latex_str.startswith("{")
            and s.endswith("}") is False
            and out_latex_str.endswith("}")
        ):
            out_latex_str = f" {out_latex_str[1:-1]} "

        if "ensuremath" in out_latex_str:
            out_latex_str = out_latex_str.replace("\\ensuremath{", " ")
            out_latex_str = out_latex_str.replace("}", " ")

        return out_latex_str

    def do_r(self, elm):
        """
        Get text from 'r' element,And try convert them to latex symbols
        """
        _str = []
        _base_str = []
        found_text = elm.findtext(f"./{OMML_NS}t")
        if found_text:
            for s in found_text:
                out_latex_str = self.process_unicode(s)
                _str.append(out_latex_str)
                _base_str.append(s)

        proc_str = escape_latex(BLANK.join(_str))
        base_proc_str = BLANK.join(_base_str)

        if "{" not in base_proc_str and "\\{" in proc_str:
            proc_str = proc_str.replace("\\{", "{")

        if "}" not in base_proc_str and "\\}" in proc_str:
            proc_str = proc_str.replace("\\}", "}")

        # Handle <m:scr> math font style
        rPr = elm.find(f"{OMML_NS}rPr")
        if rPr is not None:
            scr_elem = rPr.find(f"{OMML_NS}scr")
            if scr_elem is not None:
                scr_val = scr_elem.get(f"{OMML_NS}val")
                latex_template = SCR_TO_LATEX.get(scr_val)
                if latex_template and proc_str.strip():
                    proc_str = latex_template.format(proc_str.strip())

        return proc_str

    tag2meth = {
        "acc": do_acc,
        "r": do_r,
        "bar": do_bar,
        "sub": do_sub,
        "sup": do_sup,
        "f": do_f,
        "func": do_func,
        "fName": do_fname,
        "groupChr": do_groupchr,
        "d": do_d,
        "rad": do_rad,
        "eqArr": do_eqarr,
        "limLow": do_limlow,
        "limUpp": do_limupp,
        "lim": do_lim,
        "m": do_m,
        "mr": do_mr,
        "nary": do_nary,
    }
