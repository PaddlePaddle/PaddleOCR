import codecs
import operator
import os
import re
import warnings
from collections import OrderedDict
from html import escape, unescape
from urllib.parse import urljoin, urlparse, unquote

import cssutils
import requests
from lxml import etree
from lxml.cssselect import CSSSelector

from premailer.cache import function_cache
from premailer.merge_style import csstext_to_pairs, merge_styles


__all__ = ["PremailerError", "Premailer", "transform"]


class PremailerError(Exception):
    pass


class ExternalNotFoundError(ValueError):
    pass


class ExternalFileLoadingError(Exception):
    pass


def make_important(bulk):
    """makes every property in a string !important."""
    return ";".join(
        "%s !important" % p if not p.endswith("!important") else p
        for p in bulk.split(";")
    )


def get_or_create_head(root):
    """Ensures that `root` contains a <head> element and returns it."""
    head = _create_cssselector("head")(root)
    if not head:
        head = etree.Element("head")
        body = _create_cssselector("body")(root)[0]
        body.getparent().insert(0, head)
        return head
    else:
        return head[0]


@function_cache()
def _cache_parse_css_string(css_body, validate=True):
    """
    This function will cache the result from cssutils
    It is a big gain when number of rules is big
    Maximum cache entries are 1000. This is mainly for
    protecting memory leak in case something gone wild.
    Be aware that you can turn the cache off in Premailer

    Args:
        css_body(str): css rules in string format
        validate(bool): if cssutils should validate

    Returns:
        cssutils.css.cssstylesheet.CSSStyleSheet

    """
    return cssutils.parseString(css_body, validate=validate)


@function_cache()
def _create_cssselector(selector):
    return CSSSelector(selector)


def capitalize_float_margin(css_body):
    """Capitalize float and margin CSS property names"""

    def _capitalize_property(match):
        return "{0}:{1}{2}".format(
            match.group("property").capitalize(),
            match.group("value"),
            match.group("terminator"),
        )

    return _lowercase_margin_float_rule.sub(_capitalize_property, css_body)


_element_selector_regex = re.compile(r"(^|\s)\w")
_cdata_regex = re.compile(r"\<\!\[CDATA\[(.*?)\]\]\>", re.DOTALL)
_lowercase_margin_float_rule = re.compile(
    r"""(?P<property>margin(-(top|bottom|left|right))?|float)
        :
        (?P<value>.*?)
        (?P<terminator>$|;)""",
    re.IGNORECASE | re.VERBOSE,
)
_importants = re.compile(r"\s*!important")
#: The short (3-digit) color codes that cause issues for IBM Notes
_short_color_codes = re.compile(r"^#([0-9a-f])([0-9a-f])([0-9a-f])$", re.I)

# These selectors don't apply to all elements. Rather, they specify
# which elements to apply to.
FILTER_PSEUDOSELECTORS = [":last-child", ":first-child", ":nth-child"]


class Premailer(object):

    attribute_name = "data-premailer"

    def __init__(
        self,
        html=None,
        base_url=None,
        disable_link_rewrites=False,
        preserve_internal_links=False,
        preserve_inline_attachments=True,
        preserve_handlebar_syntax=False,
        exclude_pseudoclasses=True,
        keep_style_tags=False,
        include_star_selectors=False,
        remove_classes=False,
        capitalize_float_margin=False,
        strip_important=True,
        external_styles=None,
        css_text=None,
        method="html",
        base_path=None,
        disable_basic_attributes=None,
        disable_validation=False,
        cache_css_parsing=True,
        cssutils_logging_handler=None,
        cssutils_logging_level=None,
        disable_leftover_css=False,
        align_floating_images=True,
        remove_unset_properties=True,
        allow_network=True,
        allow_insecure_ssl=False,
        allow_loading_external_files=False,
        session=None,
    ):
        self.html = html
        self.base_url = base_url

        # If base_url is specified, it is used for loading external stylesheets
        # via relative URLs.
        #
        # Also, if base_url is specified, premailer will transform all URLs by
        # joining them with the base_url. Setting preserve_internal_links to
        # True will disable this behavior for links to named anchors. Setting
        # preserve_inline_attachments to True will disable this behavior for
        # any links with cid: scheme. Setting disable_link_rewrites to True
        # will disable this behavior altogether.
        self.disable_link_rewrites = disable_link_rewrites
        self.preserve_internal_links = preserve_internal_links
        self.preserve_inline_attachments = preserve_inline_attachments
        self.preserve_handlebar_syntax = preserve_handlebar_syntax
        self.exclude_pseudoclasses = exclude_pseudoclasses
        # whether to delete the <style> tag once it's been processed
        # this will always preserve the original css
        self.keep_style_tags = keep_style_tags
        self.remove_classes = remove_classes
        self.capitalize_float_margin = capitalize_float_margin
        # whether to process or ignore selectors like '* { foo:bar; }'
        self.include_star_selectors = include_star_selectors
        if isinstance(external_styles, str):
            external_styles = [external_styles]
        self.external_styles = external_styles
        if isinstance(css_text, str):
            css_text = [css_text]
        self.css_text = css_text
        self.strip_important = strip_important
        self.method = method
        self.base_path = base_path
        if disable_basic_attributes is None:
            disable_basic_attributes = []
        self.disable_basic_attributes = disable_basic_attributes
        self.disable_validation = disable_validation
        self.cache_css_parsing = cache_css_parsing
        self.disable_leftover_css = disable_leftover_css
        self.align_floating_images = align_floating_images
        self.remove_unset_properties = remove_unset_properties
        self.allow_network = allow_network
        self.allow_insecure_ssl = allow_insecure_ssl
        self.allow_loading_external_files = allow_loading_external_files
        self.session = session or requests

        if cssutils_logging_handler:
            cssutils.log.addHandler(cssutils_logging_handler)
        if cssutils_logging_level:
            cssutils.log.setLevel(cssutils_logging_level)

    def _parse_css_string(self, css_body, validate=True):
        if self.cache_css_parsing:
            return _cache_parse_css_string(css_body, validate=validate)

        return cssutils.parseString(css_body, validate=validate)

    def _parse_style_rules(self, css_body, ruleset_index):
        """Returns a list of rules to apply to this doc and a list of rules
        that won't be used because e.g. they are pseudoclasses. Rules
        look like: (specificity, selector, bulk)
        for example: ((0, 1, 0, 0, 0), u'.makeblue', u'color:blue').
        The bulk of the rule should not end in a semicolon.
        """

        def format_css_property(prop):
            if self.strip_important or prop.priority != "important":
                return "{0}:{1}".format(prop.name, prop.value)
            else:
                return "{0}:{1} !important".format(prop.name, prop.value)

        def join_css_properties(properties):
            """Accepts a list of cssutils Property objects and returns
            a semicolon delimitted string like 'color: red; font-size: 12px'
            """
            return ";".join(format_css_property(prop) for prop in properties)

        leftover = []
        rules = []
        # empty string
        if not css_body:
            return rules, leftover
        sheet = self._parse_css_string(css_body, validate=not self.disable_validation)
        for rule in sheet:
            # handle media rule
            if rule.type == rule.MEDIA_RULE:
                leftover.append(rule)
                continue
            # only proceed for things we recognize
            if rule.type != rule.STYLE_RULE:
                continue

            # normal means it doesn't have "!important"
            normal_properties = [
                prop
                for prop in rule.style.getProperties()
                if prop.priority != "important"
            ]
            important_properties = [
                prop
                for prop in rule.style.getProperties()
                if prop.priority == "important"
            ]

            # Create three strings that we can use to add to the `rules`
            # list later as ready blocks of css.
            bulk_normal = join_css_properties(normal_properties)
            bulk_important = join_css_properties(important_properties)
            bulk_all = join_css_properties(normal_properties + important_properties)

            selectors = (
                x.strip()
                for x in rule.selectorText.split(",")
                if x.strip() and not x.strip().startswith("@")
            )
            for selector in selectors:
                if (
                    ":" in selector
                    and self.exclude_pseudoclasses
                    and ":" + selector.split(":", 1)[1] not in FILTER_PSEUDOSELECTORS
                ):
                    # a pseudoclass
                    leftover.append((selector, bulk_all))
                    continue
                elif "*" in selector and not self.include_star_selectors:
                    continue
                elif selector.startswith(":"):
                    continue

                # Crudely calculate specificity
                id_count = selector.count("#")
                class_count = selector.count(".")
                element_count = len(_element_selector_regex.findall(selector))

                # Within one rule individual properties have different
                # priority depending on !important.
                # So we split each rule into two: one that includes all
                # the !important declarations and another that doesn't.
                for is_important, bulk in ((1, bulk_important), (0, bulk_normal)):
                    if not bulk:
                        # don't bother adding empty css rules
                        continue
                    specificity = (
                        is_important,
                        id_count,
                        class_count,
                        element_count,
                        ruleset_index,
                        len(rules),  # this is the rule's index number
                    )
                    rules.append((specificity, selector, bulk))

        return rules, leftover

    def transform(self, html=None, pretty_print=True, **kwargs):
        """change the html and return it with CSS turned into style
        attributes.
        """
        if html is not None and self.html is not None:
            raise TypeError("Can't pass html argument twice")
        elif html is None and self.html is None:
            raise TypeError("must pass html as first argument")
        elif html is None:
            html = self.html
        if hasattr(html, "getroottree"):
            # skip the next bit
            root = html.getroottree()
            page = root
            tree = root
        else:
            if self.method == "xml":
                parser = etree.XMLParser(ns_clean=False, resolve_entities=False)
            else:
                parser = etree.HTMLParser()
            stripped = html.strip()

            # Escape all characters in handlebars in HTML attributes.
            # Without this step, if handlebars were to include a character such as ",
            # etree.fromstring() would not be able to differentiate the "'s in the value
            # from the "'s for the attribute.
            # --------------------------------------------------------------------------
            # Provided the input below:
            # <a href="{{ "<Test>" }}"></a>
            # --------------------------------------------------------------------------
            # Decoded result without preservation:
            # <a href="%7B%7B%20">" }}"&gt;</a>
            # Everything between the first two quotes were treated as the value of the
            # attribute. Then, the characters between the second quote and the second
            # > were treated as invalid attributes and discarded. Lastly, the value of
            # the original attribute after the second and </a> were treated as the
            # contents of the HTML tag.
            # ---
            # Result:
            # <a href="%7B%7B%20">" }}"&gt;</a>
            # --------------------------------------------------------------------------
            # Decoded result with preservation (prior to unescape() & unquote()):
            # <a href="%7B%7B%20%22&lt;Test&gt;%22%20%7D%7D"></a>
            # No value was lost in the encoding process.
            # ---
            # Result after unquote() and unescape():
            # <a href="{{ "<Test>" }}"></a>
            if self.preserve_handlebar_syntax:
                stripped = re.sub(
                    r'="{{(.*?)}}"',
                    lambda match: '="{{' + escape(match.groups()[0]) + '}}"',
                    stripped,
                )

            tree = etree.fromstring(stripped, parser).getroottree()
            page = tree.getroot()
            # lxml inserts a doctype if none exists, so only include it in
            # the root if it was in the original html.
            root = tree if stripped.startswith(tree.docinfo.doctype) else page

        assert page is not None

        if self.disable_leftover_css:
            head = None
        else:
            head = get_or_create_head(tree)
        #
        # style selectors
        #

        rules = []
        index = 0

        cssselector = ["style"]
        if self.allow_network:
            cssselector.append("link[rel~=stylesheet]")
        for element in _create_cssselector(",".join(cssselector))(page):
            # If we have a media attribute whose value is anything other than
            # 'all' or 'screen', ignore the ruleset.
            media = element.attrib.get("media")
            if media and media not in ("all", "screen"):
                continue

            data_attribute = element.attrib.get(self.attribute_name)
            if data_attribute:
                if data_attribute == "ignore":
                    del element.attrib[self.attribute_name]
                    continue
                else:
                    warnings.warn(
                        "Unrecognized %s attribute (%r)"
                        % (self.attribute_name, data_attribute)
                    )

            is_style = element.tag == "style"
            if is_style:
                css_body = element.text
            else:
                href = element.attrib.get("href")
                css_body = self._load_external(href)

            these_rules, these_leftover = self._parse_style_rules(css_body, index)

            index += 1
            rules.extend(these_rules)
            parent_of_element = element.getparent()
            if these_leftover or self.keep_style_tags:
                if is_style:
                    style = element
                else:
                    style = etree.Element("style")
                    style.attrib["type"] = "text/css"
                if self.keep_style_tags:
                    style.text = css_body
                else:
                    style.text = self._css_rules_to_string(these_leftover)

                if self.strip_important:
                    style.text = _importants.sub("", style.text)

                if self.method == "xml":
                    style.text = etree.CDATA(style.text)

                if not is_style:
                    element.addprevious(style)
                    parent_of_element.remove(element)

            elif not self.keep_style_tags or not is_style:
                parent_of_element.remove(element)

        # external style files
        if self.external_styles and self.allow_network:
            for stylefile in self.external_styles:
                css_body = self._load_external(stylefile)
                self._process_css_text(css_body, index, rules, head)
                index += 1

        # css text
        if self.css_text:
            for css_body in self.css_text:
                self._process_css_text(css_body, index, rules, head)
                index += 1

        # rules is a tuple of (specificity, selector, styles), where
        # specificity is a tuple ordered such that more specific
        # rules sort larger.
        rules.sort(key=operator.itemgetter(0))

        # collecting all elements that we need to apply rules on
        # id is unique for the lifetime of the object
        # and lxml should give us the same everytime during this run
        # item id -> {item: item, classes: [], style: []}
        elements = {}
        for _, selector, style in rules:
            new_selector = selector
            class_ = ""
            if ":" in selector:
                new_selector, class_ = re.split(":", selector, 1)
                class_ = ":%s" % class_
            # Keep filter-type selectors untouched.
            if class_ in FILTER_PSEUDOSELECTORS or class_.startswith(":nth-child"):
                class_ = ""
            else:
                selector = new_selector

            assert selector
            sel = _create_cssselector(selector)
            items = sel(page)
            if len(items):
                # same so process it first
                processed_style = csstext_to_pairs(
                    style, validate=not self.disable_validation
                )

                for item in items:
                    item_id = id(item)
                    if item_id not in elements:
                        elements[item_id] = {"item": item, "classes": [], "style": []}

                    elements[item_id]["style"].append(processed_style)
                    elements[item_id]["classes"].append(class_)

        # Now apply inline style
        # merge style only once for each element
        # crucial when you have a lot of pseudo/classes
        # and a long list of elements
        for _, element in elements.items():
            final_style = merge_styles(
                element["item"].attrib.get("style", ""),
                element["style"],
                element["classes"],
                remove_unset_properties=self.remove_unset_properties,
            )
            if final_style:
                # final style could be empty string because of
                # remove_unset_properties
                element["item"].attrib["style"] = final_style
            self._style_to_basic_html_attributes(
                element["item"], final_style, force=True
            )

        if self.remove_classes:
            # now we can delete all 'class' attributes
            for item in page.xpath("//@class"):
                parent = item.getparent()
                del parent.attrib["class"]

        # Capitalize Margin properties
        # To fix weird outlook bug
        # https://www.emailonacid.com/blog/article/email-development/outlook.com-does-support-margins
        if self.capitalize_float_margin:
            for item in page.xpath("//@style"):
                mangled = capitalize_float_margin(item)
                item.getparent().attrib["style"] = mangled

        # Add align attributes to images if they have a CSS float value of
        # right or left. Outlook (both on desktop and on the web) are bad at
        # understanding floats, but they do understand the HTML align attrib.
        if self.align_floating_images:
            for item in page.xpath("//img[@style]"):
                image_css = cssutils.parseStyle(item.attrib["style"])
                if image_css.float == "right":
                    item.attrib["align"] = "right"
                elif image_css.float == "left":
                    item.attrib["align"] = "left"

        #
        # URLs
        #
        if self.base_url and not self.disable_link_rewrites:
            if not urlparse(self.base_url).scheme:
                raise ValueError("Base URL must have a scheme")
            for attr in ("href", "src"):
                for item in page.xpath("//@%s" % attr):
                    parent = item.getparent()
                    url = parent.attrib[attr]
                    if (
                        attr == "href"
                        and self.preserve_internal_links
                        and url.startswith("#")
                    ):
                        continue
                    if (
                        attr == "src"
                        and self.preserve_inline_attachments
                        and url.startswith("cid:")
                    ):
                        continue
                    if attr == "href" and url.startswith("tel:"):
                        continue
                    parent.attrib[attr] = urljoin(self.base_url, url)

        if hasattr(html, "getroottree"):
            return root
        else:
            kwargs.setdefault("method", self.method)
            kwargs.setdefault("pretty_print", pretty_print)
            kwargs.setdefault("encoding", "utf-8")  # As Ken Thompson intended
            out = etree.tostring(root, **kwargs).decode(kwargs["encoding"])
            if self.method == "xml":
                out = _cdata_regex.sub(
                    lambda m: "/*<![CDATA[*/%s/*]]>*/" % m.group(1), out
                )
            # Replace %xx escapes and HTML entities, within handlebars in HTML
            # attributes, with their single-character equivalents.
            if self.preserve_handlebar_syntax:
                out = re.sub(
                    r'="%7B%7B(.+?)%7D%7D"',
                    lambda match: '="{{' + unescape(unquote(match.groups()[0])) + '}}"',
                    out,
                )
            return out

    def _load_external_url(self, url):
        response = self.session.get(url, verify=not self.allow_insecure_ssl)
        response.raise_for_status()
        return response.text

    def _load_external(self, url):
        """loads an external stylesheet from a remote url or local path"""
        if url.startswith("//"):
            # then we have to rely on the base_url
            if self.base_url and "https://" in self.base_url:
                url = "https:" + url
            else:
                url = "http:" + url

        if url.startswith("http://") or url.startswith("https://"):
            css_body = self._load_external_url(url)
        elif not self.allow_loading_external_files:
            raise ExternalFileLoadingError(
                "Unable to load external file {!r} because it's explicitly not allowed"
                "".format(url)
            )
        else:
            stylefile = url
            base_path = os.path.abspath(self.base_path or os.curdir)
            if not os.path.isabs(stylefile):
                stylefile = os.path.abspath(os.path.join(base_path, stylefile))
            if os.path.exists(stylefile):
                with codecs.open(stylefile, encoding="utf-8") as f:
                    css_body = f.read()
            elif self.base_url:
                url = urljoin(self.base_url, url)
                return self._load_external(url)
            else:
                raise ExternalNotFoundError(stylefile)

        return css_body

    @staticmethod
    def six_color(color_value):
        """Fix background colors for Lotus Notes

        Notes which fails to handle three character ``bgcolor`` codes well.
        see <https://github.com/peterbe/premailer/issues/114>"""

        # Turn the color code from three to six digits
        retval = _short_color_codes.sub(r"#\1\1\2\2\3\3", color_value)
        return retval

    def _style_to_basic_html_attributes(self, element, style_content, force=False):
        """given an element and styles like
        'background-color:red; font-family:Arial' turn some of that into HTML
        attributes. like 'bgcolor', etc.

        Note, the style_content can contain pseudoclasses like:
        '{color:red; border:1px solid green} :visited{border:1px solid green}'
        """
        if style_content.count("}") and style_content.count("{") == style_content.count(
            "}"
        ):
            style_content = style_content.split("}")[0][1:]

        attributes = OrderedDict()
        for key, value in [
            x.split(":") for x in style_content.split(";") if len(x.split(":")) == 2
        ]:
            key = key.strip()

            if key == "text-align":
                attributes["align"] = value.strip()
            elif key == "vertical-align":
                attributes["valign"] = value.strip()
            elif key == "background-color" and "transparent" not in value.lower():
                # Only add the 'bgcolor' attribute if the value does not
                # contain the word "transparent"; before we add it possibly
                # correct the 3-digit color code to its 6-digit equivalent
                # ("abc" to "aabbcc") so IBM Notes copes.
                attributes["bgcolor"] = self.six_color(value.strip())
            elif key == "width" or key == "height":
                value = value.strip()
                if value.endswith("px"):
                    value = value[:-2]
                attributes[key] = value

        for key, value in attributes.items():
            if (
                key in element.attrib
                and not force
                or key in self.disable_basic_attributes
            ):
                # already set, don't dare to overwrite
                continue
            element.attrib[key] = value

    def _css_rules_to_string(self, rules):
        """given a list of css rules returns a css string"""
        lines = []
        for item in rules:
            if isinstance(item, tuple):
                k, v = item
                lines.append("%s {%s}" % (k, make_important(v)))
            # media rule
            else:
                for rule in item.cssRules:
                    if isinstance(
                        rule,
                        (
                            cssutils.css.csscomment.CSSComment,
                            cssutils.css.cssunknownrule.CSSUnknownRule,
                        ),
                    ):
                        continue
                    for key in rule.style.keys():
                        rule.style[key] = (
                            rule.style.getPropertyValue(key, False),
                            "!important",
                        )
                lines.append(item.cssText)
        return "\n".join(lines)

    def _process_css_text(self, css_text, index, rules, head):
        """processes the given css_text by adding rules that can be
        in-lined to the given rules list and adding any that cannot
        be in-lined to the given `<head>` element.
        """
        these_rules, these_leftover = self._parse_style_rules(css_text, index)
        rules.extend(these_rules)
        if head is not None and (these_leftover or self.keep_style_tags):
            style = etree.Element("style")
            style.attrib["type"] = "text/css"
            if self.keep_style_tags:
                style.text = css_text
            else:
                style.text = self._css_rules_to_string(these_leftover)
            head.append(style)


def transform(html, pretty_print=False, **kwargs):
    return Premailer(**kwargs).transform(html, pretty_print=pretty_print)


if __name__ == "__main__":  # pragma: no cover
    html = """<html>
        <head>
        <title>Test</title>
        <style>
        h1, h2 { color:red; }
        strong {
          text-decoration:none
          }
        p { font-size:2px }
        p.footer { font-size: 1px}
        </style>
        </head>
        <body>
        <h1>Hi!</h1>
        <p><strong>Yes!</strong></p>
        <p class="footer" style="color:red">Feetnuts</p>
        </body>
        </html>"""
    p = Premailer(html)
    print(p.transform())
