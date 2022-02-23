import cssutils
import threading
from collections import OrderedDict

from premailer.cache import function_cache


def format_value(prop):
    if prop.priority == "important":
        return prop.propertyValue.cssText.strip() + " !important"
    else:
        return prop.propertyValue.cssText.strip()


@function_cache()
def csstext_to_pairs(csstext, validate=True):
    """
    csstext_to_pairs takes css text and make it to list of
    tuple of key,value.
    """
    # The lock is required to avoid ``cssutils`` concurrency
    # issues documented in issue #65
    with csstext_to_pairs._lock:
        return [
            (prop.name.strip(), format_value(prop))
            for prop in cssutils.parseStyle(csstext, validate=validate)
        ]


csstext_to_pairs._lock = threading.RLock()


def merge_styles(inline_style, new_styles, classes, remove_unset_properties=False):
    """
    This will merge all new styles where the order is important
    The last one will override the first
    When that is done it will apply old inline style again
    The old inline style is always important and override
    all new ones. The inline style must be valid.

    Args:
        inline_style(str): the old inline style of the element if there
            is one
        new_styles: a list of new styles, each element should be
            a list of tuple
        classes: a list of classes which maps new_styles, important!
        remove_unset_properties(bool): Allow us to remove certain CSS
            properties with rules that set their value to 'unset'

    Returns:
        str: the final style
    """
    # building classes
    styles = OrderedDict([("", OrderedDict())])
    for pc in set(classes):
        styles[pc] = OrderedDict()

    for i, style in enumerate(new_styles):
        for k, v in style:
            styles[classes[i]][k] = v

    # keep always the old inline style
    if inline_style:
        # inline should be a declaration list as I understand
        # ie property-name:property-value;...
        for k, v in csstext_to_pairs(inline_style):
            styles[""][k] = v

    normal_styles = []
    pseudo_styles = []
    for pseudoclass, kv in styles.items():
        if remove_unset_properties:
            # Remove rules that we were going to have value 'unset' because
            # they effectively are the same as not saying anything about the
            # property when inlined
            kv = OrderedDict(
                (k, v) for (k, v) in kv.items() if not v.lower() == "unset"
            )
        if not kv:
            continue
        if pseudoclass:
            pseudo_styles.append(
                "%s{%s}"
                % (pseudoclass, "; ".join("%s:%s" % (k, v) for k, v in kv.items()))
            )
        else:
            normal_styles.append("; ".join("%s:%s" % (k, v) for k, v in kv.items()))

    if pseudo_styles:
        # if we do or code thing correct this should not happen
        # inline style definition: declarations without braces
        all_styles = (
            (["{%s}" % "".join(normal_styles)] + pseudo_styles)
            if normal_styles
            else pseudo_styles
        )
    else:
        all_styles = normal_styles

    return " ".join(all_styles).strip()
