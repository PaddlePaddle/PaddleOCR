import py

from .compat import Iterable
from .utils import TIME_UNITS
from .utils import slugify

try:
    from pygal.graph.box import Box
    from pygal.style import DefaultStyle
except ImportError as exc:
    raise ImportError(exc.args, "Please install pygal and pygaljs or pytest-benchmark[histogram]")


class CustomBox(Box):
    def _box_points(self, serie, _):
        return serie, [serie[0], serie[6]]

    def _value_format(self, x):
        return "Min: {0[0]:.4f}\n" \
               "Q1-1.5IQR: {0[1]:.4f}\n" \
               "Q1: {0[2]:.4f}\nMedian: {0[3]:.4f}\nQ3: {0[4]:.4f}\n" \
               "Q3+1.5IQR: {0[5]:.4f}\n" \
               "Max: {0[6]:.4f}".format(x[:7])

    def _format(self, x, *args):
        sup = super(CustomBox, self)._format
        if args:
            val = x.values
        else:
            val = x
        if isinstance(val, Iterable):
            return self._value_format(val), val[7]
        else:
            return sup(x, *args)

    def _tooltip_data(self, node, value, x, y, classes=None, xlabel=None):
        super(CustomBox, self)._tooltip_data(node, value[0], x, y, classes=classes, xlabel=None)
        self.svg.node(node, 'desc', class_="x_label").text = value[1]


def make_plot(benchmarks, title, adjustment):
    class Style(DefaultStyle):
        colors = ["#000000" if row["path"] else DefaultStyle.colors[1]
                  for row in benchmarks]
        font_family = 'Consolas, "Deja Vu Sans Mono", "Bitstream Vera Sans Mono", "Courier New", monospace'

    minimum = int(min(row["min"] * adjustment for row in benchmarks))
    maximum = int(max(
        min(row["max"], row["hd15iqr"]) * adjustment
        for row in benchmarks
    ) + 1)

    try:
        import pygaljs
    except ImportError:
        opts = {}
    else:
        opts = {
            "js": [
                pygaljs.uri("2.0.x", "pygal-tooltips.js")
            ]
        }

    plot = CustomBox(
        box_mode='tukey',
        x_label_rotation=-90,
        x_labels=["{0[name]}".format(row) for row in benchmarks],
        show_legend=False,
        title=title,
        x_title="Trial",
        y_title="Duration",
        style=Style,
        min_scale=20,
        max_scale=20,
        truncate_label=50,
        range=(minimum, maximum),
        zero=minimum,
        css=[
            "file://style.css",
            "file://graph.css",
            """inline:
                .tooltip .value {
                    font-size: 1em !important;
                }
                .axis text {
                    font-size: 9px !important;
                }
            """
        ],
        **opts
    )

    for row in benchmarks:
        serie = [row[field] * adjustment for field in ["min", "ld15iqr", "q1", "median", "q3", "hd15iqr", "max"]]
        serie.append(row["path"])
        plot.add("{0[fullname]} - {0[rounds]} rounds".format(row), serie)
    return plot


def make_histogram(output_prefix, name, benchmarks, unit, adjustment):
    if name:
        path = "{0}-{1}.svg".format(output_prefix, slugify(name))
        title = "Speed in {0} of {1}".format(TIME_UNITS[unit], name)
    else:
        path = "{0}.svg".format(output_prefix)
        title = "Speed in {0}".format(TIME_UNITS[unit])

    output_file = py.path.local(path).ensure()

    plot = make_plot(
        benchmarks=benchmarks,
        title=title,
        adjustment=adjustment,
    )
    plot.render_to_file(str(output_file))
    return output_file
