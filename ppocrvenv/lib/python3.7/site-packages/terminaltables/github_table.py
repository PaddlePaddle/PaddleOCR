"""GithubFlavoredMarkdownTable class."""

from terminaltables.ascii_table import AsciiTable
from terminaltables.build import combine


class GithubFlavoredMarkdownTable(AsciiTable):
    """Github flavored markdown table.

    https://help.github.com/articles/github-flavored-markdown/#tables

    :ivar iter table_data: List (empty or list of lists of strings) representing the table.
    :ivar dict justify_columns: Horizontal justification. Keys are column indexes (int). Values are right/left/center.
    """

    def __init__(self, table_data):
        """Constructor.

        :param iter table_data: List (empty or list of lists of strings) representing the table.
        """
        # Github flavored markdown table won't support title.
        super(GithubFlavoredMarkdownTable, self).__init__(table_data)

    def horizontal_border(self, _, outer_widths):
        """Handle the GitHub heading border.

        E.g.:
        |:---|:---:|---:|----|

        :param _: Unused.
        :param iter outer_widths: List of widths (with padding) for each column.

        :return: Prepared border strings in a generator.
        :rtype: iter
        """
        horizontal = str(self.CHAR_INNER_HORIZONTAL)
        left = self.CHAR_OUTER_LEFT_VERTICAL
        intersect = self.CHAR_INNER_VERTICAL
        right = self.CHAR_OUTER_RIGHT_VERTICAL

        columns = list()
        for i, width in enumerate(outer_widths):
            justify = self.justify_columns.get(i)
            width = max(3, width)  # Width should be at least 3 so justification can be applied.
            if justify == 'left':
                columns.append(':' + horizontal * (width - 1))
            elif justify == 'right':
                columns.append(horizontal * (width - 1) + ':')
            elif justify == 'center':
                columns.append(':' + horizontal * (width - 2) + ':')
            else:
                columns.append(horizontal * width)

        return combine(columns, left, intersect, right)

    def gen_table(self, inner_widths, inner_heights, outer_widths):
        """Combine everything and yield every line of the entire table with borders.

        :param iter inner_widths: List of widths (no padding) for each column.
        :param iter inner_heights: List of heights (no padding) for each row.
        :param iter outer_widths: List of widths (with padding) for each column.
        :return:
        """
        for i, row in enumerate(self.table_data):
            # Yield the row line by line (e.g. multi-line rows).
            for line in self.gen_row_lines(row, 'row', inner_widths, inner_heights[i]):
                yield line
            # Yield heading separator.
            if i == 0:
                yield self.horizontal_border(None, outer_widths)
