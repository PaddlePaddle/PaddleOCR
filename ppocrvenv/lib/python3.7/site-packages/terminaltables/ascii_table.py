"""AsciiTable is the main table class. To be inherited by other tables. Define convenience methods here."""

from terminaltables.base_table import BaseTable
from terminaltables.terminal_io import terminal_size
from terminaltables.width_and_alignment import column_max_width, max_dimensions, table_width


class AsciiTable(BaseTable):
    """Draw a table using regular ASCII characters, such as ``+``, ``|``, and ``-``.

    :ivar iter table_data: List (empty or list of lists of strings) representing the table.
    :ivar str title: Optional title to show within the top border of the table.
    :ivar bool inner_column_border: Separates columns.
    :ivar bool inner_footing_row_border: Show a border before the last row.
    :ivar bool inner_heading_row_border: Show a border after the first row.
    :ivar bool inner_row_border: Show a border in between every row.
    :ivar bool outer_border: Show the top, left, right, and bottom border.
    :ivar dict justify_columns: Horizontal justification. Keys are column indexes (int). Values are right/left/center.
    :ivar int padding_left: Number of spaces to pad on the left side of every cell.
    :ivar int padding_right: Number of spaces to pad on the right side of every cell.
    """

    def column_max_width(self, column_number):
        """Return the maximum width of a column based on the current terminal width.

        :param int column_number: The column number to query.

        :return: The max width of the column.
        :rtype: int
        """
        inner_widths = max_dimensions(self.table_data)[0]
        outer_border = 2 if self.outer_border else 0
        inner_border = 1 if self.inner_column_border else 0
        padding = self.padding_left + self.padding_right
        return column_max_width(inner_widths, column_number, outer_border, inner_border, padding)

    @property
    def column_widths(self):
        """Return a list of integers representing the widths of each table column without padding."""
        if not self.table_data:
            return list()
        return max_dimensions(self.table_data)[0]

    @property
    def ok(self):  # Too late to change API. # pylint: disable=invalid-name
        """Return True if the table fits within the terminal width, False if the table breaks."""
        return self.table_width <= terminal_size()[0]

    @property
    def table_width(self):
        """Return the width of the table including padding and borders."""
        outer_widths = max_dimensions(self.table_data, self.padding_left, self.padding_right)[2]
        outer_border = 2 if self.outer_border else 0
        inner_border = 1 if self.inner_column_border else 0
        return table_width(outer_widths, outer_border, inner_border)
