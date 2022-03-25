"""Additional simple tables defined here."""

from terminaltables.ascii_table import AsciiTable
from terminaltables.terminal_io import IS_WINDOWS


class UnixTable(AsciiTable):
    """Draw a table using box-drawing characters on Unix platforms. Table borders won't have any gaps between lines.

    Similar to the tables shown on PC BIOS boot messages, but not double-lined.
    """

    CHAR_F_INNER_HORIZONTAL = '\033(0\x71\033(B'
    CHAR_F_INNER_INTERSECT = '\033(0\x6e\033(B'
    CHAR_F_INNER_VERTICAL = '\033(0\x78\033(B'
    CHAR_F_OUTER_LEFT_INTERSECT = '\033(0\x74\033(B'
    CHAR_F_OUTER_LEFT_VERTICAL = '\033(0\x78\033(B'
    CHAR_F_OUTER_RIGHT_INTERSECT = '\033(0\x75\033(B'
    CHAR_F_OUTER_RIGHT_VERTICAL = '\033(0\x78\033(B'
    CHAR_H_INNER_HORIZONTAL = '\033(0\x71\033(B'
    CHAR_H_INNER_INTERSECT = '\033(0\x6e\033(B'
    CHAR_H_INNER_VERTICAL = '\033(0\x78\033(B'
    CHAR_H_OUTER_LEFT_INTERSECT = '\033(0\x74\033(B'
    CHAR_H_OUTER_LEFT_VERTICAL = '\033(0\x78\033(B'
    CHAR_H_OUTER_RIGHT_INTERSECT = '\033(0\x75\033(B'
    CHAR_H_OUTER_RIGHT_VERTICAL = '\033(0\x78\033(B'
    CHAR_INNER_HORIZONTAL = '\033(0\x71\033(B'
    CHAR_INNER_INTERSECT = '\033(0\x6e\033(B'
    CHAR_INNER_VERTICAL = '\033(0\x78\033(B'
    CHAR_OUTER_BOTTOM_HORIZONTAL = '\033(0\x71\033(B'
    CHAR_OUTER_BOTTOM_INTERSECT = '\033(0\x76\033(B'
    CHAR_OUTER_BOTTOM_LEFT = '\033(0\x6d\033(B'
    CHAR_OUTER_BOTTOM_RIGHT = '\033(0\x6a\033(B'
    CHAR_OUTER_LEFT_INTERSECT = '\033(0\x74\033(B'
    CHAR_OUTER_LEFT_VERTICAL = '\033(0\x78\033(B'
    CHAR_OUTER_RIGHT_INTERSECT = '\033(0\x75\033(B'
    CHAR_OUTER_RIGHT_VERTICAL = '\033(0\x78\033(B'
    CHAR_OUTER_TOP_HORIZONTAL = '\033(0\x71\033(B'
    CHAR_OUTER_TOP_INTERSECT = '\033(0\x77\033(B'
    CHAR_OUTER_TOP_LEFT = '\033(0\x6c\033(B'
    CHAR_OUTER_TOP_RIGHT = '\033(0\x6b\033(B'

    @property
    def table(self):
        """Return a large string of the entire table ready to be printed to the terminal."""
        ascii_table = super(UnixTable, self).table
        optimized = ascii_table.replace('\033(B\033(0', '')
        return optimized


class WindowsTable(AsciiTable):
    """Draw a table using box-drawing characters on Windows platforms. This uses Code Page 437. Single-line borders.

    From: http://en.wikipedia.org/wiki/Code_page_437#Characters
    """

    CHAR_F_INNER_HORIZONTAL = b'\xc4'.decode('ibm437')
    CHAR_F_INNER_INTERSECT = b'\xc5'.decode('ibm437')
    CHAR_F_INNER_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_F_OUTER_LEFT_INTERSECT = b'\xc3'.decode('ibm437')
    CHAR_F_OUTER_LEFT_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_F_OUTER_RIGHT_INTERSECT = b'\xb4'.decode('ibm437')
    CHAR_F_OUTER_RIGHT_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_H_INNER_HORIZONTAL = b'\xc4'.decode('ibm437')
    CHAR_H_INNER_INTERSECT = b'\xc5'.decode('ibm437')
    CHAR_H_INNER_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_H_OUTER_LEFT_INTERSECT = b'\xc3'.decode('ibm437')
    CHAR_H_OUTER_LEFT_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_H_OUTER_RIGHT_INTERSECT = b'\xb4'.decode('ibm437')
    CHAR_H_OUTER_RIGHT_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_INNER_HORIZONTAL = b'\xc4'.decode('ibm437')
    CHAR_INNER_INTERSECT = b'\xc5'.decode('ibm437')
    CHAR_INNER_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_OUTER_BOTTOM_HORIZONTAL = b'\xc4'.decode('ibm437')
    CHAR_OUTER_BOTTOM_INTERSECT = b'\xc1'.decode('ibm437')
    CHAR_OUTER_BOTTOM_LEFT = b'\xc0'.decode('ibm437')
    CHAR_OUTER_BOTTOM_RIGHT = b'\xd9'.decode('ibm437')
    CHAR_OUTER_LEFT_INTERSECT = b'\xc3'.decode('ibm437')
    CHAR_OUTER_LEFT_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_OUTER_RIGHT_INTERSECT = b'\xb4'.decode('ibm437')
    CHAR_OUTER_RIGHT_VERTICAL = b'\xb3'.decode('ibm437')
    CHAR_OUTER_TOP_HORIZONTAL = b'\xc4'.decode('ibm437')
    CHAR_OUTER_TOP_INTERSECT = b'\xc2'.decode('ibm437')
    CHAR_OUTER_TOP_LEFT = b'\xda'.decode('ibm437')
    CHAR_OUTER_TOP_RIGHT = b'\xbf'.decode('ibm437')


class WindowsTableDouble(AsciiTable):
    """Draw a table using box-drawing characters on Windows platforms. This uses Code Page 437. Double-line borders."""

    CHAR_F_INNER_HORIZONTAL = b'\xcd'.decode('ibm437')
    CHAR_F_INNER_INTERSECT = b'\xce'.decode('ibm437')
    CHAR_F_INNER_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_F_OUTER_LEFT_INTERSECT = b'\xcc'.decode('ibm437')
    CHAR_F_OUTER_LEFT_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_F_OUTER_RIGHT_INTERSECT = b'\xb9'.decode('ibm437')
    CHAR_F_OUTER_RIGHT_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_H_INNER_HORIZONTAL = b'\xcd'.decode('ibm437')
    CHAR_H_INNER_INTERSECT = b'\xce'.decode('ibm437')
    CHAR_H_INNER_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_H_OUTER_LEFT_INTERSECT = b'\xcc'.decode('ibm437')
    CHAR_H_OUTER_LEFT_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_H_OUTER_RIGHT_INTERSECT = b'\xb9'.decode('ibm437')
    CHAR_H_OUTER_RIGHT_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_INNER_HORIZONTAL = b'\xcd'.decode('ibm437')
    CHAR_INNER_INTERSECT = b'\xce'.decode('ibm437')
    CHAR_INNER_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_OUTER_BOTTOM_HORIZONTAL = b'\xcd'.decode('ibm437')
    CHAR_OUTER_BOTTOM_INTERSECT = b'\xca'.decode('ibm437')
    CHAR_OUTER_BOTTOM_LEFT = b'\xc8'.decode('ibm437')
    CHAR_OUTER_BOTTOM_RIGHT = b'\xbc'.decode('ibm437')
    CHAR_OUTER_LEFT_INTERSECT = b'\xcc'.decode('ibm437')
    CHAR_OUTER_LEFT_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_OUTER_RIGHT_INTERSECT = b'\xb9'.decode('ibm437')
    CHAR_OUTER_RIGHT_VERTICAL = b'\xba'.decode('ibm437')
    CHAR_OUTER_TOP_HORIZONTAL = b'\xcd'.decode('ibm437')
    CHAR_OUTER_TOP_INTERSECT = b'\xcb'.decode('ibm437')
    CHAR_OUTER_TOP_LEFT = b'\xc9'.decode('ibm437')
    CHAR_OUTER_TOP_RIGHT = b'\xbb'.decode('ibm437')


class SingleTable(WindowsTable if IS_WINDOWS else UnixTable):
    """Cross-platform table with single-line box-drawing characters.

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

    pass


class DoubleTable(WindowsTableDouble):
    """Cross-platform table with box-drawing characters. On Windows it's double borders, on Linux/OSX it's unicode.

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

    pass


class PorcelainTable(AsciiTable):
    """An AsciiTable stripped to a minimum.

    Meant to be machine passable and roughly follow format set by git --porcelain option (hence the name).

    :ivar iter table_data: List (empty or list of lists of strings) representing the table.
    """

    def __init__(self, table_data):
        """Constructor.

        :param iter table_data: List (empty or list of lists of strings) representing the table.
        """
        # Porcelain table won't support title since it has no outer birders.
        super(PorcelainTable, self).__init__(table_data)

        # Removes outer border, and inner footing and header row borders.
        self.inner_footing_row_border = False
        self.inner_heading_row_border = False
        self.outer_border = False
