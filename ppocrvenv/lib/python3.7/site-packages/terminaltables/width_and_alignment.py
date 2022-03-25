"""Functions that handle alignment, padding, widths, etc."""

import re
import unicodedata

from terminaltables.terminal_io import terminal_size

RE_COLOR_ANSI = re.compile(r'(\033\[[\d;]+m)')


def visible_width(string):
    """Get the visible width of a unicode string.

    Some CJK unicode characters are more than one byte unlike ASCII and latin unicode characters.

    From: https://github.com/Robpol86/terminaltables/pull/9

    :param str string: String to measure.

    :return: String's width.
    :rtype: int
    """
    if '\033' in string:
        string = RE_COLOR_ANSI.sub('', string)

    # Convert to unicode.
    try:
        string = string.decode('u8')
    except (AttributeError, UnicodeEncodeError):
        pass

    width = 0
    for char in string:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1

    return width


def align_and_pad_cell(string, align, inner_dimensions, padding, space=' '):
    """Align a string horizontally and vertically. Also add additional padding in both dimensions.

    :param str string: Input string to operate on.
    :param tuple align: Tuple that contains one of left/center/right and/or top/middle/bottom.
    :param tuple inner_dimensions: Width and height ints to expand string to without padding.
    :param iter padding: Number of space chars for left, right, top, and bottom (4 ints).
    :param str space: Character to use as white space for resizing/padding (use single visible chars only).

    :return: Padded cell split into lines.
    :rtype: list
    """
    if not hasattr(string, 'splitlines'):
        string = str(string)

    # Handle trailing newlines or empty strings, str.splitlines() does not satisfy.
    lines = string.splitlines() or ['']
    if string.endswith('\n'):
        lines.append('')

    # Vertically align and pad.
    if 'bottom' in align:
        lines = ([''] * (inner_dimensions[1] - len(lines) + padding[2])) + lines + ([''] * padding[3])
    elif 'middle' in align:
        delta = inner_dimensions[1] - len(lines)
        lines = ([''] * (delta // 2 + delta % 2 + padding[2])) + lines + ([''] * (delta // 2 + padding[3]))
    else:
        lines = ([''] * padding[2]) + lines + ([''] * (inner_dimensions[1] - len(lines) + padding[3]))

    # Horizontally align and pad.
    for i, line in enumerate(lines):
        new_width = inner_dimensions[0] + len(line) - visible_width(line)
        if 'right' in align:
            lines[i] = line.rjust(padding[0] + new_width, space) + (space * padding[1])
        elif 'center' in align:
            lines[i] = (space * padding[0]) + line.center(new_width, space) + (space * padding[1])
        else:
            lines[i] = (space * padding[0]) + line.ljust(new_width + padding[1], space)

    return lines


def max_dimensions(table_data, padding_left=0, padding_right=0, padding_top=0, padding_bottom=0):
    """Get maximum widths of each column and maximum height of each row.

    :param iter table_data: List of list of strings (unmodified table data).
    :param int padding_left: Number of space chars on left side of cell.
    :param int padding_right: Number of space chars on right side of cell.
    :param int padding_top: Number of empty lines on top side of cell.
    :param int padding_bottom: Number of empty lines on bottom side of cell.

    :return: 4-item tuple of n-item lists. Inner column widths and row heights, outer column widths and row heights.
    :rtype: tuple
    """
    inner_widths = [0] * (max(len(r) for r in table_data) if table_data else 0)
    inner_heights = [0] * len(table_data)

    # Find max width and heights.
    for j, row in enumerate(table_data):
        for i, cell in enumerate(row):
            if not hasattr(cell, 'count') or not hasattr(cell, 'splitlines'):
                cell = str(cell)
            if not cell:
                continue
            inner_heights[j] = max(inner_heights[j], cell.count('\n') + 1)
            inner_widths[i] = max(inner_widths[i], *[visible_width(l) for l in cell.splitlines()])

    # Calculate with padding.
    outer_widths = [padding_left + i + padding_right for i in inner_widths]
    outer_heights = [padding_top + i + padding_bottom for i in inner_heights]

    return inner_widths, inner_heights, outer_widths, outer_heights


def column_max_width(inner_widths, column_number, outer_border, inner_border, padding):
    """Determine the maximum width of a column based on the current terminal width.

    :param iter inner_widths: List of widths (no padding) for each column.
    :param int column_number: The column number to query.
    :param int outer_border: Sum of left and right outer border visible widths.
    :param int inner_border: Visible width of the inner border character.
    :param int padding: Total padding per cell (left + right padding).

    :return: The maximum width the column can be without causing line wrapping.
    """
    column_count = len(inner_widths)
    terminal_width = terminal_size()[0]

    # Count how much space padding, outer, and inner borders take up.
    non_data_space = outer_border
    non_data_space += inner_border * (column_count - 1)
    non_data_space += column_count * padding

    # Exclude selected column's width.
    data_space = sum(inner_widths) - inner_widths[column_number]

    return terminal_width - data_space - non_data_space


def table_width(outer_widths, outer_border, inner_border):
    """Determine the width of the entire table including borders and padding.

    :param iter outer_widths: List of widths (with padding) for each column.
    :param int outer_border: Sum of left and right outer border visible widths.
    :param int inner_border: Visible width of the inner border character.

    :return: The width of the table.
    :rtype: int
    """
    column_count = len(outer_widths)

    # Count how much space outer and inner borders take up.
    non_data_space = outer_border
    if column_count:
        non_data_space += inner_border * (column_count - 1)

    # Space of all columns and their padding.
    data_space = sum(outer_widths)
    return data_space + non_data_space
