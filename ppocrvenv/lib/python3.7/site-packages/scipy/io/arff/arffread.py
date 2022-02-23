# Last Change: Mon Aug 20 08:00 PM 2007 J
import re
import datetime

import numpy as np

import csv
import ctypes

"""A module to read arff files."""

__all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError']

# An Arff file is basically two parts:
#   - header
#   - data
#
# A header has each of its components starting by @META where META is one of
# the keyword (attribute of relation, for now).

# TODO:
#   - both integer and reals are treated as numeric -> the integer info
#    is lost!
#   - Replace ValueError by ParseError or something

# We know can handle the following:
#   - numeric and nominal attributes
#   - missing values for numeric attributes

r_meta = re.compile(r'^\s*@')
# Match a comment
r_comment = re.compile(r'^%')
# Match an empty line
r_empty = re.compile(r'^\s+$')
# Match a header line, that is a line which starts by @ + a word
r_headerline = re.compile(r'^\s*@\S*')
r_datameta = re.compile(r'^@[Dd][Aa][Tt][Aa]')
r_relation = re.compile(r'^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\s*(\S*)')
r_attribute = re.compile(r'^\s*@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\s*(..*$)')

r_nominal = re.compile(r'{(.+)}')
r_date = re.compile(r"[Dd][Aa][Tt][Ee]\s+[\"']?(.+?)[\"']?$")

# To get attributes name enclosed with ''
r_comattrval = re.compile(r"'(..+)'\s+(..+$)")
# To get normal attributes
r_wcomattrval = re.compile(r"(\S+)\s+(..+$)")

# ------------------------
# Module defined exception
# ------------------------


class ArffError(IOError):
    pass


class ParseArffError(ArffError):
    pass


# ----------
# Attributes
# ----------
class Attribute:

    type_name = None

    def __init__(self, name):
        self.name = name
        self.range = None
        self.dtype = np.object_

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.
        """
        return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        return None

    def __str__(self):
        """
        Parse a value of this type.
        """
        return self.name + ',' + self.type_name


class NominalAttribute(Attribute):

    type_name = 'nominal'

    def __init__(self, name, values):
        super().__init__(name)
        self.values = values
        self.range = values
        self.dtype = (np.string_, max(len(i) for i in values))

    @staticmethod
    def _get_nom_val(atrv):
        """Given a string containing a nominal type, returns a tuple of the
        possible values.

        A nominal type is defined as something framed between braces ({}).

        Parameters
        ----------
        atrv : str
           Nominal type definition

        Returns
        -------
        poss_vals : tuple
           possible values

        Examples
        --------
        >>> get_nom_val("{floup, bouga, fl, ratata}")
        ('floup', 'bouga', 'fl', 'ratata')
        """
        m = r_nominal.match(atrv)
        if m:
            attrs, _ = split_data_line(m.group(1))
            return tuple(attrs)
        else:
            raise ValueError("This does not look like a nominal string")

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For nominal attributes, the attribute string would be like '{<attr_1>,
         <attr2>, <attr_3>}'.
        """
        if attr_string[0] == '{':
            values = cls._get_nom_val(attr_string)
            return cls(name, values)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        if data_str in self.values:
            return data_str
        elif data_str == '?':
            return data_str
        else:
            raise ValueError("%s value not in %s" % (str(data_str),
                                                     str(self.values)))

    def __str__(self):
        msg = self.name + ",{"
        for i in range(len(self.values)-1):
            msg += self.values[i] + ","
        msg += self.values[-1]
        msg += "}"
        return msg


class NumericAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'numeric'
        self.dtype = np.float_

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For numeric attributes, the attribute string would be like
        'numeric' or 'int' or 'real'.
        """

        attr_string = attr_string.lower().strip()

        if(attr_string[:len('numeric')] == 'numeric' or
           attr_string[:len('int')] == 'int' or
           attr_string[:len('real')] == 'real'):
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.

        Parameters
        ----------
        data_str : str
           string to convert

        Returns
        -------
        f : float
           where float can be nan

        Examples
        --------
        >>> atr = NumericAttribute('atr')
        >>> atr.parse_data('1')
        1.0
        >>> atr.parse_data('1\\n')
        1.0
        >>> atr.parse_data('?\\n')
        nan
        """
        if '?' in data_str:
            return np.nan
        else:
            return float(data_str)

    def _basic_stats(self, data):
        nbfac = data.size * 1. / (data.size - 1)
        return (np.nanmin(data), np.nanmax(data),
                np.mean(data), np.std(data) * nbfac)


class StringAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'string'

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For string attributes, the attribute string would be like
        'string'.
        """

        attr_string = attr_string.lower().strip()

        if attr_string[:len('string')] == 'string':
            return cls(name)
        else:
            return None


class DateAttribute(Attribute):

    def __init__(self, name, date_format, datetime_unit):
        super().__init__(name)
        self.date_format = date_format
        self.datetime_unit = datetime_unit
        self.type_name = 'date'
        self.range = date_format
        self.dtype = np.datetime64(0, self.datetime_unit)

    @staticmethod
    def _get_date_format(atrv):
        m = r_date.match(atrv)
        if m:
            pattern = m.group(1).strip()
            # convert time pattern from Java's SimpleDateFormat to C's format
            datetime_unit = None
            if "yyyy" in pattern:
                pattern = pattern.replace("yyyy", "%Y")
                datetime_unit = "Y"
            elif "yy":
                pattern = pattern.replace("yy", "%y")
                datetime_unit = "Y"
            if "MM" in pattern:
                pattern = pattern.replace("MM", "%m")
                datetime_unit = "M"
            if "dd" in pattern:
                pattern = pattern.replace("dd", "%d")
                datetime_unit = "D"
            if "HH" in pattern:
                pattern = pattern.replace("HH", "%H")
                datetime_unit = "h"
            if "mm" in pattern:
                pattern = pattern.replace("mm", "%M")
                datetime_unit = "m"
            if "ss" in pattern:
                pattern = pattern.replace("ss", "%S")
                datetime_unit = "s"
            if "z" in pattern or "Z" in pattern:
                raise ValueError("Date type attributes with time zone not "
                                 "supported, yet")

            if datetime_unit is None:
                raise ValueError("Invalid or unsupported date format")

            return pattern, datetime_unit
        else:
            raise ValueError("Invalid or no date format")

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """

        attr_string_lower = attr_string.lower().strip()

        if attr_string_lower[:len('date')] == 'date':
            date_format, datetime_unit = cls._get_date_format(attr_string)
            return cls(name, date_format, datetime_unit)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        date_str = data_str.strip().strip("'").strip('"')
        if date_str == '?':
            return np.datetime64('NaT', self.datetime_unit)
        else:
            dt = datetime.datetime.strptime(date_str, self.date_format)
            return np.datetime64(dt).astype(
                "datetime64[%s]" % self.datetime_unit)

    def __str__(self):
        return super().__str__() + ',' + self.date_format


class RelationalAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'relational'
        self.dtype = np.object_
        self.attributes = []
        self.dialect = None

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """

        attr_string_lower = attr_string.lower().strip()

        if attr_string_lower[:len('relational')] == 'relational':
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        # Copy-pasted
        elems = list(range(len(self.attributes)))

        escaped_string = data_str.encode().decode("unicode-escape")

        row_tuples = []

        for raw in escaped_string.split("\n"):
            row, self.dialect = split_data_line(raw, self.dialect)

            row_tuples.append(tuple(
                [self.attributes[i].parse_data(row[i]) for i in elems]))

        return np.array(row_tuples,
                        [(a.name, a.dtype) for a in self.attributes])

    def __str__(self):
        return (super().__str__() + '\n\t' +
                '\n\t'.join(str(a) for a in self.attributes))


# -----------------
# Various utilities
# -----------------
def to_attribute(name, attr_string):
    attr_classes = (NominalAttribute, NumericAttribute, DateAttribute,
                    StringAttribute, RelationalAttribute)

    for cls in attr_classes:
        attr = cls.parse_attribute(name, attr_string)
        if attr is not None:
            return attr

    raise ParseArffError("unknown attribute %s" % attr_string)


def csv_sniffer_has_bug_last_field():
    """
    Checks if the bug https://bugs.python.org/issue30157 is unpatched.
    """

    # We only compute this once.
    has_bug = getattr(csv_sniffer_has_bug_last_field, "has_bug", None)

    if has_bug is None:
        dialect = csv.Sniffer().sniff("3, 'a'")
        csv_sniffer_has_bug_last_field.has_bug = dialect.quotechar != "'"
        has_bug = csv_sniffer_has_bug_last_field.has_bug

    return has_bug


def workaround_csv_sniffer_bug_last_field(sniff_line, dialect, delimiters):
    """
    Workaround for the bug https://bugs.python.org/issue30157 if is unpatched.
    """
    if csv_sniffer_has_bug_last_field():
        # Reuses code from the csv module
        right_regex = r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'

        for restr in (r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?P=delim)',  # ,".*?",
                      r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?P<delim>[^\w\n"\'])(?P<space> ?)',  # .*?",
                      right_regex,  # ,".*?"
                      r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'):  # ".*?" (no delim, no space)
            regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
            matches = regexp.findall(sniff_line)
            if matches:
                break

        # If it does not match the expression that was bugged, then this bug does not apply
        if restr != right_regex:
            return

        groupindex = regexp.groupindex

        # There is only one end of the string
        assert len(matches) == 1
        m = matches[0]

        n = groupindex['quote'] - 1
        quote = m[n]

        n = groupindex['delim'] - 1
        delim = m[n]

        n = groupindex['space'] - 1
        space = bool(m[n])

        dq_regexp = re.compile(
            r"((%(delim)s)|^)\W*%(quote)s[^%(delim)s\n]*%(quote)s[^%(delim)s\n]*%(quote)s\W*((%(delim)s)|$)" %
            {'delim': re.escape(delim), 'quote': quote}, re.MULTILINE
        )

        doublequote = bool(dq_regexp.search(sniff_line))

        dialect.quotechar = quote
        if delim in delimiters:
            dialect.delimiter = delim
        dialect.doublequote = doublequote
        dialect.skipinitialspace = space


def split_data_line(line, dialect=None):
    delimiters = ",\t"

    # This can not be done in a per reader basis, and relational fields
    # can be HUGE
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

    # Remove the line end if any
    if line[-1] == '\n':
        line = line[:-1]
    
    # Remove potential trailing whitespace
    line = line.strip()
    
    sniff_line = line

    # Add a delimiter if none is present, so that the csv.Sniffer
    # does not complain for a single-field CSV.
    if not any(d in line for d in delimiters):
        sniff_line += ","

    if dialect is None:
        dialect = csv.Sniffer().sniff(sniff_line, delimiters=delimiters)
        workaround_csv_sniffer_bug_last_field(sniff_line=sniff_line,
                                              dialect=dialect,
                                              delimiters=delimiters)

    row = next(csv.reader([line], dialect))

    return row, dialect


# --------------
# Parsing header
# --------------
def tokenize_attribute(iterable, attribute):
    """Parse a raw string in header (e.g., starts by @attribute).

    Given a raw string attribute, try to get the name and type of the
    attribute. Constraints:

    * The first line must start with @attribute (case insensitive, and
      space like characters before @attribute are allowed)
    * Works also if the attribute is spread on multilines.
    * Works if empty lines or comments are in between

    Parameters
    ----------
    attribute : str
       the attribute string.

    Returns
    -------
    name : str
       name of the attribute
    value : str
       value of the attribute
    next : str
       next line to be parsed

    Examples
    --------
    If attribute is a string defined in python as r"floupi real", will
    return floupi as name, and real as value.

    >>> iterable = iter([0] * 10) # dummy iterator
    >>> tokenize_attribute(iterable, r"@attribute floupi real")
    ('floupi', 'real', 0)

    If attribute is r"'floupi 2' real", will return 'floupi 2' as name,
    and real as value.

    >>> tokenize_attribute(iterable, r"  @attribute 'floupi 2' real   ")
    ('floupi 2', 'real', 0)

    """
    sattr = attribute.strip()
    mattr = r_attribute.match(sattr)
    if mattr:
        # atrv is everything after @attribute
        atrv = mattr.group(1)
        if r_comattrval.match(atrv):
            name, type = tokenize_single_comma(atrv)
            next_item = next(iterable)
        elif r_wcomattrval.match(atrv):
            name, type = tokenize_single_wcomma(atrv)
            next_item = next(iterable)
        else:
            # Not sure we should support this, as it does not seem supported by
            # weka.
            raise ValueError("multi line not supported yet")
    else:
        raise ValueError("First line unparsable: %s" % sattr)

    attribute = to_attribute(name, type)

    if type.lower() == 'relational':
        next_item = read_relational_attribute(iterable, attribute, next_item)
    #    raise ValueError("relational attributes not supported yet")

    return attribute, next_item


def tokenize_single_comma(val):
    # XXX we match twice the same string (here and at the caller level). It is
    # stupid, but it is easier for now...
    m = r_comattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError("Error while tokenizing attribute") from e
    else:
        raise ValueError("Error while tokenizing single %s" % val)
    return name, type


def tokenize_single_wcomma(val):
    # XXX we match twice the same string (here and at the caller level). It is
    # stupid, but it is easier for now...
    m = r_wcomattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError("Error while tokenizing attribute") from e
    else:
        raise ValueError("Error while tokenizing single %s" % val)
    return name, type


def read_relational_attribute(ofile, relational_attribute, i):
    """Read the nested attributes of a relational attribute"""

    r_end_relational = re.compile(r'^@[Ee][Nn][Dd]\s*' +
                                  relational_attribute.name + r'\s*$')

    while not r_end_relational.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                attr, i = tokenize_attribute(ofile, i)
                relational_attribute.attributes.append(attr)
            else:
                raise ValueError("Error parsing line %s" % i)
        else:
            i = next(ofile)

    i = next(ofile)
    return i


def read_header(ofile):
    """Read the header of the iterable ofile."""
    i = next(ofile)

    # Pass first comments
    while r_comment.match(i):
        i = next(ofile)

    # Header is everything up to DATA attribute ?
    relation = None
    attributes = []
    while not r_datameta.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                attr, i = tokenize_attribute(ofile, i)
                attributes.append(attr)
            else:
                isrel = r_relation.match(i)
                if isrel:
                    relation = isrel.group(1)
                else:
                    raise ValueError("Error parsing line %s" % i)
                i = next(ofile)
        else:
            i = next(ofile)

    return relation, attributes


class MetaData:
    """Small container to keep useful information on a ARFF dataset.

    Knows about attributes names and types.

    Examples
    --------
    ::

        data, meta = loadarff('iris.arff')
        # This will print the attributes names of the iris.arff dataset
        for i in meta:
            print(i)
        # This works too
        meta.names()
        # Getting attribute type
        types = meta.types()

    Methods
    -------
    names
    types

    Notes
    -----
    Also maintains the list of attributes in order, i.e., doing for i in
    meta, where meta is an instance of MetaData, will return the
    different attribute names in the order they were defined.
    """
    def __init__(self, rel, attr):
        self.name = rel
        self._attributes = {a.name: a for a in attr}

    def __repr__(self):
        msg = ""
        msg += "Dataset: %s\n" % self.name
        for i in self._attributes:
            msg += "\t%s's type is %s" % (i, self._attributes[i].type_name)
            if self._attributes[i].range:
                msg += ", range is %s" % str(self._attributes[i].range)
            msg += '\n'
        return msg

    def __iter__(self):
        return iter(self._attributes)

    def __getitem__(self, key):
        attr = self._attributes[key]

        return (attr.type_name, attr.range)

    def names(self):
        """Return the list of attribute names.

        Returns
        -------
        attrnames : list of str
            The attribute names.
        """
        return list(self._attributes)

    def types(self):
        """Return the list of attribute types.

        Returns
        -------
        attr_types : list of str
            The attribute types.
        """
        attr_types = [self._attributes[name].type_name
                      for name in self._attributes]
        return attr_types


def loadarff(f):
    """
    Read an arff file.

    The data is returned as a record array, which can be accessed much like
    a dictionary of NumPy arrays. For example, if one of the attributes is
    called 'pressure', then its first 10 data points can be accessed from the
    ``data`` record array like so: ``data['pressure'][0:10]``


    Parameters
    ----------
    f : file-like or str
       File-like object to read from, or filename to open.

    Returns
    -------
    data : record array
       The data of the arff file, accessible by attribute names.
    meta : `MetaData`
       Contains information about the arff file such as name and
       type of attributes, the relation (name of the dataset), etc.

    Raises
    ------
    ParseArffError
        This is raised if the given file is not ARFF-formatted.
    NotImplementedError
        The ARFF file has an attribute which is not supported yet.

    Notes
    -----

    This function should be able to read most arff files. Not
    implemented functionality include:

    * date type attributes
    * string type attributes

    It can read files with numeric and nominal attributes. It cannot read
    files with sparse data ({} in the file). However, this function can
    read files with missing data (? in the file), representing the data
    points as NaNs.

    Examples
    --------
    >>> from scipy.io import arff
    >>> from io import StringIO
    >>> content = \"\"\"
    ... @relation foo
    ... @attribute width  numeric
    ... @attribute height numeric
    ... @attribute color  {red,green,blue,yellow,black}
    ... @data
    ... 5.0,3.25,blue
    ... 4.5,3.75,green
    ... 3.0,4.00,red
    ... \"\"\"
    >>> f = StringIO(content)
    >>> data, meta = arff.loadarff(f)
    >>> data
    array([(5.0, 3.25, 'blue'), (4.5, 3.75, 'green'), (3.0, 4.0, 'red')],
          dtype=[('width', '<f8'), ('height', '<f8'), ('color', '|S6')])
    >>> meta
    Dataset: foo
    \twidth's type is numeric
    \theight's type is numeric
    \tcolor's type is nominal, range is ('red', 'green', 'blue', 'yellow', 'black')

    """
    if hasattr(f, 'read'):
        ofile = f
    else:
        ofile = open(f, 'rt')
    try:
        return _loadarff(ofile)
    finally:
        if ofile is not f:  # only close what we opened
            ofile.close()


def _loadarff(ofile):
    # Parse the header file
    try:
        rel, attr = read_header(ofile)
    except ValueError as e:
        msg = "Error while parsing header, error was: " + str(e)
        raise ParseArffError(msg) from e

    # Check whether we have a string attribute (not supported yet)
    hasstr = False
    for a in attr:
        if isinstance(a, StringAttribute):
            hasstr = True

    meta = MetaData(rel, attr)

    # XXX The following code is not great
    # Build the type descriptor descr and the list of convertors to convert
    # each attribute to the suitable type (which should match the one in
    # descr).

    # This can be used once we want to support integer as integer values and
    # not as numeric anymore (using masked arrays ?).

    if hasstr:
        # How to support string efficiently ? Ideally, we should know the max
        # size of the string before allocating the numpy array.
        raise NotImplementedError("String attributes not supported yet, sorry")

    ni = len(attr)

    def generator(row_iter, delim=','):
        # TODO: this is where we are spending time (~80%). I think things
        # could be made more efficiently:
        #   - We could for example "compile" the function, because some values
        #   do not change here.
        #   - The function to convert a line to dtyped values could also be
        #   generated on the fly from a string and be executed instead of
        #   looping.
        #   - The regex are overkill: for comments, checking that a line starts
        #   by % should be enough and faster, and for empty lines, same thing
        #   --> this does not seem to change anything.

        # 'compiling' the range since it does not change
        # Note, I have already tried zipping the converters and
        # row elements and got slightly worse performance.
        elems = list(range(ni))

        dialect = None
        for raw in row_iter:
            # We do not abstract skipping comments and empty lines for
            # performance reasons.
            if r_comment.match(raw) or r_empty.match(raw):
                continue

            row, dialect = split_data_line(raw, dialect)

            yield tuple([attr[i].parse_data(row[i]) for i in elems])

    a = list(generator(ofile))
    # No error should happen here: it is a bug otherwise
    data = np.array(a, [(a.name, a.dtype) for a in attr])
    return data, meta


# ----
# Misc
# ----
def basic_stats(data):
    nbfac = data.size * 1. / (data.size - 1)
    return np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac


def print_attribute(name, tp, data):
    type = tp.type_name
    if type == 'numeric' or type == 'real' or type == 'integer':
        min, max, mean, std = basic_stats(data)
        print("%s,%s,%f,%f,%f,%f" % (name, type, min, max, mean, std))
    else:
        print(str(tp))


def test_weka(filename):
    data, meta = loadarff(filename)
    print(len(data.dtype))
    print(data.size)
    for i in meta:
        print_attribute(i, meta[i], data[i])


# make sure nose does not find this as a test
test_weka.__test__ = False


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    test_weka(filename)
