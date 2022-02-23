# period frequency constants corresponding to scikits timeseries
# originals
from enum import Enum


cdef class PeriodDtypeBase:
    """
    Similar to an actual dtype, this contains all of the information
    describing a PeriodDtype in an integer code.
    """
    # cdef readonly:
    #    PeriodDtypeCode _dtype_code

    def __cinit__(self, PeriodDtypeCode code):
        self._dtype_code = code

    def __eq__(self, other):
        if not isinstance(other, PeriodDtypeBase):
            return False
        if not isinstance(self, PeriodDtypeBase):
            # cython semantics, this is a reversed op
            return False
        return self._dtype_code == other._dtype_code

    @property
    def freq_group_code(self) -> int:
        # See also: libperiod.get_freq_group
        return (self._dtype_code // 1000) * 1000

    @property
    def date_offset(self):
        """
        Corresponding DateOffset object.

        This mapping is mainly for backward-compatibility.
        """
        from .offsets import to_offset

        freqstr = _reverse_period_code_map.get(self._dtype_code)

        return to_offset(freqstr)

    @classmethod
    def from_date_offset(cls, offset):
        code = offset._period_dtype_code
        return cls(code)


_period_code_map = {
    # Annual freqs with various fiscal year ends.
    # eg, 2005 for A-FEB runs Mar 1, 2004 to Feb 28, 2005
    "A-DEC": 1000,  # Annual - December year end
    "A-JAN": 1001,  # Annual - January year end
    "A-FEB": 1002,  # Annual - February year end
    "A-MAR": 1003,  # Annual - March year end
    "A-APR": 1004,  # Annual - April year end
    "A-MAY": 1005,  # Annual - May year end
    "A-JUN": 1006,  # Annual - June year end
    "A-JUL": 1007,  # Annual - July year end
    "A-AUG": 1008,  # Annual - August year end
    "A-SEP": 1009,  # Annual - September year end
    "A-OCT": 1010,  # Annual - October year end
    "A-NOV": 1011,  # Annual - November year end

    # Quarterly frequencies with various fiscal year ends.
    # eg, Q42005 for Q-OCT runs Aug 1, 2005 to Oct 31, 2005
    "Q-DEC": 2000,    # Quarterly - December year end
    "Q-JAN": 2001,    # Quarterly - January year end
    "Q-FEB": 2002,    # Quarterly - February year end
    "Q-MAR": 2003,    # Quarterly - March year end
    "Q-APR": 2004,    # Quarterly - April year end
    "Q-MAY": 2005,    # Quarterly - May year end
    "Q-JUN": 2006,    # Quarterly - June year end
    "Q-JUL": 2007,    # Quarterly - July year end
    "Q-AUG": 2008,    # Quarterly - August year end
    "Q-SEP": 2009,    # Quarterly - September year end
    "Q-OCT": 2010,    # Quarterly - October year end
    "Q-NOV": 2011,    # Quarterly - November year end

    "M": 3000,        # Monthly

    "W-SUN": 4000,    # Weekly - Sunday end of week
    "W-MON": 4001,    # Weekly - Monday end of week
    "W-TUE": 4002,    # Weekly - Tuesday end of week
    "W-WED": 4003,    # Weekly - Wednesday end of week
    "W-THU": 4004,    # Weekly - Thursday end of week
    "W-FRI": 4005,    # Weekly - Friday end of week
    "W-SAT": 4006,    # Weekly - Saturday end of week

    "B": 5000,        # Business days
    "D": 6000,        # Daily
    "H": 7000,        # Hourly
    "T": 8000,        # Minutely
    "S": 9000,        # Secondly
    "L": 10000,       # Millisecondly
    "U": 11000,       # Microsecondly
    "N": 12000,       # Nanosecondly
}

_reverse_period_code_map = {
    _period_code_map[key]: key for key in _period_code_map}

# Yearly aliases; careful not to put these in _reverse_period_code_map
_period_code_map.update({"Y" + key[1:]: _period_code_map[key]
                         for key in _period_code_map
                         if key.startswith("A-")})

_period_code_map.update({
    "Q": 2000,   # Quarterly - December year end (default quarterly)
    "A": 1000,   # Annual
    "W": 4000,   # Weekly
    "C": 5000,   # Custom Business Day
})

cdef set _month_names = {
    x.split("-")[-1] for x in _period_code_map.keys() if x.startswith("A-")
}

# Map attribute-name resolutions to resolution abbreviations
_attrname_to_abbrevs = {
    "year": "A",
    "quarter": "Q",
    "month": "M",
    "day": "D",
    "hour": "H",
    "minute": "T",
    "second": "S",
    "millisecond": "L",
    "microsecond": "U",
    "nanosecond": "N",
}
cdef dict attrname_to_abbrevs = _attrname_to_abbrevs
cdef dict _abbrev_to_attrnames = {v: k for k, v in attrname_to_abbrevs.items()}


class FreqGroup(Enum):
    # Mirrors c_FreqGroup in the .pxd file
    FR_ANN = 1000
    FR_QTR = 2000
    FR_MTH = 3000
    FR_WK = 4000
    FR_BUS = 5000
    FR_DAY = 6000
    FR_HR = 7000
    FR_MIN = 8000
    FR_SEC = 9000
    FR_MS = 10000
    FR_US = 11000
    FR_NS = 12000
    FR_UND = -10000  # undefined

    @staticmethod
    def get_freq_group(code: int) -> "FreqGroup":
        # See also: PeriodDtypeBase.freq_group_code
        code = (code // 1000) * 1000
        return FreqGroup(code)


class Resolution(Enum):

    # Note: cython won't allow us to reference the cdef versions at the
    # module level
    RESO_NS = 0
    RESO_US = 1
    RESO_MS = 2
    RESO_SEC = 3
    RESO_MIN = 4
    RESO_HR = 5
    RESO_DAY = 6
    RESO_MTH = 7
    RESO_QTR = 8
    RESO_YR = 9

    def __lt__(self, other):
        return self.value < other.value

    def __ge__(self, other):
        return self.value >= other.value

    @property
    def freq_group(self) -> FreqGroup:
        if self == Resolution.RESO_NS:
            return FreqGroup.FR_NS
        elif self == Resolution.RESO_US:
            return FreqGroup.FR_US
        elif self == Resolution.RESO_MS:
            return FreqGroup.FR_MS
        elif self == Resolution.RESO_SEC:
            return FreqGroup.FR_SEC
        elif self == Resolution.RESO_MIN:
            return FreqGroup.FR_MIN
        elif self == Resolution.RESO_HR:
            return FreqGroup.FR_HR
        elif self == Resolution.RESO_DAY:
            return FreqGroup.FR_DAY
        elif self == Resolution.RESO_MTH:
            return FreqGroup.FR_MTH
        elif self == Resolution.RESO_QTR:
            return FreqGroup.FR_QTR
        elif self == Resolution.RESO_YR:
            return FreqGroup.FR_ANN
        else:
            raise ValueError(self)

    @property
    def attrname(self) -> str:
        """
        Return datetime attribute name corresponding to this Resolution.

        Examples
        --------
        >>> Resolution.RESO_SEC.attrname
        'second'
        """
        return _reso_str_map[self.value]

    @classmethod
    def from_attrname(cls, attrname: str) -> "Resolution":
        """
        Return resolution str against resolution code.

        Examples
        --------
        >>> Resolution.from_attrname('second')
        2

        >>> Resolution.from_attrname('second') == Resolution.RESO_SEC
        True
        """
        return cls(_str_reso_map[attrname])

    @classmethod
    def get_reso_from_freq(cls, freq: str) -> "Resolution":
        """
        Return resolution code against frequency str.

        `freq` is given by the `offset.freqstr` for some DateOffset object.

        Examples
        --------
        >>> Resolution.get_reso_from_freq('H')
        4

        >>> Resolution.get_reso_from_freq('H') == Resolution.RESO_HR
        True
        """
        try:
            attr_name = _abbrev_to_attrnames[freq]
        except KeyError:
            # For quarterly and yearly resolutions, we need to chop off
            #  a month string.
            split_freq = freq.split("-")
            if len(split_freq) != 2:
                raise
            if split_freq[1] not in _month_names:
                # i.e. we want e.g. "Q-DEC", not "Q-INVALID"
                raise
            attr_name = _abbrev_to_attrnames[split_freq[0]]

        return cls.from_attrname(attr_name)


cdef dict _reso_str_map = {
    Resolution.RESO_NS.value: "nanosecond",
    Resolution.RESO_US.value: "microsecond",
    Resolution.RESO_MS.value: "millisecond",
    Resolution.RESO_SEC.value: "second",
    Resolution.RESO_MIN.value: "minute",
    Resolution.RESO_HR.value: "hour",
    Resolution.RESO_DAY.value: "day",
    Resolution.RESO_MTH.value: "month",
    Resolution.RESO_QTR.value: "quarter",
    Resolution.RESO_YR.value: "year",
}

cdef dict _str_reso_map = {v: k for k, v in _reso_str_map.items()}
