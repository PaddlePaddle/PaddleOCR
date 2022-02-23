"""
Data IO api
"""

# flake8: noqa

from pandas.io.clipboards import read_clipboard
from pandas.io.excel import (
    ExcelFile,
    ExcelWriter,
    read_excel,
)
from pandas.io.feather_format import read_feather
from pandas.io.gbq import read_gbq
from pandas.io.html import read_html
from pandas.io.json import read_json
from pandas.io.orc import read_orc
from pandas.io.parquet import read_parquet
from pandas.io.parsers import (
    read_csv,
    read_fwf,
    read_table,
)
from pandas.io.pickle import (
    read_pickle,
    to_pickle,
)
from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)
from pandas.io.sas import read_sas
from pandas.io.spss import read_spss
from pandas.io.sql import (
    read_sql,
    read_sql_query,
    read_sql_table,
)
from pandas.io.stata import read_stata
from pandas.io.xml import read_xml
