__all__ = ['python_to_notebook', 'Notebook']

import json
import copy
import warnings


# Skeleton notebook in JSON format
skeleton_nb = """{
    "metadata": {
    "name":""
    },
    "nbformat": 3,
    "nbformat_minor": 0,
    "worksheets": [
        {
            "cells": [
            {
                "cell_type": "code",
                "collapsed": false,
                "input": [
                    "%matplotlib inline"
                ],
                "language": "python",
                "metadata": {},
                "outputs": []
            }
            ],
          "metadata": {}
        }
    ]
}"""


class Notebook(object):
    """
    Notebook object for building an IPython notebook cell-by-cell.
    """

    def __init__(self):
        # cell type code
        self.cell_code = {
            'cell_type': 'code',
            'collapsed': False,
            'input': [
                '# Code Goes Here'
            ],
            'language': 'python',
            'metadata': {},
            'outputs': []
        }

        # cell type markdown
        self.cell_md = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                'Markdown Goes Here'
            ]
        }

        self.template = json.loads(skeleton_nb)
        self.cell_type = {'input': self.cell_code, 'source': self.cell_md}
        self.valuetype_to_celltype = {'code': 'input', 'markdown': 'source'}

    def add_cell(self, value, cell_type='code'):
        """Add a notebook cell.

        Parameters
        ----------
        value : str
            Cell content.
        cell_type : {'code', 'markdown'}
            Type of content (default is 'code').

        """
        if cell_type in ['markdown', 'code']:
            key = self.valuetype_to_celltype[cell_type]
            cells = self.template['worksheets'][0]['cells']
            cells.append(copy.deepcopy(self.cell_type[key]))
            # assign value to the last cell
            cells[-1][key] = value
        else:
            warnings.warn('Ignoring unsupported cell type (%s)' % cell_type)

    def json(self):
        """Return a JSON representation of the notebook.

        Returns
        -------
        str
            JSON notebook.

        """
        return json.dumps(self.template, indent=2)


