import json

from ..notebook_doc import Notebook, skeleton_nb


def test_notebook_basic():
    nb = Notebook()
    assert json.loads(nb.json()) == json.loads(skeleton_nb)


def test_notebook_add():
    nb = Notebook()

    str1 = 'hello world'
    str2 = 'f = lambda x: x * x'

    nb.add_cell(str1, cell_type='markdown')
    nb.add_cell(str2, cell_type='code')

    d = json.loads(nb.json())
    cells = d['worksheets'][0]['cells']
    values = [c['input'] if c['cell_type'] == 'code' else c['source']
              for c in cells]

    assert values[1] == str1
    assert values[2] == str2

    assert cells[1]['cell_type'] == 'markdown'
    assert cells[2]['cell_type'] == 'code'

