"""
Example generation from python files.

Generate the rst files for the examples by iterating over the python
example files. Files that generate images should start with 'plot'.

To generate your own examples, add this extension to the list of
``extensions``in your Sphinx configuration file. In addition, make sure the
example directory(ies) in `plot2rst_paths` (see below) points to a directory
with examples named `plot_*.py` and include an `index.rst` file.

This code was adapted from scikit-image, which took it from scikit-learn.

Options
-------
The ``plot2rst`` extension accepts the following options:

plot2rst_paths : length-2 tuple, or list of tuples
    Tuple or list of tuples of paths to (python plot, generated rst) files,
    i.e. (source, destination).  Note that both paths are relative to Sphinx
    'source' directory. Defaults to ('../examples', 'auto_examples')

plot2rst_rcparams : dict
    Matplotlib configuration parameters. See
    https://matplotlib.org/tutorials/introductory/customizing.html for details.

plot2rst_default_thumb : str
    Path (relative to doc root) of default thumbnail image.

plot2rst_thumb_shape : float
    Shape of thumbnail in pixels. The image is resized to fit within this shape
    and the excess is filled with white pixels. This fixed size ensures that
    that gallery images are displayed in a grid.

plot2rst_plot_tag : str
    When this tag is found in the example file, the current plot is saved and
    tag is replaced with plot path. Defaults to 'PLOT2RST.current_figure'.


Suggested CSS definitions
-------------------------

    div.body h2 {
        border-bottom: 1px solid #BBB;
        clear: left;
    }

    /*---- example gallery ----*/

    .gallery.figure {
        float: left;
        margin: 1em;
    }

    .gallery.figure img{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px;
    }

    .gallery.figure .caption {
        width: 200px;
        text-align: center !important;
    }

"""
import os
import re
import shutil
import token
import tokenize
import traceback
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import io
from skimage import transform
from skimage.util.dtype import dtype_range

from notebook_doc import Notebook

from docutils.core import publish_parts
from sphinx.domains.python import PythonDomain

LITERALINCLUDE = """
.. literalinclude:: {src_name}
    :lines: {code_start}-

"""

CODE_LINK = """

**Python source code:** :download:`download <{0}>`
(generated using ``skimage`` |version|)

"""

NOTEBOOK_LINK = """

**IPython Notebook:** :download:`download <{0}>`
(generated using ``skimage`` |version|)

"""

TOCTREE_TEMPLATE = """
.. toctree::
   :hidden:

   %s

"""

IMAGE_TEMPLATE = """
.. image:: images/%s
    :align: center

"""

GALLERY_IMAGE_TEMPLATE = """
.. figure:: %(thumb)s
   :figclass: gallery
   :target: ./%(source)s.html

   :ref:`example_%(link_name)s`

"""


class Path(str):
    """Path object for manipulating directory and file paths."""

    def __new__(self, path):
        return str.__new__(self, path)

    @property
    def isdir(self):
        return os.path.isdir(self)

    @property
    def exists(self):
        """Return True if path exists"""
        return os.path.exists(self)

    def pjoin(self, *args):
        """Join paths. `p` prefix prevents confusion with string method."""
        return self.__class__(os.path.join(self, *args))

    def psplit(self):
        """Split paths. `p` prefix prevents confusion with string method."""
        return [self.__class__(p) for p in os.path.split(self)]

    def makedirs(self):
        if not self.exists:
            os.makedirs(self)

    def listdir(self):
        return os.listdir(self)

    def format(self, *args, **kwargs):
        return self.__class__(super(Path, self).format(*args, **kwargs))

    def __add__(self, other):
        return self.__class__(super(Path, self).__add__(other))

    def __iadd__(self, other):
        return self.__add__(other)


def setup(app):
    app.connect('builder-inited', generate_example_galleries)

    app.add_config_value('plot2rst_paths',
                         ('../examples', 'auto_examples'), True)
    app.add_config_value('plot2rst_rcparams', {}, True)
    app.add_config_value('plot2rst_default_thumb', None, True)
    app.add_config_value('plot2rst_thumb_shape', (250, 300), True)
    app.add_config_value('plot2rst_plot_tag', 'PLOT2RST.current_figure', True)
    app.add_config_value('plot2rst_index_name', 'index', True)


def generate_example_galleries(app):
    cfg = app.builder.config

    if isinstance(cfg.source_suffix, list):
        cfg.source_suffix_str = cfg.source_suffix[0]
    else:
        cfg.source_suffix_str = cfg.source_suffix

    doc_src = Path(os.path.abspath(app.builder.srcdir)) # path/to/doc/source

    if isinstance(cfg.plot2rst_paths, tuple):
        cfg.plot2rst_paths = [cfg.plot2rst_paths]
    for src_dest in cfg.plot2rst_paths:
        plot_path, rst_path = [Path(p) for p in src_dest]
        example_dir = doc_src.pjoin(plot_path)
        rst_dir = doc_src.pjoin(rst_path)
        generate_examples_and_gallery(example_dir, rst_dir, cfg)


def generate_examples_and_gallery(example_dir, rst_dir, cfg):
    """Generate rst from examples and create gallery to showcase examples."""
    if not example_dir.exists:
        print("No example directory found at", example_dir)
        return
    rst_dir.makedirs()

    # we create an index.rst with all examples
    with open(rst_dir.pjoin('index'+cfg.source_suffix_str), 'w') as gallery_index:
        # Here we don't use an os.walk, but we recurse only twice: flat is
        # better than nested.
        write_gallery(gallery_index, example_dir, rst_dir, cfg)
        for d in sorted(example_dir.listdir()):
            example_sub = example_dir.pjoin(d)
            if example_sub.isdir:
                rst_sub = rst_dir.pjoin(d)
                rst_sub.makedirs()
                write_gallery(gallery_index, example_sub, rst_sub, cfg, depth=1)
        gallery_index.flush()


def write_gallery(gallery_index, src_dir, rst_dir, cfg, depth=0):
    """Generate the rst files for an example directory, i.e. gallery.

    Write rst files from python examples and add example links to gallery.

    Parameters
    ----------
    gallery_index : file
        Index file for plot gallery.
    src_dir : 'str'
        Source directory for python examples.
    rst_dir : 'str'
        Destination directory for rst files generated from python examples.
    cfg : config object
        Sphinx config object created by Sphinx.
    """
    index_name = cfg.plot2rst_index_name + cfg.source_suffix_str
    gallery_template = src_dir.pjoin(index_name)
    if not os.path.exists(gallery_template):
        print(src_dir)
        print(80*'_')
        print('Example directory %s does not have a %s file'
                        % (src_dir, index_name))
        print('Skipping this directory')
        print(80*'_')
        return

    with open(gallery_template) as f:
        gallery_description = f.read()
    gallery_index.write('\n\n%s\n\n' % gallery_description)

    rst_dir.makedirs()
    examples = [fname for fname in sorted(src_dir.listdir(), key=_plots_first)
                      if fname.endswith('py')]
    ex_names = [ex[:-3] for ex in examples] # strip '.py' extension
    if depth == 0:
        sub_dir = Path('')
    else:
        sub_dir_list = src_dir.psplit()[-depth:]
        sub_dir = Path('/'.join(sub_dir_list) + '/')
    joiner = '\n   %s' % sub_dir
    gallery_index.write(TOCTREE_TEMPLATE % (sub_dir + joiner.join(ex_names)))

    for src_name in examples:

        try:
            write_example(src_name, src_dir, rst_dir, cfg)
        except Exception:
            print("Exception raised while running:")
            print("%s in %s" % (src_name, src_dir))
            print('~' * 60)
            traceback.print_exc()
            print('~' * 60)
            continue

        link_name = sub_dir.pjoin(src_name)
        link_name = link_name.replace(os.path.sep, '_')
        if link_name.startswith('._'):
            link_name = link_name[2:]

        info = {}
        info['thumb'] = sub_dir.pjoin('images/thumb', src_name[:-3] + '.png')
        info['source'] = sub_dir + src_name[:-3]
        info['link_name'] = link_name
        gallery_index.write(GALLERY_IMAGE_TEMPLATE % info)


def _plots_first(fname):
    """Decorate filename so that examples with plots are displayed first."""
    if not (fname.startswith('plot') and fname.endswith('.py')):
        return 'zz' + fname
    return fname


def write_example(src_name, src_dir, rst_dir, cfg):
    """Write rst file from a given python example.

    Parameters
    ----------
    src_name : str
        Name of example file.
    src_dir : 'str'
        Source directory for python examples.
    rst_dir : 'str'
        Destination directory for rst files generated from python examples.
    cfg : config object
        Sphinx config object created by Sphinx.
    """
    last_dir = src_dir.psplit()[-1]
    # to avoid leading . in file names, and wrong names in links
    if last_dir == '.' or last_dir == 'examples':
        last_dir = Path('')
    else:
        last_dir += '_'

    src_path = src_dir.pjoin(src_name)
    example_file = rst_dir.pjoin(src_name)
    shutil.copyfile(src_path, example_file)

    image_dir = rst_dir.pjoin('images')
    thumb_dir = image_dir.pjoin('thumb')
    notebook_dir = rst_dir.pjoin('notebook')
    image_dir.makedirs()
    thumb_dir.makedirs()
    notebook_dir.makedirs()

    base_image_name = os.path.splitext(src_name)[0]
    image_path = image_dir.pjoin(base_image_name + '_{0}.png')

    basename, py_ext = os.path.splitext(src_name)

    rst_path = rst_dir.pjoin(basename + cfg.source_suffix_str)
    notebook_path = notebook_dir.pjoin(basename + '.ipynb')

    if _plots_are_current(src_path, image_path) and rst_path.exists and \
        notebook_path.exists:
        return

    print('plot2rst: %s' % basename)

    blocks = split_code_and_text_blocks(example_file)
    if blocks[0][2].startswith('#!'):
        blocks.pop(0) # don't add shebang line to rst file.

    rst_link = '.. _example_%s:\n\n' % (last_dir + src_name)
    figure_list, rst = process_blocks(blocks, src_path, image_path, cfg)

    has_inline_plots = any(cfg.plot2rst_plot_tag in b[2] for b in blocks)
    if has_inline_plots:
        example_rst = ''.join([rst_link, rst])
    else:
        # print first block of text, display all plots, then display code.
        first_text_block = [b for b in blocks if b[0] == 'text'][0]
        label, (start, end), content = first_text_block
        figure_list = save_all_figures(image_path)
        rst_blocks = [IMAGE_TEMPLATE % f.lstrip('/') for f in figure_list]

        example_rst = rst_link
        example_rst += eval(content)
        example_rst += ''.join(rst_blocks)
        code_info = dict(src_name=src_name, code_start=end)
        example_rst += LITERALINCLUDE.format(**code_info)

    example_rst += CODE_LINK.format(src_name)
    ipnotebook_name = src_name.replace('.py', '.ipynb')
    ipnotebook_name = './notebook/' + ipnotebook_name
    example_rst += NOTEBOOK_LINK.format(ipnotebook_name)

    with open(rst_path, 'w') as f:
        f.write(example_rst)

    thumb_path = thumb_dir.pjoin(src_name[:-3] + '.png')
    first_image_file = image_dir.pjoin(figure_list[0].lstrip('/'))
    if first_image_file.exists:
        first_image = io.imread(first_image_file)
        save_thumbnail(first_image, thumb_path, cfg.plot2rst_thumb_shape)

    if not thumb_path.exists:
        if cfg.plot2rst_default_thumb is None:
            print("WARNING: No plots found and default thumbnail not defined.")
            print("Specify 'plot2rst_default_thumb' in Sphinx config file.")
        else:
            shutil.copy(cfg.plot2rst_default_thumb, thumb_path)

    # Export example to IPython notebook
    nb = Notebook()

    # Add sphinx roles to the examples, otherwise docutils
    # cannot compile the ReST for the notebook
    sphinx_roles = PythonDomain.roles.keys()
    preamble = '\n'.join(f'.. role:: py:{role}(literal)\n'
                         for role in sphinx_roles)

    # Grab all references to inject them in cells where needed
    ref_regexp = re.compile('\n(\\.\\. \\[(\\d+)\\].*(?:\n[ ]{7,8}.*)+)')
    math_role_regexp = re.compile(':math:`(.*?)`')

    text = '\n'.join((content for (cell_type, _, content) in blocks
                     if cell_type != 'code'))

    references = re.findall(ref_regexp, text)

    for (cell_type, _, content) in blocks:
        if cell_type == 'code':
            nb.add_cell(content, cell_type='code')
        else:
            if content.startswith('r'):
                content = content.replace('r"""', '')
                escaped = False
            else:
                content = content.replace('"""', '')
                escaped = True

            if not escaped:
                content = content.replace("\\", "\\\\")

            content = content.replace('.. seealso::', '**See also:**')
            content = re.sub(math_role_regexp, r'$\1$', content)

            # Remove math directive when rendering notebooks
            # until we implement a smarter way of capturing and replacing
            # its content
            content = content.replace('.. math::', '')

            if not content.strip():
                continue

            content = (preamble + content).rstrip('\n')
            content = '\n'.join([line for line in content.split('\n') if
                                 not line.startswith('.. image')])

            # Remove reference links until we can figure out a better way to
            # preserve them
            for (reference, ref_id) in references:
                ref_tag = f'[{ref_id}]_'
                if ref_tag in content:
                    content = content.replace(ref_tag, ref_tag[:-1])

            html = publish_parts(content, writer_name='html')['html_body']
            nb.add_cell(html, cell_type='markdown')

    with open(notebook_path, 'w') as f:
        f.write(nb.json())


def save_thumbnail(image, thumb_path, shape):
    """Save image as a thumbnail with the specified shape.

    The image is first resized to fit within the specified shape and then
    centered in an array of the specified shape before saving.
    """
    rescale = min(float(w_1) / w_2 for w_1, w_2 in zip(shape, image.shape))
    small_shape = (rescale * np.asarray(image.shape[:2])).astype(int)
    small_image = transform.resize(image, small_shape)

    if len(image.shape) == 3:
        shape = shape + (image.shape[2],)
    background_value = dtype_range[small_image.dtype.type][1]
    thumb = background_value * np.ones(shape, dtype=small_image.dtype)

    i = (shape[0] - small_shape[0]) // 2
    j = (shape[1] - small_shape[1]) // 2
    thumb[i:i+small_shape[0], j:j+small_shape[1]] = small_image

    io.imsave(thumb_path, thumb)


def _plots_are_current(src_path, image_path):
    first_image_file = Path(image_path.format(1))
    needs_replot = (not first_image_file.exists or
                    _mod_time(first_image_file) <= _mod_time(src_path))
    return not needs_replot


def _mod_time(file_path):
    return os.stat(file_path).st_mtime


def split_code_and_text_blocks(source_file):
    """Return list with source file separated into code and text blocks.

    Returns
    -------
    blocks : list of (label, (start, end+1), content)
        List where each element is a tuple with the label ('text' or 'code'),
        the (start, end+1) line numbers, and content string of block.
    """
    block_edges, idx_first_text_block = get_block_edges(source_file)

    with open(source_file) as f:
        source_lines = f.readlines()

    # Every other block should be a text block
    idx_text_block = np.arange(idx_first_text_block, len(block_edges), 2)
    blocks = []
    slice_ranges = zip(block_edges[:-1], block_edges[1:])
    for i, (start, end) in enumerate(slice_ranges):
        block_label = 'text' if i in idx_text_block else 'code'
        # subtract 1 from indices b/c line numbers start at 1, not 0
        content = ''.join(source_lines[start-1:end-1])
        blocks.append((block_label, (start, end), content))
    return blocks


def get_block_edges(source_file):
    """Return starting line numbers of code and text blocks

    Returns
    -------
    block_edges : list of int
        Line number for the start of each block. Note the
    idx_first_text_block : {0 | 1}
        0 if first block is text then, else 1 (second block better be text).
    """
    block_edges = []
    with open(source_file) as f:
        token_iter = tokenize.generate_tokens(f.readline)
        for token_tuple in token_iter:
            t_id, t_str, (srow, scol), (erow, ecol), src_line = token_tuple
            if (token.tok_name[t_id] == 'STRING' and scol == 0):
                # Add one point to line after text (for later slicing)
                block_edges.extend((srow, erow+1))
    idx_first_text_block = 0
    # when example doesn't start with text block.
    if not block_edges[0] == 1:
        block_edges.insert(0, 1)
        idx_first_text_block = 1
    # when example doesn't end with text block.
    if not block_edges[-1] == erow: # iffy: I'm using end state of loop
        block_edges.append(erow)
    return block_edges, idx_first_text_block


def process_blocks(blocks, src_path, image_path, cfg):
    """Run source, save plots as images, and convert blocks to rst.

    Parameters
    ----------
    blocks : list of block tuples
        Code and text blocks from example. See `split_code_and_text_blocks`.
    src_path : str
        Path to example file.
    image_path : str
        Path where plots are saved (format string which accepts figure number).
    cfg : config object
        Sphinx config object created by Sphinx.

    Returns
    -------
    figure_list : list
        List of figure names saved by the example.
    rst_text : str
        Text with code wrapped code-block directives.
    """
    src_dir, src_name = src_path.psplit()
    if not src_name.startswith('plot'):
        return [], ''

    # index of blocks which have inline plots
    inline_tag = cfg.plot2rst_plot_tag
    idx_inline_plot = [i for i, b in enumerate(blocks)
                       if inline_tag in b[2]]

    image_dir, image_fmt_str = image_path.psplit()

    figure_list = []
    plt.rcdefaults()
    plt.rcParams.update(cfg.plot2rst_rcparams)
    plt.close('all')

    example_globals = {}
    rst_blocks = []
    fig_num = 1
    for i, (blabel, brange, bcontent) in enumerate(blocks):
        if blabel == 'code':
            exec(bcontent, example_globals)
            rst_blocks.append(codestr2rst(bcontent))
        else:
            if i in idx_inline_plot:
                plt.savefig(image_path.format(fig_num))
                figure_name = image_fmt_str.format(fig_num)
                fig_num += 1
                figure_list.append(figure_name)
                figure_link = os.path.join('images', figure_name)
                bcontent = bcontent.replace(inline_tag, figure_link)
            rst_blocks.append(docstr2rst(bcontent))
    return figure_list, '\n'.join(rst_blocks)


def codestr2rst(codestr):
    """Return reStructuredText code block from code string"""
    code_directive = ".. code-block:: python\n\n"
    indented_block = '\t' + codestr.replace('\n', '\n\t')
    return code_directive + indented_block


def docstr2rst(docstr):
    """Return reStructuredText from docstring"""
    idx_whitespace = len(docstr.rstrip()) - len(docstr)
    whitespace = docstr[idx_whitespace:]
    return eval(docstr) + whitespace


def save_all_figures(image_path):
    """Save all matplotlib figures.

    Parameters
    ----------
    image_path : str
        Path where plots are saved (format string which accepts figure number).
    """
    figure_list = []
    image_dir, image_fmt_str = image_path.psplit()
    fig_mngr = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
    for fig_num in (m.num for m in fig_mngr):
        # Set the fig_num figure as the current figure as we can't
        # save a figure that's not the current figure.
        plt.figure(fig_num)
        plt.savefig(image_path.format(fig_num))
        figure_list.append(image_fmt_str.format(fig_num))
    return figure_list
