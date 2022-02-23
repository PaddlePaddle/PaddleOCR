
import sys
from unittest import TestCase
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO  # doesn't accept 'str' in Py2

from .. import Options
from ..CmdLine import parse_command_line


class CmdLineParserTest(TestCase):
    def setUp(self):
        backup = {}
        for name, value in vars(Options).items():
            backup[name] = value
        self._options_backup = backup

    def tearDown(self):
        no_value = object()
        for name, orig_value in self._options_backup.items():
            if getattr(Options, name, no_value) != orig_value:
                setattr(Options, name, orig_value)

    def test_short_options(self):
        options, sources = parse_command_line([
            '-V', '-l', '-+', '-t', '-v', '-v', '-v', '-p', '-D', '-a', '-3',
        ])
        self.assertFalse(sources)
        self.assertTrue(options.show_version)
        self.assertTrue(options.use_listing_file)
        self.assertTrue(options.cplus)
        self.assertTrue(options.timestamps)
        self.assertTrue(options.verbose >= 3)
        self.assertTrue(Options.embed_pos_in_docstring)
        self.assertFalse(Options.docstrings)
        self.assertTrue(Options.annotate)
        self.assertEqual(options.language_level, 3)

        options, sources = parse_command_line([
            '-f', '-2', 'source.pyx',
        ])
        self.assertTrue(sources)
        self.assertTrue(len(sources) == 1)
        self.assertFalse(options.timestamps)
        self.assertEqual(options.language_level, 2)

    def test_long_options(self):
        options, sources = parse_command_line([
            '--version', '--create-listing', '--cplus', '--embed', '--timestamps',
            '--verbose', '--verbose', '--verbose',
            '--embed-positions', '--no-docstrings', '--annotate', '--lenient',
        ])
        self.assertFalse(sources)
        self.assertTrue(options.show_version)
        self.assertTrue(options.use_listing_file)
        self.assertTrue(options.cplus)
        self.assertEqual(Options.embed, 'main')
        self.assertTrue(options.timestamps)
        self.assertTrue(options.verbose >= 3)
        self.assertTrue(Options.embed_pos_in_docstring)
        self.assertFalse(Options.docstrings)
        self.assertTrue(Options.annotate)
        self.assertFalse(Options.error_on_unknown_names)
        self.assertFalse(Options.error_on_uninitialized)

        options, sources = parse_command_line([
            '--force', 'source.pyx',
        ])
        self.assertTrue(sources)
        self.assertTrue(len(sources) == 1)
        self.assertFalse(options.timestamps)

    def test_options_with_values(self):
        options, sources = parse_command_line([
            '--embed=huhu',
            '-I/test/include/dir1', '--include-dir=/test/include/dir2',
            '--include-dir', '/test/include/dir3',
            '--working=/work/dir',
            'source.pyx',
            '--output-file=/output/dir',
            '--pre-import=/pre/import',
            '--cleanup=3',
            '--annotate-coverage=cov.xml',
            '--gdb-outdir=/gdb/outdir',
            '--directive=wraparound=false',
        ])
        self.assertEqual(sources, ['source.pyx'])
        self.assertEqual(Options.embed, 'huhu')
        self.assertEqual(options.include_path, ['/test/include/dir1', '/test/include/dir2', '/test/include/dir3'])
        self.assertEqual(options.working_path, '/work/dir')
        self.assertEqual(options.output_file, '/output/dir')
        self.assertEqual(Options.pre_import, '/pre/import')
        self.assertEqual(Options.generate_cleanup_code, 3)
        self.assertTrue(Options.annotate)
        self.assertEqual(Options.annotate_coverage_xml, 'cov.xml')
        self.assertTrue(options.gdb_debug)
        self.assertEqual(options.output_dir, '/gdb/outdir')

    def test_errors(self):
        def error(*args):
            old_stderr = sys.stderr
            stderr = sys.stderr = StringIO()
            try:
                self.assertRaises(SystemExit, parse_command_line, list(args))
            finally:
                sys.stderr = old_stderr
            self.assertTrue(stderr.getvalue())

        error('-1')
        error('-I')
        error('--version=-a')
        error('--version=--annotate=true')
        error('--working')
        error('--verbose=1')
        error('--verbose=1')
        error('--cleanup')
