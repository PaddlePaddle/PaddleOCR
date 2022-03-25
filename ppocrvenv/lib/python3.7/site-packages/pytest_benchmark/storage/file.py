import json
import os

from ..stats import normalize_stats
from ..utils import Path
from ..utils import commonpath
from ..utils import safe_dumps
from ..utils import short_filename


class FileStorage(object):
    def __init__(self, path, logger, default_machine_id=None):
        self.path = Path(path)
        self.default_machine_id = default_machine_id
        try:
            self.path.mkdir(parents=True)
        except OSError:
            pass
        self.path = self.path.resolve()
        self.logger = logger
        self._cache = {}

    def __str__(self):
        return str(self.path)

    @property
    def location(self):
        return str(self.path.relative_to(os.getcwd()))

    def get(self, name):
        path = self.path.joinpath(self.default_machine_id) if self.default_machine_id else self.path
        try:
            path.mkdir(parents=True)
        except OSError:
            pass
        return path.joinpath(name)

    @property
    def _next_num(self):
        files = self.query("[0-9][0-9][0-9][0-9]_*")
        files.sort(reverse=True)
        if not files:
            return "0001"
        for f in files:
            try:
                return "%04i" % (int(str(f.name).split('_')[0]) + 1)
            except ValueError:
                raise

    def save(self, output_json, save):
        output_file = self.get("%s_%s.json" % (self._next_num, save))
        assert not output_file.exists()
        with output_file.open('wb') as fh:
            fh.write(safe_dumps(output_json, ensure_ascii=True, indent=4).encode())
        self.logger.info("Saved benchmark data in: %s" % output_file)

    def query(self, *globs_or_files):
        files = []
        globs = []
        if not globs_or_files:
            globs_or_files = "*",

        for globish in globs_or_files:
            candidate = Path(globish)
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            if is_file:
                files.append(candidate)
                continue

            parts = candidate.parts
            if len(parts) > 2:
                raise ValueError("{0!r} isn't an existing file or acceptable glob. "
                                 "Expected 'platform-glob/filename-glob' or 'filename-glob'.".format(globish))
            elif len(parts) == 2:
                platform_glob, filename_glob = parts
            else:
                platform_glob = self.default_machine_id or "*"
                filename_glob, = parts or ['']

            filename_glob = filename_glob.rstrip("*") + "*.json"
            globs.append((platform_glob, filename_glob))

        files.extend((
            file
            for platform_glob, filename_glob in globs
            for path in self.path.glob(platform_glob)
            for file in path.glob(filename_glob)
        ))
        files.extend((
            file
            for _, filename_glob in globs
            for file in self.path.glob(filename_glob)
        ))
        return sorted(files, key=lambda file: (file.name, file.parent))

    def load(self, *globs_or_files):
        if not globs_or_files:
            globs_or_files = '[0-9][0-9][0-9][0-9]_',

        for file in self.query(*globs_or_files):
            if file in self._cache:
                data = self._cache[file]
            else:
                try:
                    data = json.loads(file.read_text(encoding='utf8'))
                    for bench in data["benchmarks"]:
                        normalize_stats(bench["stats"])
                except Exception as exc:
                    self.logger.warn("Failed to load {0}: {1}".format(file, exc))
                    continue
                self._cache[file] = data
            try:
                relpath = file.relative_to(self.path)
            except ValueError:
                relpath = file
            yield relpath, data

    def load_benchmarks(self, *globs_or_files):
        sources = [
            (short_filename(path), path, data)
            for path, data in self.load(*globs_or_files)
        ]
        common = len(commonpath([src for src, _, _ in sources])) if sources else 0
        for source, path, data in sources:
            source = source[common:].lstrip(r'\/')

            for bench in data["benchmarks"]:
                bench.update(bench.pop("stats"))
                bench['path'] = str(path)
                bench['source'] = source
                yield bench
