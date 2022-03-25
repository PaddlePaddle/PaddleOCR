""" brain-dead simple parser for ini-style files.
(C) Ronny Pfannschmidt, Holger Krekel -- MIT licensed
"""
__all__ = ['IniConfig', 'ParseError']

COMMENTCHARS = "#;"


class ParseError(Exception):
    def __init__(self, path, lineno, msg):
        Exception.__init__(self, path, lineno, msg)
        self.path = path
        self.lineno = lineno
        self.msg = msg

    def __str__(self):
        return "%s:%s: %s" % (self.path, self.lineno+1, self.msg)


class SectionWrapper(object):
    def __init__(self, config, name):
        self.config = config
        self.name = name

    def lineof(self, name):
        return self.config.lineof(self.name, name)

    def get(self, key, default=None, convert=str):
        return self.config.get(self.name, key,
                               convert=convert, default=default)

    def __getitem__(self, key):
        return self.config.sections[self.name][key]

    def __iter__(self):
        section = self.config.sections.get(self.name, [])

        def lineof(key):
            return self.config.lineof(self.name, key)
        for name in sorted(section, key=lineof):
            yield name

    def items(self):
        for name in self:
            yield name, self[name]


class IniConfig(object):
    def __init__(self, path, data=None):
        self.path = str(path)  # convenience
        if data is None:
            f = open(self.path)
            try:
                tokens = self._parse(iter(f))
            finally:
                f.close()
        else:
            tokens = self._parse(data.splitlines(True))

        self._sources = {}
        self.sections = {}

        for lineno, section, name, value in tokens:
            if section is None:
                self._raise(lineno, 'no section header defined')
            self._sources[section, name] = lineno
            if name is None:
                if section in self.sections:
                    self._raise(lineno, 'duplicate section %r' % (section, ))
                self.sections[section] = {}
            else:
                if name in self.sections[section]:
                    self._raise(lineno, 'duplicate name %r' % (name, ))
                self.sections[section][name] = value

    def _raise(self, lineno, msg):
        raise ParseError(self.path, lineno, msg)

    def _parse(self, line_iter):
        result = []
        section = None
        for lineno, line in enumerate(line_iter):
            name, data = self._parseline(line, lineno)
            # new value
            if name is not None and data is not None:
                result.append((lineno, section, name, data))
            # new section
            elif name is not None and data is None:
                if not name:
                    self._raise(lineno, 'empty section name')
                section = name
                result.append((lineno, section, None, None))
            # continuation
            elif name is None and data is not None:
                if not result:
                    self._raise(lineno, 'unexpected value continuation')
                last = result.pop()
                last_name, last_data = last[-2:]
                if last_name is None:
                    self._raise(lineno, 'unexpected value continuation')

                if last_data:
                    data = '%s\n%s' % (last_data, data)
                result.append(last[:-1] + (data,))
        return result

    def _parseline(self, line, lineno):
        # blank lines
        if iscommentline(line):
            line = ""
        else:
            line = line.rstrip()
        if not line:
            return None, None
        # section
        if line[0] == '[':
            realline = line
            for c in COMMENTCHARS:
                line = line.split(c)[0].rstrip()
            if line[-1] == "]":
                return line[1:-1], None
            return None, realline.strip()
        # value
        elif not line[0].isspace():
            try:
                name, value = line.split('=', 1)
                if ":" in name:
                    raise ValueError()
            except ValueError:
                try:
                    name, value = line.split(":", 1)
                except ValueError:
                    self._raise(lineno, 'unexpected line: %r' % line)
            return name.strip(), value.strip()
        # continuation
        else:
            return None, line.strip()

    def lineof(self, section, name=None):
        lineno = self._sources.get((section, name))
        if lineno is not None:
            return lineno + 1

    def get(self, section, name, default=None, convert=str):
        try:
            return convert(self.sections[section][name])
        except KeyError:
            return default

    def __getitem__(self, name):
        if name not in self.sections:
            raise KeyError(name)
        return SectionWrapper(self, name)

    def __iter__(self):
        for name in sorted(self.sections, key=self.lineof):
            yield SectionWrapper(self, name)

    def __contains__(self, arg):
        return arg in self.sections


def iscommentline(line):
    c = line.lstrip()[:1]
    return c in COMMENTCHARS
