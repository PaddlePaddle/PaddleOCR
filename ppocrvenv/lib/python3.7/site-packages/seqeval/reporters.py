import abc


class Reporter(abc.ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def report(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, row_name: str, precision: float, recall: float, f1: float, support: int):
        raise NotImplementedError


class DictReporter(Reporter):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.report_dict = {}

    def report(self):
        return self.report_dict

    def write(self, row_name: str, precision: float, recall: float, f1: float, support: int):
        self.report_dict[row_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }

    def write_blank(self):
        pass


class StringReporter(Reporter):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.buffer = []
        self.row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}'
        self.width = kwargs.get('width', 10)
        self.digits = kwargs.get('digits', 4)

    def report(self):
        report = self.write_header()
        report += '\n'.join(self.buffer)
        return report

    def write(self, row_name: str, precision: float, recall: float, f1: float, support: int):
        row = self.row_fmt.format(
            *[row_name, precision, recall, f1, support],
            width=self.width,
            digits=self.digits
        )
        self.buffer.append(row)

    def write_header(self):
        headers = ['precision', 'recall', 'f1-score', 'support']
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=self.width)
        report += '\n\n'
        return report

    def write_blank(self):
        self.buffer.append('')
