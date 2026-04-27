import collections
import os
import csv


class CsvWriter:
    def __init__(self, fname: str):
        if fname is not None and fname != '':
            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        self._fname = fname
        self._header_written = False
        self._fieldnames = None

    def write(self, values: collections.OrderedDict) -> None:
        if self._fname is None or self._fname == '':
            return

        if self._fieldnames is None:
            self._fieldnames = values.keys()

        with open(self._fname, 'a', encoding='utf8') as file_:

            writer = csv.DictWriter(file_, fieldnames=self._fieldnames)

            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(values)

    def close(self) -> None:
        pass
