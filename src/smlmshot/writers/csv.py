# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv

from torch import Tensor

from smlmshot.utils.activations import DEFAULT_COLUMNS

from .writer import WriterInterface


class CSVWriter(WriterInterface):
    """Write SMLM data to a .csv file, in a ThunderSTORM like format."""

    def __init__(self, filepath: str, columns: list = DEFAULT_COLUMNS):
        self.columns = columns
        self.filepath = filepath

    def open(self):
        """Open the CSV file."""
        self.file = open(self.filepath, mode="w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.columns)

    def close(self):
        """Close the CSV file."""
        if self.file:
            self.file.close()

    def _write(self, data: Tensor):
        """Write one chunk of data."""
        data = data.round().cpu().int().tolist()
        self.writer.writerows(data)
