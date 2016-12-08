"""Encapsulates the outputs from a sector model

.. inheritance-diagram:: smif.outputs

"""
import logging
import os

from smif.abstract import ModelElementCollection

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

LOGGER = logging.getLogger(__name__)


class OutputList(ModelElementCollection):
    """Defines the types of outputs to a sector model

    """
    def __init__(self, results):
        super().__init__()
        self.values = results

    ##
    # File interaction methods like load_results_from_files and
    # replace_{line,cell} might live somewhere else
    # - maybe sectormodel, if that deals with replacing file data?
    # - maybe a utility file handler class?
    # seem related to outputs for now
    ##
    @staticmethod
    def replace_line(file_name, line_num, new_data):
        """Replaces a line in a file with new data

        Arguments
        =========
        file_name: str
            The path to the input file
        line_num: int
            The number of the line to replace
        new_data: str
            The data to replace in the line

        """
        lines = open(file_name, 'r').readlines()
        lines[line_num] = new_data
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()

    @staticmethod
    def replace_cell(file_name, line_num, column_num, new_data,
                     delimiter=None):
        """Replaces a cell in a delimited file with new data

        Arguments
        =========
        file_name: str
            The path to the input file
        line_num: int
            The number of the line to replace (0-index)
        column_num: int
            The number of the column to replace (0-index)
        new_data: str
            The data to replace in the line
        delimiter: str, default=','
            The delimiter of the columns
        """
        line_num -= 1
        column_num -= 1

        with open(file_name, 'r') as input_file:
            lines = input_file.readlines()

        columns = lines[line_num].split(delimiter)
        columns[column_num] = new_data
        lines[line_num] = " ".join(columns) + "\n"

        with open(file_name, 'w') as out_file:
            out_file.writelines(lines)

    @property
    def file_locations(self):
        """Allows ordered searching through file to extract results from file
        """
        locations = {}

        # For each file return tuple of row/column
        for name in self.names:
            row_num = self.values[name]['row_num']
            col_num = self.values[name]['col_num']
            filename = self.values[name]['file_name']

            if filename not in locations:
                # add filename to locations
                locations[filename] = {}

            locations[filename][name] = (row_num, col_num)

        return locations

    def load_results_from_files(self, dirname):
        """Load model outputs from specified files
        """

        result_list = self.file_locations

        for filename in result_list:
            file_path = os.path.join(dirname, filename)

            with open(file_path, 'r') as results_set:
                lines = results_set.readlines()

            for name, rowcol in result_list[filename].items():
                row_num, col_num = rowcol
                result = lines[row_num][col_num:].strip()
                self.values[name]['value'] = result

    @property
    def values(self):
        """The value of the outputs
        """
        return self._values

    @values.setter
    def values(self, values):
        self._values = {output['name']: output for output in values}
        self.names = [output['name'] for output in values]

    def __getitem__(self, key):
        return self.values[key]


class ModelOutputs(object):
    """A container for all the model outputs

    """
    def __init__(self, results):
        if 'metrics' not in results:
            results['metrics'] = []
        if 'model outputs' not in results:
            results['model outputs'] = []

        self._metrics = OutputList(results['metrics'])
        self._outputs = OutputList(results['model outputs'])

    @property
    def metrics(self):
        """A list of the model result metrics

        Returns
        =======
        :class:`smif.outputs.MetricList`
        """
        return self._metrics

    @property
    def outputs(self):
        """A list of the model outputs

        Returns
        =======
        :class:`smif.outputs.OutputList`
        """
        return self._outputs

    def load_results_from_files(self, dirname):
        """Load model outputs from specified files
        """
        self._metrics.load_results_from_files(dirname)
        self._outputs.load_results_from_files(dirname)
