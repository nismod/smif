"""Encapsulates the outputs from a sector model

.. inheritance-diagram:: smif.outputs

"""
import logging
import os
from collections import OrderedDict

import numpy as np
from smif.inputs import ModelElement

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class OutputList(ModelElement):
    """Defines the types of outputs to a sector model

    """

    @staticmethod
    def getelement(output_type):
        """Implements the factory method to return subclasses of
        :class:`smif.outputFactory`
        """
        if output_type == 'metrics':
            return MetricList()
        elif output_type == 'outputs':
            return OutputList()
        else:
            raise ValueError("That output type is not defined")

    def get_outputs(self, results):
        """Gets a dictionary of results and returns a list of the relevant types
        """
        self.names = sorted(list(results[0].keys()))
        number_outputs = len(self.names)

        number_timesteps = len(results)

        values = np.zeros((number_outputs, number_timesteps))

        for index in range(number_timesteps):

            sorted_results = OrderedDict(sorted(results[index].items()))
            sorted_values = list(sorted_results.values())

            logger.debug("Values in timestep {}: {}".format(index,
                                                            sorted_values))

            values[:, index] = np.array(sorted_values)
        self.values = values

    @property
    def extract_iterable(self):
        """Allows ordered searching through file to extract results from file
        """
        entities = {}
        extract_iterable = {}

        names = set(self.values.keys())
        # For each file return tuple of row/column
        for name in names:
            row_num = self.values[name]['row_num']
            col_num = self.values[name]['col_num']
            filename = self.values[name]['file_name']

            entities[name] = {name: (row_num, col_num)}

            if filename in extract_iterable.keys():
                # Append the entity to the existing entry
                existing = extract_iterable[filename]
                existing[name] = (row_num, col_num)
                extract_iterable[filename] = existing
            else:
                # Add the new key to the dict and the entry
                extract_iterable[filename] = entities[name]

        return extract_iterable

    def get_results(self, path_to_model_root):

        result_list = self.extract_iterable
        results_instance = {}

        for filename in result_list.keys():
            file_path = os.path.join(path_to_model_root, filename)
            with open(file_path, 'r') as results_set:
                lines = results_set.readlines()

            for name, rowcol in result_list[filename].items():
                row_num, col_num = rowcol
                result = lines[row_num][col_num:].strip()
                results_instance[name] = result

        return results_instance


class MetricList(OutputList):

    def __init__(self, results):
        super().__init__()
        self.values = results


class ModelOutputList(OutputList):

    def __init__(self, results):
        super().__init__()
        self.values = results


class ModelOutputs(object):
    """A container for all the model outputs

    """
    def __init__(self, results):
        if 'metrics' not in results:
            results['metrics'] = []
        if 'model outputs' not in results:
            results['model outputs'] = []

        self._metrics = MetricList(results['metrics'])
        self._outputs = ModelOutputList(results['model outputs'])


    @property
    def metrics(self):
        """A list of the model result metrics

        Returns
        =======
        :class:`smif.outputs.MetricList`
        """
        return sorted(list(self._metrics.values.keys()))

    @property
    def outputs(self):
        """A list of the vanilla outputs

        Returns
        =======
        :class:`smif.outputs.OutputList`
        """
        return sorted(list(self._outputs.values.keys()))


    ##
    # replace_{line,cell} methods might live somewhere else
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