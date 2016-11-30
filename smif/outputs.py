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


class OutputFactory(ModelElement):
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

    def populate(self):
        pass

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


class MetricList(OutputFactory):

    def populate(self, results):
        self.values = results['metrics']


class OutputList(OutputFactory):

    def populate(self, results):
        self.values = results['results']


class ModelOutputs(object):
    """A container for all the model outputs

    """
    def __init__(self, results):

        self._results = OutputFactory()
        self._metrics = self._results.getelement('metrics')
        self._metrics.populate(results)
        self._outputs = self._results.getelement('outputs')
        self._outputs.populate(results)

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
