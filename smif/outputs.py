"""Encapsulates the outputs from a sector model

.. inheritance-diagram:: smif.outputs

"""
import logging
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


class MetricList(OutputFactory):

    pass


class OutputList(OutputFactory):

    pass


class ModelOutputs(object):
    """A container for all the model outputs

    """
    def __init__(self, results):

        self._results = OutputFactory()
        self._metrics = self._results.getelement('metrics')
        self._metrics.get_outputs(results)
        self._outputs = self._results.getelement('outputs')
        self._outputs.get_outputs(results)

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
        """A list of the vanilla outputs

        Returns
        =======
        :class:`smif.outputs.OutputList`
        """
        return self._outputs
