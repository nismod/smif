# -*- coding: utf-8 -*-
"""Encapsulates the outputs from a sector model

.. inheritance-diagram:: smif.outputs

"""
import logging
import os

from smif.inputs import ModelElementCollection

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


class OutputList(object):
    """Defines the types of outputs to a sector model

    """
    def __init__(self, results):
        super().__init__()
        self.values = results



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
