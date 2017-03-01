# -*- coding: utf-8 -*-
"""Encapsulates the outputs from a sector model

The output definitions are read in from ``outputs.yaml``.  For example::

        metrics:
        - name: total_cost
        - name: water_demand

"""
import logging
import os

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


class OutputList(object):
    """Defines the types of outputs to a sector model

    Parameters
    ----------
    outputs: dict

    """
    def __init__(self, outputs):
        self.logger = logging.getLogger(__name__)

        names = []
        for output in outputs:
            names.append(output['name'])
        self.names = names

    def __getitem__(self, key):
        return self.names[key]


class ModelOutputs(object):
    """A container for all the model outputs

    """
    def __init__(self, results):
        if 'metrics' not in results:
            results['metrics'] = []
        self._metrics = OutputList(results['metrics'])

    @property
    def metrics(self):
        """A list of the model result metrics

        Returns
        =======
        :class:`smif.outputs.MetricList`
        """
        return self._metrics
