# -*- coding: utf-8 -*-
"""Encapsulates the outputs from a sector model

The output definitions are read in from ``outputs.yaml``.  For example::

        metrics:
        - name: total_cost
        - name: water_demand

"""
import logging
from smif.inputs import ParameterList

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell, University of Oxford 2017"
__license__ = "mit"


class ModelOutputs(object):
    """A container for all the model outputs

    """
    def __init__(self, results):
        if 'metrics' not in results:
            results['metrics'] = []
        self._metrics = ParameterList(results['metrics'])
        self.logger = logging.getLogger(__name__)
        self.logger.debug(results)

    @property
    def metrics(self):
        """A list of the model result metrics

        Returns
        =======
        :class:`smif.outputs.ParameterList`
        """
        return self._metrics

    def get_spatial_res(self, name):
        for metric in self._metrics:
            if metric.name == name:
                spatial_resolution = metric.spatial_resolution
                break
        else:
            raise ValueError("No output found for name {}".format(name))
        return spatial_resolution

    def get_temporal_res(self, name):
        for metric in self._metrics:
            if metric.name == name:
                temporal_resolution = metric.temporal_resolution
                break
        else:
            raise ValueError("No output found for name '{}'".format(name))
        return temporal_resolution

    @property
    def spatial_resolutions(self):
        return self._metrics.spatial_resolutions

    @property
    def temporal_resolutions(self):
        return self._metrics.temporal_resolutions
