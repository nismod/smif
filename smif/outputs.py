# -*- coding: utf-8 -*-
"""Encapsulates the outputs from a sector model

The output definitions are read in from ``outputs.yaml``.  For example::

        metrics:
        - name: total_cost
        - name: water_demand

"""
import logging
from collections import namedtuple

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


Output = namedtuple(
    "Output",
    [
        "name",
        "spatial_resolution",
        "temporal_resolution"
    ]
)


class OutputList(object):
    """Defines the types of outputs to a sector model

    Parameters
    ----------
    outputs: dict

    """
    def __init__(self, outputs):
        self.logger = logging.getLogger(__name__)

        names = []
        spatial_resolutions = []
        temporal_resolutions = []
        for output in outputs:
            names.append(output['name'])
            spatial_resolutions.append(output['spatial_resolution'])
            temporal_resolutions.append(output['temporal_resolution'])
        self.names = names
        self.temporal_resolutions = temporal_resolutions
        self.spatial_resolutions = spatial_resolutions

    def __getitem__(self, key):

        output = Output(self.names[key],
                        self.spatial_resolutions[key],
                        self.temporal_resolutions[key])
        return output


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

    @property
    def spatial_resolutions(self):
        return self._metrics.spatial_resolutions

    @property
    def temporal_resolutions(self):
        return self._metrics.temporal_resolutions
