# -*- coding: utf-8 -*-
"""Encapsulates the input or output parameters of a sector model,
for example::

        - name: petrol_price
          spatial_resolution: GB
          temporal_resolution: annual
        - name: diesel_price
          spatial_resolution: GB
          temporal_resolution: annual
        - name: LPG_price
          spatial_resolution: GB
          temporal_resolution: annual

"""
from __future__ import absolute_import, division, print_function

import logging
from smif import Parameter

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell, University of Oxford 2017"
__license__ = "mit"


class ParameterList(object):
    """Holds model parameters

    Parameters have several attributes: ``name``, ``spatial_resolution``
    and ``temporal_resolution``.

    The ``name`` entry denotes the unique identifier of a model or scenario
    parameter.
    The ``spatial_resolution`` and ``temporal_resolution`` are references to the
    catalogues held by the :class:`~smif.sector_model.SosModel` which define the
    available conversion formats.

    Parameters
    ----------
    parameters: dict
        A dictionary of parameters

    """

    def __init__(self, parameters):

        self.logger = logging.getLogger(__name__)

        names = []
        spatial_resolutions = []
        temporal_resolutions = []

        for param in parameters:
            names.append(param['name'])
            spatial_resolutions.append(param['spatial_resolution'])
            temporal_resolutions.append(param['temporal_resolution'])

        self.names = names
        self.spatial_resolutions = spatial_resolutions
        self.temporal_resolutions = temporal_resolutions

    def __repr__(self):
        """Return literal string representation of this instance
        """
        template = "{{'name': {}, 'spatial_resolution': {}, " + \
                   "'temporal_resolution': {}}}"

        return template.format(
            self.names,
            self.spatial_resolutions,
            self.temporal_resolutions,
        )

    def __getitem__(self, key):
        """Implement __getitem__ to make this class iterable

        Example
        =======
        >>> parameter_list = ParameterList()
        >>> for dep in parameter_list:
        >>>     # do something with each parameter

        - uses :class:`smif.Parameter` (a namedtuple) to wrap the data
        """
        parameter = Parameter(
            self.names[key],
            self.spatial_resolutions[key],
            self.temporal_resolutions[key]
        )
        return parameter

    def __len__(self):
        return len(self.names)


class ModelInputs(object):
    """A container for all the model inputs

    Arguments
    =========
    inputs : dict
        A dictionary of key: val pairs including a list of input types and
        names, followed by nested dictionaries of input attributes
    """
    def __init__(self, parameters):
        if 'dependencies' not in parameters:
            parameters['dependencies'] = []

        self._parameters = ParameterList(parameters['dependencies'])

        if 'metrics' not in parameters:
            parameters['metrics'] = []

        self._metrics = ParameterList(parameters['metrics'])
        self.logger = logging.getLogger(__name__)

    @property
    def metrics(self):
        """A list of the model result metrics

        Returns
        =======
        :class:`smif.inputs.ParameterList`
        """
        return self._metrics

    @property
    def dependencies(self):
        """A list of the model parameters

        Returns
        =======
        :class:`smif.inputs.ParameterList`
        """
        return self._parameters

    def __len__(self):
        return len(self.dependencies)

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
