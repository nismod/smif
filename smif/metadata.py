# -*- coding: utf-8 -*-
"""Encapsulates the input or output parameters of a sector model,
for example::

        - name: petrol_price
          spatial_resolution: GB
          temporal_resolution: annual
          units: £/l
        - name: diesel_price
          spatial_resolution: GB
          temporal_resolution: annual
        - name: LPG_price
          spatial_resolution: GB
          temporal_resolution: annual

"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell, University of Oxford 2017"
__license__ = "mit"


Metadata = namedtuple(
    "Metadata",
    [
        "name",
        "spatial_resolution",
        "temporal_resolution",
        "units"
    ]
)


class ModelMetadata(object):
    """A container for metadata about model inputs or outputs

    Arguments
    =========
    metadata: list
        A list of dicts like ::

                {
                    "name": "heat_demand"
                    "spatial_resolution": "household"
                    "temporal_resolution": "hourly"
                    "units": "kW"
                }
    """
    def __init__(self, metadata_list):
        self._metadata = [
            Metadata(
                metadata_item['name'],
                metadata_item['spatial_resolution'],
                metadata_item['temporal_resolution'],
                metadata_item['units']
            )
            for metadata_item in metadata_list
        ]
        self.logger = logging.getLogger(__name__)

    @property
    def metadata(self):
        """A list of the model parameters

        Returns
        =======
        parameters: list
            A list of :class:`smif.Metadata`
        """
        return self._metadata

    def __len__(self):
        return len(self._metadata)

    def get_spatial_res(self, name):
        """The spatial resolution for parameter `name`

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        for parameter in self._metadata:
            if parameter.name == name:
                spatial_resolution = parameter.spatial_resolution
                break
        else:
            raise ValueError("No output found for name '{}'".format(name))
        return spatial_resolution

    def get_temporal_res(self, name):
        """The temporal resolution for parameter `name`

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        for parameter in self._metadata:
            if parameter.name == name:
                temporal_resolution = parameter.temporal_resolution
                break
        else:
            raise ValueError("No output found for name '{}'".format(name))
        return temporal_resolution

    @property
    def spatial_resolutions(self):
        """A list of the spatial resolutions

        Returns
        -------
        list
            A list of the spatial resolutions associated with the model
            parameters
        """
        return [parameter.spatial_resolution for parameter in self._metadata]

    @property
    def temporal_resolutions(self):
        """A list of the temporal resolutions

        Returns
        -------
        list
            A list of the temporal resolutions associated with the model
            parameters
        """
        return [parameter.temporal_resolution for parameter in self._metadata]

    @property
    def names(self):
        return [parameter.name for parameter in self._metadata]
