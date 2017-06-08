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

from smif.convert.unit import parse_unit

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
        self.logger = logging.getLogger(__name__)
        self._metadata = {
            metadata_item['name']: Metadata(
                metadata_item['name'],
                metadata_item['spatial_resolution'],
                metadata_item['temporal_resolution'],
                self.normalise_unit(metadata_item['units'], metadata_item['name'])
            )
            for metadata_item in metadata_list
        }

    def normalise_unit(self, unit_string, param_name):
        """Parse unit and return standard string representation
        """
        unit = parse_unit(unit_string)
        if unit is not None:
            # if parsed successfully
            normalised = str(unit)
        else:
            normalised = unit_string

        self.logger.debug("Using unit for %s: %s", param_name, normalised)
        return normalised

    @property
    def metadata(self):
        """A list of the model parameters

        Returns
        =======
        parameters: list
            A list of :class:`smif.Metadata`, sorted by name
        """
        sorted_keys = sorted(list(self._metadata.keys()))
        return [self._metadata[key] for key in sorted_keys]

    def __len__(self):
        return len(self._metadata)

    def get_metadata_item(self, name):
        if name in self._metadata:
            metadata_item = self._metadata[name]
        else:
            raise ValueError("No metadata found for name '{}'".format(name))
        return metadata_item

    def get_spatial_res(self, name):
        """The spatial resolution for parameter `name`

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        return self.get_metadata_item(name).spatial_resolution

    def get_temporal_res(self, name):
        """The temporal resolution for parameter `name`

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        return self.get_metadata_item(name).temporal_resolution

    def get_units(self, name):
        """The units used for parameter 'name'

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        return self.get_metadata_item(name).units

    @property
    def spatial_resolutions(self):
        """A list of the spatial resolutions

        Returns
        -------
        list
            A list of the spatial resolutions associated with the model
            parameters
        """
        return [parameter.spatial_resolution for name, parameter in self._metadata.items()]

    @property
    def temporal_resolutions(self):
        """A list of the temporal resolutions

        Returns
        -------
        list
            A list of the temporal resolutions associated with the model
            parameters
        """
        return [parameter.temporal_resolution for name, parameter in self._metadata.items()]

    @property
    def names(self):
        """A list of the parameter names
        """
        return list(self._metadata.keys())
