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
          units: £/l
        - name: LPG_price
          spatial_resolution: GB
          temporal_resolution: annual
          units: £/l

"""
from __future__ import absolute_import, division, print_function

import collections.abc
import logging

from smif.convert.unit import get_register as get_unit_register

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell, University of Oxford 2017"
__license__ = "mit"


class Metadata(object):
    """All metadata about a single dataset, typically model input or output

    Arguments
    =========
    name: str
        The dataset name
    spatial_resolution: :class:`smif.convert.region.RegionSet`
        The region set that defines the spatial resolution
    temporal_resolution: :class:`smif.convert.interval.IntervalSet`
       The interval set that defines the temporal resolution
    units: str
        Name of the units for the dataset values

    """
    def __init__(self, name, spatial_resolution, temporal_resolution, units):
        self.unit_register = get_unit_register()
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.units = self.normalise_unit(units, name)

    def __repr__(self):
        return repr(self.as_dict())

    def __eq__(self, other):
        return self.name == other.name \
            and self.spatial_resolution == other.spatial_resolution \
            and self.temporal_resolution == other.temporal_resolution \
            and self.units == other.units

    def as_dict(self):
        config = {
            'name': self.name,
            'spatial_resolution': self.spatial_resolution.name,
            'temporal_resolution': self.temporal_resolution.name,
            'units': self.units
        }
        return config

    def normalise_unit(self, unit_string, param_name):
        """Parse unit and return standard string representation
        """
        unit = self.unit_register.parse_unit(unit_string)
        if unit is not None:
            # if parsed successfully
            normalised = str(unit)
        else:
            normalised = unit_string
        self.logger.debug("Using unit for %s: %s", param_name, normalised)
        return normalised

    def get_region_names(self):
        """The list of region names for this spatial resolution
        """
        return self.spatial_resolution.get_entry_names()

    def get_interval_names(self):
        """The list of interval names for this temporal resolution
        """
        return self.temporal_resolution.get_entry_names()


class MetadataSet(collections.abc.Mapping):
    """A container for metadata about model inputs or outputs

    Arguments
    =========
    metadata: list
        A list of dicts like ::

                {
                    'name': 'heat_demand'
                    'spatial_resolution': smif.convert.ResolutionSet
                    'temporal_resolution': smif.convert.ResolutionSet
                    'units': 'kW'
                }

        Or, a list of smif.metadata.Metadata

                Metadata('heat_demand',
                         smif.convert.area.RegionSet,
                         smif.convert.interval.IntervalSet,
                         'kW')

    """
    def __init__(self, metadata_list=None):
        self._metadata = {}
        if metadata_list is not None:
            for metadata_item in metadata_list:
                self.add_metadata(metadata_item)

    def __repr__(self):
        return "MetadataSet({})".format(repr(list(self._metadata.values())))

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

    def __getitem__(self, name):
        if name in self._metadata:
            metadata_item = self._metadata[name]
        else:
            raise KeyError("No metadata found for name '{}'".format(name))
        return metadata_item

    def __iter__(self):
        for item in self._metadata:
            yield item

    def add_metadata(self, item):
        """Add an item to the set

        Arguments
        ---------
        item: Metadata or dict
            Metadata object or dictionary with keys 'name', 'spatial_resolution',
            'temporal_resolution' and 'units'
        """
        if isinstance(item, dict):
            item = Metadata(
                item['name'],
                item['spatial_resolution'],
                item['temporal_resolution'],
                item['units']
            )
        assert isinstance(item, Metadata)
        self._metadata[item.name] = item

    def get_spatial_res(self, name):
        """The spatial resolution for parameter `name`

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        return self[name].spatial_resolution

    def get_temporal_res(self, name):
        """The temporal resolution for parameter `name`

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        return self[name].temporal_resolution

    def get_units(self, name):
        """The units used for parameter 'name'

        Arguments
        ---------
        name: str
            The name of a model parameter
        """
        return self[name].units

    @property
    def spatial_resolutions(self):
        """A list of the spatial resolutions

        Returns
        -------
        list
            A list of the spatial resolutions associated with the model
            parameters
        """
        return [parameter.spatial_resolution for parameter in self._metadata.values()]

    @property
    def temporal_resolutions(self):
        """A list of the temporal resolutions

        Returns
        -------
        list
            A list of the temporal resolutions associated with the model
            parameters
        """
        return [parameter.temporal_resolution for parameter in self._metadata.values()]

    @property
    def names(self):
        """A list of the parameter names
        """
        return [parameter.name for parameter in self._metadata.values()]

    @property
    def units(self):
        """A list of the units

        Returns
        -------
        list
            A list of the units associated with the model
            parameters
        """
        return [parameter.units for parameter in self._metadata.values()]
