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

import logging

from smif.convert.unit import parse_unit

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
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.units = self.normalise_unit(units, name)

    def __eq__(self, other):
        return self.name == other.name \
            and self.spatial_resolution == other.spatial_resolution \
            and self.temporal_resolution == other.temporal_resolution \
            and self.units == other.units

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

    def get_region_names(self):
        """The list of region names for this spatial resolution
        """
        return self.spatial_resolution.get_entry_names()

    def get_interval_names(self):
        """The list of interval names for this temporal resolution
        """
        return self.temporal_resolution.get_entry_names()


class MetadataSet(object):
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
    def __init__(self, metadata_list):
        self._metadata = {}
        for metadata_item in metadata_list:
            if isinstance(metadata_item, dict):
                self.add_metadata(metadata_item)
            elif isinstance(metadata_item, Metadata):
                self.add_metadata_object(metadata_item)

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
        return iter(self.metadata)

    def add_metadata(self, metadata_item):
        """Add an item to the set

        Arguments
        ---------
        metadata_item: dict
            A dictionary with keys 'name', 'spatial resolution', 'temporal
            resolution' and 'units'
        """
        metadata = Metadata(metadata_item['name'],
                            metadata_item['spatial_resolution'],
                            metadata_item['temporal_resolution'],
                            metadata_item['units'])
        self._metadata[metadata.name] = metadata

    def add_metadata_object(self, metadata_object):
        """Add an metadata object to the set

        Arguments
        ---------
        metadata_object : Metadata
            A smif.metadata.Metadata instance

        """
        assert isinstance(metadata_object, Metadata)
        self._metadata[metadata_object.name] = metadata_object

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

    @property
    def units(self):
        """A list of the units

        Returns
        -------
        list
            A list of the units associated with the model
            parameters
        """
        return [parameter.units for name, parameter in self._metadata.items()]
