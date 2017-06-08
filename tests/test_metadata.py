# -*- coding: utf-8 -*-
"""Test metadata access
"""
from pytest import fixture, raises

from smif.convert.area import RegionRegister, RegionSet
from smif.convert.interval import TimeIntervalRegister
from smif.metadata import Metadata, MetadataSet


@fixture(scope='function')
def two_output_metrics():
    """Returns a model output dictionary with two metrics
    """
    outputs = [
        {
            'name': 'total_cost',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual',
            'units': 'count'
        },
        {
            'name': 'water_demand',
            'spatial_resolution': 'watershed',
            'temporal_resolution': 'daily',
            'units': 'count'
        }
    ]
    return outputs


@fixture(scope='function')
def region_register_squares():
    """Return a register with a region set of two square regions::

        |```|```|
        | a | b |
        |...|...|

    """
    rset = RegionSet('half_squares', [
        {
            'type': 'Feature',
            'properties': {'name': 'a'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'b'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 1], [0, 2], [1, 2], [1, 1]]]
            }
        },
    ])
    rreg = RegionRegister()
    rreg.register(rset)
    return rreg


@fixture(scope='function')
def interval_register_seasons():
    """Return a register with an interval set of the four seasons
    """
    seasons = [
        {'id': 'winter', 'start': 'P0M', 'end': 'P2M'},
        {'id': 'spring', 'start': 'P2M', 'end': 'P5M'},
        {'id': 'summer', 'start': 'P5M', 'end': 'P8M'},
        {'id': 'autumn', 'start': 'P8M', 'end': 'P11M'},
        {'id': 'winter', 'start': 'P11M', 'end': 'P12M'}
    ]
    ireg = TimeIntervalRegister()
    ireg.register(seasons, 'seasons')
    return ireg


class TestMetadata(object):
    """Test Metadata objects
    """
    def test_create_metadata(self):
        """Create Metadata to hold name, spatial and temporal resolution, and units
        """
        metadata = Metadata("total_lane_kilometres", "country", "month", "kilometer")
        assert metadata.name == "total_lane_kilometres"
        assert metadata.spatial_resolution == "country"
        assert metadata.temporal_resolution == "month"
        assert metadata.units == "kilometer"

    def test_metadata_equality(self):
        """Metadata with same attributes should compare equal
        """
        one_m = Metadata("total_lane_kilometres", "country", "month", "kilometer")
        other = Metadata("total_lane_kilometres", "country", "month", "kilometer")
        assert one_m == other

    def test_unit_normalisation(self):
        """Expect units to be set to full names from abbreviation
        """
        metadata = Metadata("total_lane_kilometres", "country", "month", "km")
        assert metadata.units == "kilometer"

    def test_unit_unparseable(self):
        """Expect unrecognised units to be passed through unchanged
        """
        metadata = Metadata("total_lane_kilometres", "country", "month", "unparseable")
        assert metadata.units == "unparseable"

    def test_access_region_names_with_register(self, region_register_squares):
        rreg = region_register_squares
        metadata = Metadata("total_lane_kilometres", "half_squares", "seasons", "unparseable")
        metadata.region_register = rreg
        assert metadata.get_region_names() == ["a", "b"]

    def test_access_interval_names_with_register(self, interval_register_seasons):
        ireg = interval_register_seasons
        metadata = Metadata("total_lane_kilometres", "half_squares", "seasons", "unparseable")
        metadata.interval_register = ireg
        assert metadata.get_interval_names() == ["winter", "spring", "summer", "autumn"]


class TestMetadataSet(object):
    """Test MetadataSet objects
    """
    def test_create_metadata_set(self):
        """Create MetadataSet to hold a list of Metadata
        """
        metadata_list = [{
            "name": "heat_demand",
            "spatial_resolution": "household",
            "temporal_resolution": "hourly",
            "units": "kilowatt"
        }]
        metadata_set = MetadataSet(metadata_list)

        # direct access to single metadata
        metadata = metadata_set["heat_demand"]
        assert metadata.name == "heat_demand"
        assert metadata.spatial_resolution == "household"
        assert metadata.temporal_resolution == "hourly"
        assert metadata.units == "kilowatt"

        # direct access to list of contained metadata
        assert len(metadata_set) == 1
        assert metadata_set.metadata == [
            Metadata("heat_demand", "household", "hourly", "kilowatt")
        ]

        # access single metadata attribute
        assert metadata_set.get_spatial_res("heat_demand") == "household"
        assert metadata_set.get_temporal_res("heat_demand") == "hourly"
        assert metadata_set.get_units("heat_demand") == "kilowatt"

        # access list of metadata attributes
        assert metadata_set.names == ["heat_demand"]
        assert metadata_set.spatial_resolutions == ["household"]
        assert metadata_set.temporal_resolutions == ["hourly"]
        assert metadata_set.units == ["kilowatt"]

    def test_get_spatial_property(self, two_output_metrics):
        """Access different spatial resolutions for different metadata
        """
        outputs = MetadataSet(two_output_metrics)
        assert outputs.get_spatial_res('total_cost') == 'LSOA'
        assert outputs.get_spatial_res('water_demand') == 'watershed'

        with raises(KeyError) as ex:
            outputs.get_spatial_res('missing')
        assert "No metadata found for name 'missing'" in str(ex.value)

    def test_get_temporal_property(self, two_output_metrics):
        """Access different temporal resolutions for different metadata
        """
        outputs = MetadataSet(two_output_metrics)
        assert outputs.get_temporal_res('total_cost') == 'annual'
        assert outputs.get_temporal_res('water_demand') == 'daily'

        with raises(KeyError) as ex:
            outputs.get_temporal_res('missing')
        assert "No metadata found for name 'missing'" in str(ex.value)

    def test_key_error(self):
        """Expect a KeyError on trying to access missing Metadata
        """
        metadata_set = MetadataSet([])
        with raises(KeyError) as ex:
            metadata_set["missing"]
        assert "No metadata found for name 'missing'" in str(ex.value)

    def test_metadata_order(self):
        """Expect list of Metadata to be sorted by name
        """
        metadata_list = [
            {
                "name": "total_lane_kilometres",
                "spatial_resolution": "country",
                "temporal_resolution": "month",
                "units": "kilometer"
            },
            {
                "name": "heat_demand",
                "spatial_resolution": "household",
                "temporal_resolution": "hourly",
                "units": "kilowatt"
            },
        ]
        metadata_set = MetadataSet(metadata_list)

        assert len(metadata_set) == 2
        assert metadata_set.metadata == [
            Metadata("heat_demand", "household", "hourly", "kilowatt"),
            Metadata("total_lane_kilometres", "country", "month", "kilometer")
        ]
