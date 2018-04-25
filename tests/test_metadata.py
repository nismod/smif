# -*- coding: utf-8 -*-
"""Test metadata access
"""
from pytest import fixture, raises
from smif.convert.area import RegionSet
from smif.convert.interval import IntervalSet
from smif.metadata import Metadata, MetadataSet


@fixture(scope='function')
def two_output_metrics(interval_set, region_set):
    """Returns a model output dictionary with two metrics
    """
    outputs = [
        {
            'name': 'total_cost',
            'spatial_resolution': region_set,
            'temporal_resolution': interval_set,
            'units': 'count'
        },
        {
            'name': 'water_demand',
            'spatial_resolution': region_set,
            'temporal_resolution': interval_set,
            'units': 'count'
        }
    ]
    return outputs


@fixture(scope='function')
def region_set():
    """Return a region set of two square regions::

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
    return rset


@fixture(scope='function')
def interval_set():
    """Return an interval set of the four seasons
    """
    seasons = [
        ('winter', [('P0M', 'P2M'), ('P11M', 'P12M')]),
        ('spring', [('P2M', 'P5M')]),
        ('summer', [('P5M', 'P8M')]),
        ('autumn', [('P8M', 'P11M')])
    ]

    return IntervalSet('seasons', seasons)


class TestMetadata(object):
    """Test Metadata objects
    """
    def test_create_metadata(self, interval_set, region_set):
        """Create Metadata to hold name, spatial and temporal resolution, and units
        """
        metadata = Metadata("total_lane_kilometres", region_set, interval_set,
                            "kilometer")
        assert metadata.name == "total_lane_kilometres"
        assert metadata.spatial_resolution == region_set
        assert metadata.temporal_resolution == interval_set
        assert metadata.units == "kilometer"

    def test_serialise_metadata(self, interval_set, region_set):
        """Create Metadata to hold name, spatial and temporal resolution, and units
        """
        metadata = Metadata("total_lane_kilometres", region_set, interval_set,
                            "kilometer")
        actual = metadata.as_dict()
        expected = {'name': 'total_lane_kilometres',
                    'spatial_resolution': 'half_squares',
                    'temporal_resolution': 'seasons',
                    'units': 'kilometer'}
        assert actual == expected

    def test_metadata_equality(self):
        """Metadata with same attributes should compare equal
        """
        one_m = Metadata("total_lane_kilometres", region_set, interval_set, "kilometer")
        other = Metadata("total_lane_kilometres", region_set, interval_set, "kilometer")
        assert one_m == other

    def test_unit_normalisation(self):
        """Expect units to be set to full names from abbreviation
        """
        metadata = Metadata("total_lane_kilometres", region_set, interval_set,
                            "km")
        assert metadata.units == "kilometer"

    def test_unit_unparseable(self):
        """Expect unrecognised units to be passed through unchanged
        """
        metadata = Metadata("total_lane_kilometres", region_set, interval_set,
                            "unparseable")
        assert metadata.units == "unparseable"

    def test_access_region_names_with_register(self, region_set, interval_set):
        """Metadata should expose region names when a register is available
        """
        rreg = region_set
        metadata = Metadata("total_lane_kilometres", region_set, interval_set, "unparseable")
        metadata.region_register = rreg
        assert metadata.get_region_names() == ["a", "b"]

    def test_access_interval_names(self, interval_set, region_set):
        """Metadata should expose interval names when a register is available
        """
        metadata = Metadata("total_lane_kilometres", region_set, interval_set,
                            "unparseable")
        assert metadata.get_interval_names() == ["winter", "spring", "summer", "autumn"]


class TestMetadataSet(object):
    """Test MetadataSet objects
    """
    def test_create_metadata_set(self, interval_set, region_set):
        """Create MetadataSet to hold a list of Metadata
        """
        metadata_list = [{
            "name": "heat_demand",
            "spatial_resolution": region_set,
            "temporal_resolution": interval_set,
            "units": "kilowatt"
        }]
        metadata_set = MetadataSet(metadata_list)

        # direct access to single metadata
        metadata = metadata_set["heat_demand"]
        assert metadata.name == "heat_demand"
        assert metadata.spatial_resolution == region_set
        assert metadata.temporal_resolution == interval_set
        assert metadata.units == "kilowatt"

        # direct access to list of contained metadata
        assert len(metadata_set) == 1
        assert metadata_set.metadata == [
            Metadata("heat_demand", region_set, interval_set, "kilowatt")
        ]

        # access single metadata attribute
        assert metadata_set.get_spatial_res("heat_demand") == region_set
        assert metadata_set.get_temporal_res("heat_demand") == interval_set
        assert metadata_set.get_units("heat_demand") == "kilowatt"

        # access list of metadata attributes
        assert metadata_set.names == ["heat_demand"]
        assert metadata_set.spatial_resolutions == [region_set]
        assert metadata_set.temporal_resolutions == [interval_set]
        assert metadata_set.units == ["kilowatt"]

    def test_get_spatial_property(self, two_output_metrics, region_set):
        """Access different spatial resolutions for different metadata
        """
        outputs = MetadataSet(two_output_metrics)
        assert outputs.get_spatial_res('total_cost') == region_set
        assert outputs.get_spatial_res('water_demand') == region_set

        with raises(KeyError) as ex:
            outputs.get_spatial_res('missing')
        assert "No metadata found for name 'missing'" in str(ex.value)

    def test_get_temporal_property(self, two_output_metrics, interval_set):
        """Access different temporal resolutions for different metadata
        """
        outputs = MetadataSet(two_output_metrics)
        assert outputs.get_temporal_res('total_cost') == interval_set
        assert outputs.get_temporal_res('water_demand') == interval_set

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

    def test_metadata_order(self, interval_set, region_set):
        """Expect list of Metadata to be sorted by name
        """
        metadata_list = [
            {
                "name": "total_lane_kilometres",
                "spatial_resolution": region_set,
                "temporal_resolution": interval_set,
                "units": "kilometer"
            },
            {
                "name": "heat_demand",
                "spatial_resolution": region_set,
                "temporal_resolution": interval_set,
                "units": "kilowatt"
            },
        ]
        metadata_set = MetadataSet(metadata_list)

        assert len(metadata_set) == 2
        assert metadata_set.metadata == [
            Metadata("heat_demand", region_set, interval_set, "kilowatt"),
            Metadata("total_lane_kilometres", region_set, interval_set, "kilometer")
        ]

    def test_access_registers(self, region_set, interval_set):
        """Individual Metadata in a Set should provide region and interval names
        when registers are available
        """
        metadata_list = [
            {
                "name": "total_lane_kilometres",
                "spatial_resolution": region_set,
                "temporal_resolution": interval_set,
                "units": "kilometers"
            }
        ]
        metadata_set = MetadataSet(metadata_list)
        metadata = metadata_set["total_lane_kilometres"]

        assert metadata.get_region_names() == ["a", "b"]
        assert metadata.get_interval_names() == ["winter", "spring", "summer", "autumn"]

    def test_iterate_over_empty(self):
        """Should initialise as empty by default
        """
        metadata_set = MetadataSet()
        assert metadata_set.metadata == []
        assert [x for x in metadata_set] == []

    def test_iterate_over_populated(self):
        """Should initialise with list of metadata if provided
        """
        metadata_list = [
            {
                "name": "total_lane_kilometres",
                "spatial_resolution": region_set,
                "temporal_resolution": interval_set,
                "units": "kilometers"
            }
        ]
        metadata_set = MetadataSet(metadata_list)
        expected_metadata = Metadata(
            "total_lane_kilometres",
            region_set,
            interval_set,
            "kilometers"
        )

        actual = metadata_set.metadata
        assert actual == [expected_metadata]

        actual = [(k, v) for k, v in metadata_set.items()]
        assert actual == [("total_lane_kilometres", expected_metadata)]

    def test_add_meta_object(self):
        """Should allow adding a Metadata object
        """
        metadata = Metadata("total_lane_kilometres", region_set, interval_set,
                            "kilometer")
        metadata_set = MetadataSet()
        metadata_set.add_metadata(metadata)

        # access list of metadata attributes
        assert metadata_set.names == ["total_lane_kilometres"]
        assert metadata_set.spatial_resolutions == [region_set]
        assert metadata_set.temporal_resolutions == [interval_set]
        assert metadata_set.units == ["kilometer"]

    def test_add_meta_dict(self):
        """Should allow adding a dict with required keys
        """
        metadata = {
            "name": "total_lane_kilometres",
            "spatial_resolution": region_set,
            "temporal_resolution": interval_set,
            "units": "kilometers"
        }
        metadata_set = MetadataSet()
        metadata_set.add_metadata(metadata)

        # access list of metadata attributes
        assert metadata_set.names == ["total_lane_kilometres"]
        assert metadata_set.spatial_resolutions == [region_set]
        assert metadata_set.temporal_resolutions == [interval_set]
        assert metadata_set.units == ["kilometer"]
