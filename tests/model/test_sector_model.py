"""Test SectorModel and SectorModelBuilder
"""
import smif.sample_project.models.water_supply
from pytest import fixture, mark, raises
from smif.metadata import Spec
from smif.model.sector_model import SectorModel
from smif.sample_project.models.water_supply import WaterSupplySectorModel


@fixture(scope='function')
def sector_model_dict():
    path = smif.sample_project.models.water_supply.__file__
    config = {
        "name": "water_supply",
        "description": 'a description',
        "path": path,
        "classname": "WaterSupplySectorModel",
        "inputs": [
            {
                'name': 'raininess',
                'description': None,
                'abs_range': None,
                'exp_range': None,
                'default': None,
                'dims': ['LSOA'],
                'coords': {'LSOA': [1, 2, 3]},
                'dtype': 'float',
                'unit': 'milliliter'
            }
        ],
        "outputs": [
            {
                'name': 'cost',
                'description': None,
                'abs_range': None,
                'exp_range': None,
                'default': None,
                'dims': ['LSOA'],
                'coords': {'LSOA': [1, 2, 3]},
                'dtype': 'float',
                'unit': 'million GBP'
            },
            {
                'name': 'water',
                'description': None,
                'abs_range': None,
                'exp_range': None,
                'default': None,
                'dims': ['LSOA'],
                'coords': {'LSOA': [1, 2, 3]},
                'dtype': 'float',
                'unit': 'megaliter'
            }
        ],
        "parameters": [
            {
                'name': 'assump_diff_floorarea_pp',
                'description': 'Difference in floor area per person',
                'dims': ['national'],
                'coords': {'national': ['GB']},
                'abs_range': (0.5, 2),
                'exp_range': (0.5, 2),
                'dtype': 'float',
                'default': 1,
                'unit': '%'
            }
        ]
    }
    return config


@fixture(scope='function')
def sector_model():
    model = WaterSupplySectorModel('water_supply')
    model.description = 'a description'
    model.add_input(
        Spec.from_dict({
            'name': 'raininess',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'milliliter'
        })
    )
    model.add_parameter(
        Spec.from_dict({
            'name': 'assump_diff_floorarea_pp',
            'description': 'Difference in floor area per person',
            'dims': ['national'],
            'coords': {'national': ['GB']},
            'abs_range': (0.5, 2),
            'exp_range': (0.5, 2),
            'dtype': 'float',
            'default': 1,
            'unit': '%'
        })
    )
    model.add_output(
        Spec.from_dict({
            'name': 'cost',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'million GBP'
        })
    )
    model.add_output(
        Spec.from_dict({
            'name': 'water',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'megaliter'
        })
    )
    return model


class EmptySectorModel(SectorModel):
    """Simulate nothing
    """
    def simulate(self, data):
        return data


@fixture(scope='function')
def empty_sector_model():
    return EmptySectorModel('test_model')


class TestSectorModel():
    """A SectorModel has inputs, outputs, and an implementation
    of simulate
    """
    def test_abstract(self):
        with raises(TypeError) as ex:
            SectorModel('cannot_instantiate')
        assert "Can't instantiate abstract class" in str(ex)

    def test_construct(self, sector_model):
        assert sector_model.description == 'a description'
        assert sector_model.inputs == {
            'raininess': Spec.from_dict({
                'name': 'raininess',
                'dims': ['LSOA'],
                'coords': {'LSOA': [1, 2, 3]},
                'dtype': 'float',
                'unit': 'milliliter'
            })
        }
        assert sector_model.parameters == {
            'assump_diff_floorarea_pp': Spec.from_dict({
                'name': 'assump_diff_floorarea_pp',
                'description': 'Difference in floor area per person',
                'dims': ['national'],
                'coords': {'national': ['GB']},
                'abs_range': (0.5, 2),
                'exp_range': (0.5, 2),
                'dtype': 'float',
                'default': 1,
                'unit': '%'
            })
        }
        assert sector_model.outputs == {
            'cost': Spec.from_dict({
                'name': 'cost',
                'dims': ['LSOA'],
                'coords': {'LSOA': [1, 2, 3]},
                'dtype': 'float',
                'unit': 'million GBP'
            }),
            'water': Spec.from_dict({
                'name': 'water',
                'dims': ['LSOA'],
                'coords': {'LSOA': [1, 2, 3]},
                'dtype': 'float',
                'unit': 'megaliter'
            })
        }

    def test_from_dict(self, sector_model_dict):
        """Create using classmethod from config
        """
        sector_model = EmptySectorModel.from_dict(sector_model_dict)
        assert sector_model.name == 'water_supply'

    def test_from_dict_no_inputs(self, sector_model_dict):
        """Default sensibly with missing config values
        """
        sector_model_dict['inputs'] = []
        model = EmptySectorModel.from_dict(sector_model_dict)
        assert model.inputs == {}

    def test_as_dict(self, sector_model, sector_model_dict):
        """Serialise back to dict
        """
        actual = sector_model.as_dict()
        # indifferent up to order of inputs/outputs
        actual['inputs'].sort(key=lambda m: m['name'])
        actual['outputs'].sort(key=lambda m: m['name'])
        actual['parameters'].sort(key=lambda m: m['name'])
        assert actual == sector_model_dict

    def test_add_input(self, empty_sector_model):
        """Add an input spec
        """
        spec = Spec(
            name='input_name',
            dims=['dim'],
            coords={'dim': [0]},
            dtype='int'
        )
        empty_sector_model.add_input(spec)
        assert empty_sector_model.inputs == {'input_name': spec}

    def test_add_output(self, empty_sector_model):
        """Add an output spec
        """
        spec = Spec(
            name='output_name',
            dims=['dim'],
            coords={'dim': [0]},
            dtype='int'
        )
        empty_sector_model.add_output(spec)
        assert empty_sector_model.outputs == {'output_name': spec}

    def test_add_parameter(self, empty_sector_model):
        """Adding a parameter adds a reference to the parameter list entry to
        the model that contains it.
        """
        spec = Spec.from_dict({
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'abs_range': (0, 100),
            'exp_range': (3, 10),
            'dims': ['national'],
            'coords': {'national': ['GB']},
            'dtype': 'float',
            'default': 3,
            'unit': '%'
        })
        empty_sector_model.add_parameter(spec)
        assert empty_sector_model.parameters['smart_meter_savings'] == spec

    def test_simulate_exists(self, empty_sector_model):
        """Call simulate
        """
        empty_sector_model.simulate(None)

    def test_before_model_run_exists(self, empty_sector_model):
        """Call before_model_run
        """
        empty_sector_model.before_model_run(None)


@mark.xfail()
class TestSectorModelDimensions():
    """SectorModels should have access to dimension metadata, including regions
    (name, geometry and centroid) and intervals.
    """
    def test_access_intervals(self, empty_sector_model):
        """Access names
        """
        interval_names = empty_sector_model.get_interval_names('annual')
        assert interval_names == ['1']

    def test_access_region_names(self, empty_sector_model):
        """Access names
        """
        region_names = empty_sector_model.get_region_names('half_squares')
        assert region_names == ['a', 'b']

    def test_access_region_geometries(self, empty_sector_model):
        """Access geometries
        """
        actual = empty_sector_model.get_regions('half_squares')

        expected = [
            {
                'type': 'Feature',
                'properties': {'name': 'a'},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                                     (1.0, 0.0), (0.0, 0.0),),)
                }
            },
            {
                'type': 'Feature',
                'properties': {'name': 'b'},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': (((0.0, 1.0), (0.0, 2.0), (1.0, 2.0),
                                     (1.0, 1.0), (0.0, 1.0),),)
                }
            },
        ]
        assert actual == expected

    def test_access_region_centroids(self, empty_sector_model):
        """Access geometry centroids
        """
        actual = empty_sector_model.get_region_centroids('half_squares')

        expected = [
            {
                'type': 'Feature',
                'properties': {'name': 'a'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': (0.5, 0.5)
                }
            },
            {
                'type': 'Feature',
                'properties': {'name': 'b'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': (0.5, 1.5)
                }
            }
        ]
        assert actual == expected
