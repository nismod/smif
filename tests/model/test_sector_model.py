"""Test SectorModel and SectorModelBuilder
"""
import smif.sample_project.models.water_supply
from pytest import fixture, mark
from smif.decision.intervention import Intervention
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
        "initial_conditions": [
            {"name": "water_asset_a", "build_year": 2010},
            {"name": "water_asset_b", "build_year": 2010},
            {"name": "water_asset_c", "build_year": 2010}],
        "interventions": [
            {"name": "water_asset_a", "location": "oxford", "sector": ""},
            {"name": "water_asset_b", "location": "oxford", "sector": ""},
            {"name": "water_asset_c", "location": "oxford", "sector": ""}
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
    model.initial_conditions = [
        {"name": "water_asset_a", "build_year": 2010},
        {"name": "water_asset_b", "build_year": 2010},
        {"name": "water_asset_c", "build_year": 2010}
    ]
    model.add_interventions([
        Intervention.from_dict({"name": "water_asset_a", "location": "oxford"}),
        Intervention.from_dict({"name": "water_asset_b", "location": "oxford"}),
        Intervention.from_dict({"name": "water_asset_c", "location": "oxford"})
    ])
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

        actual = sorted(sector_model.initial_conditions, key=lambda k: k['name'])
        expected = [
            {"name": "water_asset_a", "build_year": 2010},
            {"name": "water_asset_b", "build_year": 2010},
            {"name": "water_asset_c", "build_year": 2010}
        ]
        assert actual == expected

        actual = sorted(sector_model.interventions, key=lambda k: k.name)
        expected = [
            Intervention.from_dict(
                {"name": "water_asset_a", "location": "oxford"}
            ),
            Intervention.from_dict(
                {"name": "water_asset_b", "location": "oxford"}
            ),
            Intervention.from_dict(
                {"name": "water_asset_c", "location": "oxford"}
            )
        ]
        assert actual == expected

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
        actual['interventions'].sort(key=lambda m: m['name'])
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


class TestSectorModelInterventions(object):
    """Interventions can be attached to a model
    - interventions are possible/potential
    - initial_conditions are the initial set of interventions
    """
    def test_add_interventions(self, empty_sector_model):
        """Add interventions
        """
        assets = [
            Intervention.from_dict({
                'name': 'water_asset_a',
                'capital_cost': 1000,
                'economic_lifetime': 25,
                'operational_lifetime': 25,
                'location': 'Narnia'
            })
        ]
        empty_sector_model.add_interventions(assets)
        assert empty_sector_model.intervention_names == ['water_asset_a']

    def test_interventions_names(self, empty_sector_model):
        """Access list of names
        """
        a = Intervention('water_asset_a')
        b = Intervention('water_asset_b')
        c = Intervention('water_asset_c')
        empty_sector_model.add_interventions([a, b, c])

        actual = sorted(empty_sector_model.intervention_names)
        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        assert actual == expected

    def test_get_interventions(self, empty_sector_model):
        a = Intervention.from_dict({
            'name': 'water_asset_a',
            'capacity': 50
        })
        b = Intervention.from_dict({
            'name': 'water_asset_b',
            'capacity': 150
        })
        c = Intervention.from_dict({
            'name': 'water_asset_c',
            'capacity': 100
        })
        empty_sector_model.add_interventions([a, b, c])

        state = [
            {'name': 'water_asset_a', 'build_year': 2010},
            {'name': 'water_asset_b', 'build_year': 2015}
        ]
        actual = empty_sector_model.get_current_interventions(state)
        actual.sort(key=lambda m: m['name'])
        expected = [
            {
                'name': 'water_asset_a',
                'build_year': 2010,
                'capacity': 50,
                'location': None,
                'sector': ''
            },
            {
                'name': 'water_asset_b',
                'build_year': 2015,
                'capacity': 150,
                'location': None,
                'sector': ''
            }
        ]
        assert actual == expected


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
