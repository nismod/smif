"""Test SectorModel and SectorModelBuilder
"""
from copy import copy
from unittest.mock import Mock

from pytest import fixture, raises
from smif.convert.area import get_register
from smif.metadata import Metadata, MetadataSet
from smif.model.sector_model import SectorModel, SectorModelBuilder
from smif.parameters import ParameterList


@fixture(scope='function')
def get_sector_model_config(setup_folder_structure, setup_runpy_file, setup_registers):

    path = setup_folder_structure
    water_supply_wrapper_path = str(
        path.join(
            'models', 'water_supply', '__init__.py'
        )
    )

    config = {
        "name": "water_supply",
        "description": 'a description',
        "path": water_supply_wrapper_path,
        "classname": "WaterSupplySectorModel",
        "inputs": [
            {
                'name': 'raininess',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'units': 'milliliter'
            }
        ],
        "outputs": [
            {
                'name': 'cost',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'units': 'million GBP'
            },
            {
                'name': 'water',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'units': 'megaliter'
            }
        ],
        "initial_conditions": [
            {"name": "water_asset_a", "build_year": 2010},
            {"name": "water_asset_b", "build_year": 2010},
            {"name": "water_asset_c", "build_year": 2010}],
        "interventions": [
            {"name": "water_asset_a", "location": "oxford"},
            {"name": "water_asset_b", "location": "oxford"},
            {"name": "water_asset_c", "location": "oxford"}
        ],
        "parameters": [
            {
                'name': 'assump_diff_floorarea_pp',
                'description': 'Difference in floor area per person \
                                in end year compared to base year',
                'absolute_range': (0.5, 2),
                'suggested_range': (0.5, 2),
                'default_value': 1,
                'units': '%'
            }
        ]
    }

    return config


class EmptySectorModel(SectorModel):
    """Simulate nothing
    """
    def simulate(self, timestep, data=None):
        return {}

    def extract_obj(self, results):
        return 0


class TestCompositeSectorModel():

    def test_add_input(self):

        model = EmptySectorModel('test_model')
        model.add_input('input_name', [], [], 'units')

        inputs = model.inputs

        assert inputs.names == ['input_name']
        assert inputs.units == ['units']

        assert inputs['input_name'] == Metadata('input_name', [], [], 'units')

    def test_add_output(self):

        model = EmptySectorModel('test_model')
        model.add_output('output_name', Mock(), Mock(), 'units')

        outputs = model.outputs

        assert outputs.names == ['output_name']
        assert outputs.units == ['units']

    def test_run_sector_model(self):

        model = EmptySectorModel('test_model')
        model.add_input('input_name', [], [], 'units')
        data = {'input_name': [0]}
        actual = model.simulate(2010, data)
        assert actual == {}


class TestSectorModelBuilder():

    def test_add_inputs(self, setup_folder_structure, setup_runpy_file):

        model_path = str(setup_folder_structure.join('models', 'water_supply',
                                                     '__init__.py'))

        builder = SectorModelBuilder('test')
        builder.load_model(model_path, 'WaterSupplySectorModel')

        inputs = [{'name': 'an_input',
                   'spatial_resolution': 'LSOA',
                   'temporal_resolution': 'annual',
                   'units': 'tonnes'}]

        builder.add_inputs(inputs)

        assert 'an_input' in builder._sector_model.inputs.names

    def test_sector_model_builder(self, setup_folder_structure, setup_runpy_file):
        model_path = str(setup_folder_structure.join('models', 'water_supply',
                                                     '__init__.py'))

        register = Mock()
        register.get_entry = Mock(return_value='a_resolution_set')

        registers = {'regions': register,
                     'intervals': register}

        builder = SectorModelBuilder('water_supply', registers)
        builder.load_model(model_path, 'WaterSupplySectorModel')

        assets = [
            {
                'name': 'water_asset_a',
                'capital_cost': 1000,
                'economic_lifetime': 25,
                'operational_lifetime': 25,
                'location': 'Narnia'
            }
        ]
        builder.add_interventions(assets)

        # builder.add_inputs(inputs)
        # builder.add_outputs(outputs)

        model = builder.finish()
        assert isinstance(model, SectorModel)

        assert model.name == 'water_supply'
        assert model.intervention_names == ['water_asset_a']

    def test_path_not_found(self):
        builder = SectorModelBuilder('water_supply', Mock())
        with raises(FileNotFoundError) as ex:
            builder.load_model('/fictional/path/to/model.py', 'WaterSupplySectorModel')
        msg = "Cannot find '/fictional/path/to/model.py' for the 'water_supply' model"
        assert msg in str(ex.value)

    def test_build_from_config(self, get_sector_model_config):
        config = get_sector_model_config
        builder = SectorModelBuilder('test_sector_model')
        timesteps = [2015, 2020]
        builder.construct(config, timesteps)
        sector_model = builder.finish()
        assert sector_model.name == 'water_supply'
        assert sector_model.timesteps == timesteps

        actual = sector_model.as_dict()
        # sort to match expected output
        actual['inputs'].sort(key=lambda m: m['name'])
        actual['outputs'].sort(key=lambda m: m['name'])
        assert actual == config


class TestInputs:

    def test_add_no_inputs(self, setup_folder_structure, setup_runpy_file):
        model_path = str(setup_folder_structure.join('models', 'water_supply', '__init__.py'))
        registers = {'regions': Mock(),
                     'intervals': Mock()}

        builder = SectorModelBuilder('water_supply_test', registers)
        builder.load_model(model_path, 'WaterSupplySectorModel')
        builder.add_inputs(None)
        sector_model = builder.finish()
        assert isinstance(sector_model.inputs, MetadataSet)
        actual_inputs = sector_model.inputs.names
        assert actual_inputs == []


class TestSectorModelInterventions(object):

    def test_interventions_names(self):
        mock_asset_a = Mock()
        mock_asset_a.name = 'water_asset_a'
        mock_asset_b = Mock()
        mock_asset_b.name = 'water_asset_b'
        mock_asset_c = Mock()
        mock_asset_c.name = 'water_asset_c'

        assets = [mock_asset_a, mock_asset_b, mock_asset_c]
        model = EmptySectorModel('test_model')
        model.interventions = assets

        actual = model.intervention_names
        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']

        assert len(actual) == 3
        assert actual == expected

    def test_get_interventions(self):
        mock_asset_a = Mock()
        mock_asset_a.name = 'water_asset_a'
        mock_asset_a.as_dict = Mock(return_value={'name': 'water_asset_a'})
        mock_asset_b = Mock()
        mock_asset_b.name = 'water_asset_b'
        mock_asset_b.as_dict = Mock(return_value={'name': 'water_asset_b'})
        mock_asset_c = Mock()
        mock_asset_c.name = 'water_asset_c'

        assets = [mock_asset_a, mock_asset_b, mock_asset_c]
        model = EmptySectorModel('test_model')
        model.interventions = assets

        state = [{'name': 'water_asset_a', 'build_year': 2010},
                 {'name': 'water_asset_b', 'build_year': 2015}]
        actual = model.get_current_interventions(state)
        expected = [{'name': 'water_asset_a', 'build_year': 2010},
                    {'name': 'water_asset_b', 'build_year': 2015}]
        assert actual == expected


class TestParameters():

    def test_add_parameter(self):
        """Adding a parameter adds a reference to the parameter list entry to
        the model that contains it.
        """

        model = copy(EmptySectorModel('test_model'))
        model.simulate = lambda x, y: {'savings': y['smart_meter_savings']}

        param_config = {'name': 'smart_meter_savings',
                        'description': 'The savings from smart meters',
                        'absolute_range': (0, 100),
                        'suggested_range': (3, 10),
                        'default_value': 3,
                        'units': '%'}
        model.add_parameter(param_config)

        assert isinstance(model.parameters, ParameterList)
        assert model.parameters['smart_meter_savings'].as_dict() == param_config

        actual = model.simulate(2010, {'smart_meter_savings': 3})
        expected = {'savings': 3}
        assert actual == expected


class TestSectorModelIntervals():
    """SectorModels should have access to intervals (name)
    """

    def test_access_region_names(self):
        """Access names
        """
        model = EmptySectorModel('region_test')
        interval_names = model.get_interval_names('annual')
        assert interval_names == ['1']


class TestSectorModelRegions():
    """SectorModels should have access to regions (name, geometry and centroid)
    """
    def get_model(self):
        """Get a model with region register as setup in conftest
        """
        rreg = get_register()
        model = EmptySectorModel('region_test')
        model.regions = rreg
        return model

    def test_access_region_names(self):
        """Access names
        """
        model = self.get_model()
        region_names = model.get_region_names('half_squares')
        assert region_names == ['a', 'b']

    def test_access_region_geometries(self):
        model = self.get_model()
        actual = model.get_regions('half_squares')

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

    def test_access_region_centroids(self):
        model = self.get_model()
        actual = model.get_region_centroids('half_squares')

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
            },
        ]
        assert actual == expected
