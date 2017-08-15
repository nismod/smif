"""Test SectorModel and SectorModelBuilder
"""
from unittest.mock import Mock

from pytest import raises
from smif.metadata import Metadata, MetadataSet
from smif.sector_model import SectorModel, SectorModelBuilder


class EmptySectorModel(SectorModel):

    def initialise(self, initial_conditions):
        pass

    def simulate(self, decisions, state, data):
        return state, {}

    def extract_obj(self, results):
        return 0


class TestCompositeSectorModel():

    def test_add_input(self):

        model = EmptySectorModel('test_model')
        model.add_input('input_name', [], [], 'units')

        inputs = model.model_inputs

        assert inputs.names == ['input_name']
        assert inputs.units == ['units']

        assert inputs['input_name'] == Metadata('input_name', [], [], 'units')

    def test_add_output(self):

        model = EmptySectorModel('test_model')
        model.add_output('output_name', Mock(), Mock(), 'units')

        outputs = model.model_outputs

        assert outputs.names == ['output_name']
        assert outputs.units == ['units']

    def test_run_sector_model(self):

        model = EmptySectorModel('test_model')
        model.add_input('input_name', [], [], 'units')
        data = {'input_name': [0]}
        actual = model.simulate({}, {}, data)
        assert actual == ({}, {})


class TestSectorModelBuilder():

    def test_add_inputs(self, setup_project_folder):

        model_path = str(setup_project_folder.join('models', 'water_supply',
                                                   '__init__.py'))

        region = Mock()
        region.get_entry = Mock(return_value='a_resolution_set')

        interval = Mock()
        interval.get_entry = Mock(return_value='a_resolution_set')

        registers = {'regions': region,
                     'intervals': interval}

        builder = SectorModelBuilder('test', registers)
        builder.load_model(model_path, 'WaterSupplySectorModel')

        inputs = [{'name': 'an_input',
                   'spatial_resolution': 'big',
                   'temporal_resolution': 'short',
                   'units': 'tonnes'}]

        builder.add_inputs(inputs)

        assert region.get_entry.call_count == 1
        assert interval.get_entry.call_count == 1

    def test_sector_model_builder(self, setup_project_folder):
        model_path = str(setup_project_folder.join('models', 'water_supply',
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
                'type': 'water_pump',
                'attributes': {
                    'capital_cost': 1000,
                    'economic_lifetime': 25,
                    'operational_lifetime': 25
                }
            }
        ]
        builder.add_interventions(assets)

        # builder.add_inputs(inputs)
        # builder.add_outputs(outputs)

        model = builder.finish()
        assert isinstance(model, SectorModel)

        assert model.name == 'water_supply'
        assert model.intervention_names == ['water_asset_a']
        assert model.interventions == assets

    def test_path_not_found(self):
        builder = SectorModelBuilder('water_supply', Mock())
        with raises(FileNotFoundError) as ex:
            builder.load_model('/fictional/path/to/model.py', 'WaterSupplySectorModel')
        msg = "Cannot find '/fictional/path/to/model.py' for the 'water_supply' model"
        assert msg in str(ex.value)


class TestInputs:

    def test_add_no_inputs(self, setup_project_folder):
        model_path = str(setup_project_folder.join('models', 'water_supply', '__init__.py'))
        registers = {'regions': Mock(),
                     'intervals': Mock()}

        builder = SectorModelBuilder('water_supply_test', registers)
        builder.load_model(model_path, 'WaterSupplySectorModel')
        builder.add_inputs(None)
        sector_model = builder.finish()
        assert isinstance(sector_model.model_inputs, MetadataSet)
        actual_inputs = sector_model.model_inputs
        assert [x.name for x in actual_inputs.metadata] == 0


class TestSectorModel(object):

    def test_interventions_names(self):
        assets = [
            {'name': 'water_asset_a'},
            {'name': 'water_asset_b'},
            {'name': 'water_asset_c'}
        ]
        model = EmptySectorModel('test_model')
        model.interventions = assets

        intervention_names = model.intervention_names

        assert len(intervention_names) == 3
        assert 'water_asset_a' in intervention_names
        assert 'water_asset_b' in intervention_names
        assert 'water_asset_c' in intervention_names

    def test_interventions(self):
        interventions = [
            {
                'name': 'water_asset_a',
                'capital_cost': 1000,
                'economic_lifetime': 25,
                'operational_lifetime': 25
            },
            {
                'name': 'water_asset_b',
                'capital_cost': 1500,
            },
            {
                'name': 'water_asset_c',
                'capital_cost': 3000,
            }
        ]
        model = EmptySectorModel('test_model')
        model.interventions = interventions
        actual = model.interventions

        assert actual == interventions

        assert sorted(model.intervention_names) == [
            'water_asset_a',
            'water_asset_b',
            'water_asset_c'
        ]
