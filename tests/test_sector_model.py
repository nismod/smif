"""Test SectorModel and SectorModelBuilder
"""
from pytest import raises
from smif.parameters import ModelParameters
from smif.sector_model import SectorModel, SectorModelBuilder


class EmptySectorModel(SectorModel):
    def initialise(self, initial_conditions):
        pass

    def simulate(self, decisions, state, data):
        return state, {}

    def extract_obj(self, results):
        return 0


class TestSectorModelBuilder():

    def test_sector_model_builder(self, setup_project_folder):
        model_path = str(setup_project_folder.join('models', 'water_supply', '__init__.py'))
        builder = SectorModelBuilder('water_supply')
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
        builder = SectorModelBuilder('water_supply')
        with raises(FileNotFoundError) as ex:
            builder.load_model('/fictional/path/to/model.py', 'WaterSupplySectorModel')
        msg = "Cannot find '/fictional/path/to/model.py' for the 'water_supply' model"
        assert msg in str(ex.value)

    def test_add_no_inputs(self, setup_project_folder):
        model_path = str(setup_project_folder.join('models', 'water_supply', '__init__.py'))
        builder = SectorModelBuilder('water_supply')
        builder.load_model(model_path, 'WaterSupplySectorModel')
        builder.add_inputs(None)
        assert isinstance(builder._sector_model.inputs, ModelParameters)
        assert len(builder._sector_model.inputs) == 0


class TestSectorModel(object):

    def test_interventions_names(self):
        assets = [
            {'name': 'water_asset_a'},
            {'name': 'water_asset_b'},
            {'name': 'water_asset_c'}
        ]
        model = EmptySectorModel()
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
        model = EmptySectorModel()
        model.interventions = interventions
        actual = model.interventions

        assert actual == interventions

        assert sorted(model.intervention_names) == [
            'water_asset_a',
            'water_asset_b',
            'water_asset_c'
        ]
