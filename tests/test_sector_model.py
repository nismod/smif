import pytest
from smif.intervention import Asset
from smif.sector_model import SectorModel, SectorModelBuilder
from . fixtures.water_supply import WaterSupplySectorModelWithAssets

class EmptySectorModel(SectorModel):
    def simulate(self, static_inputs, decision_variables):
        pass

    def extract_obj(self, results):
        return 0

class TestSectorModelBuilder():

    def test_sector_model_builder(self, setup_project_folder):
        project_path = setup_project_folder

        model_path = str(project_path.join('models',
                                           'water_supply',
                                           '__init__.py'))
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

class TestSectorModel(object):
    def test_assets_load_names(self):
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


    def test_assets_load(self):
        assets = [
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
        model.assets = assets
        actual = model.assets

        assert actual == assets

        for asset_name in model.intervention_names:
            assert asset_name in [
                'water_asset_a',
                'water_asset_b',
                'water_asset_c'
            ]
