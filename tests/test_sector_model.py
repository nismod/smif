import pytest
from smif.abstract import ConcreteAsset as Asset
from smif.abstract import State
from smif.sector_model import SectorModel, SectorModelBuilder
from fixtures.water_supply import WaterSupplySectorModelWithAssets

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
                                           'water_supply.py'))
        builder = SectorModelBuilder('water_supply')
        builder.load_model(model_path, 'WaterSupplySectorModel')


        assets = [
            {
                'name': 'water_asset_a',
                'capital_cost': {
                    'unit': '£/kW',
                    'value': 1000
                    },
                'economic_lifetime': 25,
                'operational_lifetime': 25
            }
        ]
        builder.add_assets(assets)

        # builder.add_inputs(inputs)
        # builder.add_outputs(outputs)

        model = builder.finish()
        assert isinstance(model, SectorModel)

        assert model.name == 'water_supply'
        assert model.asset_names == ['water_asset_a']
        assert model.assets == assets

class TestSectorModel(object):
    def test_assets_load(self):
        assets = {
            'water_asset_a': {},
            'water_asset_b': {},
            'water_asset_c': {}
        }
        model = EmptySectorModel()
        model.assets = assets

        asset_names = model.asset_names

        assert len(asset_names) == 3
        assert 'water_asset_a' in asset_names
        assert 'water_asset_b' in asset_names
        assert 'water_asset_c' in asset_names


    def test_attributes_load(self):

        assets = {
            'water_asset_a': {
                'capital_cost': {
                    'value': 1000,
                    'unit': '£/kW'
                },
                'economic_lifetime': 25,
                'operational_lifetime': 25
            },
            'water_asset_b': {
                'capital_cost': {
                    'value': 1500,
                    'unit': '£/kW'
                }
            },
            'water_asset_c': {
                'capital_cost': {
                    'value': 3000,
                    'unit': '£/kW'
                }
            }
        }
        model = EmptySectorModel()
        model.assets = assets
        actual = model.assets
        expected = {
            'water_asset_a': {
                'capital_cost': {
                    'value': 1000,
                    'unit': '£/kW'
                },
                'economic_lifetime': 25,
                'operational_lifetime': 25
            },
            'water_asset_b': {
                'capital_cost': {
                    'value': 1500,
                    'unit': '£/kW'
                }
            },
            'water_asset_c': {
                'capital_cost': {
                    'value': 3000,
                    'unit': '£/kW'
                }
            }
        }

        assert actual == expected

        for asset in model.asset_names:
            assert asset in expected.keys()


@pytest.mark.skip(reason="no way of currently testing this")
class TestAssets:
    """
    """

    def test_declaring_an_asset(self):

        name, capacity = ('dec asset treatment plant', 100.)
        asset = Asset(name, capacity)
        state = asset.get_state()
        assert state == [{'dec asset treatment plant': 100.}]

        name, capacity = ('dec asset pipes', 100.)
        asset = Asset(name, capacity)
        state = asset.get_state()
        assert state == [{'dec asset treatment plant': 100.},
                         {'dec asset pipes': 100.}]

    def test_add_a_list_of_assets(self):

        all_assets = []
        list_of_assets = [{'name': 'add_a_list treatment plant',
                           'capacity': 100.},
                          {'name': 'add_a_list pipes',
                           'capacity': 100.}]
        for asset in list_of_assets:
            all_assets.append(Asset(asset['name'], asset['capacity']))


@pytest.mark.skip(reason="no way of currently testing this")
class TestStates:
    """
    """

    def test_declaring_a_state(self):

        state_dict = {'mega plant': 100.,
                      'wobbly pipes': 100.}

        list_of_assets = [('mega plant', 100.,
                           'oxford', 2010, 'water_supply'),
                          ('wobbly pipes', 100.,
                           'oxford', 2010, 'water_supply')
                          ]
        state_parameter_map = {'mega plant': state_dict['mega plant'],
                               'wobbly pipes': state_dict['wobbly pipes']}
        state = State('oxford', 2010, 'water_supply', state_parameter_map)
        state.initialise_from_tuples(list_of_assets)
        actual_current_state = state.current_state
        expected_current_state = {'model': 'water_supply',
                                  'region': 'oxford',
                                  'timestep': 2010,
                                  'assets': {'mega plant': 100.0,
                                             'wobbly pipes': 100.0}
                                  }
        assert actual_current_state == expected_current_state


@pytest.mark.skip(reason="no way of currently testing this")
class TestSectorModel:

    def test_adding_an_asset_to_sector_model(self):
        """New capacity updates the state of the sector model

        """
        assets = [('treatment plant', 1)]
        data = {'raininess': 2, 'plants': 1}
        ws = WaterSupplySectorModelWithAssets()
        ws.initialise(data, assets)
        expected_pre_state = {'treatment plant': 1}
        actual_pre_state = ws.state.current_state

        assert actual_pre_state['assets'] == expected_pre_state
        assert ws.model.number_of_treatment_plants == 1

        ws.simulate()
        assert ws.results['water'] == 1
        assert ws.results['cost'] == 1

        additional_asset = [{'name': 'treatment plant', 'capacity': 1}]
        ws.state.add_new_capacity(additional_asset)
        ws.state.update_state()
        actual_post_state = ws.state.current_state
        expected_post_state = {'treatment plant': 2}
        assert actual_post_state['assets'] == expected_post_state

        assert ws.state._assets['treatment plant'].capacity == 2
        # Note that because we pass by value, the state method cannot update
        # sector model directly.
        ws.simulate()
        assert ws.model.number_of_treatment_plants == 2
        assert ws.results['water'] == 2
        assert ws.results['cost'] == 2
