from smif.abstract import ConcreteAsset as Asset
from smif.abstract import State
from tests.fixtures.water_supply import WaterSupplyPythonAssets


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


class TestStates:
    """
    """

    def test_declaring_a_state(self):

        list_of_assets = [('mega plant', 100.,
                           'oxford', 2010, 'water_supply'),
                          ('wobbly pipes', 100.,
                           'oxford', 2010, 'water_supply')
                          ]
        state_parameter_map = {'mega plant': None,
                               'wobbly pipes': None}
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


class TestSectorModel:

    def test_adding_an_asset_to_sector_model(self):
        """New capacity updates the state of the sector model

        """
        assets = [('treatment plant', 1)]
        data = {'raininess': 2, 'plants': 1}
        ws = WaterSupplyPythonAssets()
        ws.initialise(data, assets)
        expected_pre_state = {'treatment plant': 1}
        actual_pre_state = ws.state.current_state

        assert actual_pre_state['assets'] == expected_pre_state

        additional_asset = [{'name': 'treatment plant', 'capacity': 1}]
        ws.state.add_new_capacity(additional_asset)
        ws.state.update_asset_capacities()
        actual_post_state = ws.state.current_state
        expected_post_state = {'treatment plant': 2}
        assert actual_post_state['assets'] == expected_post_state
