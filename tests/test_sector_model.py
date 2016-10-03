import logging

import pytest
from smif.abstract import ConcreteAsset as Asset
from smif.abstract import State
from tests.fixtures.water_supply import WaterSupplyPythonAssets

_log_format = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
logging.basicConfig(filename='test_sector_model.log',
                    level=logging.DEBUG,
                    format=_log_format,
                    filemode='w')


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
        ws = WaterSupplyPythonAssets()
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
