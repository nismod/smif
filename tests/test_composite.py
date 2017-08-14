from unittest.mock import MagicMock

from pytest import raises
from smif.composite import ScenarioModel, SectorModel, SosModel


class TestSosModel:

    def test_simulate_data_not_present(self):
        """Raise a ValueError if an input is defined but no dependency links
        it to a data source
        """

        sos_model = SosModel('test')
        model = SectorModel('test_model', ['input'], ['output'])
        sos_model.add_model(model)
        with raises(ValueError):
            data = {'input_not_here': 0}
            sos_model.simulate(data)

    def test_simulate_data_not_present_with_mock(self):
        """Raise a ValueError if an input is defined but no dependency links
        it to a data source
        """

        sos_model = SosModel('test')
        mock_model = MagicMock(model_inputs=['input'])
        sos_model._models = {'test_model': mock_model}
        with raises(ValueError):
            data = {'input_not_here': 0}
            sos_model.simulate(data)


class TestCompositeIntegration:

    def test_simplest_case(self):
        """One scenario only
        """
        elec_scenario = ScenarioModel('electricity_demand_scenario', [
                                 'electricity_demand_output'])
        elec_scenario.add_data({'electricity_demand_output': 123})
        sos_model = SosModel('simple')
        sos_model.add_model(elec_scenario)
        actual = sos_model.simulate()
        expected = {'electricity_demand_scenario':
                    {'electricity_demand_output': 123}}
        assert actual == expected

    def test_sector_model_null_model(self):
        no_inputs = SectorModel('energy_sector_model', [], [])
        no_inputs.add_executable(lambda x: x)
        actual = no_inputs.simulate()
        expected = None
        assert actual == expected

    def test_sector_model_one_input(self):
        elec_scenario = ScenarioModel('scenario', [
                                 'output'])
        elec_scenario.add_data({'output': 123})

        energy_model = SectorModel('model', [], [])
        energy_model.add_input('input')
        energy_model.add_dependency(elec_scenario, 'output', 'input')
        energy_model.add_executable(lambda x: x)

        sos_model = SosModel('blobby')
        sos_model.add_model(elec_scenario)
        sos_model.add_model(energy_model)

        actual = sos_model.simulate()

        expected = {'model': {'input': 123}, 'scenario': {'output': 123}}
        assert actual == expected

    def test_composite_nested_sos_model(self):
        """System of systems example with two nested SosModels, two Scenarios
        and one SectorModel. One dependency is defined at the SectorModel
        level, another at the lower SosModel level
        """

        elec_scenario = ScenarioModel('electricity_demand_scenario', [
                                 'electricity_demand_output'])
        elec_scenario.add_data({'electricity_demand_output': 123})

        energy_model = SectorModel('energy_sector_model', [], [])
        energy_model.add_input('electricity_demand_input')
        energy_model.add_input('fluffiness_input')
        energy_model.add_output('cost')
        energy_model.add_output('fluffyness')

        def energy_function(input_data):
            """Mimics the running of a sector model
            """
            results = {}
            demand = input_data['electricity_demand_input']
            fluff = input_data['fluffiness_input']
            results['cost'] = demand * 1.2894
            results['fluffyness'] = fluff * 22
            return results

        energy_model.add_executable(energy_function)
        energy_model.add_dependency(elec_scenario,
                                    'electricity_demand_output',
                                    'electricity_demand_input')

        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(elec_scenario)
        sos_model_lo.add_model(energy_model)

        fluf_scenario = ScenarioModel('fluffiness_scenario', ['fluffiness'])
        fluf_scenario.add_data({'fluffiness': 12})
        sos_model_lo.add_dependency(fluf_scenario,
                                    'fluffiness',
                                    'fluffiness_input')

        sos_model_high = SosModel('higher')
        sos_model_high.add_model(sos_model_lo)
        sos_model_high.add_model(fluf_scenario)

        actual = sos_model_high.simulate()
        expected = {'fluffiness_scenario': {'fluffiness': 12},
                    'lower': {'electricity_demand_scenario':
                              {'electricity_demand_output': 123},
                              'energy_sector_model': {'cost': 158.5962,
                                                      'fluffyness': 264}
                              }
                    }

        assert actual == expected

    def test_loop(self):
        """Fails because no functionality to deal with loops
        """
        energy_model = SectorModel('energy_sector_model', [], [])
        energy_model.add_input('electricity_demand_input')
        energy_model.add_output('fluffiness')

        def energy_function(input_data):
            """Mimics the running of a sector model
            """
            results = {}
            fluff = input_data['electricity_demand_input']
            results['fluffiness'] = fluff * 22
            return results

        energy_model.add_executable(energy_function)

        water_model = SectorModel('water_sector_model', [], [])
        water_model.add_input('fluffyness')
        water_model.add_output('electricity_demand')

        def water_function(input_data):
            results = {}
            fluff = input_data['fluffyness']
            results['electricity_demand'] = fluff / 1.23
            return results

        water_model.add_executable(water_function)

        sos_model = SosModel('energy_water_model')
        water_model.add_dependency(energy_model, 'fluffiness', 'fluffyness')
        energy_model.add_dependency(water_model, 'electricity_demand',
                                    'electricity_demand_input')

        sos_model.add_model(water_model)
        sos_model.add_model(energy_model)

        # sos_model.simulate()
