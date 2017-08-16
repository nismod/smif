from unittest.mock import Mock

import networkx
from pytest import raises
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel as AbstractSectorModel
from smif.model.sos_model import ModelSet, SosModel


class SectorModel(AbstractSectorModel):

    def simulate(self, timestep, data=None):
        return data

    def extract_obj(self):
        pass

    def initialise(self):
        pass


class TestModelSet:

    def test_model_set(self):
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')
        elec_scenario.add_data({2010: {'output': 123}})

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        model_set = ModelSet([elec_scenario], sos_model)
        model_set.run(2010)


class TestBasics:

    def test_dependency_not_present(self):
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')
        elec_scenario.add_data({'output': 123})

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        with raises(ValueError):
            energy_model.add_dependency(elec_scenario, 'not_present',
                                        'input')

        with raises(ValueError):
            energy_model.add_dependency(elec_scenario, 'output',
                                        'not_correct_input_name')


class TestDependencyGraph:

    def test_simple_graph(self):
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')
        elec_scenario.add_data({'output': 123})

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        # Builds the dependency graph
        sos_model._check_dependencies()

        graph = sos_model.dependency_graph

        assert energy_model in graph
        assert elec_scenario in graph

        assert graph.edges() == [(elec_scenario, energy_model)]

    def test_get_model_sets(self):

        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')

        elec_scenario.add_data({'output': 123})

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        sos_model._check_dependencies()

        actual = sos_model._get_model_sets_in_run_order()
        expected = [{'scenario'}, {'model'}]

        for modelset, name in zip(actual, expected):
            assert modelset._model_names == name

    def test_topological_sort(self):
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')

        elec_scenario.add_data({'output': 123})

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        sos_model._check_dependencies()

        graph = sos_model.dependency_graph
        actual = networkx.topological_sort(graph, reverse=False)
        assert actual == [elec_scenario, energy_model]


class TestSosModel:

    def test_simulate_data_not_present(self):
        """Raise a ValueError if an input is defined but no dependency links
        it to a data source
        """

        sos_model = SosModel('test')
        model = SectorModel('test_model')
        model.add_input('input', Mock(), Mock(), 'units')
        sos_model.add_model(model)
        data = {'input_not_here': 0}
        with raises(AssertionError):
            sos_model.simulate(2010, data)


class TestCompositeIntegration:

    def test_simplest_case(self):
        """One scenario only
        """
        elec_scenario = ScenarioModel('electricity_demand_scenario')
        elec_scenario.add_output('electricity_demand_output',
                                 Mock(), Mock(), 'unit')

        elec_scenario.add_data({2010: {'electricity_demand_output': 123}})
        sos_model = SosModel('simple')
        sos_model.add_model(elec_scenario)
        actual = sos_model.simulate(2010)
        expected = {2010: {'electricity_demand_scenario':
                    {'electricity_demand_output': 123}}}
        assert actual == expected

    def test_sector_model_null_model(self):
        no_inputs = SectorModel('energy_sector_model', [], [])
        # no_inputs.add_executable(lambda x: x)
        actual = no_inputs.simulate(2010)
        expected = None
        assert actual == expected

    def test_sector_model_one_input(self):
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')
        elec_scenario.add_data({2010: {'output': 123}})

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('blobby')
        sos_model.add_model(elec_scenario)
        sos_model.add_model(energy_model)

        actual = sos_model.simulate(2010)

        expected = {2010: {'model': {'input': 123},
                           'scenario': {'output': 123}
                           }
                    }
        assert actual == expected

    def test_composite_nested_sos_model(self):
        """System of systems example with two nested SosModels, two Scenarios
        and one SectorModel. One dependency is defined at the SectorModel
        level, another at the lower SosModel level
        """

        elec_scenario = ScenarioModel('electricity_demand_scenario')
        elec_scenario.add_output('electricity_demand_output',
                                 Mock(), Mock(), 'unit')
        elec_scenario.add_data({'electricity_demand_output': 123})

        energy_model = SectorModel('energy_sector_model')
        energy_model.add_input('electricity_demand_input', Mock(), Mock(), 'unit')
        energy_model.add_input('fluffiness_input', Mock(), Mock(), 'unit')
        energy_model.add_output('cost', Mock(), Mock(), 'unit')
        energy_model.add_output('fluffyness', Mock(), Mock(), 'unit')

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

        actual = sos_model_high.simulate(2010)
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
        energy_model = SectorModel('energy_sector_model')
        energy_model.add_input('electricity_demand_input', Mock(), Mock(), 'unit')
        energy_model.add_output('fluffiness', Mock(), Mock(), 'unit')

        def energy_function(input_data):
            """Mimics the running of a sector model
            """
            results = {}
            fluff = input_data['electricity_demand_input']
            results['fluffiness'] = fluff * 22
            return results

        energy_model.add_executable(energy_function)

        water_model = SectorModel('water_sector_model', [], [])
        water_model.add_input('fluffyness', Mock(), Mock(), 'unit')
        water_model.add_output('electricity_demand', Mock(), Mock(), 'unit')

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

        with raises(NotImplementedError):
            sos_model.simulate(2010)
