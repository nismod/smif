from unittest.mock import Mock

import networkx
import numpy as np
import pytest
from pytest import fixture, raises
from smif.data_layer import DataHandle, MemoryInterface
from smif.metadata import Metadata, MetadataSet
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel as AbstractSectorModel
from smif.model.sos_model import ModelSet, SosModel


@fixture
def get_scenario():
    scenario = ScenarioModel('electricity_demand_scenario')
    scenario.scenario_name = 'Arbitrary Demand Scenario'
    scenario.add_output('electricity_demand_output',
                        scenario.regions.get_entry('LSOA'),
                        scenario.intervals.get_entry('annual'),
                        'unit')
    return scenario


@fixture
def get_sector_model():
    class SectorModel(AbstractSectorModel):
        """A no-op sector model
        """
        def simulate(self, timestep, data=None):
            return data

        def extract_obj(self):
            pass

    return SectorModel


def get_data_handle(model):
    """Return a data handle for the model
    """
    store = MemoryInterface()
    store.write_sos_model_run({
        'name': 'test',
        'narratives': {}
    })
    store.write_scenario_data(
        'Arbitrary Demand Scenario',
        'electricity_demand_output',
        np.array([[123]]),
        'LSOA',
        'annual',
        2010)
    return DataHandle(
        store,
        'test',  # modelrun_name
        2010,  # current_timestep
        [2010],  # timesteps
        model
    )


@fixture
def get_water_sector_model():
    class SectorModel(AbstractSectorModel):
        """
        fluffyness -> electricity_demand
        """

        def simulate(self, data_handle=None):
            x = data_handle['fluffyness']
            data_handle['electricity_demand'] = (x**3) - (6 * x**2) + (0.9 * x) + 0.15
            return data_handle

        def extract_obj(self):
            pass

    water_model = SectorModel('water_supply_model')
    water_model.add_input('fluffyness',
                          water_model.regions.get_entry('LSOA'),
                          water_model.intervals.get_entry('annual'),
                          'unit')
    water_model.add_output('electricity_demand',
                           water_model.regions.get_entry('LSOA'),
                           water_model.intervals.get_entry('annual'),
                           'unit')

    return water_model


@fixture
def get_energy_sector_model():
    class SectorModel(AbstractSectorModel):
        """
        electricity_demand_input -> fluffiness
        """

        def simulate(self, data_handle):
            """Mimics the running of a sector model
            """
            fluff = data_handle['electricity_demand_input']
            data_handle['fluffiness'] = fluff * 0.819
            return data_handle

        def extract_obj(self):
            pass

    energy_model = SectorModel('energy_sector_model')
    energy_model.add_input('electricity_demand_input',
                           energy_model.regions.get_entry('LSOA'),
                           energy_model.intervals.get_entry('annual'),
                           'unit')
    energy_model.add_output('fluffiness',
                            energy_model.regions.get_entry('LSOA'),
                            energy_model.intervals.get_entry('annual'),
                            'unit')

    return energy_model


class TestModelSet:

    def test_model_set(self):
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output',
                                 elec_scenario.regions.get_entry('LSOA'),
                                 elec_scenario.intervals.get_entry('annual'),
                                 'unit')
        ModelSet({elec_scenario.name: elec_scenario})

    def test_model_set_deps(self, get_water_sector_model, get_energy_sector_model):
        pop_scenario = ScenarioModel('population')
        pop_scenario.add_output('population',
                                pop_scenario.regions.get_entry('LSOA'),
                                pop_scenario.intervals.get_entry('annual'),
                                'unit')
        energy_model = get_energy_sector_model
        energy_model.add_input('population',
                               energy_model.regions.get_entry('LSOA'),
                               energy_model.intervals.get_entry('annual'),
                               'unit')
        water_model = get_water_sector_model

        energy_model.add_dependency(pop_scenario, 'population', 'population')
        energy_model.add_dependency(water_model, 'electricity_demand',
                                    'electricity_demand_input')
        water_model.add_dependency(energy_model, 'fluffiness', 'fluffyness')

        model_set = ModelSet({
            energy_model.name: energy_model,
            water_model.name: water_model
        })
        # ModelSet should derive inputs as any input to one of its models which
        # is not met by an internal dependency
        print(model_set.inputs)
        assert len(model_set.inputs) == 1
        assert 'population' in model_set.inputs.names

        # ModelSet should derive dependencies as links to any model which
        # supplies a dependency not met internally
        assert len(model_set.deps) == 1
        assert model_set.deps['population'].source_model is pop_scenario


class TestBasics:

    def test_dependency_not_present(self, get_sector_model):
        SectorModel = get_sector_model
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', Mock(), Mock(), 'unit')

        energy_model = SectorModel('model')
        energy_model.add_input('input', Mock(), Mock(), 'unit')
        with raises(ValueError):
            energy_model.add_dependency(elec_scenario, 'not_present',
                                        'input')

        with raises(ValueError):
            energy_model.add_dependency(elec_scenario, 'output',
                                        'not_correct_input_name')


class TestDependencyGraph:

    def test_simple_graph(self, get_sector_model):
        regions = Mock()
        regions.name = 'test_regions'
        intervals = Mock()
        intervals.name = 'test_intervals'

        SectorModel = get_sector_model
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', regions, intervals, 'unit')

        energy_model = SectorModel('model')
        energy_model.add_input('input', regions, intervals, 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        # Builds the dependency graph
        sos_model.make_dependency_graph()

        graph = sos_model.dependency_graph

        assert energy_model in graph
        assert elec_scenario in graph

        assert list(graph.edges()) == [(elec_scenario, energy_model)]

    def test_get_model_sets(self, get_sector_model):
        regions = Mock()
        regions.name = 'test_regions'
        intervals = Mock()
        intervals.name = 'test_intervals'

        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', regions, intervals, 'unit')

        SectorModel = get_sector_model
        energy_model = SectorModel('model')
        energy_model.add_input('input', regions, intervals, 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        sos_model.make_dependency_graph()

        actual = sos_model._get_model_sets_in_run_order()
        expected = ['scenario', 'model']

        for model, name in zip(actual, expected):
            assert model.name == name

    def test_topological_sort(self, get_sector_model):
        regions = Mock()
        regions.name = 'test_regions'
        intervals = Mock()
        intervals.name = 'test_intervals'

        SectorModel = get_sector_model
        elec_scenario = ScenarioModel('scenario')
        elec_scenario.add_output('output', regions, intervals, 'unit')

        energy_model = SectorModel('model')
        energy_model.add_input('input', regions, intervals, 'unit')
        energy_model.add_dependency(elec_scenario, 'output', 'input')

        sos_model = SosModel('energy_sos_model')
        sos_model.add_model(energy_model)
        sos_model.add_model(elec_scenario)

        sos_model.make_dependency_graph()

        graph = sos_model.dependency_graph
        actual = list(networkx.topological_sort(graph))
        assert actual == [elec_scenario, energy_model]


class TestCompositeIntegration:

    def test_simplest_case(self, get_scenario):
        """One scenario only
        """
        elec_scenario = get_scenario
        sos_model = SosModel('simple')
        sos_model.add_model(elec_scenario)
        data_handle = get_data_handle(sos_model)
        sos_model.simulate(data_handle)
        # no results available, as only scenario model ran

    def test_sector_model_null_model(self, get_energy_sector_model):
        no_inputs = get_energy_sector_model
        no_inputs._inputs = MetadataSet([])
        no_inputs.simulate = lambda x: x

        data_handle = Mock()
        no_inputs.simulate(data_handle)
        data_handle.assert_not_called()

    def test_sector_model_one_input(self, get_energy_sector_model,
                                    get_scenario):
        elec_scenario = get_scenario

        energy_model = get_energy_sector_model

        sos_model = SosModel('blobby')
        sos_model.add_model(elec_scenario)
        sos_model.add_model(energy_model)
        energy_model.add_dependency(elec_scenario,
                                    'electricity_demand_output',
                                    'electricity_demand_input')

        data_handle = get_data_handle(sos_model)
        results = sos_model.simulate(data_handle)

        expected = np.array([[100.737]])
        actual = results.get_results('fluffiness', 'energy_sector_model')
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


class TestNestedModels():

    def test_one_free_input(self, get_sector_model):
        SectorModel = get_sector_model
        energy_model = SectorModel('energy_sector_model')
        expected = Metadata('electricity_demand_input', Mock(), Mock(), 'unit')
        energy_model._inputs = MetadataSet([expected])

        actual = energy_model.free_inputs['electricity_demand_input']
        assert actual == expected

    def test_hanging_inputs(self, get_sector_model):
        """
        sos_model_high
            sos_model_lo
               -> em

        """
        SectorModel = get_sector_model
        energy_model = SectorModel('energy_sector_model')
        input_metadata = {
            'name': 'electricity_demand_input',
            'spatial_resolution': Mock(),
            'temporal_resolution': Mock(),
            'units': 'unit'
        }

        energy_model._inputs = MetadataSet([input_metadata])

        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(energy_model)

        expected = Metadata(input_metadata['name'],
                            input_metadata['spatial_resolution'],
                            input_metadata['temporal_resolution'],
                            input_metadata['units'])

        assert energy_model.free_inputs.names == ['electricity_demand_input']
        assert sos_model_lo.free_inputs.names == ['electricity_demand_input']

        sos_model_high = SosModel('higher')
        sos_model_high.add_model(sos_model_lo)
        actual = sos_model_high.free_inputs['electricity_demand_input']

        assert actual == expected

    @pytest.mark.xfail(reason="Nested sosmodels not yet implemented")
    def test_nested_graph(self, get_sector_model):
        """If we add a nested model, all Sectormodel and ScenarioModel objects
        are added as nodes in the graph with edges along dependencies.

        SosModel objects are not included, as they are just containers for the
        SectorModel and ScenarioModel objects, passing up inputs for deferred
        linkages to dependencies.

        Not implemented yet:
        """
        SectorModel = get_sector_model

        energy_model = SectorModel('energy_sector_model')

        input_metadata = {'name': 'electricity_demand_input',
                          'spatial_resolution': Mock(),
                          'temporal_resolution': Mock(),
                          'units': 'unit'}

        energy_model._inputs = MetadataSet([input_metadata])

        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(energy_model)

        sos_model_high = SosModel('higher')
        sos_model_high.add_model(sos_model_lo)

        with raises(NotImplementedError):
            sos_model_high.make_dependency_graph()
        graph = sos_model_high.dependency_graph
        assert graph.edges() == []

        expected = networkx.DiGraph()
        expected.add_node(sos_model_lo)
        expected.add_node(energy_model)

        assert energy_model in graph.nodes()

        scenario = ScenarioModel('electricity_demand')
        scenario.add_output('elec_demand_output', Mock(), Mock(), 'kWh')

        sos_model_high.add_dependency(scenario, 'elec_demand_output',
                                      'electricity_demand_input')

        sos_model_high.make_dependency_graph()
        assert graph.edges() == [(scenario, sos_model_high)]

    @pytest.mark.xfail(reason="Nested sosmodels not yet implemented")
    def test_composite_nested_sos_model(self, get_sector_model):
        """System of systems example with two nested SosModels, two Scenarios
        and one SectorModel. One dependency is defined at the SectorModel
        level, another at the lower SosModel level
        """
        SectorModel = get_sector_model

        elec_scenario = ScenarioModel('electricity_demand_scenario')
        elec_scenario.add_output('electricity_demand_output',
                                 Mock(), Mock(), 'unit')

        energy_model = SectorModel('energy_sector_model')
        energy_model.add_input(
            'electricity_demand_input', Mock(), Mock(), 'unit')
        energy_model.add_input('fluffiness_input', Mock(), Mock(), 'unit')
        energy_model.add_output('cost', Mock(), Mock(), 'unit')
        energy_model.add_output('fluffyness', Mock(), Mock(), 'unit')

        def energy_function(timestep, input_data):
            """Mimics the running of a sector model
            """
            results = {}
            demand = input_data['electricity_demand_input']
            fluff = input_data['fluffiness_input']
            results['cost'] = demand * 1.2894
            results['fluffyness'] = fluff * 22
            return results

        energy_model.simulate = energy_function
        energy_model.add_dependency(elec_scenario,
                                    'electricity_demand_output',
                                    'electricity_demand_input')

        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(elec_scenario)
        sos_model_lo.add_model(energy_model)

        fluf_scenario = ScenarioModel('fluffiness_scenario')
        fluf_scenario.add_output('fluffiness', Mock(), Mock(), 'unit')
        # fluf_scenario.add_data('fluffiness', np.array([[[12]]]), [2010])

        assert sos_model_lo.free_inputs.names == ['fluffiness_input']

        sos_model_lo.add_dependency(fluf_scenario,
                                    'fluffiness',
                                    'fluffiness_input')

        assert sos_model_lo.inputs.names == []

        sos_model_high = SosModel('higher')
        sos_model_high.add_model(sos_model_lo)
        sos_model_high.add_model(fluf_scenario)

        data_handle = get_data_handle(sos_model_high)
        actual = sos_model_high.simulate(data_handle)
        expected = {
            'fluffiness_scenario': {
                'fluffiness': 12
            },
            'lower': {
                'electricity_demand_scenario': {
                    'electricity_demand_output': 123
                },
                'energy_sector_model': {
                    'cost': 158.5962,
                    'fluffyness': 264
                }
            }
        }

        assert actual == expected


class TestCircularDependency:

    def test_loop(self, get_energy_sector_model, get_water_sector_model):
        """Fails because no functionality to deal with loops
        """
        energy_model = get_energy_sector_model
        water_model = get_water_sector_model

        sos_model = SosModel('energy_water_model')
        water_model.add_dependency(energy_model, 'fluffiness', 'fluffyness')
        energy_model.add_dependency(water_model, 'electricity_demand',
                                    'electricity_demand_input')
        sos_model.add_model(water_model)
        sos_model.add_model(energy_model)

        assert energy_model.inputs.names == ['electricity_demand_input']
        assert water_model.inputs.names == ['fluffyness']
        assert sos_model.inputs.names == []

        assert energy_model.free_inputs.names == []
        assert water_model.free_inputs.names == []
        assert sos_model.free_inputs.names == []

        sos_model.make_dependency_graph()
        graph = sos_model.dependency_graph

        assert (water_model, energy_model) in graph.edges()
        assert (energy_model, water_model) in graph.edges()

        sos_model.max_iterations = 100
        data_handle = get_data_handle(sos_model)
        results = sos_model.simulate(data_handle)

        expected = np.array([[0.13488114]], dtype=np.float)
        actual = results.get_results('fluffiness', model_name='energy_sector_model',
                                     modelset_iteration=35)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

        expected = np.array([[0.16469004]], dtype=np.float)
        actual = results.get_results('electricity_demand', model_name='water_supply_model',
                                     modelset_iteration=35)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
