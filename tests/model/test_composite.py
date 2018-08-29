from unittest.mock import Mock

import networkx
import numpy as np
import pytest
from pytest import fixture, raises
from smif.data_layer import DataHandle, MemoryInterface
from smif.metadata import Spec
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel
from smif.model.sos_model import ModelSet, SosModel


@fixture
def scenario_model():
    scenario = ScenarioModel('electricity_demand')
    scenario.scenario = 'Arbitrary Demand Scenario'
    scenario.add_output(
        Spec(
            name='electricity_demand_output',
            dims=['LSOA'],
            coords={'LSOA': ['E090001', 'E090002']},
            dtype='float',
        )
    )
    return scenario


class ExampleSectorModel(SectorModel):
    """A no-op sector model
    """
    def simulate(self, data):
        return data


def get_data_handle(model):
    """Return a data handle for the model
    """
    store = MemoryInterface()
    store.write_model_run({
        'name': 'test',
        'narratives': {}
    })
    store.write_scenario_variant_data(
        np.array([1.0, 2.0]),
        'electricity_demand',
        'Arbitrary Demand Scenario',
        'electricity_demand_output',
        2010
    )
    return DataHandle(
        store,
        'test',  # modelrun_name
        2010,  # current_timestep
        [2010],  # timesteps
        model
    )


@fixture
def water_model():
    class WaterModel(SectorModel):
        """
        fluffyness -> electricity_demand
        """
        def simulate(self, data_handle):
            x = data_handle['fluffyness']
            data_handle['electricity_demand'] = (x**3) - (6 * x**2) + (0.9 * x) + 0.15
            return data_handle
    water_model = WaterModel('water_supply_model')
    water_model.add_input(
        Spec(
            name='fluffyness',
            dims=['LSOA'],
            coords={'LSOA': ['E090001', 'E090002']},
            dtype='float',
        )
    )
    water_model.add_output(
        Spec(
            name='electricity_demand',
            dims=['LSOA'],
            coords={'LSOA': ['E090001', 'E090002']},
            dtype='float',
        )
    )

    return water_model


@fixture
def energy_model():
    class EnergyModel(SectorModel):
        """
        electricity_demand_input -> fluffiness
        """

        def simulate(self, data_handle):
            """Mimics the running of a sector model
            """
            fluff = data_handle['electricity_demand_input']
            data_handle['fluffiness'] = fluff * 0.819
            return data_handle

    energy_model = EnergyModel('energy_model')
    energy_model.add_input(
        Spec(
            name='electricity_demand_input',
            dims=['LSOA'],
            coords={'LSOA': ['E090001', 'E090002']},
            dtype='float',
        )
    )
    energy_model.add_output(
        Spec(
            name='fluffiness',
            dims=['LSOA'],
            coords={'LSOA': ['E090001', 'E090002']},
            dtype='float',
        )
    )

    return energy_model


@fixture
def sos_model(scenario_model, energy_model):
    """Build a simple SosModel
    """
    energy_model.add_dependency(
        scenario_model, 'electricity_demand_output', 'electricity_demand_input')

    sos_model = SosModel('energy_sos_model')
    sos_model.add_model(energy_model)
    sos_model.add_model(scenario_model)
    return sos_model


class TestModelSet:

    def test_model_set(self, scenario_model):
        ModelSet({scenario_model.name: scenario_model})

    def test_model_set_deps(self, water_model, energy_model):
        pop_scenario = ScenarioModel('population')
        pop_spec = Spec(
            name='population',
            dims=['LSOA'],
            coords={'LSOA': ['E090001', 'E090002']},
            dtype='float',
        )
        pop_scenario.add_output(pop_spec)
        energy_model.add_input(pop_spec)

        energy_model.add_dependency(pop_scenario, 'population', 'population')
        energy_model.add_dependency(
            water_model, 'electricity_demand', 'electricity_demand_input')
        water_model.add_dependency(energy_model, 'fluffiness', 'fluffyness')

        model_set = ModelSet({
            energy_model.name: energy_model,
            water_model.name: water_model
        })
        # ModelSet should derive inputs as any input to one of its models which
        # is not met by an internal dependency
        print(model_set.inputs)
        assert len(model_set.inputs) == 1
        assert 'population' in model_set.inputs

        # ModelSet should derive dependencies as links to any model which
        # supplies a dependency not met internally
        assert len(model_set.deps) == 1
        assert model_set.deps['population'].source_model is pop_scenario


class TestBasics:
    def test_dependency_duplicate(self, scenario_model, energy_model):
        energy_model.add_dependency(
            scenario_model, 'electricity_demand_output', 'electricity_demand_input')

        with raises(ValueError) as ex:
            energy_model.add_dependency(
                scenario_model, 'electricity_demand_output', 'electricity_demand_input')
        assert "dependency already defined" in str(ex)

    def test_dependency_not_present(self, scenario_model, energy_model):
        """Should fail with missing input/output
        """
        with raises(ValueError) as ex:
            energy_model.add_dependency(
                scenario_model, 'not_present', 'electricity_demand_input')
        msg = "Output 'not_present' is not defined in '{}'".format(scenario_model.name)
        assert msg in str(ex)

        with raises(ValueError) as ex:
            energy_model.add_dependency(
                scenario_model, 'electricity_demand_output', 'not_correct_input_name')
        msg = "Input 'not_correct_input_name' is not defined in '{}'".format(energy_model.name)
        assert msg in str(ex)

    def test_composite_dependency(self, scenario_model):
        """No dependencies on CompositeModels
        """
        sos_model = SosModel('test')
        with raises(NotImplementedError) as ex:
            sos_model.add_dependency(scenario_model, 'a', 'b')
        assert "Dependencies cannot be added to a CompositeModel" in str(ex)


class TestDependencyGraph:

    def test_simple_graph(self, sos_model, scenario_model, energy_model):
        """Build the dependency graph
        """
        graph = SosModel.make_dependency_graph(sos_model.models)
        nodes = sorted(node.name for node in graph.nodes())
        models = sorted(list(sos_model.models.keys()))
        assert nodes == models
        assert list(graph.edges()) == [(scenario_model, energy_model)]

    def test_get_model_set_simple_order(self, sos_model, scenario_model, energy_model):
        """Single dependency edge order
        """
        graph = SosModel.make_dependency_graph(sos_model.models)
        actual = SosModel.get_model_sets_in_run_order(graph, 1, 1, 1)
        expected = [scenario_model, energy_model]
        assert actual == expected

    def test_complex_order(self):
        """Single models upstream and downstream of an interdependency
        """
        a = ExampleSectorModel('a')
        a.add_output(Spec(name='a_out', dtype='bool'))

        b = ExampleSectorModel('b')
        b.add_input(Spec(name='b_in', dtype='bool'))
        b.add_input(Spec(name='b_in_from_c', dtype='bool'))
        b.add_output(Spec(name='b_out', dtype='bool'))
        b.add_dependency(a, 'a_out', 'b_in')

        c = ExampleSectorModel('c')
        c.add_input(Spec(name='c_in_from_b', dtype='bool'))
        c.add_output(Spec(name='c_out', dtype='bool'))
        c.add_dependency(b, 'b_out', 'c_in_from_b')
        b.add_dependency(c, 'c_out', 'b_in_from_c')

        d = ExampleSectorModel('d')
        d.add_input(Spec(name='d_in', dtype='bool'))
        d.add_output(Spec(name='d_out', dtype='bool'))
        d.add_dependency(c, 'c_out', 'd_in')

        graph = SosModel.make_dependency_graph({'a': a, 'b': b, 'c': c, 'd': d})
        actual = SosModel.get_model_sets_in_run_order(graph, 1, 1, 1)
        # two sets
        assert len(actual) == 3
        # first is just a
        assert actual[0] == a
        # second is the modelset containing interdependent b and c
        assert sorted(actual[1].models.keys()) == ['b', 'c']
        # third is just d
        assert actual[2] == d


class TestCompositeIntegration:

    def test_simplest_case(self, scenario_model):
        """One scenario only
        """
        sos_model = SosModel('simple')
        sos_model.add_model(scenario_model)
        data_handle = get_data_handle(sos_model)
        sos_model.simulate(data_handle)
        # no results available, as only scenario model ran

    def test_sector_model_null_model(self, energy_model):
        no_inputs = energy_model
        no_inputs._inputs = {}
        no_inputs.simulate = lambda x: x

        data_handle = Mock()
        no_inputs.simulate(data_handle)
        data_handle.assert_not_called()

    def test_sector_model_one_input(self, sos_model):
        data_handle = get_data_handle(sos_model)
        results = sos_model.simulate(data_handle)

        expected = np.array([0.819,  1.638])
        actual = results.get_results(('energy_model', 'fluffiness'), 'energy_model')
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


class TestNestedModels():

    def test_one_free_input(self, energy_model):
        expected = list(energy_model.inputs.values())
        actual = [energy_model.free_inputs['electricity_demand_input']]
        assert actual == expected

    def test_hanging_inputs(self, energy_model):
        """
        sos_model_high
            sos_model_lo
               -> em

        """
        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(energy_model)

        assert list(energy_model.free_inputs.keys()) == ['electricity_demand_input']
        assert list(sos_model_lo.free_inputs.keys()) == [
            ('energy_model', 'electricity_demand_input')]

        sos_model_high = SosModel('higher')
        # work around add_model, which would fail as nesting is not fully implemented
        sos_model_high.models[sos_model_lo.name] = sos_model_lo
        actual = list(sos_model_high.free_inputs.keys())
        expected = [('lower', ('energy_model', 'electricity_demand_input'))]

        assert actual == expected

    @pytest.mark.xfail(reason="Nested sosmodels not yet implemented")
    def test_nested_graph(self, energy_model, scenario_model):
        """If we add a nested model, all Sectormodel and ScenarioModel objects
        are added as nodes in the graph with edges along dependencies.

        SosModel objects are not included, as they are just containers for the
        SectorModel and ScenarioModel objects, passing up inputs for deferred
        linkages to dependencies.

        Not implemented yet:
        """
        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(energy_model)

        sos_model_high = SosModel('higher')
        sos_model_high.add_model(sos_model_lo)

        with raises(NotImplementedError):
            graph = sos_model_lo.make_dependency_graph(sos_model_lo.models)
        assert graph.edges() == []
        assert energy_model in graph.nodes()

        expected = networkx.DiGraph()
        expected.add_node(sos_model_lo)
        expected.add_node(energy_model)

        sos_model_high.add_model(scenario_model)
        sos_model_lo.add_dependency(
            scenario_model, 'electricity_demand_output', 'electricity_demand_input')

        graph = sos_model_high.make_dependency_graph(sos_model_high.models)
        assert graph.edges() == [(scenario_model, sos_model_lo)]

    @pytest.mark.xfail(reason="Nested sosmodels not yet implemented")
    def test_composite_nested_sos_model(self, energy_model, scenario_model):
        """System of systems example with two nested SosModels, two Scenarios
        and one SectorModel. One dependency is defined at the SectorModel
        level, another at the lower SosModel level
        """
        sos_model_lo = SosModel('lower')
        sos_model_lo.add_model(scenario_model)
        sos_model_lo.add_model(energy_model)

        fluff_spec = Spec(
            name='fluffiness',
            dtype='float'
        )
        fluf_scenario = ScenarioModel('fluffiness_scenario')
        fluf_scenario.add_output(fluff_spec)
        energy_model.add_input(fluff_spec)

        assert list(sos_model_lo.free_inputs.keys()) == ['fluffiness']
        assert list(sos_model_lo.inputs.keys()) == []

        sos_model_high = SosModel('higher')
        sos_model_high.add_model(sos_model_lo)
        sos_model_high.add_model(fluf_scenario)
        sos_model_lo.add_dependency(fluf_scenario, 'fluffiness', 'fluffiness')

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
                'energy_model': {
                    'cost': 158.5962,
                    'fluffyness': 264
                }
            }
        }

        assert actual == expected


class TestCircularDependency:

    def test_loop(self, energy_model, water_model):
        """Fails because no functionality to deal with loops
        """
        energy_model = energy_model
        water_model = water_model

        sos_model = SosModel('energy_water_model')
        water_model.add_dependency(energy_model, 'fluffiness', 'fluffyness')
        energy_model.add_dependency(
            water_model, 'electricity_demand', 'electricity_demand_input')
        sos_model.add_model(water_model)
        sos_model.add_model(energy_model)

        assert list(energy_model.inputs.keys()) == ['electricity_demand_input']
        assert list(water_model.inputs.keys()) == ['fluffyness']
        assert list(sos_model.inputs.keys()) == []

        assert list(energy_model.free_inputs.keys()) == []
        assert list(water_model.free_inputs.keys()) == []
        assert list(sos_model.free_inputs.keys()) == []

        graph = sos_model.make_dependency_graph(sos_model.models)

        assert (water_model, energy_model) in graph.edges()
        assert (energy_model, water_model) in graph.edges()

        sos_model.max_iterations = 100
        data_handle = get_data_handle(sos_model)
        results = sos_model.simulate(data_handle)

        expected = np.array([0.13488114, 0.13488114], dtype=np.float)
        actual = results.get_results(
            ('energy_model', 'fluffiness'), model_name='energy_model', modelset_iteration=35)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

        expected = np.array([0.16469004, 0.16469004], dtype=np.float)
        actual = results.get_results(
            ('water_supply_model', 'electricity_demand'),
            model_name='water_supply_model', modelset_iteration=35)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
