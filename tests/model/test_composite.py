import numpy as np
from pytest import fixture, raises
from smif.data_layer import DataHandle, MemoryInterface
from smif.metadata import Spec
from smif.model.model_set import ModelSet
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel
from smif.model.sos_model import SosModel


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
