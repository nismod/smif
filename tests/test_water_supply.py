"""Tests to ensure that the example simulation model fixtures are behaving


"""
import subprocess

from pytest import raises
from smif.abstract import SectorModel

from .fixtures.water_supply import (ExampleWaterSupplySimulation,
                                    ExampleWaterSupplySimulationReservoir,
                                    raininess_oracle)


class WaterSupplyPython(SectorModel):
    """A concrete instance of the water supply model wrapper for testing

    Inherits :class:`SectorModel` to wrap the example simulation tool.

    """

    def initialise(self, data):
        """Set up the model

        Parameters
        ==========
        data : dict
            A dictionary of which one key 'raininess' must contain the amount
            of rain
        """
        self.inputs['raininess'] = data['raininess']

        self.model = ExampleWaterSupplySimulation(data['raininess'])
        self.results = None
        self.run_successful = None

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        """Runs the model and stores the results in the results parameter

        """
        self.results = self.model.simulate()
        self.run_successful = True

    def model_executable(self):
        pass


class WaterSupplyExecutable(SectorModel):
    """A concrete instance of the water supply which wraps a command line model

    """

    def initialise(self, data):
        self.model = self.model_executable
        self.data = data
        self.results = None
        self.run_successful = None
        self.model_executable = './tests/fixtures/water_supply_exec.py'

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        executable = self.model_executable
        raininess = self.data['raininess']
        argument = "--raininess={}".format(str(raininess))
        output = subprocess.check_output([executable, argument])
        self.results = process_results(output)
        self.run_successful = True


def process_results(output):
    """Utility function which decodes stdout text from the water supply model

    Returns
    =======
    results : dict
        A dictionary where keys are the results e.g. `cost` and `water`

    """
    results = {}
    raw_results = output.decode('utf-8').split('\n')
    for result in raw_results[0:2]:
        values = result.split(',')
        if len(values) == 2:
            results[str(values[0])] = float(values[1])
    return results


def test_water_supply_with_reservoir():
    raininess = 1
    reservoir_level = 2
    model = ExampleWaterSupplySimulationReservoir(raininess, reservoir_level)
    actual = model.simulate()
    expected = {'cost': 1.2, 'water': 3, 'reservoir level': 2}
    assert actual == expected


def test_water_supply_with_reservoir_negative_level():
    raininess = 1
    reservoir_level = -2
    with raises(ValueError, message="Reservoir level cannot be negative"):
        ExampleWaterSupplySimulationReservoir(raininess, reservoir_level)


def test_process_results():
    input_bytes = b"cost,1\nwater,1\n"
    actual = process_results(input_bytes)
    expected = {'water': 1, 'cost': 1}
    assert actual == expected


def test_raininess_oracle():
    time = [2010, 2020, 2030, 2042, 2050]
    expected = [1, 2, 3, 4, 5]

    for result in zip(time, expected):
        actual = raininess_oracle(result[0])
        assert actual == result[1]


def test_raininess_oracle_out_of_range():
    msg = "timestep 2051 is outside of the range [2010, 2050]"
    with raises(AssertionError, message=msg):
        raininess_oracle(2051)


def test_simulate_rain_python():
    ws = WaterSupplyPython()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results["water"] == 1


def test_simulate_rain_cost_python():
    ws = WaterSupplyPython()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results["cost"] == 1


def test_simulate_rain_executable():
    ws = WaterSupplyExecutable()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results['water'] == 1


def test_simulate_rain_cost_executable():
    ws = WaterSupplyExecutable()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results['cost'] == 1
