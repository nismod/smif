import subprocess

import numpy as np
from fixtures.water_supply import ExampleWaterSupplySimulation as WaterMod
from fixtures.water_supply import process_results
from smif.abstract import AbstractModelWrapper


class WaterSupplySimulationWrapper(AbstractModelWrapper):
    """Provides an interface for :class:`ExampleWaterSupplySimulation`
    """

    def simulate(self, static_inputs, decision_variables):
        """

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is capacity of water treatment plants
        """
        raininess = static_inputs
        instance = self.model(raininess)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']


class WaterModelWrapExec(AbstractModelWrapper):

    def simulate(self, static_inputs, decision_variables):
        """Runs the executable version of the water supply model

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is capacity of water treatment plants
        """
        model_executable = './tests/fixtures/water_supply_exec.py'
        argument = "--raininess={}".format(str(static_inputs))
        output = subprocess.check_output([model_executable, argument])
        results = process_results(output)
        return results

    def extract_obj(self, results):
        return results


class TestModelWrapping:

    def test_model_adapter(self):
        adapter = WaterSupplySimulationWrapper(WaterMod)
        static = np.array(2)
        decision = np.array(3)
        results = adapter.simulate(static, decision)
        assert results == {'water': 2, 'cost': 1}

    def test_model_adapter_numpy(self):
        adapter = WaterSupplySimulationWrapper(WaterMod)
        static = np.array(2)
        decision = np.array(3)
        results = adapter.simulate(static, decision)
        assert results == {'water': 2, 'cost': 1}

    def test_simulate_rain_cost_executable(self):
        adapter = WaterModelWrapExec('dummy')
        static = 1
        decision = 1
        results = adapter.simulate(static, decision)
        assert results['cost'] == 1
