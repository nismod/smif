import os
import subprocess
import sys
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
    def get_model_executable(self):
        """Return path of current python interpreter
        """
        return sys.executable

    def get_model_script(self):
        """Return relative path to water_supply_exec.py script
        """
        return os.path.join(os.getcwd(), "tests", "fixtures", "water_supply_exec.py")

    def simulate(self, static_inputs, decision_variables):
        """Runs the executable version of the water supply model

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is capacity of water treatment plants
        """
        model_executable = self.get_model_executable()
        model_script = self.get_model_script()

        if model_executable != "" and model_executable is not None:
            argument = "--raininess={}".format(str(static_inputs))
            output = subprocess.check_output([model_executable, model_script, argument])
            results = process_results(output)
        else:
            results = None

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
