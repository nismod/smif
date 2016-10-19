"""Establishes the tests which outline the operation of the optimisation
feature

The optimisation must run a model, observe the output of the objective function
for that model, then adjust the inputs within the specified ranges and repeat,
to find the minimum.

The optimisation features requires:

- a dictionary of (continuous) decision variables with upper and lower bounds
- the definition of an objective function, which either extracts the required
  objective information from the model, or processes the outputs to produce a
  scalar value

"""
from unittest.mock import Mock

import numpy as np
from fixtures.water_supply import ExampleWaterSupplySimulationAsset as WaterMod
from fixtures.water_supply import WaterSupplySimulationAssetWrapper, one_input
from numpy.testing import assert_allclose
from pytest import fixture
from smif.system import WaterModelAsset


@fixture(scope='function')
def mock_model(input_value):
    """Returns the mocked objective function value of `1` whatever is passed as
    an input

    Arguments
    =========
    input_value : any

    Returns
    =======
    int

    """
    model = Mock(return_value=1)
    return model(input_value)


def adapter_function(self, inputs):
    """
    Arguments
    =========
    inputs : numpy.ndarray

    Returns
    =======
    results = numpy.ndarray
    """
    model_instance = self.model(inputs[1], inputs[0])
    results = model_instance.simulate()
    return np.array(results['cost'])


class TestTest:

    def test_model_adapter(self):
        adapter = WaterSupplySimulationAssetWrapper(WaterMod)
        static = np.array(2)
        decision = np.array(3)
        results = adapter.simulate(static, decision)
        assert results == {'water': 2, 'cost': 3.792}

    def test_model_adapter_numpy(self):
        adapter = WaterSupplySimulationAssetWrapper(WaterMod)
        static = np.array(2)
        decision = np.array(3)
        results = adapter.simulate(static, decision)
        assert results == {'water': 2, 'cost': 3.792}


class TestWaterModel:

    def test_water_model_optimisation(self):
        wrapped = WaterSupplySimulationAssetWrapper(WaterMod)

        model = WaterModelAsset(wrapped, wrapped.simulate)
        model.inputs = one_input()
        actual_value = model.optimise()
        expected_value = {'water treatment capacity': 3}
        for actual, expected in zip(actual_value.values(),
                                    expected_value.values()):
            assert_allclose(actual, expected)
