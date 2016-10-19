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
from numpy.testing import assert_allclose, assert_equal
from pytest import fixture
from smif.abstract import ModelInputs
from smif.system import AbstractModelWrapper, WaterModelAsset


def one_input():
    inputs = {'decision variables': ['water treatment capacity'],
              'parameters': ['raininess'],
              'water treatment capacity': {'bounds': (0, 20),
                                           'index': 0,
                                           'init': 10
                                           },
              'raininess': {'bounds': (0, 5),
                            'index': 0,
                            'value': 3
                            }
              }

    return inputs


def two_inputs():
    inputs = {'decision variables': ['water treatment capacity',
                                     'reservoir pumpiness'],
              'parameters': ['raininess'],
              'water treatment capacity': {'bounds': (0, 20),
                                           'index': 1,
                                           'init': 10
                                           },
              'reservoir pumpiness': {'bounds': (0, 100),
                                      'index': 0,
                                      'init': 24.583
                                      },
              'raininess': {'bounds': (0, 5),
                            'index': 0,
                            'value': 3
                            }
              }
    return inputs


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


class ModelWrapper(AbstractModelWrapper):
    """
    """

    def simulate(self, static_inputs, decision_variables):
        """

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x=1 is raininess
            x=2 is capacity of water treatment plants
        """
        raininess = static_inputs
        capacity = decision_variables
        instance = self.model(raininess, capacity)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']


class TestTest:

    def test_model_adapter(self):
        adapter = ModelWrapper(WaterMod)
        static = np.array(2)
        decision = np.array(3)
        results = adapter.simulate(static, decision)
        assert results == {'water': 2, 'cost': 3.792}

    def test_model_adapter_numpy(self):
        adapter = ModelWrapper(WaterMod)
        static = np.array(2)
        decision = np.array(3)
        results = adapter.simulate(static, decision)
        assert results == {'water': 2, 'cost': 3.792}


class TestWaterModel:

    def test_water_model_optimisation(self):
        wrapped = ModelWrapper(WaterMod)

        model = WaterModelAsset(wrapped, wrapped.simulate)
        model.inputs = one_input()
        actual_value = model.optimise()
        expected_value = {'water treatment capacity': 3}
        for actual, expected in zip(actual_value.values(),
                                    expected_value.values()):
            assert_allclose(actual, expected)


class TestInputs:

    def test_one_input_decision_variables(self):

        inputs = ModelInputs(one_input())
        act_names = inputs.decision_variable_names
        act_initial = inputs.decision_variable_values
        act_bounds = inputs.decision_variable_bounds

        exp_names = np.array(['water treatment capacity'], dtype=str)
        exp_initial = np.array([10], dtype=float)
        exp_bounds = np.array([(0, 20)], dtype=(float, 2))

        assert_equal(act_names, exp_names)
        assert_equal(act_initial, exp_initial)
        assert_equal(act_bounds, exp_bounds)

    def test_two_inputs_decision_variables(self):

        inputs = ModelInputs(two_inputs())
        act_names = inputs.decision_variable_names
        act_initial = inputs.decision_variable_values
        act_bounds = inputs.decision_variable_bounds

        exp_names = np.array(['reservoir pumpiness',
                              'water treatment capacity'], dtype='U30')
        exp_initial = np.array([24.583, 10], dtype=float)
        exp_bounds = np.array([(0, 100), (0, 20)], dtype=(float, 2))

        assert_equal(act_names, exp_names)
        assert_equal(act_initial, exp_initial)
        assert_equal(act_bounds, exp_bounds)

    def test_one_input_parameters(self):

        inputs = ModelInputs(one_input())
        act_names = inputs.parameter_names
        act_values = inputs.parameter_values
        act_bounds = inputs.parameter_bounds

        exp_names = np.array(['raininess'], dtype='U30')
        exp_values = np.array([3], dtype=float)
        exp_bounds = np.array([(0, 5)], dtype=(float, 2))

        assert_equal(act_names, exp_names)
        assert_equal(act_values, exp_values)
        assert_equal(act_bounds, exp_bounds)
