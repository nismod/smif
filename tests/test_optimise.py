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
import numpy as np
import pytest
from fixtures.water_supply import DynamicWaterSupplyModel as DynMod
from fixtures.water_supply import ExampleWaterSupplySimulationAsset as WaterMod
from fixtures.water_supply import dynamic_data, one_input
from numpy.testing import assert_allclose
from smif.abstract import AbstractModelWrapper
from smif.sectormodel import SectorModel


class WaterSupplySimulationAssetWrapper(AbstractModelWrapper):
    """Provides an interface for :class:`ExampleWaterSupplyAssetSimulation
    """

    def simulate(self, static_inputs, decision_variables):
        """

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
        decision_variables : :class:`numpy.ndarray`
            x_0 is capacity of water treatment plants
        """
        raininess = static_inputs
        capacity = decision_variables
        instance = self.model(raininess, capacity)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']

    def constraints(self, parameters):
        """

        Notes
        =====
        This constraint below expresses that water supply must be greater than
        or equal to 3.  ``x[0]`` is the decision variable for water treatment
        capacity, while the value ``parameters[0]`` in the min term is the
        value of the raininess parameter.
        """
        constraints = ({'type': 'ineq',
                        'fun': lambda x: min(x[0], parameters[0]) - 3}
                       )
        return constraints


class TestWaterModelOptimisation:

    def test_water_model_optimisation(self, one_input):
        wrapped = WaterSupplySimulationAssetWrapper(WaterMod)
        model = SectorModel(wrapped)
        model.inputs = one_input
        actual_value = model.optimise()
        expected_value = {'water': np.array([3.], dtype=float),
                          'cost': np.array([3.792], dtype=float)}
        keys = list(actual_value.keys())
        for act_key in keys:
            assert act_key in ['cost', 'water', 'capacity']
        for key in keys:
            assert_allclose(actual_value[key], expected_value[key])

    def test_optimisation_fail_no_input(self, one_input):
        """Raise an error if no inputs are specified
        """
        wrapped = WaterSupplySimulationAssetWrapper(WaterMod)
        model = SectorModel(wrapped)
        with pytest.raises(AssertionError):
            model.optimise()


class DynamicModelWrapper(AbstractModelWrapper):

    def simulate(self, static_inputs, decision_variables):
        """

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is existing capacity
        decision_variables : :class:`numpy.ndarray`
            x_0 is new capacity of water treatment plants
        """
        new_capacity = decision_variables
        instance = self.model(static_inputs[0], static_inputs[1], new_capacity)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']

    def constraints(self, parameters):
        """

        Notes
        =====
        This constraint below expresses that water supply must be greater than
        or equal to 3.  ``x[0]`` is the decision variable for water treatment
        capacity, while the value ``parameters[0]`` in the min term is the
        value of the raininess parameter.
        """
        constraints = ({'type': 'ineq',
                        'fun': lambda x: min(x[0] + parameters[1],
                                             parameters[0]) - 3}
                       )
        return constraints


class TestMultiYearOptimisation:
    """Dynamic Optimisation over multiple years

    - Requires input data to be define over years
    - Requires a sequence of simulation models to be instantiated with this
      data
    - Established temporal dependencies between the models
    - Requires storage of model state between model instances
    """

    def test_dynamic_water_model_one_off(self, dynamic_data):
        wrapped = DynamicModelWrapper(DynMod)
        model = SectorModel(wrapped)
        model.inputs = dynamic_data
        actual_value = model.optimise()
        expected_value = {'water': np.array([3.], dtype=float),
                          'cost': np.array([1.264 * 2], dtype=float),
                          'capacity': np.array([3.], dtype=float)}
        keys = list(actual_value.keys())
        for act_key in keys:
            assert act_key in ['cost', 'water', 'capacity']
        for key in keys:
            assert_allclose(actual_value[key], expected_value[key])

    def test_dynamic_water_model_two_off(self, dynamic_data):
        wrapped = DynamicModelWrapper(DynMod)
        model = SectorModel(wrapped)
        model.inputs = dynamic_data
        first_results = model.optimise()

        model.inputs.update_parameter_value('existing capacity',
                                            first_results['capacity'])
        second_results = model.optimise()

        expected_value = {'water': np.array([3.], dtype=float),
                          'cost': np.array([0.], dtype=float),
                          'capacity': np.array([3.], dtype=float)}
        keys = list(second_results.keys())
        for act_key in keys:
            assert act_key in ['cost', 'water', 'capacity']
        for key in keys:
            assert_allclose(second_results[key], expected_value[key],
                            rtol=1e-6, atol=1e-6)
