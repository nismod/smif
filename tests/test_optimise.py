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
from numpy.testing import assert_allclose
import pytest

from fixtures.water_supply import WaterSupplySectorModel, DynamicWaterSupplySectorModel
from fixtures.water_supply import dynamic_data, one_input
from smif.controller import SosModel


class TestWaterModelOptimisation:

    def test_water_model_optimisation(self, one_input):
        model = WaterSupplySectorModel()
        model.inputs = one_input
        model.attributes = {}

        actual_value = model.optimise()
        expected_value = {'water': np.array([3.], dtype=float),
                          'cost': np.array([3.792], dtype=float)}
        keys = list(actual_value.keys())
        for act_key in keys:
            assert act_key in ['cost', 'water', 'capacity']
        for key in keys:
            assert_allclose(actual_value[key], expected_value[key])

    def test_optimisation_fail_no_input(self):
        """Raise an error if no inputs are specified
        """
        model = WaterSupplySectorModel()
        model.attributes = {}

        with pytest.raises(AssertionError):
            model.optimise()


class TestMultiYearOptimisation:
    """Dynamic Optimisation over multiple years

    - Requires input data to be define over years
    - Requires a sequence of simulation models to be instantiated with this
      data
    - Established temporal dependencies between the models
    - Requires storage of model state between model instances
    """

    def test_dynamic_water_model_one_off(self, dynamic_data):
        model = DynamicWaterSupplySectorModel()
        model.attributes = {}
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
        model = DynamicWaterSupplySectorModel()
        model.attributes = {}
        model.inputs = dynamic_data

        first_results = model.optimise()

        # Updates model state (existing capacity) with total capacity from
        # previous iteration
        model.inputs.parameters.update_value('existing capacity',
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

    def test_sequential_simulation(self, dynamic_data):
        # Instantiate a sector model
        model = DynamicWaterSupplySectorModel()
        model.attributes = {}

        # Instantiate a system-of-system instance
        sos_model = SosModel()
        # Attach the sector model to the system-of-system model
        sos_model.model_list = {'water_supply': model}
        sos_model.timesteps = [2010, 2015, 2020]
        decisions = np.array([[2], [0], [0]], dtype=float)
        results = sos_model.sequential_simulation(model,
                                                  dynamic_data,
                                                  decisions)

        expected_results = [{'water': np.array([3.], dtype=float),
                             'cost': np.array([2.528], dtype=float),
                             'capacity': np.array([3.], dtype=float)},
                            {'water': np.array([3.], dtype=float),
                             'cost': np.array([0.], dtype=float),
                             'capacity': np.array([3.], dtype=float)},
                            {'water': np.array([3.], dtype=float),
                             'cost': np.array([0.], dtype=float),
                             'capacity': np.array([3.], dtype=float)}]
        assert results == expected_results

    @pytest.mark.skip(reason="should SectorModel be able to run sequential simulation?")
    def test_sector_model_sequential_simulation(self, dynamic_data):
        # Instantiate a sector model
        model = DynamicWaterSupplySectorModel()
        model.attributes = {}
        model.inputs = dynamic_data
        timesteps = [2010, 2015, 2020]
        decisions = np.array([[5, 0, 0]], dtype=float)
        results = model.sequential_simulation(timesteps, decisions)
        expected = [{'capacity': 5.0, 'cost': 6.32, 'water': 3.0},
                    {'capacity': 5.0, 'cost': 0.0, 'water': 3.0},
                    {'capacity': 5.0, 'cost': 0.0, 'water': 3.0}]
        assert results == expected

    @pytest.mark.skip(reason="should SectorModel be able to run sequential optimisation?")
    def test_sector_model_sequential_optimisation(self, dynamic_data):
        # Instantiate a sector model
        model = DynamicWaterSupplySectorModel()
        model.attributes = {}
        model.inputs = dynamic_data
        timesteps = [2010, 2015, 2020]
        results = model.sequential_optimisation(timesteps)
        expected = [{'capacity': 3.0, 'cost': 3.792, 'water': 3.0},
                    {'capacity': 4.0, 'cost': 1.264, 'water': 3.0},
                    {'capacity': 5.0, 'cost': 1.264, 'water': 3.0}]
        for act, exp in zip(results, expected):
            for key in act.keys():
                assert_allclose(act[key], exp[key],
                                rtol=1e-6, atol=1e-6)
