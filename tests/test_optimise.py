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
from fixtures.water_supply import ExampleWaterSupplySimulationAsset as WaterMod
from fixtures.water_supply import one_input
from numpy.testing import assert_allclose
from smif.abstract import AbstractModelWrapper, SectorModel
# from smif.system import WaterModelAsset


class WaterSupplySimulationAssetWrapper(AbstractModelWrapper):
    """Provides an interface for :class:`ExampleWaterSupplyAssetSimulation
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
        capacity = decision_variables
        instance = self.model(raininess, capacity)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']

    def constraints(self, parameters):
        constraints = ({'type': 'ineq',
                        'fun': lambda x: min(x[0], parameters[0]) - 3}
                       )
        return constraints


class TestWaterModelOptimisation:

    def test_water_model_optimisation(self, one_input):
        wrapped = WaterSupplySimulationAssetWrapper(WaterMod)

        model = SectorModel(wrapped, wrapped.simulate)
        model.inputs = one_input
        actual_value = model.optimise()
        expected_value = {'water treatment capacity': 3}
        for actual, expected in zip(actual_value.values(),
                                    expected_value.values()):
            assert_allclose(actual, expected)
