"""Tests the definition and solution of the planning problem optimisation
"""
from skopt import dummy_minimize
from smif.optimisation import OptimizerTemplate


class Optimizer(OptimizerTemplate):
    """A simple algorithm which minimizes the optimization function
    """

    def run(self):

        dimensions = [(0, 1)] * self._action_space
        results = dummy_minimize(self.optimization_function,
                                 dimensions,
                                 random_state=1,
                                 n_calls=1000,
                                 verbose=True
                                 )
        self.results['optimal_decision'] = results.x
        self.results['objective_value'] = results.fun


def dummy_function(binary_vector):
    """A dummy function, representing a system-of-systems model

    The function accepts a list of 0-1 intervention decisions and returns
    a scalar

    Arguments
    ---------
    binary_vector : list
        A list of 0-1 values

    """

    def binary_to_integer(binary_vector):
        """Convert list of binary integers to integer scalar
        """
        binary_string = "".join(str(x) for x in binary_vector)
        return float(int('0b{}'.format(binary_string), 2))

    scalar = binary_to_integer(binary_vector)

    return scalar ** 2


class TestOptimizationImplementation:

    def test_optimise_function(self):
        """Demonstrates the steps to run the optimization algorithm
        """

        optimiser = Optimizer()
        number_dimensions = 8  # Action space
        optimiser.initialize(number_dimensions)
        optimiser.optimization_function = dummy_function
        optimiser.run()
        assert optimiser.results['optimal_decision'] == [0, 0, 0, 0, 0, 0, 0, 0]
        assert optimiser.results['objective_value'] == 0


class TestOptimizationTemplate:

    def test_implement_template(self):

        class MyOpt(OptimizerTemplate):

            def run(self):
                self.results['optimal_decision'] = [1] * self._action_space
                self.results['objective_value'] = 0

        myopt = MyOpt()
        myopt.initialize(1)
        assert myopt._action_space == 1

        myopt.optimization_function = lambda x: x

        myopt.run()

        assert myopt.results == {'optimal_decision': [1],
                                 'objective_value': 0}
