"""Implements the interface to optimization functionality

This module defines an :class:`OptimizerTemplate` which lays out the fundamental
steps of any optimization algorithm. To implement your own algorithm, create a
child class which inherits OptimizerTemplate and implement the methods.

"""
__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


from abc import ABC, abstractmethod


class OptimizerTemplate(ABC):
    """
    """

    def __init__(self):
        self._action_space = None
        self._optimization_function = None
        self._results = {'optimal_decision': None,
                         'objective_value': None}

    def initialize(self, available_interventions):
        """Setup the optimization problem

        Arguments
        ---------
        available_interventions : int
            The number of dimensions
        """
        assert isinstance(available_interventions, int)
        self._action_space = available_interventions

    @property
    def optimization_function(self):
        """The function which will be minimized by the algorithm

        The optimization function should return a scalar and accept a vector
        of binary decision variables
        """
        return self._optimization_function

    @optimization_function.setter
    def optimization_function(self, function):
        self._optimization_function = function

    @property
    def results(self):
        """The results from a successful optimization
        """
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @abstractmethod
    def run(self):
        """Override to implement an optimization algorithm

        The optimization algorithm should return the optimal decision vector
        and objective function
        """
        pass
