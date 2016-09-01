#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Abstract base classes for the scalable modelling of infrastructure systems

"""
from __future__ import absolute_import, division, print_function

import logging
from abc import ABC, abstractmethod, abstractproperty

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class Commodity(ABC):
    """
    """
    def __init__(self, name):
        self._name = name

    def name(self):
        pass


class SectorModel(ABC):
    """An abstract representation of the sector model with inputs and outputs.

    Attributes
    ==========
    input_hooks :
    output_hooks :

    """
    @abstractmethod
    def optimise(self, method, decision_vars, objective_function):
        """Performs an optimisation of the sector model assets

        Arguments
        =========
        method : function
            Provides the
        decision_vars : list
            Defines the decision variables
        objective_function : function
            Defines the objective function
        """
        pass

    @abstractmethod
    def simulate(self):
        """Performs an operational simulation of the sector model

        Note
        ====
        The term simulation may refer to operational optimisation, rather than
        simulation-only. This process is described as simulation to distinguish
        from the definition of investments in capacity, versus operation using
        the given capacity

        """
        pass

    @abstractproperty
    def model_executable(self):
        """The path to the model executable
        """
        pass


class Interface(ABC):
    """Provides the interface between a sector model and other interfaces

    Attributes
    ==========
    inputs : :class:`abstract.Input`
    region_id : int
    timestep : int

    Returns
    =======

    Methods
    =======

    """
    def __init__(self, inputs):
        self._inputs = inputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, list_of_inputs):
        for single_input in list_of_inputs:
            assert isinstance(single_input, Input)
        self._inputs = list_of_inputs


class Input(ABC):
    """An input is a dependency which is used in the sector model
    """

    def __init__(self):
        """
        """
        pass


class Asset(ABC):
    """
    """
    def __init__(self):
        """
        """
        pass


if __name__ == "__main__":
    pass
