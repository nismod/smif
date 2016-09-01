#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Abstract base classes for the scalable modelling of infrastructure systems

"""
from __future__ import absolute_import, division, print_function

import logging
from abc import ABC, abstractmethod

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class Commodity(ABC):
    """
    """
    names = []

    def __init__(self, name, emission_factor):
        self._name = name
        self.names.append(name)
        self._emissions_factor = emission_factor

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @classmethod
    def print_commodities(self):
        for commodity in self.names:
            print("{}".format(commodity))


class SectorModel(ABC):
    """An abstract representation of the sector model with inputs and outputs.

    Attributes
    ==========
    input_hooks : dict
        A mapping between :class:`Dependency` and sector model inputs
    output_hooks : dict
        A mapping between :class:`Output` and sector model outputs
    run_successful

    """
    def __init__(self):
        self._run_successful = None

    @property
    def run_successful(self):
        """Indicates whether the simulation or optimisation run was successful

        """
        return self._run_successful

    @run_successful.setter
    def run_successful(self, outcome):
        self._run_successful = outcome

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

    @property
    @abstractmethod
    def model_executable(self):
        """The path to the model executable
        """
        return self._model_executable

    @model_executable.setter
    @abstractmethod
    def model_executable(self, value):
        """
        """
        self._model_executable = value


class Interface(ABC):
    """Provides the interface between a sector model and other interfaces

    Attributes
    ==========
    inputs : :class:`Input`
    region : str
        The unique identifier for the region
    timestep : int
        The current timestep
    sector_model : :class:`SectorModel`
        The sector model wrapped by the Interface

    """
    def __init__(self, inputs, region, timestep, sector_model):
        self._inputs = inputs
        self._region = region
        self._timestep = timestep
        self._sector_model = sector_model

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, list_of_inputs):
        for single_input in list_of_inputs:
            assert isinstance(single_input, Input)
        self._inputs = list_of_inputs


class Input(ABC):
    """An input is a sector model input exposed to the :class:`Interface`
    """

    inputs = []

    def __init__(self, name, region, timestep):
        self._name = name
        self._region = region
        self._timestep = timestep
        input_tuple = (name, region, timestep)
        self.inputs.append(input_tuple)

    @classmethod
    def list_inputs(self):
        for input_tuple in self.inputs:
            print('{}'.format(input_tuple))


class Dependency(Input):
    """A dependency is a type of input which links interfaces
    """


class Decision(ABC):
    """A decision denotes an Asset target, and bounds for the current Interface

    Normally, an instance of a decision corresponds to an investment decision
    or policy to increase capacity, or decrease demand

    """
    def __init__(self, target_asset, lower_bound, upper_bound):
        """The asset which is targetted by the decision
        """
        self._target_asset = target_asset
        """The upper bound on the value of the investment in the target asset
        """
        self._upper_bound = upper_bound
        """The lower bound on the value of the investment in the target asset
        """
        self._lower_bound = lower_bound


class Asset(ABC):
    """An asset is a decision targetted capacity that persists across timesteps

    Examples of assets include power stations, water treatment plants, roads,
    railway tracks, airports, ports, centres of demand such as houses or
    factories, waste processing plants etc.

    An Asset is targetted by and influenced by a :class:`Decision` but only
    need to be defined in the :class:`Interface` if targetted
    by a :class:`Decision`

    The inter-timestep state of a model is a function of the Asset-base.

    The Asset-state is also persisted (written to the datastore)

    """
    def __init__(self, name, capacity):
        """
        """
        self._name = name
        self._capacity = capacity
        self._new_capacity = None

    @classmethod
    @abstractmethod
    def write_assets_to_datastore(self):
        """Writes the current :class:`State` of the Asset to the datastore
        """
        pass

    @abstractmethod
    def get_decisions(self):
        """Returns a container of :class:`Decision`
        """
        pass

    @property
    def new_capacity(self):
        return self._new_capacity

    @classmethod
    def get_state(self):
        return self


class AbstractState(ABC):
    """
    """
    @abstractmethod
    def update_state(self):
        pass


class State(AbstractState):
    """A static representation of an Interface's assets

    The state is used to record (and persist) the inter-temporal transition
    from one time-step to the next of the :class:`Interface`
    """
    def __init__(self, region, timestep, sector_model):
        self._region = region
        self._timestep = timestep
        self._sector_model = sector_model
        self.assets = Asset.get_state()

    def update_state(self):
        self.increment_timestep()
        self.update_asset_capacities()

    def increment_timestep(self, increment=1):
        """Increments the timestep by step-length
        """
        self._timestep += increment

    def update_asset_capacities(self):
        """
        """
        for asset in self.assets:
            asset_decisions = asset.get_decisions()
            asset.capacity += asset_decisions.new_capacity
            asset.capacity -= asset.retiring_capacity(self._timestep)


class AbstractModel(ABC):
    """An instance of a model contains Interfaces wrapped around sector models
    """
    def __init__(self):
        self._commodities = []
        self._regions = []
        self.timesteps = []

    @property
    def commodities(self):
        return self._commodities

    @commodities.setter
    def commodities(self, commodities):
        self._commodities = commodities

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, value):
        self._regions = value

    @property
    def timesteps(self):
        return self.timesteps

    @timesteps.setter
    def timesteps(self, value):
        self.timesteps = value

    @abstractmethod
    def attach_interface(self):
        """
        """
        pass


class Model(AbstractModel):
    def attach_interface(self):
        pass


class AbstractModelBuilder(ABC):
    """Instantiates and validates a model before releasing it for operation
    """
    def __init__(self):
        self.model = None

    def initialise_model(self):
        self.model = Model()

    @abstractmethod
    def add_commodities(self):
        pass

    @abstractmethod
    def add_regions(self):
        pass

    @abstractmethod
    def add_timeseteps(self):
        pass

    def validate(self):
        self.validate_commodities()
        self.validate_regions()
        self.validate_timesteps()

    @abstractmethod
    def validate_commodities(self):
        pass

    @abstractmethod
    def validate_regions(self):
        pass

    @abstractmethod
    def validate_timesteps(self):
        pass


if __name__ == "__main__":
    pass
