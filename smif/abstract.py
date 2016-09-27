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

logger = logging.getLogger(__name__)


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
    def print_all_commodities(cls):
        for commodity in cls.names:
            print("{}".format(commodity))


class SectorModel(ABC):
    """An abstract representation of the sector model with inputs and outputs.

    Parameters
    ==========


    Attributes
    ==========
    state : :class:`State`
        The state of the sector model
    results
        The output from the simulation model
    model
        An instance of the sector model
    input_hooks : dict
        A mapping between :class:`Dependency` and sector model inputs
    output_hooks : dict
        A mapping between :class:`Output` and sector model outputs

    """
    def __init__(self):
        self._run_successful = None
        self.results = None
        self.model = None
        self._model_executable = None
        self.state = None

    @abstractmethod
    def initialise(self):
        """Use this to initialise the model
        """
        pass

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
    def model_executable(self):
        """The path to the model executable
        """
        return self._model_executable

    @model_executable.setter
    def model_executable(self, value):
        """The path to the model executable
        """
        self._model_executable = value


class Interface(ABC):
    """Provides the interface between a sector model and other interfaces

    Parameters
    ==========
    inputs : collection of :class:`Input`
        The commodities required as inputs to the `sector_model`
    outputs : collection of :class:`Output`
        The outputs produced by `sector_model`
    metrics : collection of :class:`Metric`
        Metrics computed after running the `sector_model`
    region : str
        The unique identifier for the region
    timestep : int
        The current timestep
    sector_model : :class:`SectorModel`
        The sector model wrapped by the Interface

    Attributes
    ==========
    state : :class:`State`

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


class Dependency(Input, ABC):
    """A dependency is a type of input which links interfaces
    """


class Decision(ABC):
    """A decision denotes an Asset target, and bounds for the current Interface

    Normally, an instance of a decision corresponds to an investment decision
    or policy to increase capacity, or decrease demand

    Parameters
    ==========
    target_asset : str
        The asset which is targetted by the decision
    lower_bound : float
        The upper bound on the value of the investment in the target asset
    upper_bound : float
        The lower bound on the value of the investment in the target asset
    """
    def __init__(self, target_asset, lower_bound, upper_bound):
        assert isinstance(target_asset, Asset)
        self._target_asset = target_asset
        assert upper_bound >= lower_bound
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound


class Asset(ABC):
    """An asset is a decision targeted capacity that persists across timesteps

    Examples of assets include power stations, water treatment plants, roads,
    railway tracks, airports, ports, centres of demand such as houses or
    factories, waste processing plants etc.

    An Asset is targetted by and influenced by a :class:`Decision` but only
    need to be defined in the :class:`Interface` if targetted
    by a :class:`Decision`.

    A snapshot of the current Assets in a model is represented by
    :class:`State` and is persisted across model-years.

    The Asset-state is also persisted (written to the datastore).

    Parameters
    ==========
    name : str
        The name of the asset
    capacity : float
        The initial capacity of the asset

    """
    assets = []

    def __init__(self, name, capacity):
        """
        """
        self._name = name
        self._capacity = capacity
        self._new_capacity = 0
        self._retiring_capacity = 0
        self._model_asset = None
        self.assets.append(self)

    def get_decisions(self):
        """Returns a container of :class:`Decision`
        """
        return self._new_capacity

    @property
    def capacity(self):
        """The current capacity of the asset
        """
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = value

    @property
    def name(self):
        """The name of the asset
        """
        return self._name

    @property
    def new_capacity(self):
        """The capacity of the asset which will be added when updating state
        """
        return self._new_capacity

    @new_capacity.setter
    def new_capacity(self, value):
        self._new_capacity = value

    @property
    def retiring_capacity(self):
        """The capacity of the asset which will be removed when updating state
        """
        return self._retiring_capacity

    @retiring_capacity.setter
    def retiring_capacity(self, value):
        self._retiring_capacity = value

    @classmethod
    def get_state(cls):
        """Returns the state of all assets
        """
        return [{asset.name: asset.capacity} for asset in cls.assets]


class ConcreteAsset(Asset):

    def write_assets_to_datastore(self):
        """Writes the current :class:`State` of the Asset to the datastore
        """
        pass

    def get_decisions(self):
        """Returns a container of :class:`Decision`
        """
        pass


class AbstractState(ABC):
    """

    Parameters
    ==========
    region : str
        The name of the region
    timestep : int
        The timestep of the state
    sector_model : str
        The name of the sector model
    state_parameter_map : dict
        The mapping of the asset names (key) and the sector model parameters
        (value) which can be edited in the model to add
        or remove asset capacity
    """

    def __init__(self, region, timestep, sector_model, state_parameter_map):
        self._assets = {}
        self._region = region
        self._timestep = timestep
        self._sector_model = sector_model
        self._state_parameter_map = state_parameter_map

    def update_state(self, timestep_increment=1):
        self._increment_timestep(timestep_increment)
        self._update_asset_capacities()

    def _increment_timestep(self, increment=1):
        """Increments the timestep by step-length
        """
        self._timestep += increment

    @abstractmethod
    def _update_asset_capacities(self):
        pass

    @abstractmethod
    def write_state_to_datastore(self):
        """Writes the current state of the sector model to the datastore
        """
        pass


class State(AbstractState):
    """A static snapshot of a sector model's assets

    The state is used to record (and persist) the inter-temporal transition
    from one time-step to the next of the collection of :class:`Asset`

    """

    def initialise_from_tuples(self, list_of_assets):
        """Initialise the state

        Parameters
        ==========
        list_of_assets : list
            A list of asset dictionaries with which to initialise
            the SectorModel state.
        """
        for asset in list_of_assets:
            self._assets[asset[0]] = ConcreteAsset(asset[0], asset[1])
            if asset[0] not in self._state_parameter_map.keys():
                msg = "{} is not defined in the state parameter."
                logger.error(msg.format(asset[0]))
                raise ValueError(msg.format(asset[0]))

    @property
    def sector_model(self):
        """

        Returns
        =======
        str
            The name of the sector model
        """
        return self._sector_model

    @property
    def current_state(self):
        """Returns the current state of the wrapped simulation model

        Returns
        =======
        dict
            A dictionary of the current state including information on the
            model name, region, timestep, and asset capacities
        """
        assets = {key: val.capacity for key, val in self._assets.items()}

        return {'model': self._sector_model,
                'region': self._region,
                'timestep': self._timestep,
                'assets': assets
                }

    def write_state_to_datastore(self):
        """Writes the current state of the sector model to the datastore
        """
        raise NotImplementedError()

    def add_new_capacity(self, list_of_new_assets):
        """
        """
        for new_asset in list_of_new_assets:
            name = new_asset['name']
            new_capacity = new_asset['capacity']
            self._assets[name].new_capacity = new_capacity

    def _update_asset_capacities(self):
        """Pushes the changes to the simulation model
        """
        logger.info("Updating state variable for {}".format(self.sector_model))
        for name, asset in self._assets.items():
            self._add_capacity_to_asset(asset)
            self._remove_capacity_of_asset(asset)

    def _add_capacity_to_asset(self, asset):
        """Pushes the addition of asset capacity into the simulation model

        Arguments
        =========
        asset : :class:`Asset`
        """
        msg = "Existing capacity of {} is {}"
        logger.debug(msg.format(asset.name,
                                self._state_parameter_map[asset.name]))
        self._state_parameter_map[asset.name] += asset.new_capacity
        asset.capacity = self._state_parameter_map[asset.name]
        msg = "Added {} capacity to asset: {}"
        logger.debug(msg.format(asset.new_capacity, asset.name))
        msg = "Capacity of {} is now {}"
        logger.debug(msg.format(asset.name, asset.capacity))

    def _remove_capacity_of_asset(self, asset):
        """Pushes the removal of asset capacity into the simulation model

        Arguments
        =========
        asset : :class:`Asset`
        """
        if asset.retiring_capacity <= self._state_parameter_map[asset.name]:
            self._state_parameter_map[asset.name] -= asset.retiring_capacity
            asset.capacity = self._state_parameter_map[asset.name]
            msg = "Removed {} capacity for asset: {}"
            logger.debug(msg.format(asset.retiring_capacity, asset.name))
            msg = "Capacity of {} is now {}"
            logger.debug(msg.format(asset.name, asset.capacity))
        else:
            msg = "Retiring capacity exceeds existing capacity for asset: {}"
            logger.error(msg.format(asset.name))
            raise ValueError("Retiring capacity exceeds existing capacity")


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
    """A model is a collection of sector models joined through dependencies


    """
    sector_models = []

    def attach_interface(self, interface):
        """Adds an interface to the list of interfaces which comprise a model
        """
        assert isinstance(interface, Interface)
        self.sector_models.append(interface)


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
