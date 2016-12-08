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

LOGGER = logging.getLogger(__name__)

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
    """An asset is a decision targeted capacity that persists across timesteps.

    An Asset is otherwise an Input to a model.

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

    """

    def __init__(self, region, timestep, sector_model, parameter_map):
        self._assets = parameter_map
        self._region = region
        self._timestep = timestep
        self._sector_model = sector_model

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
    e.g. 'water treatment plant capacity', and a relevant subset
    of :class:`Output` (optional) e.g. 'reservoir level'.

    Note
    ====
    The state transition is the process by which the current state of the
    model, any decisions which affect asset capacities (e.g. investment or
    retirement) and so on are carried over from one year to the next.

    An initial (over-complicated) attempt incorporated decisions into the
    `new_capacity` variables, which then update the state in the private
    method `_update_asset_capacities`.

    This state-transition may need to be moved into :class:`Interface`.

    """

    def initialise_from_tuples(self, list_of_assets):
        """Utility function to initialise the state from a list of tuples

        Parameters
        ==========
        list_of_assets : list
            A list of asset dictionaries with which to initialise
            the SectorModel state.
        """
        for asset in list_of_assets:
            self._assets[asset[0]] = ConcreteAsset(asset[0], asset[1])

    @property
    def sector_model(self):
        """The sector model with which this state is associated

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

        Note
        ====
        This should also include anything else which is required to recorded in
        state, such as in the
        `tests.fixtures.water_supply.ExampleWaterSupplySimulationWithReservoir`
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
        raise NotImplementedError("Can't yet persist the data")

    def add_new_capacity(self, list_of_new_assets):
        """Populates the new_capacity attribute of the :class:`Asset` with
        new capacities added in this year

        Parameters
        ==========
        list_of_new_assets : list of dict
            List of new asset capacities in a dict with 'name'
            and 'capacity' keys

        """
        for new_asset in list_of_new_assets:
            name = new_asset['name']
            new_capacity = new_asset['capacity']
            self._assets[name].new_capacity = new_capacity

    def _update_asset_capacities(self):
        """Pushes the changes to the simulation model
        """
        LOGGER.info("Updating state variable for {}".format(self.sector_model))
        for name, asset in self._assets.items():
            self._add_capacity_to_asset(asset)
            self._remove_capacity_of_asset(asset)

    def _add_capacity_to_asset(self, asset):
        """Pushes the addition of asset capacity into the simulation model

        Arguments
        =========
        asset : :class:`Asset`
        """
        pass

    def _remove_capacity_of_asset(self, asset):
        """Pushes the removal of asset capacity into the simulation model

        Arguments
        =========
        asset : :class:`Asset`
        """
        pass
