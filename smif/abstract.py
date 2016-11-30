#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Abstract base classes for the scalable modelling of infrastructure systems

"""
from __future__ import absolute_import, division, print_function

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from smif.inputs import ModelInputs
from smif.outputs import ModelOutputs
from smif.sectormodel import SectorModel

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class AbstractModelWrapper(ABC):
    """Provides in interface to wrap any simulation model for optimisation

    To wrap a simulation model, subclass this wrapper, and populate the three
    methods.

    At run time, instantiate the wrapper with the ``model`` as an argument.

    Attributes
    ==========
    model
        Its useful to have the model being wrapped available via ``self.model``
        throughout the code.  Instantiate the class with an instance of the
        model your are trying to wrap

    """

    def __init__(self, model):
        self.model = model
        self._inputs = None
        self._outputs = None

    @property
    def inputs(self):
        """The inputs to the model

        Returns
        =======
        :class:`smif.inputs.ModelInputs`

        """
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """The inputs to the model

        The inputs should be specified in a dictionary, with the keys first to
        declare ``decision variables`` and ``parameters`` and corresponding
        lists.  The remainder of the dictionary should contain the bounds,
        index and values of the names decision variables and parameters.

        Arguments
        =========
        value : dict
            A dictionary of inputs to the model. This may include parameters,
            assets and exogenous data.

        """
        assert isinstance(value, dict)
        self._inputs = ModelInputs(value)

    @property
    def outputs(self):
        """The outputs from the model

        Returns
        =======
        :class:`smif.outputs.ModelOutputs`

        """
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        """
        Arguments
        =========
        value : dict
            A dictionary of outputs from the model. This may include results
            and metrics
        """
        self._outputs = ModelOutputs(value)

    @staticmethod
    def replace_line(file_name, line_num, new_data):
        """Replaces a line in a file with new data

        Arguments
        =========
        file_name: str
            The path to the input file
        line_num: int
            The number of the line to replace
        new_data: str
            The data to replace in the line

        """
        lines = open(file_name, 'r').readlines()
        lines[line_num] = new_data
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()

    @staticmethod
    def replace_cell(file_name, line_num, column_num, new_data,
                     delimiter=None):
        """Replaces a cell in a delimited file with new data

        Arguments
        =========
        file_name: str
            The path to the input file
        line_num: int
            The number of the line to replace (0-index)
        column_num: int
            The number of the column to replace (0-index)
        new_data: str
            The data to replace in the line
        delimiter: str, default=','
            The delimiter of the columns
        """
        line_num -= 1
        column_num -= 1

        with open(file_name, 'r') as input_file:
            lines = input_file.readlines()

        columns = lines[line_num].split(delimiter)
        columns[column_num] = new_data
        lines[line_num] = " ".join(columns) + "\n"

        with open(file_name, 'w') as out_file:
            out_file.writelines(lines)

    @abstractmethod
    def simulate(self, static_inputs, decision_variables):
        """This method should allow run model with inputs and outputs as arrays

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
        decision_variables : x-by-1 :class:`numpy.ndarray`
        """
        pass

    @abstractmethod
    def extract_obj(self, results):
        """Implement this method to return a scalar value objective function

        This method should take the results from the output of the `simulate`
        method, process the results, and return a scalar value which can be
        used as the objective function

        Arguments
        =========
        results : :class:`dict`
            The results from the `simulate` method

        Returns
        =======
        float
            A scalar component generated from the simulation model results
        """
        pass

    def constraints(self, parameters):
        """Express constraints for the optimisation

        Use the form outlined in :class:`scipy.optimise.minimize`, namely::

            constraints = ({'type': 'ineq',
                            'fun': lambda x: x - 3})

        Arguments
        =========
        parameters : :class:`numpy.ndarray`
            An array of parameter values passed in from the
            `SectorModel.optimise` method

        """
        constraints = ()
        return constraints


class Interface(ABC):
    """Provides the interface between a sector model and other interfaces

    Parameters
    ==========
    inputs : dict of :class:`Input`
        The commodities required as inputs to the `sector_model`
    outputs : dict of :class:`Output`
        The outputs produced by `sector_model`
    metrics : dict of :class:`Metric`
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
        self._inputs = ModelInputs(list_of_inputs)


class Dependency(ABC):
    """A dependency is a type of :class:`Input` which links sector models

    A dependency is an input to one model, which is an output from another
    model.

    Parameters
    ==========
    name : str
        The name of the input
    value : float
        The value of the input
    from_model : str
        The name of the :class:`SectorModel` which produces the output
    to_model : str
        The name of the :class:`SectorModel` which requires the output

    """
    def __init__(self, name, value, from_model, to_model):
        self._from_model = from_model
        self._to_model = to_model
        self._name = name
        self._value = value


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

    def __init__(self, region, timestep, sector_model):
        self._assets = {}
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
        `tests.fixtures.water_supply.ExampleWaterSupplySimulationReservoir`
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
        pass

    def _remove_capacity_of_asset(self, asset):
        """Pushes the removal of asset capacity into the simulation model

        Arguments
        =========
        asset : :class:`Asset`
        """
        pass


class AbstractModel(ABC):
    """An instance of a model contains Interfaces wrapped around sector models

    This is NISMOD - i.e. the system of system model which brings all of the
    sector models together.

    """
    def __init__(self):
        self._timesteps = None
        self._sector_models = []

    @property
    def timesteps(self):
        return self.timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._timesteps = value

    @property
    def model_list(self):
        return [x.name for x in self._sector_models]

    def attach_interface(self, interface):
        """Adds an interface to the list of interfaces which comprise a model
        """
        assert isinstance(interface, SectorModel)
        self._sector_models.append(interface)

    def run(self):
        """Run the system of systems model
        """
        pass


class Model(AbstractModel):
    """A model is a collection of sector models joined through dependencies

    """
    def run(self):
        """
        1. Determine running order
        2. Run each sector model
        3. Return success or failure
        """
        raise NotImplementedError("Can't run the SOS model yet")

    def optimise(self):
        """Runs a dynamic optimisation over a system-of-simulation models

        Use dynamic programming with memoization where the objective function
        :math:`Z(s)` are indexed by state :math:`s`

        if :math:`s` is in the hash table: return :math:`Z(s)`

        :math:`Z(s) = min\{Z(s) + E(Z(s'))\}`

        """
        pass

    @lru_cache(maxsize=None)
    def cost_to_go(self, state):
        value = self.model._simulate_optimised(state) + self.cost_to_go()
        return value

    def sequential_simulation(self, model, inputs, decisions):
        results = []
        for index in range(len(self._timesteps)):
            # Intialise the model
            model.model.inputs = inputs
            # Update the state from the previous year
            if index > 0:
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                logger.debug("Updating {} with {}".format(state_var,
                                                          state_res))
                model.model.inputs.parameters.update_value(state_var,
                                                           state_res)

            # Run the simulation
            decision = decisions[index]
            results.append(model.simulate(decision))
        return results


class AbstractModelBuilder(ABC):
    """Instantiates and validates a model before releasing it for operation
    """
    def __init__(self):
        self.model = None

    def initialise_model(self):
        self.model = Model()

    @abstractmethod
    def add_timesteps(self):
        pass

    def validate(self):
        self.validate_commodities()
        self.validate_regions()
        self.validate_timesteps()
        return self.model

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
