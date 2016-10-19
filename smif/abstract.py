#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Abstract base classes for the scalable modelling of infrastructure systems

"""
from __future__ import absolute_import, division, print_function

import logging
from abc import ABC, abstractmethod
import numpy as np

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class ModelInputs(object):
    """A container for all the model inputs

    """
    def __init__(self, inputs):
        self.input_dict = inputs

        (self._decision_variable_names,
         self._decision_variable_values,
         self._decision_variable_bounds) = self.get_decision_variables()

        (self._parameter_names,
         self._parameter_bounds,
         self._parameter_values) = self.get_parameter_values()

    @property
    def parameter_names(self):
        """A list of ordered parameter names
        """
        return self._parameter_names

    @property
    def parameter_bounds(self):
        """An array of tuples of parameter bounds
        """
        return self._parameter_bounds

    @property
    def parameter_values(self):
        """An array of parameter values
        """
        return self._parameter_values

    @property
    def decision_variable_names(self):
        """A list of decision variable names
        """
        return self._decision_variable_names

    @property
    def decision_variable_values(self):
        """An array of decision variable values
        """
        return self._decision_variable_values

    @property
    def decision_variable_bounds(self):
        """An array of tuples of decision variable bounds
        """
        return self._decision_variable_bounds

    def get_decision_variables(self):
        """Extracts an array of decision variables from a dictionary of inputs

        Returns
        =======
        ordered_names : :class:`numpy.ndarray`
            The names of the decision variables in the order specified by the
            'index' key in the entries of the inputs
        bounds : :class:`numpy.ndarray`
            The bounds ordered by the index key
        initial : :class:`numpy.ndarray`
            The initial values ordered by the index key

        Notes
        =====
        The inputs are expected to be defined using the following keys::

            'decision variables': [<list of decision variable names>]
            'parameters': [<list of parameter names>]
            '<decision variable name>': {'bounds': (<tuple of upper and lower
                                                     bound>),
                                         'index': <scalar showing position in
                                                   arguments>},
                                         'init': <scalar showing initial value
                                                  for solver>
                                          },
            '<parameter name>': {'bounds': (<tuple of upper and lower range for
                                            sensitivity analysis>),
                                 'index': <scalar showing position in
                                          arguments>,
                                 'value': <scalar showing value for model>
                                  },
        """

        names = self.input_dict['decision variables']
        number_of_decision_variables = len(names)

        indices = [self.input_dict[name]['index'] for name in names]
        assert len(indices) == number_of_decision_variables, \
            'Index entries do not match the number of decision variables'
        initial = np.zeros(number_of_decision_variables, dtype=np.float)
        bounds = np.zeros(number_of_decision_variables, dtype=(np.float, 2))
        ordered_names = np.zeros(number_of_decision_variables, dtype='U30')

        for name, index in zip(names, indices):
            initial[index] = self.input_dict[name]['init']
            bounds[index] = self.input_dict[name]['bounds']
            ordered_names[index] = name

        return ordered_names, initial, bounds

    def get_parameter_values(self):
        """Extracts an array of parameters from a dictionary of inputs

        Returns
        =======
        ordered_names : :class:`numpy.ndarray`
            The names of the parameters in the order specified by the
            'index' key in the entries of the inputs
        bounds : :class:`numpy.ndarray`
            The parameter bounds (or range) ordered by the index key
        values : :class:`numpy.ndarray`
            The parameter values ordered by the index key
        """
        names = self.input_dict['parameters']
        number_of_parameters = len(names)

        indices = [self.input_dict[name]['index'] for name in names]
        assert len(indices) == number_of_parameters, \
            'Index entries do not match the number of decision variables'
        values = np.zeros(number_of_parameters, dtype=np.float)
        bounds = np.zeros(number_of_parameters, dtype=(np.float, 2))
        ordered_names = np.zeros(number_of_parameters, dtype='U30')

        for name, index in zip(names, indices):
            values[index] = self.input_dict[name]['value']
            bounds[index] = self.input_dict[name]['bounds']
            ordered_names[index] = name

        return ordered_names, bounds, values


class SectorModel(ABC):
    """An abstract representation of the sector model with inputs and outputs

    Parameters
    ==========
    schema : dict
        A dictionary of parameter, asset and exogenous data names with expected
        types. Used for validating presented data.

    Attributes
    ==========
    model
        An instance of the sector model
    inputs : dict
        A dictionary of inputs to the model. This may include parameters,
        assets and exogenous data.

    """
    def __init__(self):
        self.model = None
        self._model_executable = None
        self._inputs = {}
        self._schema = None

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = ModelInputs(value)

    @abstractmethod
    def initialise(self):
        """Use this method to initialise (load the input data into) the model

        Returns
        =======
        results : dict
            A dictionary of results with keys as output names e.g. 'cost' and
            values

        """
        results = None

        return results

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
        raise NotImplemented("Optimisation is not yet implemented")

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
        self._interfaces = None
        self.timesteps = None
        self.sector_models = []

    @property
    def timesteps(self):
        return self.timesteps

    @timesteps.setter
    def timesteps(self, value):
        self.timesteps = value

    def attach_interface(self, interface):
        """Adds an interface to the list of interfaces which comprise a model
        """
        assert isinstance(interface, Interface)
        self.sector_models.append(interface)

    def run(self):
        """Run the system of systems model
        """
        pass


class Model(AbstractModel):
    """A model is a collection of sector models joined through dependencies

    """
    def __init__(self):
        super().__init__()
        self.almanac = None

    def _add_to_almanac(self, model_inputs, model_outputs):
        self.almanac['inputs'] = model_inputs
        self.almanac['outputs'] = model_outputs

    def run(self):
        """
        1. Determine running order
        2. Run each sector model
        3. Return success or failure
        """

    def _determine_running_order(self):
        model_inputs = []
        model_outputs = []
        for sector_model in self.sector_models:
            model_inputs.extend(sector_model.get_inputs())
            model_outputs.extend(sector_model.get_outputs())
        self._add_to_almanac(model_inputs, model_outputs)


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
