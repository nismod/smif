# -*- coding: utf-8 -*-
"""This module acts as a bridge to the sector models from the controller

The :class:`SectorModel` exposes several key methods for running wrapped
sector models.  To add a sector model to an instance of the framework,
first implement :class:`SectorModel`.

Utility Methods
===============
A number of utility methods are included to ease the integration of a
SectorModel wrapper within a System of Systems model.  These include::

get_scenario_data(input_name)
    Get an array of scenario data (timestep-by-region-by-interval)
get_region_names(region_set_name)
    Get a list of region names
get_interval_names(interval_set_name)
    Get a list of interval names

Key Functions
=============
This class performs several key functions which ease the integration of sector
models into the system-of-systems framework.

The user must implement the various abstract functions throughout the class to
provide an interface to the sector model, which can be called upon by the
framework. From the model's perspective, :class:`SectorModel` provides a bridge
from the sector-specific problem representation to the general representation
which allows reasoning across infrastructure systems.

The key functions include

* converting input/outputs to/from geographies/temporal resolutions
* converting control vectors from the decision layer of the framework, to
  asset Interventions specific to the sector model
* returning scaler/vector values to the framework to enable measurements of
  performance, particularly for the purposes of optimisation and rule-based
  approaches

"""
import importlib
import logging
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from smif import StateData
from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.model import Model

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SectorModel(Model, metaclass=ABCMeta):
    """A representation of the sector model with inputs and outputs

    Arguments
    ---------
    name : str
        The unique name of the sector model

    """
    def __init__(self, name):
        super().__init__(name)

        self._initial_state = defaultdict(dict)
        self.interventions = []
        self.system = []
        self._user_data = {}

        self.logger = logging.getLogger(__name__)

    @property
    def user_data(self):
        """A utility dictionary provided for use by the model wrapper
        """
        return self._user_data

    @user_data.setter
    def user_data(self, value):
        self.logger.debug("Adding %s to user data for %s", value, self.name)
        self._user_data = value

    def add_input(self, name, spatial_resolution, temporal_resolution, units):
        """Add an input to the sector model

        The inputs should be specified in a list.  For example::

                - name: electricity_price
                  spatial_resolution: GB
                  temporal_resolution: annual
                  units: £/kWh

        Arguments
        ---------
        name: str
        spatial_resolution: :class:`smif.convert.area.RegionSet`
        temporal_resolution: :class:`smif.convert.interval.IntervalSet`
        units: str

        """
        input_metadata = {"name": name,
                          "spatial_resolution": spatial_resolution,
                          "temporal_resolution": temporal_resolution,
                          "units": units}

        self._model_inputs.add_metadata(input_metadata)

    def add_output(self, name, spatial_resolution, temporal_resolution, units):
        """Add an output to the sector model

        Arguments
        ---------
        name: str
        spatial_resolution: :class:`smif.convert.area.RegionSet`
        temporal_resolution: :class:`smif.convert.interval.IntervalSet`
        units: str

        """
        output_metadata = {"name": name,
                           "spatial_resolution": spatial_resolution,
                           "temporal_resolution": temporal_resolution,
                           "units": units}

        self._model_outputs.add_metadata(output_metadata)

    def validate(self):
        """Validate that this SectorModel has been set up with sufficient data
        to run
        """
        pass

    @property
    def intervention_names(self):
        """The names of the interventions

        Returns
        =======
        list
            A list of the names of the interventions
        """
        return [intervention['name'] for intervention in self.interventions]

    @abstractmethod
    def initialise(self, initial_conditions):
        """Implement this method to set up the model system

        This method is called as the SectorModel is constructed, and prior to
        establishment of dependencies and other data links.

        Arguments
        ---------
        initial_conditions: list
            A list of past Interventions, with build dates and locations as
            necessary to specify the infrastructure system to be modelled.
        """
        pass

    def before_model_run(self):
        """Implement this method to conduct pre-model run tasks
        """
        pass

    @abstractmethod
    def simulate(self, timestep, data=None):
        """Implement this method to run the model

        Arguments
        ---------
        timestep : int
            The timestep for which to run the SectorModel
        data: dict, default=None
            A collection of state, parameter values, dependency inputs
        Returns
        -------
        results : dict
            This method should return a results dictionary

        Notes
        -----
        In the results returned from the :py:meth:`simulate` method:

        ``interval``
            should reference an id from the interval set corresponding to
            the output parameter, as specified in model configuration
        ``region``
            should reference a region name from the region set corresponding to
            the output parameter, as specified in model configuration

        """
        pass

    @abstractmethod
    def extract_obj(self, results):
        """Implement this method to return a scalar value objective function

        This method should take the results from the output of the
        :py:meth:`simulate` method, process the results,
        and return a scalar value which can be used as a component of
        the objective function by the decision layer

        Arguments
        ---------
        results : dict
            A nested dict of the results from the :py:meth:`simulate` method

        Returns
        -------
        float
            A scalar component generated from the simulation model results
        """
        pass

    def get_scenario_data(self, input_name):
        """Returns all scenario dependency data as a numpy array

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray which has the dimensions timestep-by-regions-by-intervals
        """
        if input_name not in self.deps:
            raise ValueError("Scenario data for %s not available for this input",
                             input_name)

        return self.deps[input_name].source_model._data

    def get_region_names(self, region_set_name):
        """Get the list of region names for ``region_set_name``

        Returns
        -------
        list
            A list of region names
        """
        return self.regions.get_entry(region_set_name).get_entry_names()

    def get_interval_names(self, interval_set_name):
        """Get the list of interval names for ``interval_set_name``

        Returns
        -------
        list
            A list of interval names
        """
        return self.intervals.get_entry(interval_set_name).get_entry_names()


class SectorModelBuilder(object):
    """Build the components that make up a sectormodel from the configuration

    Parameters
    ----------
    name : str
        The name of the sector model
    sector_model : smif.model.SectorModel, default=None
        The sector model object

    Returns
    -------
    :class:`~smif.sector_model.SectorModel`

    Examples
    --------
    Call :py:meth:`SectorModelBuilder.construct` to populate a
    :class:`SectorModel` object and :py:meth:`SectorModelBuilder.finish`
    to return the validated and dependency-checked system-of-systems model.

    >>> builder = SectorModelBuilder(name, secctor_model)
    >>> builder.construct(config_data)
    >>> sos_model = builder.finish()

    """

    def __init__(self, name, sector_model=None):
        self._sector_model_name = name
        self._sector_model = sector_model
        self.interval_register = get_interval_register()
        self.region_register = get_region_register()
        self.logger = logging.getLogger(__name__)

    def construct(self, sector_model_config):
        """Constructs the sector model

        Arguments
        ---------
        sector_model_config : dict
            The sector model configuration data
        """
        self.load_model(sector_model_config['path'], sector_model_config['classname'])
        self.create_initial_system(sector_model_config['initial_conditions'])
        self.add_inputs(sector_model_config['inputs'])
        self.add_outputs(sector_model_config['outputs'])
        self.add_interventions(sector_model_config['interventions'])
        self.add_initial_conditions(sector_model_config['initial_conditions'])
        self.add_parameters(sector_model_config['parameters'])

    def load_model(self, model_path, classname):
        """Dynamically load model module

        Arguments
        ---------
        model_path : str
            The path to the python module which contains the SectorModel
            implementation
        classname : str
            The name of the class of the SectorModel implementation

        """
        if os.path.exists(model_path):
            self.logger.info("Importing run module from %s", model_path)

            spec = importlib.util.spec_from_file_location(
                self._sector_model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            klass = module.__dict__[classname]

            self._sector_model = klass(self._sector_model_name)
            self._sector_model.name = self._sector_model_name

        else:
            msg = "Cannot find '{}' for the '{}' model".format(
                model_path, self._sector_model_name)
            raise FileNotFoundError(msg)

    def create_initial_system(self, initial_conditions):
        """Set up model with initial system
        """
        msg = "Sector model must be loaded before creating initial system"
        assert self._sector_model is not None, msg

        self._sector_model.initialise(initial_conditions)

    def add_parameters(self, parameter_config):
        """Add parameter configuration to sector model

        Arguments
        ---------
        parameter_config : list
            A list of dicts with keys ``name``, ``description``,
            ``absolute_range``, ``suggested_range``, ``default_value``,
            ``units``, ``parent``
        """

        for parameter in parameter_config:
            self._sector_model.add_parameter(parameter)

    def add_inputs(self, input_dicts):
        """Add inputs to the sector model
        """
        msg = "Sector model must be loaded before adding inputs"
        assert self._sector_model is not None, msg

        if input_dicts:

            for model_input in input_dicts:
                name = model_input['name']

                spatial_resolution = model_input['spatial_resolution']
                region_set = self.region_register.get_entry(spatial_resolution)

                temporal_resolution = model_input['temporal_resolution']
                interval_set = self.interval_register.get_entry(temporal_resolution)

                units = model_input['units']

                self._sector_model.add_input(name,
                                             region_set,
                                             interval_set,
                                             units)

    def add_outputs(self, output_dicts):
        """Add outputs to the sector model
        """
        msg = "Sector model must be loaded before adding outputs"
        assert self._sector_model is not None, msg

        if output_dicts:

            for model_output in output_dicts:
                name = model_output['name']

                spatial_resolution = model_output['spatial_resolution']
                region_set = self.region_register.get_entry(spatial_resolution)

                temporal_resolution = model_output['temporal_resolution']
                interval_set = self.interval_register.get_entry(temporal_resolution)

                units = model_output['units']

                self._sector_model.add_output(name,
                                              region_set,
                                              interval_set,
                                              units)

    def add_interventions(self, intervention_list):
        """Add interventions to the sector model

        Parameters
        ----------
        intervention_list : list
            A list of dicts of interventions

        """
        msg = "Sector model must be loaded before adding interventions"
        assert self._sector_model is not None, msg

        self._sector_model.interventions = intervention_list

    def add_initial_conditions(self, initial_conditions):
        """Adds initial conditions (state) for a model
        """
        state_data = [self.intervention_state_from_data(datum)
                      for datum in initial_conditions
                      if datum.data]
        self._sector_model._initial_state = list(state_data)

    @staticmethod
    def intervention_state_from_data(intervention_data):
        """Unpack an intervention from the initial system to extract StateData
        """
        target = None
        data = {}
        for key, value in intervention_data.items():
            if key == "name":
                target = value

            if isinstance(value, dict) and "is_state" in value and value["is_state"]:
                del value["is_state"]
                data[key] = value

        return StateData(target, data)

    def validate(self):
        """Check and/or assert that the sector model is correctly set up
        - should raise errors if invalid
        """
        assert self._sector_model is not None, "Sector model not loaded"
        self._sector_model.validate()
        return True

    def finish(self):
        """Validate and return the sector model
        """
        self.validate()
        return self._sector_model
