# -*- coding: utf-8 -*-
"""This module acts as a bridge to the sector models from the controller

The :class:`SectorModel` exposes several key methods for running wrapped
sector models.  To add a sector model to an instance of the framework,
first implement :class:`SectorModel`.

Utility Methods
===============
A number of utility methods are included to ease the integration of a
SectorModel wrapper within a System of Systems model.  These include:

- ``get_region_names(region_set_name)`` - gets a list of region names
- ``get_interval_names(interval_set_name)`` - gets a list of interval names

Key Functions
=============
This class performs several key functions which ease the integration of sector
models into the system-of-systems framework.

The user must implement the various abstract functions throughout the class to
provide an interface to the sector model, which can be called upon by the
framework. From the model's perspective, :class:`SectorModel` provides a bridge
from the sector-specific problem representation to the general representation
which allows reasoning across infrastructure systems.

The key functions include:

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

from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.intervention import Intervention
from smif.model import Model

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SectorModel(Model, metaclass=ABCMeta):
    """A representation of the sector model with inputs and outputs

    Implement this class to enable integration of the wrapped simulation model
    into a system-of-system model.

    Arguments
    ---------
    name : str
        The unique name of the sector model

    Notes
    -----

    Implement the various abstract functions throughout the class to
    provide an interface to the simulation model, which can then be called
    upon by the framework.

    The key methods in the SectorModel class which must be overridden are:

    - :py:meth:`SectorModel.simulate`
    - :py:meth:`SectorModel.extract_obj`

    An implementation may also override:

    - :py:meth:`SectorModel.before_model_run`

    A number of utility methods are included to ease the integration of a
    SectorModel wrapper within a System of Systems model.  These include:

    * ``get_region_names(region_set_name)`` - Get a list of region names
    * ``get_interval_names(interval_set_name)`` - Get a list of interval names

    For example, within the implementation of the simulate method, call::

        self.get_region_names('lad')

    to return a list of region names defined in the region register at runtime.

    """
    def __init__(self, name):
        super().__init__(name)

        self.path = ''
        self.initial_conditions = []
        self.interventions = []

        self.logger = logging.getLogger(__name__)

    def get_current_interventions(self, state):
        """Get the interventions the exist in the current state

        Arguments
        ---------
        state : list
            A list of tuples that represent the state of the system in the
            current planning timestep

        Returns
        -------
        list of intervention dicts with build_year attribute
        """

        interventions = []
        for decision in state:
            name = decision['name']
            build_year = decision['build_year']
            if name in self.intervention_names:
                for intervention in self.interventions:
                    if intervention.name == name:
                        serialised = intervention.as_dict()
                        serialised['build_year'] = build_year
                        interventions.append(serialised)

        msg = "State matched with %s interventions"
        self.logger.info(msg, len(interventions))

        return interventions

    def as_dict(self):
        """Serialize the SectorModel object as a dictionary

        Returns
        -------
        dict
        """
        config = {
            'name': self.name,
            'description': self.description,
            'path': self.path,
            'classname': self.__class__.__name__,
            'inputs': [inp.as_dict() for inp in self.inputs.values()],
            'outputs': [out.as_dict() for out in self.outputs.values()],
            'parameters': self.parameters.as_list(),
            'interventions': [inter.as_dict() for inter in self.interventions],
            'initial_conditions': self.initial_conditions
        }
        return config

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
        self.logger.debug("Adding input %s to %s", name, self.name)
        input_metadata = {"name": name,
                          "spatial_resolution": spatial_resolution,
                          "temporal_resolution": temporal_resolution,
                          "units": units}

        self.inputs.add_metadata(input_metadata)

    def add_output(self, name, spatial_resolution, temporal_resolution, units):
        """Add an output to the sector model

        Arguments
        ---------
        name: str
        spatial_resolution: :class:`smif.convert.area.RegionSet`
        temporal_resolution: :class:`smif.convert.interval.IntervalSet`
        units: str

        """
        self.logger.debug("Adding output %s to %s", name, self.name)
        output_metadata = {"name": name,
                           "spatial_resolution": spatial_resolution,
                           "temporal_resolution": temporal_resolution,
                           "units": units}

        self.outputs.add_metadata(output_metadata)

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
        return [intervention.name for intervention in self.interventions]

    def before_model_run(self, data):
        """Implement this method to conduct pre-model run tasks

        Arguments
        ---------
        data: smif.data_layer.DataHandle
            Access parameter values (before any model is run, no dependency
            input data or state is guaranteed to be available)
            Access decision/system state (i.e. initial_conditions)
        """
        pass

    @abstractmethod
    def simulate(self, data):
        """Implement this method to run the model

        Arguments
        ---------
        data: smif.data_layer.DataHandle
            Access state, parameter values, dependency inputs

        Notes
        -----
        See docs on :class:`smif.data_layer.DataHandle` for details of how to
        access inputs, parameters and state and how to set results.

        ``interval``
            should reference an id from the interval set corresponding to
            the output parameter, as specified in model configuration
        ``region``
            should reference a region name from the region set corresponding to
            the output parameter, as specified in model configuration

        To obtain simulation model data in this method,
        use the data_handle methods such as::

            parameter_value = data.get_parameter('my_parameter_name')

        Other useful methods are ``get_base_timestep_data(input_name)``,
        ``get_previous_timestep_data(input_name)``,
        ``get_parameter(parameter_name)``, ``get_data(input_name, timestep=None)``,
        ``get_parameters()`` and
        ``get_results(output_name, model_name=None, modelset_iteration=None,
        decision_iteration=None, timestep=None)``.

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

    def get_region_names(self, region_set_name):
        """Get the unordered list of region names for
        ``region_set_name``

        Returns
        -------
        list
            A list of region names
        """
        return self.regions.get_entry(region_set_name).get_entry_names()

    def get_regions(self, region_set_name):
        """Get the list of regions for ``region_set_name``

        Returns
        -------
        list
            A list of GeoJSON-style dicts
        """
        return self.regions.get_entry(region_set_name).as_features()

    def get_region_centroids(self, region_set_name):
        """Get the list of region centroids for ``region_set_name``

        Returns
        -------
        list
            A list of GeoJSON-style dicts, with Point features corresponding to
            region centroids
        """
        return self.regions.get_entry(region_set_name).centroids_as_features()

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

    def construct(self, sector_model_config, timesteps):
        """Constructs the sector model

        Arguments
        ---------
        sector_model_config : dict
            The sector model configuration data
        """
        self.load_model(sector_model_config['path'], sector_model_config['classname'])
        self._sector_model.name = sector_model_config['name']
        self._sector_model.description = sector_model_config['description']
        self._sector_model.timesteps = timesteps

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
            self._sector_model.path = model_path

        else:
            msg = "Cannot find '{}' for the '{}' model".format(
                model_path, self._sector_model_name)
            raise FileNotFoundError(msg)

    def add_initial_conditions(self, initial_conditions):
        """Adds initial conditions (state) for a model
        """
        self._sector_model.initial_conditions = initial_conditions

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
        for intervention in intervention_list:
            intervention_obj = Intervention(data=intervention, sector=self._sector_model_name)
            self._sector_model.interventions.append(intervention_obj)

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
