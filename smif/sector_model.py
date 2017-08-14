# -*- coding: utf-8 -*-
"""This module acts as a bridge to the sector models from the controller

The :class:`SectorModel` exposes several key methods for running wrapped
sector models.  To add a sector model to an instance of the framework,
first implement :class:`SectorModel`.

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

from smif.composite import Model
from smif.metadata import MetadataSet

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SectorModel(Model, metaclass=ABCMeta):
    """A representation of the sector model with inputs and outputs

    """
    def __init__(self):
        super().__init__(None, MetadataSet([]), MetadataSet([]))

        self.interventions = []
        self.system = []

        self._inputs = MetadataSet([])
        self._outputs = MetadataSet([])
        self.regions = None
        self.intervals = None

        self.logger = logging.getLogger(__name__)

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
        spatial_resolution: :class:`smif.convert.area.RegionRegister`
        temporal_resolution: :class:`smif.convert.interval.TimeIntervalRegister`
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
        spatial_resolution: :class:`smif.convert.area.RegionRegister`
        temporal_resolution: :class:`smif.convert.interval.TimeIntervalRegister`
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

        Arguments
        ---------
        initial_conditions: list
            A list of past Interventions, with build dates and locations as
            necessary to specify the infrastructure system to be modelled.
        """
        pass

    @abstractmethod
    def simulate(self, decisions, state, data):
        """Implement this method to run the model

        Arguments
        ---------
        decisions: list
            A list of :py:class:Intervention to apply to the modelled system
        state: list
            A list of :py:class:StateData to update the state of the modelled system
        data: dict
            A dictionary of the format:
            ``data[parameter] = [SpaceTimeValue(region, interval, value, units), ...]``
        Returns
        -------
        dict
            A dictionary of the format:
            ``results[parameter] = [SpaceTimeValue(region, interval, value, units), ...]``

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


class SectorModelBuilder(object):
    """Build the components that make up a sectormodel from the configuration

    Parameters
    ----------
    name : str
        The name of the sector model
    registers : dict
        A package of spatial, temporal, unit registers

    Returns
    -------
    :class:`~smif.sector_model.SectorModel`

    Examples
    --------
    Call :py:meth:`SectorModelBuilder.construct` to populate a
    :class:`SectorModel` object and :py:meth:`SectorModelBuilder.finish`
    to return the validated and dependency-checked system-of-systems model.

    >>> builder = SectorModelBuilder(name, registers, secctor_model)
    >>> builder.construct(config_data)
    >>> sos_model = builder.finish()

    """

    def __init__(self, name, registers, sector_model=None):
        self._sector_model_name = name
        self._sector_model = sector_model
        self.registers = registers
        self.logger = logging.getLogger(__name__)

    def construct(self, model_data):
        """Constructs the sector model

        Arguments
        ---------
        model_data : dict
            The sector model configuration data
        """
        self.load_model(model_data['path'], model_data['classname'])
        self.create_initial_system(model_data['initial_conditions'])
        self.add_inputs(model_data['inputs'])
        self.add_outputs(model_data['outputs'])
        self.add_interventions(model_data['interventions'])

    def load_model(self, model_path, classname):
        """Dynamically load model module

        """
        if os.path.exists(model_path):
            self.logger.info("Importing run module from %s", model_path)

            spec = importlib.util.spec_from_file_location(
                self._sector_model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            klass = module.__dict__[classname]

            self._sector_model = klass()
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

    def add_inputs(self, input_dicts):
        """Add inputs to the sector model
        """
        msg = "Sector model must be loaded before adding inputs"
        assert self._sector_model is not None, msg

        regions = self.registers['regions']
        intervals = self.registers['intervals']

        if input_dicts:

            for model_input in input_dicts:
                name = model_input['name']

                spatial_resolution = model_input['spatial_resolution']
                region_set = regions.get_entry(spatial_resolution)

                temporal_resolution = model_input['temporal_resolution']
                interval_set = intervals.get_entry(temporal_resolution)

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

        regions = self.registers['regions']
        intervals = self.registers['intervals']

        if output_dicts:

            for model_output in output_dicts:
                name = model_output['name']

                spatial_resolution = model_output['spatial_resolution']
                region_set = regions.get_entry(spatial_resolution)

                temporal_resolution = model_output['temporal_resolution']
                interval_set = intervals.get_entry(temporal_resolution)

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
