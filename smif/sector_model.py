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
import logging
import os
from abc import ABC, abstractmethod

import importlib
from smif.parameters import ModelParameters

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SectorModel(ABC):
    """A representation of the sector model with inputs and outputs

    """
    def __init__(self):
        self._model_name = None

        self.interventions = []
        self.system = []

        self._inputs = ModelParameters({})
        self._outputs = ModelParameters({})

        self.logger = logging.getLogger(__name__)

    def validate(self):
        """Validate that this SectorModel has been set up with sufficient data
        to run
        """
        pass

    @property
    def name(self):
        """The name of the sector model

        Returns
        =======
        str
            The name of the sector model
        """
        return self._model_name

    @name.setter
    def name(self, value):
        self._model_name = value

    @property
    def inputs(self):
        """The inputs to the model

        The inputs should be specified in a list.  For example::

                - name: eletricity_price
                  spatial_resolution: GB
                  temporal_resolution: annual

        Arguments
        =========
        value : list
            A list of dicts of inputs to the model.
            These includes parameters, assets and exogenous data

        Returns
        =======
        :class:`smif.parameters.ModelInputs`

        """
        return self._inputs

    @inputs.setter
    def inputs(self, value=None):
        if value is not None:
            assert isinstance(value, list)
        else:
            value = []

        self._inputs = ModelParameters(value)

    @property
    def outputs(self):
        """The outputs from the model

        Arguments
        =========
        value : list
            A list of dicts of outputs from the model.
            This may include results
            and metrics

        Returns
        =======
        :class:`smif.parameters.ModelParameters`

        """
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        if value is not None:
            assert isinstance(value, list)
        else:
            value = []

        self._outputs = ModelParameters(value)

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
        =========
        results : dict
            A nested dict of the results from the :py:meth:`simulate` method

        Returns
        =======
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

    """

    def __init__(self, name):
        self._sector_model_name = name
        self._sector_model = None
        self.logger = logging.getLogger(__name__)

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

    def add_inputs(self, input_dict):
        """Add inputs to the sector model
        """
        msg = "Sector model must be loaded before adding inputs"
        assert self._sector_model is not None, msg

        self._sector_model.inputs = input_dict

    def add_outputs(self, output_dict):
        """Add outputs to the sector model
        """
        msg = "Sector model must be loaded before adding outputs"
        assert self._sector_model is not None, msg

        self._sector_model.outputs = output_dict

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
