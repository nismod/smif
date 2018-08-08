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
import logging
from abc import ABCMeta, abstractmethod

from smif.intervention import Intervention
from smif.metadata import Spec
from smif.model.model import Model

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

    @classmethod
    def from_dict(cls, config):
        model = cls(config['name'])
        model.description = config['description']
        model.path = config['path']
        for input_ in config['inputs']:
            model.add_input(Spec.from_dict(input_))
        for output in config['outputs']:
            model.add_input(Spec.from_dict(output))
        for param in config['parameters']:
            model.add_input(Spec.from_dict(param))
        model.interventions = [
            Intervention.from_dict(intervention_config)
            for intervention_config in config['interventions']
        ]
        model.initial_conditions = config['initial_conditions']
        return model

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
            'parameters': [param.as_dict() for param in self.parameters.values()],
            'interventions': [inter.as_dict() for inter in self.interventions],
            'initial_conditions': self.initial_conditions
        }
        return config

    def add_input(self, spec):
        """Add an input to the sector model

        The inputs should be specified in a list.  For example::

                - name: electricity_price
                  spatial_resolution: GB
                  temporal_resolution: annual
                  units: £/kWh

        Arguments
        ---------
        spec: smif.metadata.Spec
        """
        self.logger.debug("Adding input %s to %s", spec.name, self.name)
        self.inputs[spec.name] = spec

    def add_output(self, spec):
        """Add an output to the sector model

        Arguments
        ---------
        spec: smif.metadata.Spec
        """
        self.logger.debug("Adding output %s to %s", spec.name, self.name)
        self.outputs[spec.name] = spec

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
