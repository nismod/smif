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

The key functions include:

* converting input/outputs to/from geographies/temporal resolutions
* converting control vectors from the decision layer of the framework, to
  asset Interventions specific to the sector model
* returning scalar/vector values to the framework to enable measurements of
  performance, particularly for the purposes of optimisation and rule-based
  approaches

"""
from abc import ABCMeta, abstractmethod

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
            Access state, parameter values, dependency inputs, results and
            interventions

        Notes
        -----
        See docs on :class:`~smif.data_layer.data_handle.DataHandle` for details of how to
        access inputs, parameters and state and how to set results.

        ``interval``
            should reference an id from the interval set corresponding to
            the output parameter, as specified in model configuration
        ``region``
            should reference a region name from the region set corresponding to
            the output parameter, as specified in model configuration

        To obtain simulation model data in this method,
        use the methods such as::

            parameter_value = data.get_parameter('my_parameter_name')

        Other useful methods are
        :meth:`~smif.data_layer.data_handle.DataHandle.get_base_timestep_data`,
        :meth:`~smif.data_layer.data_handle.DataHandle.get_previous_timestep_data`,
        :meth:`~smif.data_layer.data_handle.DataHandle.get_parameter`,
        :meth:`~smif.data_layer.data_handle.DataHandle.get_data`,
        :meth:`~smif.data_layer.data_handle.DataHandle.get_parameters` and
        :meth:`~smif.data_layer.data_handle.DataHandle.get_results`.

        :meth:`~smif.data_layer.data_handle.DataHandle.get_state` returns a list
        of intervention dict for the current timestep
        :meth:`~smif.data_layer.data_handle.DataHandle.get_current_interventions`
        returns a list of dict where each dict is an intervention

        """
