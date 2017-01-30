# -*- coding: utf-8 -*-
"""Encapsulates the inputs to a sector model

.. inheritance-diagram:: smif.inputs

Inputs are defined in a .yaml file.  There are three types of inputs,
decision variables, parameters and dependencies.

Decision variables and parameters have three attributes, name, bounds
and value.

Dependencies also have three attributes, name, spatial resolution
and temporal resolution.

An example yaml file::

    decision variables:
    - name: reservoir pumpiness
      bounds: [0, 100]
      value: 24.583
    - name: water treatment capacity
      bounds: [0, 20]
      value: 10
    dependencies:
    - name: electricity
      spatial_resolution: LOCAL
      temporal_resolution: HOURLY
    parameters:
    - name: raininess
      bounds: [0, 5]
      value: 3

"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple

import numpy as np

from smif.abstract import ModelElementCollection

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


class InputList(ModelElementCollection):
    """Defines the types of inputs to a sector model

    """
    def __init__(self):
        super().__init__()
        self.bounds = []
        self.logger = logging.getLogger(__name__)

    def __repr__(self):
        """Return literal string representation of this instance
        """
        return "{{'name': {}, 'value': {}, 'bounds': {}}}".format(
            self.names,
            self.values,
            self.bounds
        )

    def update_value(self, name, value):
        """Update the value of an input

        (Over rides `smif.inputs.ModelElement.update_value`)

        Arguments
        =========
        name : str
            The name of the decision variable
        value : float
            The value to which to update the decision variable

        """
        index = self._get_index(name)
        self.logger.debug("Index of {} is {}".format(name, index))
        bounds = self.bounds
        assert bounds[index][0] <= value <= bounds[index][1], \
            "Bounds exceeded"
        self.values[index] = value

    def _parse_input_dictionary(self, inputs):
        """Extracts arrays of decision variables and metadata from a list of
        inputs

        Arguments
        =========
        inputs : list
            A list of dicts which specify input attributes in key:val pairs

        Sets attributes
        ===============
        names : :class:`numpy.ndarray`
            The names of the decision variables in the order given in the
            inputs
        bounds : :class:`numpy.ndarray`
            The bounds in the same order
        values : :class:`numpy.ndarray`
            The initial values in the same order

        """

        number_of_inputs = len(inputs)

        self.values = np.zeros(number_of_inputs, dtype=np.float)
        self.bounds = np.zeros(number_of_inputs, dtype=(np.float, 2))
        self.names = np.zeros(number_of_inputs, dtype='U30')

        for index, input_data in enumerate(inputs):
            self.values[index] = input_data['value']
            self.bounds[index] = input_data['bounds']
            self.names[index] = input_data['name']

    def __getitem__(self, key):
        index = self._get_index(key)
        return {
            "name": self.names[index],
            "bounds": self.bounds[index],
            "value": self.values[index]
        }

    def __len__(self):
        return len(self.names)


class ParameterList(InputList):

    def __init__(self, parameters):
        super().__init__()
        self._parse_input_dictionary(parameters)


class DecisionVariableList(InputList):

    def __init__(self, decision_variables):
        super().__init__()
        self._parse_input_dictionary(decision_variables)



Dependency = namedtuple(
    "Dependency",
    [
        "name",
        "spatial_resolution",
        "temporal_resolution",
        "from_model"
    ]
)


class DependencyList(InputList):

    def __init__(self, dependencies):
        super().__init__()
        self._parse_input_dictionary(dependencies)


    def __repr__(self):
        """Return literal string representation of this instance
        """
        template = "{{'name': {}, 'spatial_resolution': {}, " + \
                   "'temporal_resolution': {}, 'from_model': {}}}"

        return template.format(
            self.names,
            self.spatial_resolutions,
            self.temporal_resolutions,
            self.from_models
        )

    def _parse_input_dictionary(self, inputs):
        """Extracts arrays of decision variables and metadata from a list of
        inputs

        Arguments
        =========
        inputs : list
            A list of dicts which specify input attributes in key:val pairs

        Sets attributes
        ===============
        ordered_names : :class:`numpy.ndarray`
            The names of the decision variables in the order given in the
            inputs

        """

        number_of_inputs = len(inputs)

        names = np.zeros(number_of_inputs, dtype='U30')
        spatial_resolutions = np.zeros(number_of_inputs, dtype='U30')
        temporal_resolutions = np.zeros(number_of_inputs, dtype='U30')
        from_models = np.zeros(number_of_inputs, dtype='U30')

        for index, input_data in enumerate(inputs):
            names[index] = input_data['name']
            spatial_resolutions[index] = input_data['spatial_resolution']
            temporal_resolutions[index] = input_data['temporal_resolution']
            from_models[index] = input_data['from_model']

        self.names = names
        self.spatial_resolutions = spatial_resolutions
        self.temporal_resolutions = temporal_resolutions
        self.from_models = from_models

    def __getitem__(self, key):
        """Implement __getitem__ to make this class iterable

        Example
        =======
            dependency_list = DependencyList()
            ...
            for dep in dependency_list:
                # do something with each dependency

        - uses Dependency (defined as a namedtuple) to wrap the data
        - lets the np.arrays raise TypeError or IndexError for incorrect
          or out-of-bounds accesses
        """
        dependency = Dependency(
            self.names[key],
            self.spatial_resolutions[key],
            self.temporal_resolutions[key],
            self.from_models[key]
        )
        return dependency


class ModelInputs(object):
    """A container for all the model inputs

    Arguments
    =========
    inputs : dict
        A dictionary of key: val pairs including a list of input types and
        names, followed by nested dictionaries of input attributes
    """
    def __init__(self, inputs):
        if 'decision variables' not in inputs:
            inputs['decision variables'] = []
        if 'parameters' not in inputs:
            inputs['parameters'] = []
        if 'dependencies' not in inputs:
            inputs['dependencies'] = []

        self._decision_variables = DecisionVariableList(
            inputs['decision variables'])
        self._parameters = ParameterList(inputs['parameters'])
        self._dependencies = DependencyList(inputs['dependencies'])


    @property
    def parameters(self):
        """A list of the model parameters

        Returns
        =======
        :class:`smif.inputs.ParameterList`
        """
        return self._parameters

    @property
    def decision_variables(self):
        """A list of the decision variables

        Returns
        =======
        :class:`smif.inputs.DecisionVariableList`
        """
        return self._decision_variables

    @property
    def dependencies(self):
        """A list of the model dependencies

        Returns
        =======
        :class:`smif.inputs.DependencyList`
        """
        return self._dependencies

    def __len__(self):
        return len(self.parameters) + len(self.decision_variables) + len(self.dependencies)
