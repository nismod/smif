"""Encapsulates the inputs to a sector model


.. inheritance-diagram:: smif.inputs

"""
from __future__ import absolute_import, division, print_function

import logging
from abc import ABC, abstractmethod

import numpy as np

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class ModelElement(ABC):

    def __init__(self):
        self._names = []
        self._values = []

    @property
    def names(self):
        """A descriptive name of the input
        """
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    @property
    def values(self):
        """The value of the input
        """
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    def _get_index(self, name):
        """A index values associated an element name

        Argument
        ========
        name : str
            The name of the decision variable
        """
        assert name in self.names, \
            "That name is not in the list of input names"
        return self.indices[name]

    @property
    def indices(self):
        """A dictionary of index values associated with decision variable names

        Returns
        =======
        dict
            A dictionary of index values associated with decision variable
            names
        """
        return self._enumerate_names(self.names)

    def _enumerate_names(self, names):
        """

        Arguments
        =========
        names : iterable
            A list of names

        Returns
        =======
        dict
            Key: value pairs to lookup the index of a name
        """
        return {name: index for (index, name) in enumerate(names)}

    def update_value(self, name, value):
        """Update the value of an input

        Arguments
        =========
        name : str
            The name of the decision variable
        value : float
            The value to which to update the decision variable

        """
        index = self._get_index(name)
        logger.debug("Updating {} with {}".format(name, value))
        self.values[index] = value



class InputList(ModelElement):
    """Defines the types of inputs to a sector model

    The input data are expected to be defined using the following format::

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
    def __init__(self):
        super().__init__()
        self.bounds = []

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
        logger.debug("Index of {} is {}".format(name, index))
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
            The names of the decision variables in the order given in the inputs
        bounds : :class:`numpy.ndarray`
            The bounds in the same order
        values : :class:`numpy.ndarray`
            The initial values in the same order

        """

        number_of_inputs = len(inputs)

        values = np.zeros(number_of_inputs, dtype=np.float)
        bounds = np.zeros(number_of_inputs, dtype=(np.float, 2))
        names = np.zeros(number_of_inputs, dtype='U30')

        for index, input_data in enumerate(inputs):
            values[index] = input_data['value']
            bounds[index] = input_data['bounds']
            names[index] = input_data['name']

        self.names = names
        self.values = values
        self.bounds = bounds


class ParameterList(InputList):

    def __init__(self, parameters):
        super().__init__()
        self._parse_input_dictionary(parameters)


class DecisionVariableList(InputList):

    def __init__(self, decision_variables):
        super().__init__()
        self._parse_input_dictionary(decision_variables)

class DependencyList(InputList):

    def __init__(self, dependencies):
        super().__init__()
        self._parse_input_dictionary(dependencies)

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
            The names of the decision variables in the order given in the inputs

        """

        number_of_inputs = len(inputs)

        names = np.zeros(number_of_inputs, dtype='U30')
        spatial_resolutions = np.zeros(number_of_inputs, dtype='U30')
        temporal_resolutions = np.zeros(number_of_inputs, dtype='U30')

        for index, input_data in enumerate(inputs):
            names[index] = input_data['name']
            spatial_resolutions[index] = input_data['spatial_resolution']
            temporal_resolutions[index] = input_data['temporal_resolution']

        self.names = names
        self.spatial_resolutions = spatial_resolutions
        self.temporal_resolutions = temporal_resolutions


class ModelInputs(object):
    """A container for all the model inputs

    Arguments
    =========
    inputs : dict
        A dictionary of key: val pairs including a list of input types and
        names, followed by nested dictionaries of input attributes
    """
    def __init__(self, inputs):
        self._inputs = InputList()

        self._decision_variables = DecisionVariableList(inputs['decision variables'])
        self._parameters = ParameterList(inputs['parameters'])

        if 'dependencies' in inputs:
            self._dependencies = DependencyList(inputs['dependencies'])
        else:
            self._dependencies = DependencyList([])

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
