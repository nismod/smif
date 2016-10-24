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

    @staticmethod
    @abstractmethod
    def getelement(element_type):
        pass


class InputFactory(ModelElement):
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
        self._bounds = []

    @staticmethod
    def getelement(input_type):
        """Implements the factory method to return subclasses of
        :class:`smif.InputFactory`

        Arguments
        =========
        input_type : str
            An input type name
        """
        if input_type == 'parameters':
            return ParameterList()
        elif input_type == 'decision_variables':
            return DecisionVariableList()
        else:
            raise ValueError("That input type is not defined")

    @property
    def bounds(self):
        """The bounds of the input
        """
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    def update_value(self, name, value):
        """Update the value of an input

        (Over rides :staticmethod:`ModelElement.update_value`)

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

    def _parse_input_dictionary(self, inputs, input_type, mapping):
        """Extracts an array of decision variables from a dictionary of inputs

        Arguments
        =========
        inputs : dict
            A dictionary of key: val pairs including a list of input types and
            names, followed by nested dictionaries of input attributes
        input_type : str
            A string input type
        mapping : dict
            A mapping for the expected keys `values`, `bounds` and `indices`
        Returns
        =======
        ordered_names : :class:`numpy.ndarray`
            The names of the decision variables in the order specified by the
            'index' key in the entries of the inputs
        bounds : :class:`numpy.ndarray`
            The bounds ordered by the index key
        values : :class:`numpy.ndarray`
            The initial values ordered by the index key

        """

        names = inputs[input_type]
        number_if_inputs = len(names)

        indices = [inputs[name][mapping['indices']] for name in names]
        assert len(indices) == number_if_inputs, \
            'Index entries do not match the number of {}'.format(input_type)
        values = np.zeros(number_if_inputs, dtype=np.float)
        bounds = np.zeros(number_if_inputs, dtype=(np.float, 2))
        ordered_names = np.zeros(number_if_inputs, dtype='U30')

        for name, index in zip(names, indices):
            values[index] = inputs[name][mapping['values']]
            bounds[index] = inputs[name][mapping['bounds']]
            ordered_names[index] = name

        self.names = ordered_names
        self.values = values
        self.bounds = bounds


class ParameterList(InputFactory):

    def get_inputs(self, inputs):
        mapping = {'values': 'value', 'bounds': 'bounds', 'indices': 'index'}
        self._parse_input_dictionary(inputs, 'parameters', mapping)


class DecisionVariableList(InputFactory):

    def get_inputs(self, inputs):
        mapping = {'values': 'init', 'bounds': 'bounds', 'indices': 'index'}
        self._parse_input_dictionary(inputs, 'decision variables', mapping)


class ModelInputs(object):
    """A container for all the model inputs

    """
    def __init__(self, inputs):

        self._inputs = InputFactory()
        self._decision_variables = \
            self._inputs.getelement('decision_variables')
        self._decision_variables.get_inputs(inputs)
        self._parameters = self._inputs.getelement('parameters')
        self._parameters.get_inputs(inputs)

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
