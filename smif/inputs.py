from __future__ import absolute_import, division, print_function

import logging
import numpy as np

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class Input(object):
    """An input is a sector model input exposed to the :class:`Interface`
    """

    inputs = []

    def __init__(self, name, value, bounds):
        self._name = name
        self._value = value
        self._bounds = bounds

    @property
    def name(self):
        """A descriptive name of the input
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def value(self):
        """The value of the property
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def bounds(self):
        """The bounds of the property
        """
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value


class InputFactory(object):

    def __init__(self):
        self._names = []
        self._values = []
        self._bounds = []

    @staticmethod
    def getinput(input_type):
        if input_type == 'parameters':
            return ParameterList()
        elif input_type == 'decision_variables':
            return DecisionVariableList()
        else:
            raise ValueError("That input type is not defined")

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
        """The value of the property
        """
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    @property
    def bounds(self):
        """The bounds of the property
        """
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    def _get_index(self, name):
        """A index values associated a decision variable name

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

    def _parse_input_dictionary(self, inputs, input_type, mapping):
        """Extracts an array of decision variables from a dictionary of inputs

        Returns
        =======
        ordered_names : :class:`numpy.ndarray`
            The names of the decision variables in the order specified by the
            'index' key in the entries of the inputs
        bounds : :class:`numpy.ndarray`
            The bounds ordered by the index key
        values : :class:`numpy.ndarray`
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

        names = inputs[input_type]
        number_of_decision_variables = len(names)

        indices = [inputs[name]['index'] for name in names]
        assert len(indices) == number_of_decision_variables, \
            'Index entries do not match the number of {}'.format(input_type)
        values = np.zeros(number_of_decision_variables, dtype=np.float)
        bounds = np.zeros(number_of_decision_variables, dtype=(np.float, 2))
        ordered_names = np.zeros(number_of_decision_variables, dtype='U30')

        for name, index in zip(names, indices):
            values[index] = inputs[name][mapping['values']]
            bounds[index] = inputs[name][mapping['bounds']]
            ordered_names[index] = name

        self.names = ordered_names
        self.values = values
        self.bounds = bounds


class ParameterList(InputFactory):

    def get_inputs(self, inputs):
        mapping = {'values': 'value', 'bounds': 'bounds'}
        self._parse_input_dictionary(inputs, 'parameters', mapping)


class DecisionVariableList(InputFactory):

    def get_inputs(self, inputs):
        mapping = {'values': 'init', 'bounds': 'bounds'}
        self._parse_input_dictionary(inputs, 'decision variables', mapping)


class ModelInputs(object):
    """A container for all the model inputs

    """
    def __init__(self, inputs):

        self._inputs = InputFactory()
        self._decision_variables = self._inputs.getinput('decision_variables')
        self._decision_variables.get_inputs(inputs)
        self._parameters = self._inputs.getinput('parameters')
        self._parameters.get_inputs(inputs)

    @property
    def parameters(self):
        return self._parameters

    @property
    def decision_variables(self):
        return self._decision_variables

    def update_decision_variable_value(self, name, value):
        """Update the value of a decision variable

        Arguments
        =========
        name : str
            The name of the decision variable
        value : float
            The value to which to update the decision variable

        """
        index = self._decision_variables._get_index(name)
        logger.debug("Index of {} is {}".format(name, index))
        bounds = self.decision_variables.bounds
        assert bounds[index][0] <= value <= bounds[index][1], \
            "Decision variable bounds exceeded"
        self._decision_variables.values[index] = value

    def update_parameter_value(self, name, value):
        """Update the value of a decision variable

        Arguments
        =========
        name : str
            The name of the parameter
        value : float
            The value to which to update the parameter

        """
        index = self._parameters._get_index(name)
        logger.debug("Index of {} is {}".format(name, index))
        bounds = self._parameters.bounds
        assert bounds[index][0] <= value <= bounds[index][1], \
            "Parameter bounds exceeded"
        logger.debug("Updating {} with {}".format(name, value))
        self._parameters.values[index] = value

    @property
    def decision_variable_bounds(self):
        """An array of tuples of decision variable bounds
        """
        return self._decision_variables.bounds
