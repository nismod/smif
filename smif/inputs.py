from __future__ import absolute_import, division, print_function

import logging
import numpy as np

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class ModelInputs(object):
    """A container for all the model inputs

    """
    def __init__(self, inputs):
        self.input_dict = inputs

        (self._decision_variable_names,
         self._decision_variable_values,
         self._decision_variable_bounds) = self._get_decision_variables()

        (self._parameter_names,
         self._parameter_bounds,
         self._parameter_values) = self._get_parameter_values()

    @property
    def parameter_names(self):
        """A list of ordered parameter names
        """
        return self._parameter_names

    @property
    def parameter_bounds(self):
        """An array of tuples of parameter bounds
        """
        return self._parameter_bounds

    @property
    def parameter_values(self):
        """An array of parameter values
        """
        return self._parameter_values

    @property
    def decision_variable_names(self):
        """A list of decision variable names
        """
        return self._decision_variable_names

    @property
    def decision_variable_values(self):
        """An array of decision variable values
        """
        return self._decision_variable_values

    @property
    def decision_variable_indices(self):
        """A dictionary of index values associated with decision variable names

        Returns
        =======
        dict
            A dictionary of index values associated with decision variable
            names
        """
        return self._enumerate_names(self.decision_variable_names)

    @property
    def parameter_indices(self):
        """A dictionary of index values associated with parameter names

        Returns
        =======
        dict
            A dictionary of index values associated with parameter names
        """
        return self._enumerate_names(self.parameter_names)

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

    def _get_decision_variable_index(self, name):
        """A index values associated a decision variable name

        Argument
        ========
        name : str
            The name of the decision variable
        """
        assert name in self.decision_variable_names, \
            "That name is not in the list of decision variables"
        return self.decision_variable_indices[name]

    def _get_parameter_index(self, name):
        """A index values associated a parameter name

        Argument
        ========
        name : str
            The name of the parameter
        """
        assert name in self.parameter_names, \
            "That name is not in the list of parameters"
        return self.parameter_indices[name]

    def update_decision_variable_value(self, name, value):
        """Update the value of a decision variable

        Arguments
        =========
        name : str
            The name of the decision variable
        value : float
            The value to which to update the decision variable

        """
        index = self._get_decision_variable_index(name)
        logger.debug("Index of {} is {}".format(name, index))
        bounds = self.decision_variable_bounds
        assert bounds[index][0] <= value <= bounds[index][1], \
            "Decision variable bounds exceeded"
        self._decision_variable_values[index] = value

    def update_parameter_value(self, name, value):
        """Update the value of a decision variable

        Arguments
        =========
        name : str
            The name of the parameter
        value : float
            The value to which to update the parameter

        """
        index = self._get_parameter_index(name)
        logger.debug("Index of {} is {}".format(name, index))
        bounds = self.parameter_bounds
        assert bounds[index][0] <= value <= bounds[index][1], \
            "Parameter bounds exceeded"
        logger.debug("Updating {} with {}".format(name, value))
        self._parameter_values[index] = value

    @property
    def decision_variable_bounds(self):
        """An array of tuples of decision variable bounds
        """
        return self._decision_variable_bounds

    def _get_decision_variables(self):
        """Extracts an array of decision variables from a dictionary of inputs

        Returns
        =======
        ordered_names : :class:`numpy.ndarray`
            The names of the decision variables in the order specified by the
            'index' key in the entries of the inputs
        bounds : :class:`numpy.ndarray`
            The bounds ordered by the index key
        initial : :class:`numpy.ndarray`
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

        names = self.input_dict['decision variables']
        number_of_decision_variables = len(names)

        indices = [self.input_dict[name]['index'] for name in names]
        assert len(indices) == number_of_decision_variables, \
            'Index entries do not match the number of decision variables'
        initial = np.zeros(number_of_decision_variables, dtype=np.float)
        bounds = np.zeros(number_of_decision_variables, dtype=(np.float, 2))
        ordered_names = np.zeros(number_of_decision_variables, dtype='U30')

        for name, index in zip(names, indices):
            initial[index] = self.input_dict[name]['init']
            bounds[index] = self.input_dict[name]['bounds']
            ordered_names[index] = name

        return ordered_names, initial, bounds

    def _get_parameter_values(self):
        """Extracts an array of parameters from a dictionary of inputs

        Returns
        =======
        ordered_names : :class:`numpy.ndarray`
            The names of the parameters in the order specified by the
            'index' key in the entries of the inputs
        bounds : :class:`numpy.ndarray`
            The parameter bounds (or range) ordered by the index key
        values : :class:`numpy.ndarray`
            The parameter values ordered by the index key
        """
        names = self.input_dict['parameters']
        number_of_parameters = len(names)

        indices = [self.input_dict[name]['index'] for name in names]
        assert len(indices) == number_of_parameters, \
            'Index entries do not match the number of decision variables'
        values = np.zeros(number_of_parameters, dtype=np.float)
        bounds = np.zeros(number_of_parameters, dtype=(np.float, 2))
        ordered_names = np.zeros(number_of_parameters, dtype='U30')

        for name, index in zip(names, indices):
            values[index] = self.input_dict[name]['value']
            bounds[index] = self.input_dict[name]['bounds']
            ordered_names[index] = name

        return ordered_names, bounds, values
