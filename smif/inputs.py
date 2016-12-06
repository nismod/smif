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
from abc import ABC
from collections import namedtuple

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
            The names of the decision variables in the order given in the
            inputs
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


<<<<<<< dd1fae3d39a8a58c0c05a1dabc086a28f1acbb75
=======
Dependency = namedtuple("Dependency", ["name", "spatial_resolution" ,"temporal_resolution", "from_model"])


>>>>>>> inputs and outputs as InputList or OutputList
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
        d = Dependency(
            self.names[key],
            self.spatial_resolutions[key],
            self.temporal_resolutions[key],
            self.from_models[key]
        )
        return d


class AssetList:
    """

- The set of assets (power stations etc.) should be explicitly declared
  in a yaml file.
- Assets are associated with sector models, not the integration configuration.
- Assets should be stored in a sub-folder associated with the sector model
  name.

    """

    def __init__(self, filepath):
        self._asset_list = ConfigParser(filepath)
        self._validate({
                        "type": "array",
                        "uniqueItems": True
                        })
        self._asset_attributes = None

    def _validate(self, schema):
        self._asset_list.validate(schema)

    @property
    def asset_list(self):
        return self._asset_list.data

    @property
    def asset_attributes(self):
        return self._asset_attributes.data

    def load_attributes(self, filepath):
        """
        """
        self._asset_attributes = ConfigParser(filepath)
        schema = {
                  "type": "array",
                  "oneof": self.asset_list,
                  "properties": {
                      "cost": {
                          "properties": {
                              "value": {"type": "number",
                                        "minimum": 0,
                                        "exclusiveMinimum": True},
                              "unit": {'type': 'string'}
                           }
                       }
                    }
                    }
        self._validate(schema)


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
