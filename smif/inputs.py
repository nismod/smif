# -*- coding: utf-8 -*-
"""Encapsulates the inputs to a sector model

Dependencies are defined in a file called ``inputs.yaml``.

Dependencies have several attributes: ``name``, ``spatial_resolution``,
``temporal_resolution`` and ``from_model``.

The ``name`` entry denotes the unique identifier of a model or scenario output.
The ``spatial_resolution`` and ``temporal_resolution`` are references to the
catalogue held by the :class:`~smif.sector_model.SosModel` which define the
available conversion formats.

An example yaml file::

        dependencies:
        - name: eletricity_price
          spatial_resolution: GB
          temporal_resolution: annual
          from_model: energy_supply
        - name: petrol_price
          spatial_resolution: GB
          temporal_resolution: annual
          from_model: scenario
        - name: diesel_price
          spatial_resolution: GB
          temporal_resolution: annual
          from_model: energy_supply
        - name: LPG_price
          spatial_resolution: GB
          temporal_resolution: annual
          from_model: energy_supply

"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple

import numpy as np

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell, University of Oxford 2017"
__license__ = "mit"

class ModelElementCollection(object):
    """A collection of model elements

    ModelInputs and ModelOutputs both derive from this class
    """

    def __init__(self):
        self._names = []
        self._values = []
        self.logger = logging.getLogger(__name__)

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
        if name not in self.names:
            raise IndexError("That name is not in the list of input names")
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
        self.logger.debug("Updating {} with {}".format(name, value))
        self.values[index] = value


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

    def __getitem__(self, key):
        index = self._get_index(key)
        return {
            "name": self.names[index],
            "bounds": self.bounds[index],
            "value": self.values[index]
        }

    def __len__(self):
        return len(self.names)


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
        >>> dependency_list = DependencyList()
        >>> for dep in dependency_list:
        >>>     # do something with each dependency

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
        if 'dependencies' not in inputs:
            inputs['dependencies'] = []

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
        return len(self.parameters) + len(self.decision_variables) + \
            len(self.dependencies)
