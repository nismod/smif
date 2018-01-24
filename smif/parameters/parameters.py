"""Parameter and ParameterList
"""
import collections.abc
from collections import OrderedDict


class Parameter():
    """Data class to represent a single parameter's metadata

    Parameters
    ----------
    name : str, optional
    description : str, optional
    absolute_range : tuple, optional
    suggested_range : tuple, optional
    default_value : float or int, optional
    units : str, optional

    Attributes
    ----------
    name : str
    description : str
    absolute_range : tuple
    suggested_range : tuple
    default_value : float or int
    units : str

    """
    def __init__(self, name='', description='', absolute_range=(0, 0),
                 suggested_range=(0, 0), default_value=0, units=''):
        self.name = name
        self.description = description
        self.absolute_range = absolute_range
        self.suggested_range = suggested_range
        self.default_value = default_value
        self.units = units

    @classmethod
    def from_dict(cls, data):
        """Create a Parameter from a dictionary of values

        Parameters
        ----------
        data : dict
            Dictionary with values for each attribute, like::

                {
                    'name': 'smart_meter_savings',
                    'description': 'The savings from smart meters',
                    'absolute_range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%'
                }

        """
        parameter = cls()
        parameter.name = data['name']
        parameter.description = data['description']
        parameter.absolute_range = data['absolute_range']
        parameter.suggested_range = data['suggested_range']
        parameter.default_value = data['default_value']
        parameter.units = data['units']
        return parameter

    def as_dict(self):
        """Return a dict of parameter data
        """
        return {
            'name': self.name,
            'description': self.description,
            'absolute_range': self.absolute_range,
            'suggested_range': self.suggested_range,
            'default_value': self.default_value,
            'units': self.units
        }

    def __eq__(self, other):
        return self.name == other.name and \
            self.description == other.description and \
            self.absolute_range == other.absolute_range and \
            self.suggested_range == other.suggested_range and \
            self.default_value == other.default_value and \
            self.units == other.units


class ParameterList(collections.abc.Mapping):
    """Collection of parameters

    Parameters
    ----------
    parameters : list of `smif.parameter.Parameter`, optional

    """

    def __init__(self, parameters=None):
        self._data = OrderedDict()

        if parameters is not None:
            for parameter in parameters:
                self.add_parameter(parameter)

    def add_parameter(self, parameter):
        """Add a parameter

        Parameters
        ----------
        parameter : `smif.parameter.Parameter` or dict
        """
        if isinstance(parameter, dict):
            parameter = Parameter.from_dict(parameter)

        if parameter.name in self._data:
            raise ValueError("Parameter already defined")

        self._data[parameter.name] = parameter

    def as_list(self):
        """Return a list of dicts of parameter data
        """
        return [param.as_dict() for param in self._data.values()]

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        for key in self._data:
            yield key

    def __len__(self):
        return len(self._data)

    @property
    def names(self):
        """Returns the names of all the contained parameters
        """
        return list(self._data.keys())

    @property
    def defaults(self):
        """Default parameter values
        """
        return {
            parameter.name: parameter.default_value
            for parameter in self._data.values()
        }

    def overridden(self, new_values):
        """Override parameter values, falling back to defaults

        Parameters
        ----------
        new_values : dict
            Dict with keys matching parameter names, values to override the
            defaults
        """
        return {
            parameter_name: new_values.get(parameter_name, default_value)
            for parameter_name, default_value in self.defaults.items()
        }
