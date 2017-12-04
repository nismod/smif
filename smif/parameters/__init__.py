"""
"""
from collections import UserDict
from logging import getLogger


class ParameterList(UserDict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = getLogger(__name__)

    def as_list(self):
        return list(self.data.values())

    @property
    def parameters(self):
        return self.data

    @property
    def defaults(self):
        return {name: param['default_value']
                for name, param in self.data.items()}

    def add_parameters_from_list(self, config_list):

        for parameter in config_list:
            name = parameter['name']
            self.data[name] = parameter

    def add_parameter(self, name,
                      description,
                      absolute_range,
                      suggested_range,
                      default_value,
                      units,
                      parent):
        """Add a parameter to the parameter list

        Arguments
        ---------
        name : str
        description : str
        absolute_range : tuple
        suggested_range : tuple
        default_value : float
        units : str
        parent : `smif.model.Model`
        """

        if name in self.data:
            raise ValueError("Parameter already defined")

        self.data[name] = {'name': name,
                           'description': description,
                           'absolute_range': absolute_range,
                           'suggested_range': suggested_range,
                           'default_value': default_value,
                           'units': units,
                           'parent': parent}

        msg = "Added parameter '%s' to '%s'"
        self.logger.debug(msg, name, parent.name)

    @property
    def names(self):
        """Returns the names of all the contained parameters
        """
        return list(self.data.keys())

    def __getitem__(self, key):
        return self.parameters[key]
