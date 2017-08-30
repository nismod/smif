"""
"""
from collections import UserDict
from logging import getLogger


class ParameterList(UserDict):
    """A nested dict of parameters accessed by model name, parameter name
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = getLogger(__name__)

    @property
    def parameters(self):
        return self.data

    def add_parameters_from_list(self, config_list):

        for parameter in config_list:
            model_name = parameter['parent'].name
            param_name = parameter['name']
            if model_name in self.data:
                if param_name in self.data[model_name]:
                    raise ValueError("Duplicate parameter name")
                else:
                    self.data[model_name][param_name] = parameter
            else:
                self.data[model_name] = {param_name: parameter}

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

        parameter = {
                'name': name,
                'description': description,
                'absolute_range': absolute_range,
                'suggested_range': suggested_range,
                'default_value': default_value,
                'units': units,
                'parent': parent
                }

        if parent.name not in self.data:

            self.data[parent.name] = {name: parameter}
        elif name in self.data[parent.name]:
            raise ValueError("Parameter already defined")

        else:
            self.data[parent.name].update({name: parameter})

        msg = "Added parameter '%s' to '%s'"
        self.logger.debug(msg, name, parent.name)

    @property
    def names(self):
        """Returns the names of all the contained parameters
        """
        names = {}
        for model_name, parameters in self.data.items():
            names[model_name] = list(parameters.keys())

        return names

    def __getitem__(self, model_name):
        """

        Arguments
        ---------
        model_name : str
            The name of a model
        """
        return self.parameters[model_name]
