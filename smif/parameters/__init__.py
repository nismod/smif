"""
"""


class ParameterList(object):

    def __init__(self):

        self.parameters = {}

    def add_parameters_from_list(self, config_list):

        for parameter in config_list:
            name = parameter['name']
            self.parameters[name] = parameter

    def add_parameter(self, name,
                      description,
                      absolute_range,
                      suggested_range,
                      default_value,
                      units,
                      parent):

        if name in self.parameters:
            raise ValueError("Parameter already defined")

        self.parameters[name] = {'name': name,
                                 'description': description,
                                 'range': absolute_range,
                                 'suggested_range': suggested_range,
                                 'default_value': default_value,
                                 'units': units,
                                 'parent': parent}

    @property
    def names(self):
        return list(self.parameters.keys())

    def __getitem__(self, key):

        return self.parameters[key]
