"""Tests parameter objects, passing and conversion into data dicts

- parameter objects
- that parameters are passed into the simulate method
"""

from pytest import fixture, raises
from smif.parameters import ParameterList


@fixture
def get_config_list():

    config_list = [{'name': 'smart_meter_savings',
                    'description': 'The savings from smart meters',
                    'absolute_range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%',
                    'parent': 'energy_demand'}]

    return config_list


class TestInstantiateObjectsFromConfig():
    """
    """
    def test_parameter_list_instantiation(self,
                                          get_config_list):

        config_list = get_config_list

        parameters = ParameterList()
        parameters.add_parameters_from_list(config_list)

        assert parameters.names == ['smart_meter_savings']

        expected = {'name': 'smart_meter_savings',
                    'description': 'The savings from smart meters',
                    'range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%',
                    'parent': 'energy_demand'}

        assert parameters['smart_meter_savings'] == expected

    def test_parameter_single_instantiation(self,
                                            get_config_list):

        config_list = get_config_list

        config = config_list[0]

        parameters = ParameterList()
        parameters.add_parameter(config['name'],
                                 config['description'],
                                 config['range'],
                                 config['suggested_range'],
                                 config['default_value'],
                                 config['units'],
                                 config['parent'])

        assert parameters.names == ['smart_meter_savings']

        expected = {'name': 'smart_meter_savings',
                    'description': 'The savings from smart meters',
                    'range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%',
                    'parent': 'energy_demand'}

        assert parameters['smart_meter_savings'] == expected

    def test_add_duplicate_parameter(self, get_config_list):

        config_list = get_config_list

        config = config_list[0]

        parameters = ParameterList()
        parameters.add_parameter(config['name'],
                                 config['description'],
                                 config['range'],
                                 config['suggested_range'],
                                 config['default_value'],
                                 config['units'],
                                 config['parent'])
        with raises(ValueError):
            parameters.add_parameter(config['name'],
                                     config['description'],
                                     config['range'],
                                     config['suggested_range'],
                                     config['default_value'],
                                     config['units'],
                                     config['parent'])
