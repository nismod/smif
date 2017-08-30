"""Tests parameter objects, passing and conversion into data dicts

- parameter objects
- that parameters are passed into the simulate method
"""

from unittest.mock import Mock

from pytest import fixture, raises
from smif.parameters import ParameterList


@fixture
def get_config_list():

    mock_parent = Mock()
    mock_parent.name = 'mock_parent'
    config_list = [{'name': 'smart_meter_savings',
                    'description': 'The savings from smart meters',
                    'absolute_range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%',
                    'parent': mock_parent}]

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
                    'absolute_range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%',
                    'parent': config_list[0]['parent']}

        assert parameters['smart_meter_savings'] == expected

    def test_parameter_single_instantiation(self,
                                            get_config_list):

        config_list = get_config_list

        config = config_list[0]

        parameters = ParameterList()
        parameters.add_parameter(config['name'],
                                 config['description'],
                                 config['absolute_range'],
                                 config['suggested_range'],
                                 config['default_value'],
                                 config['units'],
                                 config['parent'])

        assert parameters.names == ['smart_meter_savings']

        expected = {'name': 'smart_meter_savings',
                    'description': 'The savings from smart meters',
                    'absolute_range': (0, 100),
                    'suggested_range': (3, 10),
                    'default_value': 3,
                    'units': '%',
                    'parent': config_list[0]['parent']}

        assert parameters['smart_meter_savings'] == expected

    def test_defaults(self, get_config_list):
        config_list = get_config_list
        parameters = ParameterList()
        parameters.add_parameters_from_list(config_list)

        expected = {'smart_meter_savings': 3}

        assert parameters.defaults == expected

    def test_add_duplicate_parameter(self, get_config_list):

        config_list = get_config_list

        config = config_list[0]

        parameters = ParameterList()
        parameters.add_parameter(config['name'],
                                 config['description'],
                                 config['absolute_range'],
                                 config['suggested_range'],
                                 config['default_value'],
                                 config['units'],
                                 config['parent'])
        with raises(ValueError):
            parameters.add_parameter(config['name'],
                                     config['description'],
                                     config['absolute_range'],
                                     config['suggested_range'],
                                     config['default_value'],
                                     config['units'],
                                     config['parent'])
