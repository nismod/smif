"""Tests parameter objects, passing and conversion into data dicts

- parameter objects
- that parameters are passed into the simulate method
"""
from pytest import fixture, raises
from smif.parameters import Parameter, ParameterList


@fixture
def config_list():
    return [
        {
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        },
        {
            'name': 'min_building_size',
            'description': 'Threshold for discarding polygons as not significant buildings',
            'absolute_range': (0, 1000000),
            'suggested_range': (0, 50),
            'default_value': 30,
            'units': 'm^2'
        }
    ]


@fixture
def parameters(config_list):
    return [
        Parameter.from_dict(config)
        for config in config_list
    ]


class TestParameter():
    """A `Parameter` holds metadata and provides creation/serialisation convenience methods
    """
    def test_default_creation(self):
        """should initialise with default values
        """
        parameter = Parameter()
        assert parameter.name == ''
        assert parameter.description == ''
        assert parameter.absolute_range == (0, 0)
        assert parameter.suggested_range == (0, 0)
        assert parameter.default_value == 0
        assert parameter.units == ''

    def test_creation(self):
        """should initialise with specified values
        """
        parameter = Parameter(
            name='test',
            description='Test description',
            absolute_range=(0, 1000),
            suggested_range=(0, 10),
            default_value=10,
            units='km'
        )
        assert parameter.name == 'test'
        assert parameter.description == 'Test description'
        assert parameter.absolute_range == (0, 1000)
        assert parameter.suggested_range == (0, 10)
        assert parameter.default_value == 10
        assert parameter.units == 'km'

    def test_dict_creation(self):
        """should initialise from dict
        """
        parameter = Parameter.from_dict({
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        })
        assert parameter.name == 'smart_meter_savings'
        assert parameter.description == 'The savings from smart meters'
        assert parameter.absolute_range == (0, 100)
        assert parameter.suggested_range == (3, 10)
        assert parameter.default_value == 3
        assert parameter.units == '%'

    def test_dict_creation_fail(self):
        """should fail if any field is missing from dict
        """
        parameter = Parameter.from_dict({
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        })
        assert parameter.name == 'smart_meter_savings'
        assert parameter.description == 'The savings from smart meters'
        assert parameter.absolute_range == (0, 100)
        assert parameter.suggested_range == (3, 10)
        assert parameter.default_value == 3
        assert parameter.units == '%'

    def test_as_dict(self, config_list, parameters):
        """should serialise to plain dict
        """
        for expected, param in zip(config_list, parameters):
            actual = param.as_dict()
            assert actual == expected


class TestParameterList():
    """A `ParameterList` collects a set of parameter metadata, with convenience
    methods for deriving and overriding default values.
    """
    def test_default_creation(self):
        """should initialise empty
        """
        parameter_list = ParameterList()
        assert parameter_list.defaults == {}
        assert parameter_list.overridden({'test': 'value'}) == {}
        assert parameter_list.as_list() == []

    def test_creation_with_list_of_parameter(self, parameters):
        """should initialise with a list of `Parameter`, access dict-like by
        name
        """
        parameter_list = ParameterList(parameters)
        for parameter in parameters:
            assert parameter_list[parameter.name] == parameter

    def test_creation_with_list_of_dict(self, config_list, parameters):
        """should initialise with a list of dict suitable for `Parameter`
        creation
        """
        parameter_list = ParameterList(config_list)
        for parameter in parameters:
            assert parameter_list[parameter.name] == parameter

    def test_add_parameter(self, parameters):
        """should add a single parameter at a time
        """
        parameter = parameters[0]
        parameter_list = ParameterList()
        parameter_list.add_parameter(parameter)
        assert parameter_list[parameter.name] == parameter

    def test_add_dict(self, config_list, parameters):
        """should add a single dict at a time
        """
        config = config_list[0]
        parameter = parameters[0]
        parameter_list = ParameterList()
        parameter_list.add_parameter(config)
        assert parameter_list[parameter.name] == parameter

    def test_get_names(self, parameters):
        """should derive list of names
        """
        parameter_list = ParameterList(parameters)
        assert parameter_list.names == [parameter.name for parameter in parameters]

    def test_defaults(self, parameters):
        """should derive default values as dict of parameter_name => default_value
        """
        parameter_list = ParameterList(parameters)
        expected = {
            parameter.name: parameter.default_value
            for parameter in parameters
        }

        assert parameter_list.defaults == expected

    def test_overridden(self, parameters):
        """should override with configured values, without affecting defaults
        """
        parameter_list = ParameterList(parameters)

        defaults = {
            'smart_meter_savings': 3,
            'min_building_size': 30
        }
        override = {
            'smart_meter_savings': 5
        }
        expected = {
            'smart_meter_savings': 5,
            'min_building_size': 30
        }
        actual = parameter_list.overridden(override)

        assert parameter_list.defaults == defaults
        assert actual == expected

    def test_add_duplicate_parameter(self, parameters):
        """should fail to add the same parameter twice
        """
        parameter = parameters[0]

        parameter_list = ParameterList()
        parameter_list.add_parameter(parameter)
        with raises(ValueError):
            parameter_list.add_parameter(parameter)

    def test_as_list(self, config_list, parameters):
        """should return list of dicts suitable for serialisation or
        ParameterList creation
        """
        parameter_list = ParameterList(parameters)
        assert parameter_list.as_list() == config_list
