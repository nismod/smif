from unittest.mock import Mock

from pytest import fixture
from smif.metadata import Spec
from smif.model.scenario_model import ScenarioModel


@fixture(scope='function')
def scenario_model():
    """Sample scenario model
    """
    scenario_model = ScenarioModel('population')
    scenario_model.scenario = 'High Population (ONS)'
    scenario_model.description = 'The High ONS Forecast for UK population out to 2050'
    scenario_model.add_output(
        Spec(
            name='population_count',
            dims=['LSOA'],
            coords={'LSOA': ['a', 'b', 'c']},
            dtype='int',
            unit='people'
        )
    )
    return scenario_model


@fixture(scope='function')
def scenario_model_dict():
    """Dict serialisation of sample scenario model
    """
    return {
        'name': 'population',
        'scenario': 'High Population (ONS)',
        'description': 'The High ONS Forecast for UK population out to 2050',
        'outputs': [
            {
                'name': 'population_count',
                'dims': ['LSOA'],
                'coords': {'LSOA': ['a', 'b', 'c']},
                'dtype': 'int',
                'unit': 'people',
                'abs_range': None,
                'exp_range': None,
                'default': None,
                'description': None
            }
        ]
    }


class TestScenarioModel(object):
    """ScenarioModel should represent data available from pre-computed datasets
    """
    def test_construct(self, scenario_model):
        assert scenario_model.name == 'population'
        assert scenario_model.scenario == 'High Population (ONS)'
        assert scenario_model.description == \
            'The High ONS Forecast for UK population out to 2050'
        assert scenario_model.outputs == {
            'population_count': Spec.from_dict({
                'name': 'population_count',
                'dims': ['LSOA'],
                'coords': {'LSOA': ['a', 'b', 'c']},
                'dtype': 'int',
                'unit': 'people'
            })
        }

    def test_from_dict(self, scenario_model, scenario_model_dict):
        actual = ScenarioModel.from_dict(scenario_model_dict)
        expected = scenario_model
        assert actual.name == expected.name
        assert actual.scenario == expected.scenario
        assert actual.description == expected.description
        assert actual.outputs == expected.outputs

    def test_from_dict_option_description(self, scenario_model_dict):
        del scenario_model_dict['description']
        scenario_model = ScenarioModel.from_dict(scenario_model_dict)
        assert scenario_model.description == ''

    def test_as_dict(self, scenario_model, scenario_model_dict):
        """Serialise metadata to dict
        """
        actual = scenario_model.as_dict()
        expected = scenario_model_dict
        assert actual == expected

    def test_as_dict_multi_output(self, scenario_model, scenario_model_dict):
        # with additional output
        scenario_model.add_output(
            Spec.from_dict({
                'name': 'population_density',
                'dims': ['LSOA'],
                'coords': {'LSOA': ['a', 'b', 'c']},
                'dtype': 'int',
                'unit': 'people / kilometer ** 2'
            })
        )
        scenario_model_dict['outputs'].append({
            'name': 'population_density',
            'dims': ['LSOA'],
            'coords': {'LSOA': ['a', 'b', 'c']},
            'dtype': 'int',
            'unit': 'people / kilometer ** 2',
            'abs_range': None,
            'exp_range': None,
            'default': None,
            'description': None
        })

        actual = scenario_model.as_dict()
        expected = scenario_model_dict

        # equivalent up to order of outputs
        actual['outputs'].sort(key=lambda p: p['name'])
        expected['outputs'].sort(key=lambda p: p['name'])

        assert actual == expected

    def test_scenario_data(self, scenario_model):
        """Scenario model simulate method should be a no-op
        """
        data_handle = Mock()
        actual = scenario_model.simulate(data_handle)
        assert actual is data_handle
        data_handle.assert_not_called()
