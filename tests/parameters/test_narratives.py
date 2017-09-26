"""Tests for the narratives

Narratives hold collections of overridden parameter data. During model setup,
a user compiles a narrative file which contains a list of parameter names and
values
"""
from pytest import fixture, raises
from smif.parameters.narrative import NarrativeData


@fixture
def get_narrative():

    narrative = NarrativeData('Energy Demand - High Tech',
                              'A description',
                              'energy_demand_high_tech.yml',
                              'technology')
    return narrative


class TestNarrativeData:

    def test_narrative_data_initialise(self):

        narrative = NarrativeData('Energy Demand - High Tech',
                                  'A description',
                                  'energy_demand_high_tech.yml',
                                  'technology')

        actual = narrative.as_dict()
        expected = {'name': 'Energy Demand - High Tech',
                    'description': 'A description',
                    'filename': 'energy_demand_high_tech.yml',
                    'narrative_set': 'technology'}
        assert actual == expected

    def test_load_data(self, get_narrative):

        narrative = get_narrative
        narrative_data = {'global': [{'global_parameter': 'value'}],
                          'model_name': [{'model_parameter': 'value'},
                                         {'model_parameter_two': 'value'}
                                         ]
                          }
        narrative.add_data(narrative_data)
        actual = narrative.data

        expected = {'global': [{'global_parameter': 'value'}],
                    'model_name': [{'model_parameter': 'value'},
                                   {'model_parameter_two': 'value'}
                                   ]
                    }
        assert actual == expected

    def test_load_wrong_type(self, get_narrative):
        narrative = get_narrative
        with raises(TypeError):
            narrative.add_data(list(['should', 'b', 'a', 'dict']))
