"""Tests for the narratives

Narrative hold collections of overridden parameter data. During model setup,
a user compiles a number of narrative files which contains a list of parameter
names and values. These are assigned to a narrative set during a model run
and the NarrativeSet object holds this information at runtime.
"""
from pytest import raises
from smif.parameters import Narrative


class TestNarrative:

    def test_narrative_data_initialise(self):

        narrative = Narrative(
            'Energy Demand - High Tech',
            'A description',
            'technology'
        )

        actual = narrative.as_dict()
        expected = {
            'name': 'Energy Demand - High Tech',
            'description': 'A description',
            'narrative_set': 'technology'
        }
        assert actual == expected

    def test_load_data(self, get_narrative_obj):
        narrative = get_narrative_obj
        narrative_data = {
            'global': [
                {'global_parameter': 'value'}
            ],
            'model_name': [
                {'model_parameter': 'value'},
                {'model_parameter_two': 'value'}
            ]
        }
        expected = narrative_data.copy()
        narrative.data = narrative_data
        actual = narrative.data

        assert actual == expected

    def test_load_wrong_type(self, get_narrative_obj):
        narrative = get_narrative_obj
        with raises(TypeError):
            narrative.data = list(['should', 'b', 'a', 'dict'])
