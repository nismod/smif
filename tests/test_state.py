from unittest.mock import MagicMock

import numpy as np
from pytest import fixture
from smif.decision import Planning
from smif.intervention import Asset, Intervention, InterventionRegister
from smif.state import State


@fixture(scope='function')
def build_intervention():
    data = [{
        'sector': 'water_supply',
        'name': 'small_pumping_station_oxford',
        'capacity': {
            'units': 'ML/day',
            'value': 450
            },
        'capital_cost': {
            'units': 'M£',
            'value': 500
        },
        'location': "POINT(51.1 -1.7)"
        },
        {
        'sector': 'water_supply',
        'name': 'small_pumping_station_bicester',
        'capacity': {
            'units': 'ML/day',
            'value': 450
            },
        'capital_cost': {
            'units': 'M£',
            'value': 500
        },
        'location': "POINT(51.1 -1.7)"
        },
        {
        'sector': 'water_supply',
        'name': 'large_pumping_station_abingdon',
        'capacity': {
            'units': 'ML/day',
            'value': 450
            },
        'capital_cost': {
            'units': 'M£',
            'value': 500
        },
        'location': "POINT(51.1 -1.7)"
        }]

    return data


@fixture(scope='function')
def build_register(build_intervention):

    register = InterventionRegister()
    for entry in build_intervention:
        register.register(Intervention(data=entry))
    return register


@fixture(scope='function')
def nonintervention_state():
    data = [{'name': 'reservoir_level',
             'time_period': 2010,
             'value': 500,
             'unit': 'Ml',
             'sector': 'water_supply'}]
    return data


@fixture(scope='function')
def plan():
    planned_interventions = [
        {
            'name': 'small_pumping_station_oxford',
            'build_date': 2045,
        },
        {
            'name': 'large_pumping_station_abingdon',
            'build_date': 2035,
        }
    ]

    return Planning(planned_interventions)


class TestState:

    def test_state_intialisation(self, plan, build_register):

        state = State(plan, build_register)

        state.get_initial_state(2035)
        assert state._state[2035] == {'large_pumping_station_abingdon'}

    def test_get_current_state(self, plan, build_register):

        state = State(plan, build_register)

        actual = state.get_current_state(2035)
        expected = {'large_pumping_station_abingdon'}
        assert actual == expected

        actual = state.get_current_state(2045)
        expected = {'small_pumping_station_oxford',
                    'large_pumping_station_abingdon'}
        assert actual == expected

    def test_get_state_conditional(self, plan, build_register):
        state = State(plan, build_register)
        _, actual = state.get_all_state(2035, 'water_supply')
        expected = [Asset("large_pumping_station_abingdon",
                          {"capacity": {"units": "ML/day", "value": 450},
                           "capital_cost": {"units": "M£", "value": 500},
                           "location": "POINT(51.1 -1.7)",
                           "name": "large_pumping_station_abingdon",
                           "sector": "water_supply",
                           "build_date": 2035})]
        assert actual == expected
        _, actual = state.get_all_state(2035, 'energy_supply')
        expected = []
        assert actual == expected


class TestActionSpace:

    def test_initial_action_space(self, plan, build_register):

        state = State(plan, build_register)

        state.get_initial_action_space()
        assert state.action_space == {'small_pumping_station_bicester'}

    def test_update_action_space(self, plan, build_register):

        state = State(plan, build_register)

        state.get_initial_action_space()
        assert state.action_space == {'small_pumping_station_bicester'}

        state.build('small_pumping_station_bicester', 2045)

        state.update_action_space(2045)
        assert state.action_space == set()


class TestDecisionVector:

    def test_interventions(self, plan, build_register):

        plan = MagicMock(names={1})

        interventions = MagicMock(names={1, 2, 3})

        state = State(plan, interventions)
        assert state._interventions.names == {1, 2, 3}
        assert state._planned.names == {1}

        assert state.action_space == {2, 3}

        actual = state.get_initial_action_space()

        assert actual == {2, 3}

        assert state.action_list == [2, 3]

    def test_generate_vector_from_action_space(self, plan, build_register):

        state = State(plan, build_register)

        assert state.action_space == {'small_pumping_station_bicester'}
        actual = state.get_action_dimension()
        expected = 1

        assert actual == expected

    def test_parse_empty_decision_vector(self, plan, build_register):

        state = State(plan, build_register)

        vector = np.array([0])
        actual = state.parse_decisions(vector)

        expected = []

        assert actual == expected

    def test_parse_small_decision_vector(self, plan, build_register):

        state = State(plan, build_register)

        vector = np.array([1])
        actual = state.parse_decisions(vector)

        expected = ['small_pumping_station_bicester']

        for x, y in zip(actual, expected):
            assert x.name == y


class TestNonInterventionState:

    def test_initialise_state(self, plan, build_register,
                              nonintervention_state):

        state = State(plan, build_register)

        state.set_initial_data(nonintervention_state)

        actual = state.get_all_state(2010, 'water_supply')
        expected = nonintervention_state

        assert actual[0] == expected

        expected = []
        assert actual[1] == expected

        actual = state.get_all_state(2035, 'water_supply')
        expected = []

        assert actual[0] == expected

        expected = Asset("large_pumping_station_abingdon",
                         {"capacity": {"units": "ML/day", "value": 450},
                          "capital_cost": {"units": "M£", "value": 500},
                          "location": "POINT(51.1 -1.7)",
                          "name": "large_pumping_station_abingdon",
                          "sector": "water_supply",
                          "build_date": 2035})
        a = actual[1][0]

        for key in a.data.keys():
            assert a.data[key] == expected.data[key]

        for key in expected.data.keys():
            assert a.data[key] == expected.data[key]

        assert actual[1][0] == expected

    def test_set_state(self, plan, build_register,
                       nonintervention_state):

        state = State(plan, build_register)
        state.set_initial_data(nonintervention_state)

        state_data = [{'name': 'reservoir_level',
                       'time_period': 2015,
                       'value': 500,
                       'unit': 'Ml',
                       'sector': 'water_supply'}]

        state.set_state('water_supply', 2015, state_data)
        actual, _ = state.get_all_state(2015, 'water_supply')
        expected = [{'name': 'reservoir_level',
                     'time_period': 2015,
                     'value': 500,
                     'unit': 'Ml',
                     'sector': 'water_supply'}]
        for x, y in zip(actual, expected):
            assert sorted(x) == sorted(y)
