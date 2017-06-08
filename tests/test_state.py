from pytest import fixture
from smif.decision import Planning
from smif.intervention import Intervention, InterventionRegister
from smif.state import State


@fixture(scope='function')
def build_intervention_ws():
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
def build_register_two(build_intervention_ws):

    register = InterventionRegister()
    for entry in build_intervention_ws:
        register.register(Intervention(data=entry))
    return register


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

    def test_state_intialisation(self, plan, build_register_two):

        state = State(plan, build_register_two)

        state.get_initial_state(2035)
        assert state._state[2035] == {'large_pumping_station_abingdon'}

    def test_get_current_state(self, plan, build_register_two):

        state = State(plan, build_register_two)

        actual = state.get_current_state(2035)
        expected = {'large_pumping_station_abingdon'}
        assert actual == expected

        actual = state.get_current_state(2045)
        expected = {'small_pumping_station_oxford',
                    'large_pumping_station_abingdon'}
        assert actual == expected


class TestActionSpace:

    def test_initial_action_space(self, plan, build_register_two):

        state = State(plan, build_register_two)

        state.get_initial_action_space()
        assert state.action_space == {'small_pumping_station_bicester'}

    def test_update_action_space(self, plan, build_register_two):

        state = State(plan, build_register_two)

        state.get_initial_action_space()
        assert state.action_space == {'small_pumping_station_bicester'}

        state.build('small_pumping_station_bicester', 2045)

        state.update_action_space(2045)
        assert state.action_space == set()
