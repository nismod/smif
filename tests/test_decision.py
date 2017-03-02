from pytest import fixture
from smif.decision import Planning


@fixture(scope='function')
def plan():
    planned_interventions = [
        {
            'name': 'small_pumping_station',
            'build_date': 2045,
            'location': 'oxford',
            'capacity': {
                'value': 500,
                'units': 'mcm/day'
            }
        },
        {
            'name': 'small_pumping_station',
            'build_date': 2035,
            'location': 'bicester',
            'capacity': {
                'value': 500,
                'units': 'mcm/day'
            }
        },
        {
            'name': 'large_pumping_station',
            'build_date': 2035,
            'location': 'abingdon',
            'capacity': {
                'value': 1500,
                'units': 'mcm/day'
            }
        }
    ]

    return Planning(planned_interventions)


class TestPlanning:

    def test_intervention_names(self, plan):
        expected = {'small_pumping_station', 'large_pumping_station'}
        actual = plan.names
        assert actual == expected

    def test_timeperiods(self, plan):
        expected = {2045, 2035}
        actual = plan.timeperiods
        assert actual == expected

    def test_empty(self):
        plan = Planning()
        assert plan.planned_interventions == []
