from pytest import fixture
from smif.decision import Planning


@fixture(scope='function')
def plan():
    planned_interventions = [
        {
            'name': 'small_pumping_station_oxford',
            'build_date': 2045,
        },
        {
            'name': 'small_pumping_station_bicester',
            'build_date': 2035,
        },
        {
            'name': 'large_pumping_station_abingdon',
            'build_date': 2035,
        }
    ]

    return Planning(planned_interventions)


class TestPlanning:

    def test_intervention_names(self, plan):
        expected = {'small_pumping_station_oxford',
                    'small_pumping_station_bicester',
                    'large_pumping_station_abingdon'}
        actual = plan.names
        assert actual == expected

    def test_timeperiods(self, plan):
        expected = {2045, 2035}
        actual = plan.timeperiods
        assert actual == expected

    def test_empty(self):
        plan = Planning()
        assert plan.planned_interventions == []

    def test_return_current_interventions(self, plan):
        plan = plan
        actual = plan.current_interventions(2035)
        expected = {'small_pumping_station_bicester',
                    'large_pumping_station_abingdon'}
        assert actual == expected

    def test_return_current_interventions_two(self, plan):
        plan = plan
        actual = plan.current_interventions(2045)
        expected = {'small_pumping_station_oxford',
                    'small_pumping_station_bicester',
                    'large_pumping_station_abingdon'}
        assert actual == expected
