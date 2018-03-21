from pytest import fixture
from smif.decision import Planning, PreSpecified, RuleBased


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


class TestPreSpecified:

    def test_initialisation(self):

        timesteps = [2010, 2015, 2020]
        actual = PreSpecified(timesteps)

        assert actual.horizon == timesteps

    def test_generator(self):

        timesteps = [2010, 2015, 2020]
        dm = PreSpecified(timesteps)

        actual = next(dm)

        expected = {1: timesteps}

        assert actual == expected


class TestRuleBased:

    def test_initialisation(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps)
        assert dm.horizon == timesteps
        assert dm.satisfied is False
        assert dm.current_timestep_index == 0
        assert dm.current_iteration == 0

    def test_generator(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps)

        actual = next(dm)
        assert actual == {1: [2010]}

        dm.satisfied = True
        actual = next(dm)
        assert actual == {2: [2015]}
        assert dm.satisfied is False

        actual = next(dm)
        assert actual == {3: [2015]}
        assert dm.satisfied is False

        dm.satisfied = True
        dm.current_timestep_index = 2
        actual = next(dm)
        assert actual is None
