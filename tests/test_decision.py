from pytest import fixture, raises
from smif.decision import DecisionManager, PreSpecified, RuleBased


@fixture(scope='function')
def plan():
    planned_interventions = [
        {
         'name': 'small_pumping_station_oxford',
         'build_year': 2010
        },
        {
         'name': 'small_pumping_station_abingdon',
         'build_year': 2015

        },
        {
         'name': 'large_pumping_station_oxford',
         'build_year': 2020
        }
    ]

    return planned_interventions


@fixture(scope='function')
def get_strategies():
    strategies = [{'strategy': 'pre-specified-planning',
                   'description': 'build_nuclear',
                   'model_name': 'energy_supply',
                   'interventions': [{'name': 'nuclear_large', 'build_year': 2012},
                                     {'name': 'carrington_retire', 'build_year': 2011}]
                   }]

    return strategies


class TestPreSpecified:

    def test_initialisation(self, plan):

        timesteps = [2010, 2015, 2020]
        actual = PreSpecified(timesteps, plan)

        assert actual.timesteps == timesteps

    def test_generator(self, plan):

        timesteps = [2010, 2015, 2020]
        dm = PreSpecified(timesteps, plan)

        actual = next(dm)

        expected = {1: timesteps}

        assert actual == expected

    def test_get_state(self, plan):

        timesteps = [2010, 2015, 2020]
        dm = PreSpecified(timesteps, plan)

        actual = dm.get_state(2010)
        expected = {2010: ['small_pumping_station_oxford']}
        assert actual == expected

        actual = dm.get_state(2015)
        expected = {2010: ['small_pumping_station_oxford'],
                    2015: ['small_pumping_station_abingdon']}
        assert actual == expected

        actual = dm.get_state(2020)
        expected = {2010: ['small_pumping_station_oxford'],
                    2015: ['small_pumping_station_abingdon'],
                    2020: ['large_pumping_station_oxford']}
        assert actual == expected


class TestRuleBased:

    def test_initialisation(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps)
        assert dm.timesteps == timesteps
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


class TestDecisionManager():

    def test_null_strategy(self):
        strategy = []
        df = DecisionManager([2010, 2015], strategy)
        dm = df.decision_loop()
        bundle = next(dm)
        assert bundle == {0: [2010, 2015]}
        with raises(StopIteration):
            next(dm)

    def test_decision_manager_init(self, get_strategies):
        df = DecisionManager([2010, 2015], get_strategies)
        dm = df.decision_loop()
        bundle = next(dm)
        assert bundle == {0: [2010, 2015]}
        with raises(StopIteration):
            next(dm)

    def test_buildable(self, get_strategies):
        dm = PreSpecified([2010, 2015], get_strategies[0]['interventions'])
        assert dm.timesteps == [2010, 2015]
        assert dm.buildable(2010, 2010) is True
        assert dm.buildable(2011, 2010) is True

    def test_buildable_raises(self, get_strategies):
        dm = PreSpecified([2010, 2015], get_strategies[0]['interventions'])
        with raises(ValueError):
            dm.buildable(2015, 2014)

    def test_get_state(self, get_strategies):
        dm = PreSpecified([2010, 2015], get_strategies[0]['interventions'])
        actual = dm.get_state(2010)
        expected = {2011: ['carrington_retire'],
                    2012: ['nuclear_large']}
        assert actual == expected

        # actual = dm.get_state(2015)
        # expected = {2011: ['carrington_retire']}
        # assert actual == expected

        actual = dm.get_state(2015)
        expected = {2011: ['carrington_retire'],
                    2012: ['nuclear_large']}
        assert actual == expected
