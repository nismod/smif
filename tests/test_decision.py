from pytest import fixture, raises
from smif.decision import DecisionManager, PreSpecified, RuleBased


@fixture(scope='function')
def plan():
    planned_interventions = [
        ('small_pumping_station_oxford', 2010),
        ('small_pumping_station_abingdon', 2015),
        ('large_pumping_station_oxford', 2020)
    ]

    return planned_interventions


@fixture(scope='function')
def get_strategies():
    strategies = [{'strategy': 'pre-specified-planning',
                   'description': 'build_nuclear',
                   'interventions': [('nuclear_large', 2012),
                                     ('carrington_retire', 2011)]
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

        expected = {0: timesteps}

        assert actual == expected

    def test_get_decision(self, plan):

        timesteps = [2010, 2015, 2020]
        dm = PreSpecified(timesteps, plan)

        actual = dm.get_decision(2010)
        expected = [('small_pumping_station_oxford', 2010)]
        assert actual == expected

        actual = dm.get_decision(2015)
        expected = [
            ('small_pumping_station_oxford', 2010),
            ('small_pumping_station_abingdon', 2015),
        ]
        assert actual == expected

        actual = dm.get_decision(2020)
        expected = [
            ('small_pumping_station_oxford', 2010),
            ('small_pumping_station_abingdon', 2015),
            ('large_pumping_station_oxford', 2020),
        ]
        assert actual == expected

    def test_get_decision_two(self, get_strategies):
        dm = PreSpecified([2010, 2015], get_strategies[0]['interventions'])
        actual = dm.get_decision(2010)
        expected = [
            ('carrington_retire', 2011),
            ('nuclear_large', 2012),
        ]
        # assert actual == expected
        # we don't mind the order
        assert sorted(actual) == sorted(expected)

        # actual = dm.get_decision(2015)
        # expected = [('carrington_retire', 2011)]
        # assert actual == expected

        actual = dm.get_decision(2015)
        expected = [
            ('carrington_retire', 2011),
            ('nuclear_large', 2012),
        ]
        assert sorted(actual) == sorted(expected)



    def test_buildable(self, get_strategies):
        dm = PreSpecified([2010, 2015], get_strategies[0]['interventions'])
        assert dm.timesteps == [2010, 2015]
        assert dm.buildable(2010, 2010) is True
        assert dm.buildable(2011, 2010) is True

    def test_buildable_raises(self, get_strategies):
        dm = PreSpecified([2010, 2015], get_strategies[0]['interventions'])
        with raises(ValueError):
            dm.buildable(2015, 2014)



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
        df = DecisionManager([2010, 2015], [], [])
        dm = df.decision_loop()
        bundle = next(dm)
        assert bundle == {0: [2010, 2015]}
        with raises(StopIteration):
            next(dm)

    def test_decision_manager_init(self, get_strategies):
        df = DecisionManager([2010, 2015], get_strategies, [])
        dm = df.decision_loop()
        bundle = next(dm)
        assert bundle == {0: [2010, 2015]}
        with raises(StopIteration):
            next(dm)
