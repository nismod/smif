from unittest.mock import Mock, PropertyMock

from pytest import fixture, raises
from smif.data_layer.memory_interface import MemoryInterface
from smif.decision.decision import DecisionManager, PreSpecified, RuleBased


@fixture(scope='function')
def plan():
    planned_interventions = [
        {'name': 'small_pumping_station_oxford', 'build_year': 2010},
        {'name': 'small_pumping_station_abingdon', 'build_year': 2015},
        {'name': 'large_pumping_station_oxford', 'build_year': 2020}
    ]

    return planned_interventions


@fixture(scope='function')
def get_strategies():
    strategies = [{'strategy': 'pre-specified-planning',
                   'description': 'build_nuclear',
                   'interventions': [
                       {'name': 'nuclear_large', 'build_year': 2012},
                       {'name': 'carrington_retire', 'build_year': 2011}]
                   }]

    return strategies


@fixture(scope='function')
def get_register():
    lifetime = {'technical_lifetime': {'value': 99}}
    register = {'nuclear_large': lifetime,
                'carrington_retire': lifetime,
                'small_pumping_station_oxford': lifetime,
                'small_pumping_station_abingdon': lifetime,
                'large_pumping_station_oxford': lifetime
                }
    return register


class TestPreSpecified:

    def test_initialisation(self, plan):

        timesteps = [2010, 2015, 2020]

        actual = PreSpecified(timesteps, Mock(), plan)

        assert actual.timesteps == timesteps

    def test_generator(self, plan):

        timesteps = [2010, 2015, 2020]
        dm = PreSpecified(timesteps, Mock(), plan)

        actual = next(dm)

        expected = {0: timesteps}

        assert actual == expected

    def test_get_decision(self, plan, get_register):

        register = get_register

        timesteps = [2010, 2015, 2020]
        dm = PreSpecified(timesteps, register, plan)

        mock_handle = Mock()
        type(mock_handle).current_timestep = PropertyMock(return_value=2010)
        actual = dm.get_decision(mock_handle)
        expected = [
            {'name': 'small_pumping_station_oxford',
             'build_year': 2010}]
        assert actual == expected

        type(mock_handle).current_timestep = PropertyMock(return_value=2015)
        actual = dm.get_decision(mock_handle)
        expected = [
            {'name': 'small_pumping_station_oxford',
             'build_year': 2010},
            {'name': 'small_pumping_station_abingdon',
             'build_year': 2015}]
        assert actual == expected

        type(mock_handle).current_timestep = PropertyMock(return_value=2020)
        actual = dm.get_decision(mock_handle)
        expected = [
            {'name': 'small_pumping_station_oxford',
             'build_year': 2010},
            {'name': 'small_pumping_station_abingdon',
             'build_year': 2015},
            {'name': 'large_pumping_station_oxford',
             'build_year': 2020}
        ]
        assert actual == expected

    def test_get_decision_two(self, get_strategies, get_register):
        register = get_register
        dm = PreSpecified([2010, 2015], register, get_strategies[0]['interventions'])

        mock_handle = Mock()
        type(mock_handle).current_timestep = PropertyMock(return_value=2010)
        actual = dm.get_decision(mock_handle)
        expected = [
            {'name': 'nuclear_large', 'build_year': 2012},
            {'name': 'carrington_retire', 'build_year': 2011}
        ]
        # assert actual == expected
        # we don't mind the order
        assert (actual) == (expected)

        # actual = dm.get_decision(2015)
        # expected = [('carrington_retire', 2011)]
        # assert actual == expected

        type(mock_handle).current_timestep = PropertyMock(return_value=2015)
        actual = dm.get_decision(mock_handle)
        expected = [
            {'name': 'nuclear_large', 'build_year': 2012},
            {'name': 'carrington_retire', 'build_year': 2011}
        ]
        assert (actual) == (expected)

    def test_buildable(self, get_strategies):
        dm = PreSpecified([2010, 2015], Mock(), get_strategies[0]['interventions'])
        assert dm.timesteps == [2010, 2015]
        assert dm.buildable(2010, 2010) is True
        assert dm.buildable(2011, 2010) is True

    def test_historical_intervention_buildable(self, get_strategies):
        dm = PreSpecified([2020, 2030], Mock(), get_strategies[0]['interventions'])
        assert dm.timesteps == [2020, 2030]
        assert dm.buildable(1980, 2020) is True
        assert dm.buildable(1990, 2020) is True

    def test_buildable_raises(self, get_strategies):
        dm = PreSpecified([2010, 2015], Mock(), get_strategies[0]['interventions'])
        with raises(ValueError):
            dm.buildable(2015, 2014)

    def test_within_lifetime(self):
        dm = PreSpecified([2010, 2015], Mock(), [])
        assert dm.within_lifetime(2010, 2010, 1)

    def test_within_lifetime_does_not_check_start(self):
        """Note that the ``within_lifetime`` method does not check
        that the build year is compatible with timestep
        """
        dm = PreSpecified([2010, 2015], Mock(), [])
        assert dm.within_lifetime(2011, 2010, 1)

    def test_negative_lifetime_raises(self):
        dm = PreSpecified([2010, 2015], Mock(), [])
        with raises(ValueError):
            dm.within_lifetime(2010, 2010, -1)


class TestRuleBased:

    def test_initialisation(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        assert dm.timesteps == timesteps
        assert dm.satisfied is False
        assert dm.current_timestep_index == 0
        assert dm.current_iteration == 0

    def test_generator(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())

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
    def test_decision_manager_init(self):
        store = MemoryInterface()
        store._model_runs = {'test': {'sos_model': 'test_sos_model'}}
        store._sos_models = {'test_sos_model': {'sector_models': []}}
        store._strategies = {'test': []}

        df = DecisionManager(store, [2010, 2015], 'test', 'test_sos_model')
        dm = df.decision_loop()
        bundle = next(dm)
        assert bundle == {0: [2010, 2015]}
        with raises(StopIteration):
            next(dm)
