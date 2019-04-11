from typing import Dict, List
from unittest.mock import Mock

from pytest import fixture, raises

from smif.data_layer.store import Store
from smif.decision.decision import DecisionManager, RuleBased
from smif.exception import SmifDataNotFoundError


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


class TestRuleBasedProperties:

    def test_timesteps(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        assert dm.current_timestep == 2010
        assert dm.next_timestep == 2015

    def test_timesteps_end(self):
        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        dm.current_timestep = 2020
        assert dm.next_timestep is None

    def test_timesteps_begin(self):
        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        dm.current_timestep = 2010
        assert dm.previous_timestep is None

    def test_timesteps_first_last(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        assert dm.first_timestep == 2010
        assert dm.last_timestep == 2020

    def test_interventions(self):

        all_interventions = {'test_intervention': {'name': 'test_intervention'}}

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, all_interventions)
        assert dm.available_interventions([]) == ['test_intervention']

    def test_interventions_planned(self):

        all_interventions = {'test_intervention': {'name': 'test_intervention'},
                             'planned_intervention': {'name': 'planned_intervention'}}

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, all_interventions)
        actual = dm.available_interventions([{'name': 'planned_intervention'}])
        expected = ['test_intervention']
        assert actual == expected

    def test_get_intervention(self):

        interventions = {'a': {'name': 'a'},
                         'b': {'name': 'b'},
                         'c': {'name': 'c'}}

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, interventions)
        assert dm.get_intervention('a') == interventions['a']
        with raises(SmifDataNotFoundError) as ex:
            dm.get_intervention('z')
        msg = "Intervention 'z' is not found in the list of available interventions"
        assert msg in str(ex)


class TestRuleBasedIterationTimestepAccounting:
    """Test that the iteration and timestep accounting methods properly follow
    the path through the decision iterations

    2010 - 0, 1
    2015 - 2, 3
    """

    @fixture(scope='function')
    def dm(self):
        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        return dm

    def test_first_iteration_base_year(self, dm):

        dm.current_timestep = 2010
        dm.current_iteration = 1
        dm._max_iteration_by_timestep[2010] = 1
        assert dm.get_previous_iteration_timestep() is None

    def test_second_iteration_base_year(self, dm):

        dm.current_timestep = 2010
        dm.current_iteration = 2
        dm._max_iteration_by_timestep[2010] = 2
        assert dm.get_previous_iteration_timestep() == (2010, 1)

    def test_second_iteration_next_year(self, dm):

        dm.current_timestep = 2015
        dm.current_iteration = 3
        dm._max_iteration_by_timestep[2010] = 2
        dm._max_iteration_by_timestep[2015] = 3
        assert dm.get_previous_iteration_timestep() == (2010, 2)

    def test_third_iteration_next_year(self, dm):

        dm.current_timestep = 2015
        dm.current_iteration = 4
        dm._max_iteration_by_timestep[2010] = 2
        dm._max_iteration_by_timestep[2015] = 4
        assert dm.get_previous_iteration_timestep() == (2015, 3)


class TestRuleBased:

    def test_initialisation(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())
        assert dm.timesteps == timesteps
        assert dm.satisfied is False
        assert dm.current_timestep == 2010
        assert dm.current_iteration == 0

    def test_generator(self):

        timesteps = [2010, 2015, 2020]
        dm = RuleBased(timesteps, Mock())

        actual = next(dm)
        assert actual == {
            'decision_iterations': [1],
            'timesteps': [2010]
        }

        dm.satisfied = True
        actual = next(dm)
        assert actual == {
            'decision_iterations': [2],
            'timesteps': [2015],
            'decision_links': {
                2: 1
            }
        }
        assert dm.satisfied is False

        actual = next(dm)
        assert actual == {
            'decision_iterations': [3],
            'timesteps': [2015],
            'decision_links': {
                3: 1
            }
        }
        assert dm.satisfied is False

        dm.satisfied = True
        dm.current_timestep = 2020
        actual = next(dm)
        assert actual is None


class TestDecisionManager():

    @fixture(scope='function')
    def decision_manager(self, empty_store) -> DecisionManager:
        empty_store.write_model_run({'name': 'test', 'sos_model': 'test_sos_model'})
        empty_store.write_sos_model({'name': 'test_sos_model', 'sector_models': []})
        empty_store.write_strategies('test', [])
        sos_model = Mock()
        sos_model.name = 'test_sos_model'
        sos_model.sector_models = []

        df = DecisionManager(empty_store, [2010, 2015], 'test', sos_model)
        return df

    def test_decision_manager_init(self,  decision_manager: DecisionManager):
        df = decision_manager
        dm = df.decision_loop()
        bundle = next(dm)
        assert bundle == {
            'decision_iterations': [0],
            'timesteps': [2010, 2015]
        }
        with raises(StopIteration):
            next(dm)

    def test_available_interventions(self, decision_manager: DecisionManager):
        df = decision_manager
        df._register = {'a': {'name': 'a'},
                        'b': {'name': 'b'},
                        'c': {'name': 'c'}}

        assert df.available_interventions == df._register

        df.planned_interventions = {(2010, 'a'), (2010, 'b')}

        expected = {'c': {'name': 'c'}}

        assert df.available_interventions == expected

    def test_get_intervention(self,  decision_manager: DecisionManager):
        df = decision_manager
        df._register = {'a': {'name': 'a'},
                        'b': {'name': 'b'},
                        'c': {'name': 'c'}}

        assert df.get_intervention('a') == {'name': 'a'}

        with raises(SmifDataNotFoundError):
            df.get_intervention('z')

    def test_buildable(self, decision_manager):

        decision_manager._timesteps = [2010, 2015]
        assert decision_manager.buildable(2010, 2010) is True
        assert decision_manager.buildable(2011, 2010) is True

    def test_historical_intervention_buildable(self, decision_manager):
        decision_manager._timesteps = [2020, 2030]
        assert decision_manager.buildable(1980, 2020) is True
        assert decision_manager.buildable(1990, 2020) is True

    def test_buildable_raises(self, decision_manager):

        with raises(ValueError):
            decision_manager.buildable(2015, 2014)

    def test_within_lifetime(self, decision_manager):

        assert decision_manager.within_lifetime(2010, 2010, 1)

    def test_within_lifetime_does_not_check_start(self, decision_manager):
        """Note that the ``within_lifetime`` method does not check
        that the build year is compatible with timestep
        """
        assert decision_manager.within_lifetime(2011, 2010, 1)

    def test_negative_lifetime_raises(self, decision_manager):
        with raises(ValueError):
            decision_manager.within_lifetime(2010, 2010, -1)


class TestDecisionManagerDecisions:

    @fixture(scope='function')
    def decision_manager(self, empty_store) -> DecisionManager:
        empty_store.write_model_run({'name': 'test', 'sos_model': 'test_sos_model'})
        empty_store.write_sos_model({'name': 'test_sos_model', 'sector_models': []})
        empty_store.write_strategies('test', [])
        sos_model = Mock()
        sos_model.name = 'test_sos_model'
        sos_model.sector_models = []

        interventions = {'test': {'technical_lifetime': {'value': 99}},
                         'planned': {'technical_lifetime': {'value': 99}},
                         'decided': {'technical_lifetime': {'value': 99}}
                         }

        df = DecisionManager(empty_store, [2010, 2015], 'test', sos_model)
        df._register = interventions
        return df

    def test_get_decisions(self, decision_manager: DecisionManager):
        dm = decision_manager

        mock_handle = Mock()
        dm._decision_module = Mock()
        dm._decision_module.get_decision = Mock(
            return_value=[{'name': 'test', 'build_year': 2010}])

        actual = dm._get_decisions(dm._decision_module, mock_handle)
        expected = [(2010, 'test')]
        assert actual == expected

    def test_get_and_save_decisions_dm(self, decision_manager: DecisionManager):
        """Test that the ``get_and_save_decisions`` method updates pre-decision
        state with a new decision and writes it to store
        """
        dm = decision_manager

        dm._decision_module = Mock()
        dm._decision_module.get_decision = Mock(
            return_value=[{'name': 'decided', 'build_year': 2010}])
        dm._decision_module.get_previous_state = Mock(return_value=[])

        dm.get_and_save_decisions(0, 2010)

        actual = dm._store  # type: Store
        expected = [{'name': 'decided', 'build_year': 2010}]  # type: List[Dict]
        assert actual.read_state('test', 2010, decision_iteration=0) == expected

    def test_get_and_save_decisions_prespec(self,
                                            decision_manager: DecisionManager):
        """Test that the ``get_and_save_decisions`` method updates pre-decision
        state with a pre-specified planning and writes it to store
        """
        dm = decision_manager

        dm.planned_interventions = [(2010, 'planned')]

        dm.get_and_save_decisions(0, 2010)

        actual = dm._store  # type: Store
        expected = [{'name': 'planned', 'build_year': 2010}]  # type: List[Dict]
        assert actual.read_state('test', 2010, decision_iteration=0) == expected

    def test_pre_spec_and_decision_module(self,
                                          decision_manager: DecisionManager):
        dm = decision_manager

        dm._decision_module = Mock()
        dm._decision_module.get_decision = Mock(
            return_value=[{'name': 'decided', 'build_year': 2010}])
        dm._decision_module.get_previous_state = Mock(return_value=[])

        dm.planned_interventions = [(2010, 'planned')]

        dm.get_and_save_decisions(0, 2010)

        actual = dm._store.read_state('test', 2010, decision_iteration=0)  # type: List[Dict]

        expected = set([('decided', 2010), ('planned', 2010)])
        assert set([(x['name'], x['build_year']) for x in actual]) == expected
