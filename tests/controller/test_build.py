from unittest.mock import Mock

from smif.controller.build import (get_initial_conditions_strategies,
                                   get_pre_specified_planning_strategies)


class TestStrategies:

    def test_get_pre_specified_planning_strategies(self, get_sos_model_run, initial_system):

        handler = Mock()
        interventions = initial_system
        handler.read_strategies = Mock(return_value=interventions)

        actual = get_pre_specified_planning_strategies(get_sos_model_run, handler)

        assert handler.read_strategies.called_with("energy_supply.csv")
        assert isinstance(actual[0], dict)

        assert isinstance(actual, list)
        assert 'filename' not in actual[0].keys()
        assert 'interventions' in actual[0].keys()
        assert actual[0]['interventions'] == interventions
        assert actual[0]['strategy'] == 'pre-specified-planning'
        assert actual[0]['model_name'] == 'energy_supply'

    def test_get_initial_conditions_strategies(self, initial_system):

        sector_model = Mock()
        sector_model.name = Mock(return_value='test_model')
        sector_model.initial_conditions = initial_system

        actual = get_initial_conditions_strategies([sector_model])

        assert isinstance(actual, list)
        assert isinstance(actual[0], dict)
        assert 'filename' not in actual[0].keys()
        assert 'interventions' in actual[0].keys()
        assert actual[0]['interventions'] == initial_system
        assert actual[0]['strategy'] == 'pre-specified-planning'
