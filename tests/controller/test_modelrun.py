from copy import copy
from unittest.mock import Mock, patch

import networkx as nx
from pytest import fixture, raises
from smif.controller.modelrun import ModelRunBuilder, ModelRunner
from smif.exception import SmifModelRunError


@fixture(scope='function')
def config_data():
    """Config for a model run
    """
    sos_model = Mock()
    sos_model.name = "test_sos_model"
    sos_model.parameters = {}

    climate_scenario = Mock()
    climate_scenario.name = 'climate'
    climate_scenario.deps = {}
    sos_model.scenario_models = {'climate': climate_scenario}

    energy_supply = Mock()
    energy_supply.name = 'energy_supply'
    energy_supply.deps = {}
    energy_supply.parameters = {}
    sos_model.models = {'energy_supply': energy_supply}

    graph = nx.DiGraph()
    graph.add_node('energy_supply', model=climate_scenario)

    config = {
        'name': 'unique_model_run_name',
        'stamp': '2017-09-20T12:53:23+00:00',
        'description': 'a description of what the model run contains',
        'timesteps': [2010, 2011, 2012],
        'sos_model': sos_model,
        'scenarios': {
            'climate': 'RCP4.5'
        },
        'narratives': [
            Mock(data={'model_name': {'parameter_name': 0}}),
            Mock(data={'model_name': {'parameter_name': 0}})
        ],
        'strategies': [
            {
                'strategy': 'pre-specified-planning',
                'description': 'build_nuclear',
                'model_name': 'energy_supply',
                'interventions': [
                    {'name': 'nuclear_large', 'build_year': 2012},
                    {'name': 'carrington_retire', 'build_year': 2011}
                ]
            }
        ]
    }
    return config


@fixture(scope='function')
def model_run(config_data):
    """ModelRun built from config
    """
    builder = ModelRunBuilder()
    builder.construct(config_data)
    return builder.finish()


@fixture(scope='function')
def mock_model_run():
    """Minimal mock ModelRun
    """
    sos_model = Mock()
    sos_model.parameters = {}
    sos_model.models = []

    modelrun = Mock()
    modelrun.strategies = []
    modelrun.sos_model = sos_model
    modelrun.narratives = []
    modelrun.model_horizon = [1]
    modelrun.initialised = False
    return modelrun


@fixture(scope='function')
def mock_store():
    """Minimal mock store
    """
    store = Mock()
    store.read_model_run = Mock(return_value={'narratives': {}})
    store.read_strategies = Mock(return_value=[])
    store.read_all_initial_conditions = Mock(return_value=[])

    store.read_sos_model = Mock(return_value={'sector_models': ['sector_model_test']})
    store.read_interventions = Mock(return_value={})

    return store


class TestModelRunBuilder:
    """Build from config
    """
    def test_builder(self, config_data):
        """Test basic properties
        """
        builder = ModelRunBuilder()
        builder.construct(config_data)
        modelrun = builder.finish()

        assert modelrun.name == 'unique_model_run_name'
        assert modelrun.timestamp == '2017-09-20T12:53:23+00:00'
        assert modelrun.model_horizon == [2010, 2011, 2012]
        assert modelrun.status == 'Built'
        assert modelrun.scenarios == {'climate': 'RCP4.5'}
        assert modelrun.narratives == config_data['narratives']
        assert modelrun.strategies == config_data['strategies']

    def test_builder_scenario_sosmodelrun_not_in_sosmodel(self, config_data):
        """Error from unused scenarios
        """
        config_data['scenarios'] = {
            'climate': 'RCP4.5',
            'population': 'high_population'
        }
        builder = ModelRunBuilder()
        builder.construct(config_data)

        with raises(SmifModelRunError) as ex:
            builder.finish()
        assert "ScenarioSet 'population' is selected in the ModelRun " \
               "configuration but not found in the SosModel configuration" in str(ex)


class TestModelRun:
    """Core ModelRun
    """
    def test_run_static(self, model_run, mock_store):
        """Call run
        """
        model_run.run(mock_store)

    def test_run_timesteps(self, config_data):
        """should error that timesteps are empty
        """
        config_data['timesteps'] = []
        builder = ModelRunBuilder()
        builder.construct(config_data)
        model_run = builder.finish()
        store = Mock()
        with raises(SmifModelRunError) as ex:
            model_run.run(store)
        assert 'No timesteps specified' in str(ex)

    def test_serialize(self, config_data):
        """Serialise back to config dict
        """
        builder = ModelRunBuilder()
        builder.construct(config_data)
        model_run = builder.finish()

        expected = copy(config_data)
        expected['sos_model'] = config_data['sos_model'].name  # expect a reference by name
        assert expected == model_run.as_dict()


class TestModelRunnerJobGraphs():
    """Cover all JobGraph corner cases
    """
    @patch('smif.controller.scheduler.JobScheduler.add')
    def test_jobgraph_single_timestep(self, mock_add, mock_store, mock_model_run):
        """
        a[before]
        |
        v
        a[sim]
        """
        mock_add.return_value = (0, None)
        model_a = Mock()
        model_a.name = 'model_a'
        model_a.deps = {}
        model_a.parameters = {}

        mock_model_run.sos_model.models = {
            model_a.name: model_a
        }

        runner = ModelRunner()
        runner.solve_model(mock_model_run, mock_store)

        call_args = mock_add.call_args
        job_graph = call_args[0][0]

        actual = list(job_graph.predecessors('before_model_run_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.successors('before_model_run_model_a'))
        expected = ['simulate_1_0_model_a']
        assert actual == expected

        actual = list(job_graph.predecessors('simulate_1_0_model_a'))
        expected = ['before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('simulate_1_0_model_a'))
        expected = []
        assert actual == expected

    @patch('smif.controller.scheduler.JobScheduler.add')
    def test_jobgraph_multiple_timesteps(self, mock_add, mock_store, mock_model_run):
        """
        a[before]
        |        |
        v        V
        a[sim]  a[sim]
        t=1     t=2
        """
        mock_add.return_value = (0, None)
        model_a = Mock()
        model_a.name = 'model_a'
        model_a.deps = {}
        model_a.parameters = {}

        mock_model_run.sos_model.models = {
            model_a.name: model_a
        }

        mock_model_run.model_horizon = [1, 2]

        runner = ModelRunner()
        runner.solve_model(mock_model_run, mock_store)

        call_args = mock_add.call_args
        job_graph = call_args[0][0]

        actual = list(job_graph.predecessors('before_model_run_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.successors('before_model_run_model_a'))
        expected = ['simulate_1_0_model_a', 'simulate_2_0_model_a']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.predecessors('simulate_1_0_model_a'))
        expected = ['before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('simulate_1_0_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.predecessors('simulate_2_0_model_a'))
        expected = ['before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('simulate_2_0_model_a'))
        expected = []
        assert actual == expected

    @patch('smif.controller.scheduler.JobScheduler.add')
    def test_jobgraph_multiple_models(self, mock_add, mock_store, mock_model_run):
        """
        a[before]   b[before]   c[before]
        |           |           |
        v           V           V
        a[sim] ---> b[sim] ---> c[sim]
           |------------------>
        """
        mock_add.return_value = (0, None)
        model_a = Mock()
        model_b = Mock()
        model_c = Mock()
        model_a.name = 'model_a'
        model_b.name = 'model_b'
        model_c.name = 'model_c'
        model_a.parameters = {}
        model_b.parameters = {}
        model_c.parameters = {}

        dep_a_b = Mock()
        dep_a_b.source_model.name = 'model_a'
        dep_a_b.sink_model.name = 'model_b'
        dep_b_c = Mock()
        dep_b_c.source_model.name = 'model_b'
        dep_b_c.sink_model.name = 'model_c'
        dep_a_c = Mock()
        dep_a_c.source_model.name = 'model_a'
        dep_a_c.sink_model.name = 'model_c'

        model_a.deps = {
            'model_a': dep_a_b
        }
        model_b.deps = {
            'model_b': dep_b_c
        }
        model_c.deps = {
            'model_c': dep_a_c
        }

        mock_model_run.sos_model.models = {
            model_a.name: model_a,
            model_b.name: model_b,
            model_c.name: model_c
        }

        runner = ModelRunner()
        runner.solve_model(mock_model_run, mock_store)

        call_args = mock_add.call_args
        job_graph = call_args[0][0]

        actual = list(job_graph.predecessors('simulate_1_0_model_a'))
        expected = ['before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('simulate_1_0_model_a'))
        expected = ['simulate_1_0_model_b', 'simulate_1_0_model_c']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.predecessors('simulate_1_0_model_b'))
        expected = ['before_model_run_model_b', 'simulate_1_0_model_a']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.successors('simulate_1_0_model_b'))
        expected = ['simulate_1_0_model_c']
        assert actual == expected

        actual = list(job_graph.predecessors('simulate_1_0_model_c'))
        expected = ['before_model_run_model_c', 'simulate_1_0_model_a', 'simulate_1_0_model_b']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.successors('simulate_1_0_model_c'))
        expected = []
        assert actual == expected

    def test_jobgraph_interdependency(self, mock_store, mock_model_run):
        """
        a[before]   b[before]
        |           |
        v           V
        a[sim] ---> b[sim]
               <---
        """
        model_a = Mock()
        model_b = Mock()
        model_a.name = 'model_a'
        model_b.name = 'model_b'
        model_a.parameters = {}
        model_b.parameters = {}

        dep_a_b = Mock()
        dep_a_b.source_model.name = 'model_a'
        dep_a_b.sink_model.name = 'model_b'
        dep_b_a = Mock()
        dep_b_a.source_model.name = 'model_b'
        dep_b_a.sink_model.name = 'model_a'

        model_a.deps = {
            'model_a': dep_a_b
        }
        model_b.deps = {
            'model_b': dep_b_a
        }

        mock_model_run.sos_model.models = {
            model_a.name: model_a,
            model_b.name: model_b,
        }

        runner = ModelRunner()
        with raises(NotImplementedError):
            runner.solve_model(mock_model_run, mock_store)

    @patch('smif.controller.scheduler.JobScheduler.add')
    def test_jobgraph_with_models_initialised(self, mock_add, mock_store, mock_model_run):
        """
        a[sim]
        """
        mock_add.return_value = (0, None)
        model_a = Mock()
        model_a.name = 'model_a'
        model_a.deps = {}
        model_a.parameters = {}

        mock_model_run.sos_model.models = {
            model_a.name: model_a
        }

        mock_model_run.initialised = True

        runner = ModelRunner()
        runner.solve_model(mock_model_run, mock_store)

        call_args = mock_add.call_args
        job_graph = call_args[0][0]

        actual = list(job_graph.predecessors('simulate_1_0_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.successors('simulate_1_0_model_a'))
        expected = []
        assert actual == expected
