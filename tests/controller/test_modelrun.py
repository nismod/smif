from copy import copy
from unittest.mock import Mock

from pytest import fixture, raises
from smif.controller.modelrun import ModelRunBuilder, ModelRunner
from smif.exception import SmifModelRunError
from smif.metadata import RelativeTimestep, Spec
from smif.model import ScenarioModel, SectorModel, SosModel


class EmptySectorModel(SectorModel):
    def simulate(self, data):
        return data


@fixture(scope='function')
def config_data():
    """Config for a model run
    """
    sos_model = SosModel('sos_model')

    climate_scenario = ScenarioModel('climate')
    sos_model.add_model(climate_scenario)

    energy_supply = EmptySectorModel('energy_supply')
    sos_model.add_model(energy_supply)

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
    sos_model = SosModel('test_sos_model')
    sos_model.parameters = {}

    modelrun = Mock()
    modelrun.name = 'test'
    modelrun.strategies = []
    modelrun.sos_model = sos_model
    modelrun.narratives = {}
    modelrun.model_horizon = [1]
    modelrun.initialised = False
    return modelrun


@fixture(scope='function')
def mock_store():
    """Minimal mock store
    """
    store = Mock()
    store.read_model_run = Mock(return_value={
        'sos_model': 'test_sos_model',
        'narratives': {},
        'scenarios': {}
    })
    store.read_sos_model = Mock(return_value={
        'name': 'test_sos_model',
        'model_dependencies': [],
        'scenario_dependencies': [],
        'sector_models': ['sector_model_test']
    })
    store.read_strategies = Mock(return_value=[])
    store.read_all_initial_conditions = Mock(return_value=[])
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
        assert "ScenarioSets {'population'} are selected in the ModelRun " \
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
    def test_jobgraph_single_timestep(self, mock_model_run):
        """
        a[before]
        |
        v
        a[sim]
        """
        model_a = EmptySectorModel('model_a')
        mock_model_run.sos_model.add_model(model_a)

        runner = ModelRunner()
        bundle = {
            'decision_iterations': [0],
            'timesteps': [1]
        }
        job_graph = runner.build_job_graph(mock_model_run, bundle)
        print(job_graph.nodes, job_graph.edges)

        actual = list(job_graph.predecessors('test_before_model_run_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.successors('test_before_model_run_model_a'))
        expected = ['test_simulate_1_0_model_a']
        assert actual == expected

        actual = list(job_graph.predecessors('test_simulate_1_0_model_a'))
        expected = ['test_before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('test_simulate_1_0_model_a'))
        expected = []
        assert actual == expected

    def test_jobgraph_multiple_timesteps(self, mock_model_run):
        """
        a[before]
        |        |
        v        V
        a[sim]  a[sim]
        t=1     t=2
        """
        model_a = EmptySectorModel('model_a')
        mock_model_run.sos_model.add_model(model_a)

        mock_model_run.model_horizon = [1, 2]

        runner = ModelRunner()
        bundle = {
            'decision_iterations': [0],
            'timesteps': [1, 2]
        }
        job_graph = runner.build_job_graph(mock_model_run, bundle)

        actual = list(job_graph.predecessors('test_before_model_run_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.successors('test_before_model_run_model_a'))
        expected = ['test_simulate_1_0_model_a', 'test_simulate_2_0_model_a']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.predecessors('test_simulate_1_0_model_a'))
        expected = ['test_before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('test_simulate_1_0_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.predecessors('test_simulate_2_0_model_a'))
        expected = ['test_before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('test_simulate_2_0_model_a'))
        expected = []
        assert actual == expected

    def test_jobgraph_multiple_timesteps_with_dep(self, mock_model_run):
        """
        a[before]
        |        |
        v        V
        a[sim]   a[sim]
        t=1 ---> t=2
        """
        model_a = EmptySectorModel('model_a')
        model_a.add_input(Spec('input', dtype='float'))
        model_a.add_output(Spec('output', dtype='float'))

        mock_model_run.sos_model.add_model(model_a)
        mock_model_run.sos_model.add_dependency(
            model_a, 'output',
            model_a, 'input',
            RelativeTimestep.PREVIOUS)

        mock_model_run.model_horizon = [1, 2]

        runner = ModelRunner()
        bundle = {
            'decision_iterations': [0],
            'timesteps': [1, 2]
        }
        job_graph = runner.build_job_graph(mock_model_run, bundle)

        actual = list(job_graph.predecessors('test_simulate_1_0_model_a'))
        expected = ['test_before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.predecessors('test_simulate_2_0_model_a'))
        expected = ['test_before_model_run_model_a', 'test_simulate_1_0_model_a']
        assert sorted(actual) == sorted(expected)

    def test_jobgraph_multiple_models(self, mock_model_run):
        """
        a[before]   b[before]   c[before]
        |           |           |
        v           V           V
        a[sim] ---> b[sim] ---> c[sim]
           |------------------>
        """
        model_a = EmptySectorModel('model_a')
        model_a.add_output(Spec('a', dtype='float'))

        model_b = EmptySectorModel('model_b')
        model_b.add_input(Spec('a', dtype='float'))
        model_b.add_output(Spec('b', dtype='float'))

        model_c = EmptySectorModel('model_c')
        model_c.add_input(Spec('a', dtype='float'))
        model_c.add_input(Spec('b', dtype='float'))

        mock_model_run.sos_model.add_model(model_a)
        mock_model_run.sos_model.add_model(model_b)
        mock_model_run.sos_model.add_model(model_c)

        mock_model_run.sos_model.add_dependency(model_a, 'a', model_b, 'a')
        mock_model_run.sos_model.add_dependency(model_a, 'a', model_c, 'a')
        mock_model_run.sos_model.add_dependency(model_b, 'b', model_c, 'b')

        runner = ModelRunner()
        bundle = {
            'decision_iterations': [0],
            'timesteps': [1]
        }
        job_graph = runner.build_job_graph(mock_model_run, bundle)

        actual = list(job_graph.predecessors('test_simulate_1_0_model_a'))
        expected = ['test_before_model_run_model_a']
        assert actual == expected

        actual = list(job_graph.successors('test_simulate_1_0_model_a'))
        expected = ['test_simulate_1_0_model_b', 'test_simulate_1_0_model_c']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.predecessors('test_simulate_1_0_model_b'))
        expected = ['test_before_model_run_model_b', 'test_simulate_1_0_model_a']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.successors('test_simulate_1_0_model_b'))
        expected = ['test_simulate_1_0_model_c']
        assert actual == expected

        actual = list(job_graph.predecessors('test_simulate_1_0_model_c'))
        expected = ['test_before_model_run_model_c', 'test_simulate_1_0_model_a',
                    'test_simulate_1_0_model_b']
        assert sorted(actual) == sorted(expected)

        actual = list(job_graph.successors('test_simulate_1_0_model_c'))
        expected = []
        assert actual == expected

    def test_jobgraph_interdependency(self, mock_model_run):
        """
        a[before]   b[before]
        |           |
        v           V
        a[sim] ---> b[sim]
               <---
        """
        model_a = EmptySectorModel('model_a')
        model_a.add_input(Spec('b', dtype='float'))
        model_a.add_output(Spec('a', dtype='float'))

        model_b = EmptySectorModel('model_b')
        model_b.add_input(Spec('a', dtype='float'))
        model_b.add_output(Spec('b', dtype='float'))

        mock_model_run.sos_model.add_model(model_a)
        mock_model_run.sos_model.add_model(model_b)

        mock_model_run.sos_model.add_dependency(model_a, 'a', model_b, 'a')
        mock_model_run.sos_model.add_dependency(model_b, 'b', model_a, 'b')

        runner = ModelRunner()
        bundle = {
            'decision_iterations': [0],
            'timesteps': [1]
        }
        with raises(NotImplementedError):
            runner.build_job_graph(mock_model_run, bundle)

    def test_jobgraph_with_models_initialised(self, mock_model_run):
        """
        a[sim]
        """
        model_a = EmptySectorModel('model_a')
        mock_model_run.sos_model.add_model(model_a)
        mock_model_run.initialised = True

        runner = ModelRunner()
        bundle = {
            'decision_iterations': [0],
            'timesteps': [1]
        }
        job_graph = runner.build_job_graph(mock_model_run, bundle)

        actual = list(job_graph.predecessors('test_simulate_1_0_model_a'))
        expected = []
        assert actual == expected

        actual = list(job_graph.successors('test_simulate_1_0_model_a'))
        expected = []
        assert actual == expected
