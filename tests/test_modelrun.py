from unittest.mock import Mock

from pytest import fixture, raises
from smif.modelrun import ModelRunBuilder, ModelRunError, ModelRunner


@fixture(scope='function')
def get_model_run_config_data():

    sos_model = Mock()
    energy_supply = Mock()
    energy_supply.name = 'energy_supply'
    sos_model.models = [energy_supply]

    config = {
        'name': 'unique_model_run_name',
        'stamp': '2017-09-20T12:53:23+00:00',
        'description': 'a description of what the model run contains',
        'decision_module': None,
        'timesteps': [2010, 2011, 2012],
        'sos_model': sos_model,
        'scenarios':
            {'raininess': 'high_raininess'},
        'narratives':
            [Mock(data={'model_name': {'parameter_name': 0}}),
             Mock(data={'model_name': {'parameter_name': 0}})
             ],
        'strategies': [{'strategy': 'pre-specified-planning',
                        'description': 'build_nuclear',
                        'model_name': 'energy_supply',
                        'interventions': [
                            {'name': 'nuclear_large', 'build_year': 2012},
                            {'name': 'carrington_retire', 'build_year': 2011}]
                        }]
    }
    return config


@fixture(scope='function')
def get_model_run(get_model_run_config_data):

    config_data = get_model_run_config_data

    builder = ModelRunBuilder()
    builder.construct(config_data)
    return builder.finish()


class TestModelRunBuilder:

    def test_builder(self, get_model_run_config_data):

        config_data = get_model_run_config_data

        builder = ModelRunBuilder()
        builder.construct(config_data)

        modelrun = builder.finish()

        assert modelrun.name == 'unique_model_run_name'
        assert modelrun.timestamp == '2017-09-20T12:53:23+00:00'
        assert modelrun.model_horizon == [2010, 2011, 2012]
        assert modelrun.status == 'Built'
        assert modelrun.scenarios == {'raininess': 'high_raininess'}
        assert modelrun.narratives == config_data['narratives']
        assert modelrun.strategies == config_data['strategies']


class TestModelRun:

    def test_run_static(self, get_model_run):
        store = Mock()
        model_run = get_model_run
        model_run.run(store)

    def test_run_timesteps(self, get_model_run_config_data):
        """should error that timesteps are empty
        """
        config_data = get_model_run_config_data
        config_data['timesteps'] = []
        builder = ModelRunBuilder()
        builder.construct(config_data)
        model_run = builder.finish()
        store = Mock()
        with raises(ModelRunError) as ex:
            model_run.run(store)
        assert 'No timesteps specified' in str(ex)

    def test_serialize(self, get_model_run_config_data):
        builder = ModelRunBuilder()

        config = get_model_run_config_data

        builder.construct(config)
        model_run = builder.finish()

        config = model_run.as_dict()
        assert config == config


class TestModelRunner():

    def test_call_before_model_run(self):
        store = Mock()
        runner = ModelRunner()
        modelrun = Mock()
        modelrun.strategies = []
        sos_model = Mock()
        sos_model.models = []
        modelrun.sos_model = sos_model

        modelrun.narratives = []
        modelrun.model_horizon = [1, 2]

        runner.solve_model(modelrun, store)

        modelrun.sos_model.before_model_run.call_count == 1

    def test_call_simulate(self):
        store = Mock()
        runner = ModelRunner()
        modelrun = Mock()
        modelrun.strategies = []
        sos_model = Mock()
        sos_model.models = []
        modelrun.sos_model = sos_model
        modelrun.narratives = []
        modelrun.model_horizon = [1, 2]

        runner.solve_model(modelrun, store)

        assert modelrun.sos_model.simulate.call_count == 2
