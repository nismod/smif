from unittest.mock import Mock

from pytest import fixture
from smif.modelrun import ModelRunBuilder


@fixture(scope='function')
def get_model_run_config_data():

    config = {
        'name': 'unique_model_run_name',
        'stamp': '2017-09-20T12:53:23+00:00',
        'description': 'a description of what the model run contains',
        'decision_module': None,
        'timesteps': [2010, 2011, 2012],
        'sos_model': Mock(sector_models=[]),
        'scenarios':
            {'raininess': 'high_raininess'},
        'narratives':
            [Mock(data={'model_name': {'parameter_name': 0}}),
             Mock(data={'model_name': {'parameter_name': 0}})
             ]
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


class TestModelRun:

    def test_run_static(self, get_model_run):
        model_run = get_model_run
        model_run.run()

    def test_serialize(self, get_model_run_config_data):
        builder = ModelRunBuilder()

        config = get_model_run_config_data

        builder.construct(config)
        model_run = builder.finish()

        config = model_run.as_dict()
        assert config == config
