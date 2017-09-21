from unittest.mock import Mock

from pytest import fixture
from smif.modelrun import ModelRunBuilder


@fixture(scope='function')
def get_model_run_config_data():

    config = {
        'name': 'unique_model_run_name',
        'stamp': '2017-09-20T12:53:23+00:00',
        'description': 'a description of what the model run contains',
        'timesteps': [2010, 2011, 2012],
        'sos_model': Mock(sector_models=[]),
        'scenarios':
            {'raininess': 'high_raininess'},
        'narratives':
            {'technology': 'high tech',
             'governance': 'central plan'}
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
        assert modelrun.narratives == {'technology': 'high tech',
                                       'governance': 'central plan'}


class TestModelRun:

    def test_run_static(self, get_model_run):
        model_run = get_model_run
        model_run.run()
