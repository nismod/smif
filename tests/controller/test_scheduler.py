"""Test ModelRunScheduler and JobScheduler
"""
from unittest.mock import Mock, patch

import networkx
from pytest import fixture, raises
from smif.controller.scheduler import JobScheduler, ModelRunScheduler
from smif.model import ModelOperation, ScenarioModel, SectorModel


class TestModelRunScheduler():
    @patch('smif.controller.scheduler.subprocess.Popen')
    def test_single_modelrun(self, mock_popen):
        my_scheduler = ModelRunScheduler()
        my_scheduler.add('my_model_run', {
            'directory': 'mock/dir',
            'verbosity': 0,
            'warm_start': False,
            'output_format': 'local_csv'
        })

        mock_popen.assert_called_with(
            'smif  run my_model_run -d mock/dir -i local_csv',
            shell=True,
            stderr=-2, stdout=-1
        )

    def test_status_modelrun_never_added(self):
        my_scheduler = ModelRunScheduler()
        status = my_scheduler.get_status('my_model_run')
        assert status['status'] == 'unstarted'

    @patch('smif.controller.scheduler.subprocess.Popen')
    def test_status_model_started(self, mock_popen):
        attrs = {
            'poll.return_value': None,
            'communicate.return_value': (
                "this is a stdout".encode('utf-8'),
            ),
            'returncode': None
        }
        process_mock = Mock(**attrs)
        mock_popen.return_value = process_mock

        my_scheduler = ModelRunScheduler()
        my_scheduler.add('my_model_run', {
            'directory': 'mock/dir',
            'verbosity': 0,
            'warm_start': False,
            'output_format': 'local_csv'
        })
        my_scheduler.lock = True
        status = my_scheduler.get_status('my_model_run')
        assert status['status'] == 'running'

    @patch('smif.controller.scheduler.subprocess.Popen')
    def test_status_model_done(self, mock_popen):
        attrs = {
            'poll.return_value': 0,
            'communicate.return_value': (
                "this is a stdout".encode('utf-8')
            )
        }
        process_mock = Mock(**attrs)
        mock_popen.return_value = process_mock

        my_scheduler = ModelRunScheduler()
        my_scheduler.add('my_model_run', {
            'directory': 'mock/dir',
            'verbosity': 0,
            'warm_start': False,
            'output_format': 'local_csv'
        })
        my_scheduler.lock = True
        response = my_scheduler.get_status('my_model_run')

        assert response['status'] == 'done'

    @patch('smif.controller.scheduler.subprocess.Popen')
    def test_status_model_failed(self, mock_popen):
        attrs = {
            'poll.return_value': 1,
            'communicate.return_value': (
                "this is a stdout".encode('utf-8'),
            )
        }
        process_mock = Mock(**attrs)
        mock_popen.return_value = process_mock

        my_scheduler = ModelRunScheduler()
        my_scheduler.add('my_model_run', {
            'directory': 'mock/dir',
            'verbosity': 0,
            'warm_start': False,
            'output_format':
            'local_csv'
        })
        my_scheduler.lock = True
        response = my_scheduler.get_status('my_model_run')

        assert response['status'] == 'failed'

    @patch('smif.controller.scheduler.subprocess.Popen')
    def test_status_model_stopped(self, mock_popen):
        attrs = {
            'poll.return_value': None,
            'communicate.return_value': (
                "this is a stdout".encode('utf-8'),
            )
        }
        process_mock = Mock(**attrs)
        mock_popen.return_value = process_mock

        my_scheduler = ModelRunScheduler()
        my_scheduler.add('my_model_run', {
            'directory': 'mock/dir',
            'verbosity': 0,
            'warm_start': False,
            'output_format':
            'local_csv'
        })
        my_scheduler.lock = True
        my_scheduler.kill('my_model_run')
        response = my_scheduler.get_status('my_model_run')

        assert response['status'] == 'stopped'


class TestJobScheduler():
    @fixture
    def job_graph(self):
        G = networkx.DiGraph()
        G.add_node(
            'a',
            model=Mock(spec=ScenarioModel),
            operation=ModelOperation.BEFORE_MODEL_RUN,
            data_handle=Mock()
        )
        G.add_node(
            'b',
            model=Mock(spec=SectorModel),
            operation=ModelOperation.SIMULATE,
            data_handle=Mock()
        )
        G.add_edge('a', 'b')
        return G

    def test_add(self, job_graph):
        scheduler = JobScheduler()
        job_id, err = scheduler.add(job_graph)

        print(err)
        assert err is None
        assert scheduler.get_status(job_id)['status'] == 'done'

    def test_default_status(self):
        scheduler = JobScheduler()
        assert scheduler.get_status(0)['status'] == 'unstarted'

    def test_add_cyclic(self, job_graph):
        job_graph.add_edge('b', 'a')
        scheduler = JobScheduler()
        job_id, err = scheduler.add(job_graph)

        assert isinstance(err, NotImplementedError)
        assert scheduler.get_status(job_id)['status'] == 'failed'

    def test_kill_fails(self, job_graph):
        scheduler = JobScheduler()
        job_id, err = scheduler.add(job_graph)

        assert err is None
        with raises(NotImplementedError):
            scheduler.kill(job_id)

    def test_unknown_operation(self, job_graph):
        job_graph.add_node(
            'c',
            model=Mock(),
            operation='unknown_operation',
            data_handle=Mock()
        )
        scheduler = JobScheduler()
        job_id, err = scheduler.add(job_graph)

        assert isinstance(err, ValueError)
        assert scheduler.get_status(job_id)['status'] == 'failed'
