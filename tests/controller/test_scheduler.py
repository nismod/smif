
from unittest.mock import Mock, patch

from smif.controller.scheduler import Scheduler


@patch('smif.controller.scheduler.Popen')
def test_single_modelrun(mock_popen):
    my_scheduler = Scheduler()
    my_scheduler.add('my_model_run', {'directory': 'mock/dir'})

    mock_popen.assert_called_with(
        ['smif', 'run', 'my_model_run', '-d', 'mock/dir'],
        stderr=-1, stdout=-1
    )


def test_status_modelrun_never_added():
    my_scheduler = Scheduler()
    status = my_scheduler.get_status('my_model_run')
    assert status['status'] == 'unknown'


@patch('smif.controller.scheduler.Popen')
def test_status_model_started(mock_popen):

    attrs = {
        'poll.return_value': None,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8'),
            "this is a stderr".encode('utf-8')
        ),
        'returncode': None
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
    my_scheduler.add('my_model_run', {'directory': 'mock/dir'})
    status = my_scheduler.get_status('my_model_run')
    assert status['status'] == 'running'


@patch('smif.controller.scheduler.Popen')
def test_status_model_done(mock_popen):

    attrs = {
        'poll.return_value': True,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8'),
            "this is a stderr".encode('utf-8')
        ),
        'returncode': 0
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
    my_scheduler.add('my_model_run', {'directory': 'mock/dir'})
    status = my_scheduler.get_status('my_model_run')

    assert status == {
        'status': 'done',
        'output': 'this is a stdout',
        'err': 'this is a stderr'
    }


@patch('smif.controller.scheduler.Popen')
def test_status_model_failed(mock_popen):

    attrs = {
        'poll.return_value': True,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8'),
            "this is a stderr".encode('utf-8')
        ),
        'returncode': 1
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
    my_scheduler.add('my_model_run', {'directory': 'mock/dir'})
    status = my_scheduler.get_status('my_model_run')

    assert status == {
        'status': 'failed',
        'output': 'this is a stdout',
        'err': 'this is a stderr'
    }
