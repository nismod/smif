
from unittest.mock import Mock, patch

from smif.controller.scheduler import Scheduler


@patch('smif.controller.scheduler.subprocess.Popen')
def test_single_modelrun(mock_popen):
    my_scheduler = Scheduler()
    my_scheduler.add('my_model_run', {
        'directory': 'mock/dir',
        'verbosity': 0, 
        'warm_start': False, 
        'output_format': 'local_csv'
    })

    mock_popen.assert_called_with(
        'exec smif  run my_model_run -d mock/dir -i local_csv',
        shell=True,
        stderr=-2, stdout=-1
    )


def test_status_modelrun_never_added():
    my_scheduler = Scheduler()
    status = my_scheduler.get_status('my_model_run')
    assert status['status'] == 'unstarted'


@patch('smif.controller.scheduler.subprocess.Popen')
def test_status_model_started(mock_popen):

    attrs = {
        'poll.return_value': None,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8'),
        ),
        'returncode': None
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
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
def test_status_model_done(mock_popen):

    attrs = {
        'poll.return_value': 0,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8')
        )
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
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
def test_status_model_failed(mock_popen):

    attrs = {
        'poll.return_value': 1,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8'),
        )
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
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
def test_status_model_stopped(mock_popen):

    attrs = {
        'poll.return_value': None,
        'communicate.return_value': (
            "this is a stdout".encode('utf-8'),
        )
    }
    process_mock = Mock(**attrs)
    mock_popen.return_value = process_mock

    my_scheduler = Scheduler()
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
