"""Test command line interface
"""

import os
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import smif
from smif.cli import (confirm, get_narratives, parse_arguments,
                      setup_project_folder)
from smif.data_layer import DatafileInterface
from smif.parameters.narrative import Narrative


def get_args(args):
    """Get args object from list of strings
    """
    parser = parse_arguments()
    return parser.parse_args(args)


def test_parse_arguments():
    """Setup a project folder argument parsing
    """
    with TemporaryDirectory() as project_folder:
        args = get_args(['setup', project_folder])

        expected = project_folder
        actual = args.path
        assert actual == expected

        # Ensure that the `setup_configuration` function is called when `setup`
        # command is passed to the cli
        assert args.func.__name__ == 'setup_configuration'


def test_fixture_single_run():
    """Test running the filesystem-based single_run fixture
    """
    config_dir = os.path.join(os.path.dirname(__file__),
                              '..', 'fixtures', 'single_run')
    output = subprocess.run(["smif", "-v", "run", "-d", config_dir,
                             "20170918_energy_water_short.yml"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert "Running 20170918_energy_water_short.yml" in str(output.stderr)
    assert "Model run complete" in str(output.stdout)


def test_fixture_list_runs():
    """Test running the filesystem-based single_run fixture
    """
    config_dir = os.path.join(os.path.dirname(__file__),
                              '..', 'fixtures', 'single_run')
    output = subprocess.run(["smif", "list", "-d", config_dir], stdout=subprocess.PIPE)
    assert "20170918_energy_water.yml" in str(output.stdout)
    assert "20170918_energy_water_short.yml" in str(output.stdout)


def test_setup_project_folder():
    """Test contents of the setup project folder
    """
    with TemporaryDirectory() as project_folder:
        setup_project_folder(project_folder)

        assert os.path.exists(project_folder)

        folder_list = ['data']
        for folder in folder_list:
            folder_path = os.path.join(project_folder, folder)

            assert os.path.exists(folder_path)


@patch('builtins.input', return_value='y')
def test_confirm_yes(input):
    assert confirm()


@patch('builtins.input', return_value='n')
def test_confirm_no(input):
    assert not confirm()


@patch('builtins.input', return_value='')
def test_confirm_default_response(input):
    assert not confirm()


@patch('builtins.input', return_value='n')
@patch('builtins.print')
def test_confirm_default_message(mock_print, input):
    confirm()
    input.assert_has_calls([call('Confirm [n]|y: ')])


@patch('builtins.input', return_value='n')
@patch('builtins.print')
def test_confirm_custom_message(mock_print, input):
    confirm('Create directory?', True)
    input.assert_has_calls([call('Create directory? [y]|n: ')])


@patch('builtins.input', side_effect=['invalid', 'y'])
@patch('builtins.print')
def test_confirm_repeat_message(mock_print, input):
    confirm()
    input.assert_has_calls([call('Confirm [n]|y: '), call('Confirm [n]|y: ')])
    mock_print.assert_called_with('please enter y or n.')


def test_version_display():
    """Expect version number from `smif -V`
    """
    output = subprocess.run(['smif', '-V'], stdout=subprocess.PIPE)
    assert smif.__version__ in str(output.stdout)


def test_verbose_debug():
    """Expect debug message from `smif -vv`
    """
    output = subprocess.run(['smif', '-vv'], stderr=subprocess.PIPE)
    assert 'DEBUG' in str(output.stderr)


def test_verbose_debug_alt():
    """Expect debug message from `smif --verbose --verbose`
    """
    output = subprocess.run(['smif', '--verbose', '--verbose'], stderr=subprocess.PIPE)
    assert 'DEBUG' in str(output.stderr)


def test_verbose_info(setup_folder_structure, setup_project_folder):
    """Expect info message from `smif -v validate <config_file>`
    """
    config_file = os.path.join(str(setup_folder_structure))
    output = subprocess.run(['smif', '-v', 'run', config_file], stderr=subprocess.PIPE)
    assert 'INFO' in str(output.stderr)


class TestRunSosModelRunComponents():

    def test_narratives(self):
        config_file = os.path.join(os.path.dirname(__file__),
                                   '..', 'fixtures', 'single_run')

        handler = DatafileInterface(config_file)
        narratives = [{'technology': ['High Tech Demand Side Management']}]
        actual = get_narratives(handler, narratives)

        data = {'energy_demand': {'smart_meter_savings': 8},
                'water_supply': {'clever_water_meter_savings': 8}
                }
        name = 'High Tech Demand Side Management'
        description = 'High penetration of SMART technology on the demand side'
        narrative_set = 'technology'

        narrative_object = Narrative(name, description, narrative_set)
        narrative_object.data = data

        assert actual == [narrative_object]
