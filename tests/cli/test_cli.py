"""Test command line interface
"""

import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import smif
from pytest import fixture
from smif.cli import confirm, parse_arguments, setup_project_folder


@fixture
def tmp_sample_project(tmpdir_factory):
    test_folder = tmpdir_factory.mktemp("smif")
    subprocess.run(
        ["smif", "setup", "-d", str(test_folder), "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return str(test_folder)


def get_args(args):
    """Get args object from list of strings
    """
    parser = parse_arguments()
    return parser.parse_args(args)


def test_parse_arguments():
    """Setup a project folder argument parsing
    """
    with TemporaryDirectory() as project_folder:
        args = get_args(['setup', '-d', project_folder])

        expected = project_folder
        actual = args.directory
        assert actual == expected

        # Ensure that the `setup_project_folder` function is called when `setup`
        # command is passed to the cli
        assert args.func.__name__ == 'setup_project_folder'


def test_fixture_single_run(tmp_sample_project):
    """Test running the (default) binary-filesystem-based single_run fixture
    """
    config_dir = tmp_sample_project
    output = subprocess.run(["smif", "run", "-d", config_dir,
                             "energy_central", "-v"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert "Running energy_central" in str(output.stderr)
    assert "Model run 'energy_central' complete" in str(output.stdout)


def test_fixture_single_run_csv(tmp_sample_project):
    """Test running the csv-filesystem-based single_run fixture
    """
    output = subprocess.run(
        ["smif", "run", "-i", "local_csv", "-d", tmp_sample_project,
         "energy_central", "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert "Running energy_central" in str(output.stderr)
    assert "Model run 'energy_central' complete" in str(output.stdout)


def test_fixture_single_run_warm(tmp_sample_project):
    """Test running the (default) single_run fixture with warm setting enabled
    """
    config_dir = tmp_sample_project
    output = subprocess.run(["smif", "run", "-v", "-w", "-d", config_dir,
                             "energy_central"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert "Running energy_central" in str(output.stderr)
    assert "Model run 'energy_central' complete" in str(output.stdout)


def test_fixture_batch_run(tmp_sample_project):
    """Test running the multiple modelruns using the batch_run option
    """
    config_dir = tmp_sample_project
    output = subprocess.run(["smif", "run", "-v", "-b", "-d", config_dir,
                             os.path.join(config_dir, "batchfile")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert "Running energy_water_cp_cr" in str(output.stderr)
    assert "Model run 'energy_water_cp_cr' complete" in str(output.stdout)
    assert "Running energy_central" in str(output.stderr)
    assert "Model run 'energy_central' complete" in str(output.stdout)


def test_fixture_list_runs(tmp_sample_project):
    """Test running the filesystem-based single_run fixture
    """
    config_dir = tmp_sample_project
    output = subprocess.run(["smif", "list", "-d", config_dir], stdout=subprocess.PIPE)
    assert "energy_water_cp_cr" in str(output.stdout)
    assert "energy_central" in str(output.stdout)

    # Run energy_central and re-check output with optional flag for completed results
    subprocess.run(["smif", "run", "energy_central", "-d", config_dir], stdout=subprocess.PIPE)
    output = subprocess.run(["smif", "list", "-c", "-d", config_dir], stdout=subprocess.PIPE)
    assert "energy_central *" in str(output.stdout)


def test_fixture_available_results(tmp_sample_project):
    """Test cli for listing available results
    """
    config_dir = tmp_sample_project
    output = subprocess.run(["smif", "available_results", "energy_central", "-d", config_dir],
                            stdout=subprocess.PIPE)

    out_str = str(output.stdout)
    assert(out_str.count('model run: energy_central') == 1)
    assert(out_str.count('sos model: energy') == 1)
    assert(out_str.count('sector model:') == 1)
    assert(out_str.count('output:') == 2)
    assert(out_str.count('output: cost') == 1)
    assert(out_str.count('output: water_demand') == 1)
    assert(out_str.count('no results') == 2)
    assert(out_str.count('decision') == 0)

    # Run energy_central and re-check output with optional flag for completed results
    subprocess.run(["smif", "run", "energy_central", "-d", config_dir], stdout=subprocess.PIPE)
    output = subprocess.run(["smif", "available_results", "energy_central", "-d", config_dir],
                            stdout=subprocess.PIPE)

    out_str = str(output.stdout)
    assert(out_str.count('model run: energy_central') == 1)
    assert(out_str.count('sos model: energy') == 1)
    assert(out_str.count('sector model:') == 1)
    assert(out_str.count('output:') == 2)
    assert(out_str.count('output: cost') == 1)
    assert(out_str.count('output: water_demand') == 1)
    assert(out_str.count('no results') == 0)
    assert(out_str.count('decision') == 8)
    assert(out_str.count('decision 1') == 2)
    assert(out_str.count('decision 2') == 2)
    assert(out_str.count('decision 3') == 2)
    assert(out_str.count('decision 4') == 2)
    assert(out_str.count(': 2010') == 4)
    assert(out_str.count(': 2015') == 2)
    assert(out_str.count(': 2020') == 2)


def test_fixture_missing_results(tmp_sample_project):
    """Test cli for listing missing results
    """
    config_dir = tmp_sample_project
    output = subprocess.run(["smif", "missing_results", "energy_central", "-d", config_dir],
                            stdout=subprocess.PIPE)

    out_str = str(output.stdout)
    assert(out_str.count('model run: energy_central') == 1)
    assert(out_str.count('sos model: energy') == 1)
    assert(out_str.count('sector model:') == 1)
    assert(out_str.count('output:') == 2)
    assert(out_str.count('output: cost') == 1)
    assert(out_str.count('output: water_demand') == 1)
    assert(out_str.count('no missing results') == 0)
    assert(out_str.count('results missing for:') == 2)

    # Run energy_central and re-check output with optional flag for completed results
    subprocess.run(["smif", "run", "energy_central", "-d", config_dir], stdout=subprocess.PIPE)
    output = subprocess.run(["smif", "missing_results", "energy_central", "-d", config_dir],
                            stdout=subprocess.PIPE)

    out_str = str(output.stdout)
    assert(out_str.count('model run: energy_central') == 1)
    assert(out_str.count('sos model: energy') == 1)
    assert(out_str.count('sector model:') == 1)
    assert(out_str.count('output:') == 2)
    assert(out_str.count('output: cost') == 1)
    assert(out_str.count('output: water_demand') == 1)
    assert(out_str.count('no missing results') == 2)
    assert(out_str.count('results missing for:') == 0)


def test_setup_project_folder():
    """Test contents of the setup project folder
    """
    with TemporaryDirectory() as project_folder:
        args = get_args(['setup', '-d', project_folder])
        setup_project_folder(args)

        assert os.path.exists(project_folder)

        folder_list = ['config', 'data', 'models', 'planning']
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
    output = subprocess.run(['smif', 'list', '-vv'], stderr=subprocess.PIPE)
    assert 'DEBUG' in str(output.stderr)


def test_verbose_debug_alt():
    """Expect debug message from `smif --verbose --verbose`
    """
    output = subprocess.run(['smif', 'list', '--verbose', '--verbose'], stderr=subprocess.PIPE)
    assert 'DEBUG' in str(output.stderr)
