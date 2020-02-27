"""Test command line interface
"""

import os
import shutil
import subprocess
import sys
from itertools import product
from tempfile import TemporaryDirectory
from time import sleep
from unittest.mock import call, patch

import smif
from pytest import fixture
from smif.cli import confirm, parse_arguments, setup_project_folder


@fixture
def tmp_sample_project(tmpdir_factory):
    # copy sample_project folder to temporary directory, ignoring results
    dst = tmpdir_factory.mktemp("smif")
    src = os.path.join(os.path.dirname(smif.__file__), 'sample_project')
    shutil.copytree(src, dst, ignore=lambda _dir, _contents: ['results'], dirs_exist_ok=True)
    return dst


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
    output = subprocess.run(
        ["smif", "run", "-d", config_dir, "energy_central", "-v"],
        capture_output=True)
    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert "Running energy_central" in str(output.stderr)
    assert "Model run 'energy_central' complete" in str(output.stdout)


def test_fixture_single_run_csv(tmp_sample_project):
    """Test running the csv-filesystem-based single_run fixture
    """
    output = subprocess.run(
        ["smif", "run", "-i", "local_csv", "-d", tmp_sample_project, "energy_central", "-v"],
        capture_output=True)
    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert "Running energy_central" in str(output.stderr)
    assert "Model run 'energy_central' complete" in str(output.stdout)


def test_fixture_single_run_warm(tmp_sample_project):
    """Test running the (default) single_run fixture with warm setting enabled
    """
    config_dir = tmp_sample_project
    cold_output = subprocess.run(
        ["smif", "run", "-v", "-d", config_dir, "energy_central"],
        capture_output=True)
    print(cold_output.stdout.decode("utf-8"))
    print(cold_output.stderr.decode("utf-8"), file=sys.stderr)

    warm_output = subprocess.run(
        ["smif", "run", "-v", "-w", "-d", config_dir, "energy_central"],
        capture_output=True)
    print(warm_output.stdout.decode("utf-8"))
    print(warm_output.stderr.decode("utf-8"), file=sys.stderr)

    assert "Job energy_central_simulate_2010_1_energy_demand" in str(cold_output.stderr)
    assert "Job energy_central_simulate_2010_1_energy_demand" not in str(warm_output.stderr)


def test_fixture_run_step_no_decision(tmp_sample_project):
    """Test running model at single timestep

    Run:
        smif step -vv energy_water_cp_cr -m energy_demand -t 2010 -dn 0

    """
    output = subprocess.run(
        ["smif", "step",  "-d", tmp_sample_project, "energy_water_cp_cr", "-m",
         "energy_demand", "-t", "2010", "-dn", "0"],
        capture_output=True)

    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)
    assert output.returncode == 1
    assert "Decision state file not found for timestep 2010, decision 0" in \
        str(output.stderr)


def test_fixture_run_step_after_decision(tmp_sample_project):
    """Test running model at single timestep

    Run:
        smif decide energy_water_cp_cr -dn 0
        smif step -vv energy_water_cp_cr -m energy_demand -t 2010 -dn 0

    """
    output = subprocess.run(
        ["smif", "decide",  "-d", tmp_sample_project, "energy_water_cp_cr"],
        capture_output=True)

    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)

    assert output.returncode == 0
    assert "Got decision bundle" in str(output.stdout)
    assert "decision iterations [0]" in str(output.stdout)
    assert "timesteps [2010, 2015, 2020]" in str(output.stdout)

    output = subprocess.run(
        ["smif", "step",  "-d", tmp_sample_project, "energy_water_cp_cr", "-m",
         "energy_demand", "-t", "2010", "-dn", "0"],
        capture_output=True)

    print(output.stdout.decode("utf-8"))
    print(output.stderr.decode("utf-8"), file=sys.stderr)

    assert output.returncode == 0
    assert "" == str(output.stdout.decode("utf-8"))
    assert "" == str(output.stderr.decode("utf-8"))


def test_fixture_batch_run(tmp_sample_project):
    """Test running the multiple modelruns using the batch_run option
    """
    config_dir = tmp_sample_project
    output = subprocess.run(
        ["smif", "run", "-v", "-b", "-d", config_dir, os.path.join(config_dir, "batchfile")],
        capture_output=True)

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
    output = subprocess.run(
        ["smif", "list", "-d", config_dir],
        capture_output=True)

    assert "energy_water_cp_cr" in str(output.stdout)
    assert "energy_central" in str(output.stdout)

    # Run energy_central and re-check output with optional flag for completed results
    subprocess.run(["smif", "run", "energy_central", "-d", config_dir])
    output = subprocess.run(
        ["smif", "list", "-c", "-d", config_dir],
        capture_output=True)
    assert "energy_central *" in str(output.stdout)


def test_fixture_available_results(tmp_sample_project):
    """Test cli for listing available results
    """
    config_dir = tmp_sample_project
    output = subprocess.run(
        ["smif", "available_results", "energy_central", "-d", config_dir],
        capture_output=True)

    out_str = str(output.stdout)
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no results') == 2
    assert out_str.count('decision') == 0

    # Run energy_central and re-check output with optional flag for completed results
    subprocess.run(["smif", "run", "energy_central", "-d", config_dir])
    output = subprocess.run(
        ["smif", "available_results", "energy_central", "-d", config_dir],
        capture_output=True)

    out_str = str(output.stdout)
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no results') == 0
    assert out_str.count('decision') == 8
    assert out_str.count('decision 1') == 2
    assert out_str.count('decision 2') == 2
    assert out_str.count('decision 3') == 2
    assert out_str.count('decision 4') == 2
    assert out_str.count(': 2010') == 4
    assert out_str.count(': 2015') == 2
    assert out_str.count(': 2020') == 2


def test_fixture_missing_results(tmp_sample_project):
    """Test cli for listing missing results
    """
    config_dir = tmp_sample_project
    output = subprocess.run(
        ["smif", "missing_results", "energy_central", "-d", config_dir],
        capture_output=True)

    out_str = str(output.stdout)
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no missing results') == 0
    assert out_str.count('results missing for:') == 2

    # Run energy_central and re-check output with optional flag for completed results
    subprocess.run(["smif", "run", "energy_central", "-d", config_dir])
    output = subprocess.run(
        ["smif", "missing_results", "energy_central", "-d", config_dir],
        capture_output=True)

    out_str = str(output.stdout)
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no missing results') == 2
    assert out_str.count('results missing for:') == 0


def test_fixture_prepare_model_runs(tmp_sample_project):
    """Test cli for preparing model runs from template
    referencing scenario with 1 or more variants
    """
    config_dir = tmp_sample_project
    pop_variants = ['low', 'med', 'high']
    nb_variants = len(pop_variants)

    clear_model_runs(config_dir)

    subprocess.run(["smif", "prepare-run", "population", "energy_central", "-d", config_dir])

    for suffix in pop_variants:
        filename = 'energy_central_population_' + suffix + '.yml'
        assert os.path.isfile(os.path.join(config_dir, 'config/model_runs', filename))

    variant_range = range(0, nb_variants)
    for s, e in product(variant_range, variant_range):
        clear_model_runs(config_dir)
        subprocess.run(
            ["smif", "prepare-run", "population", "energy_central", "-s", str(s), "-e", str(e),
             "-d", config_dir])
        for suffix in pop_variants[s:e + 1]:
            filename = 'energy_central_population_' + suffix + '.yml'
            assert os.path.isfile(os.path.join(config_dir, 'config/model_runs', filename))
        for suffix in pop_variants[0:s]:
            filename = 'energy_central_population_' + suffix + '.yml'
            assert not os.path.isfile(os.path.join(config_dir, 'config/model_runs', filename))
        if e < variant_range[-1]:
            for suffix in pop_variants[e + 1:]:
                filename = 'energy_central_population_' + suffix + '.yml'
                assert not os.path.isfile(
                    os.path.join(config_dir, 'config/model_runs', filename))


def clear_model_runs(config_dir):
    """ Helper function for test function
        test_fixture_prepare_model_runs
    """
    for suffix in ['low', 'med', 'high']:
        filename = 'energy_central_population_' + suffix + '.yml'
        if os.path.isfile(os.path.join(config_dir, 'config/model_runs', filename)):
            os.remove(os.path.join(config_dir, 'config/model_runs', filename))


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


def test_prepare_convert(tmp_sample_project):
    project_folder = tmp_sample_project
    print(project_folder)
    # clean up
    # r=root, d=directories, f = files
    path = os.path.join(project_folder, 'data')
    for r, d, f in os.walk(path):
        for filename in f:
            if '.parquet' in filename:
                os.remove(os.path.join(r, filename))

    list_of_files = {
        'initial_conditions': [],
        'interventions': ['energy_supply', 'energy_supply_alt'],
        'narratives': [],
        'parameters': ['defaults'],
        'scenarios': ['population_density_low', 'population_density_med',
                      'population_density_high', 'population_low', 'population_med',
                      'population_high'],
        'strategies': ['build_nuke'],
        }

    subprocess.run(
        ["smif", "prepare-convert", "energy_central", "-d", project_folder, "-i", "local_csv"])
    # assert that correct files have been generated
    for folder in list_of_files.keys():
        for filename in list_of_files[folder]:
            path = os.path.join(project_folder, 'data', folder, filename)
            path = "{}.parquet".format(path)
            assert os.path.isfile(path)

    sleep(2)

    # Now call prepare-convert with the --noclobber option
    # all previously generated parquet files should not be modified
    subprocess.run(
        ["smif", "prepare-convert", "energy_central", "--noclobber", "-d", project_folder,
         "-i", "local_csv"])
    # assert that files have not been modified
    for folder in list_of_files.keys():
        for filename in list_of_files[folder]:
            path = os.path.join(project_folder, 'data', folder, filename)
            path = "{}.parquet".format(path)
            assert (os.path.getmtime(path) > 2)


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


def test_help():
    """Expect help from `smif` or `smif -h`
    """
    msg = "Command line tools for smif"
    output = subprocess.run(['smif'], stdout=subprocess.PIPE)
    assert msg in str(output.stdout)

    output = subprocess.run(['smif', '-h'], stdout=subprocess.PIPE)
    assert msg in str(output.stdout)


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
