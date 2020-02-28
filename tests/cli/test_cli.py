"""Test command line interface
"""

import os
import sys
from distutils.dir_util import copy_tree, remove_tree
from itertools import product
from tempfile import TemporaryDirectory
from time import sleep
from unittest.mock import call, patch

import smif
from pytest import fixture, raises
from smif.cli import confirm, main, parse_arguments, setup_project_folder
from smif.exception import SmifDataNotFoundError


@fixture
def tmp_sample_project(tmpdir_factory):
    """Copy sample_project folder to temporary directory, ignoring any results
    """
    dst = str(tmpdir_factory.mktemp("smif"))
    src = os.path.join(os.path.dirname(smif.__file__), 'sample_project')
    copy_tree(src, dst)
    try:
        remove_tree(os.path.join(dst, 'results'))
    except FileNotFoundError:
        pass
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


def test_fixture_single_run(capsys, tmp_sample_project):
    """Test running the (default) binary-filesystem-based single_run fixture
    """
    main(["run", "-d", tmp_sample_project, "energy_central", "-v"])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)
    assert "Running energy_central" in output.err
    assert "Model run 'energy_central' complete" in output.out


def test_fixture_single_run_csv(capsys, tmp_sample_project):
    """Test running the csv-filesystem-based single_run fixture
    """
    main(["run", "-i", "local_csv", "-d", tmp_sample_project, "energy_central",
          "-v"])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)
    assert "Running energy_central" in output.err
    assert "Model run 'energy_central' complete" in output.out


def test_fixture_single_run_warm(capsys, tmp_sample_project):
    """Test running the (default) single_run fixture with warm setting enabled
    """
    main(["run", "-v", "-d", tmp_sample_project, "energy_central"])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)
    assert "Job energy_central_simulate_2010_1_energy_demand" in output.err

    main(["run", "-v", "-w", "-d", tmp_sample_project, "energy_central"])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)
    assert output.err.count("Job energy_central_simulate_2010_1_energy_demand") == 1


def test_fixture_run_step_no_decision(capsys, tmp_sample_project):
    """Test running model at single timestep

    Run:
        smif step -vv energy_water_cp_cr -m energy_demand -t 2010 -dn 0

    """
    with raises(SmifDataNotFoundError) as ex:
        main(["step",  "-d", tmp_sample_project, "energy_water_cp_cr", "-m", "energy_demand",
              "-t", "2010", "-dn", "0"])

    assert "Decision state file not found for timestep 2010, decision 0" in str(ex)


def test_fixture_run_step_after_decision(capsys, tmp_sample_project):
    """Test running model at single timestep

    Run:
        smif decide energy_water_cp_cr -dn 0
        smif step -vv energy_water_cp_cr -m energy_demand -t 2010 -dn 0

    """
    main(["decide",  "-d", tmp_sample_project, "energy_water_cp_cr"])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)

    assert "Got decision bundle" in output.out
    assert "decision iterations [0]" in output.out
    assert "timesteps [2010, 2015, 2020]" in output.out
    output_len = len(output.out)

    main(["step",  "-d", tmp_sample_project, "energy_water_cp_cr", "-m", "energy_demand",
          "-t", "2010", "-dn", "0"])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)

    assert len(output.out) == output_len + 1  # one extra char for newline
    assert "\n" == output.err


def test_dry_run(capsys, tmp_sample_project):
    """Test dry run full model
    """
    main(["run", "-d", tmp_sample_project, "energy_water_cp_cr", "-n"])
    out, err = capsys.readouterr()
    print(out)
    print(err, file=sys.stderr)

    assert "smif decide energy_water_cp_cr" in out
    assert "smif before_step energy_water_cp_cr --model energy_demand" in out
    assert "smif step energy_water_cp_cr --model energy_demand --timestep 2020 --decision 0" \
        in out
    assert "smif step energy_water_cp_cr --model energy_demand --timestep 2015 --decision 0" \
        in out
    assert "smif step energy_water_cp_cr --model energy_demand --timestep 2010 --decision 0" \
        in out
    assert "smif before_step energy_water_cp_cr --model water_supply" in out
    assert "smif step energy_water_cp_cr --model water_supply --timestep 2010 --decision 0" \
        in out
    assert "smif step energy_water_cp_cr --model water_supply --timestep 2015 --decision 0" \
        in out
    assert "smif step energy_water_cp_cr --model water_supply --timestep 2020 --decision 0" \
        in out


def test_fixture_batch_run(capsys, tmp_sample_project):
    """Test running the multiple modelruns using the batch_run option
    """
    main(["run", "-v", "-b", "-d", tmp_sample_project,
          os.path.join(tmp_sample_project, "batchfile")])
    output = capsys.readouterr()
    print(output.out)
    print(output.err, file=sys.stderr)

    assert "Running energy_water_cp_cr" in output.err
    assert "Model run 'energy_water_cp_cr' complete" in output.out
    assert "Running energy_central" in output.err
    assert "Model run 'energy_central' complete" in output.out


def test_fixture_list_runs(capsys, tmp_sample_project):
    """Test running the filesystem-based single_run fixture
    """
    main(["list", "-d", tmp_sample_project])
    output = capsys.readouterr()

    assert "energy_water_cp_cr" in output.out
    assert "energy_central" in output.out

    # Run energy_central and re-check output with optional flag for completed results
    main(["run", "energy_central", "-d", tmp_sample_project])
    main(["list", "-c", "-d", tmp_sample_project])
    output = capsys.readouterr()

    assert "energy_central *" in output.out


def test_fixture_available_results(capsys, tmp_sample_project):
    """Test cli for listing available results
    """
    main(["available_results", "energy_central", "-d", tmp_sample_project])
    output = capsys.readouterr()

    out_str = output.out
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no results') == 2
    assert out_str.count('decision') == 0

    # Run energy_central and re-check output with optional flag for completed results
    main(["run", "energy_central", "-d", tmp_sample_project])
    main(["available_results", "energy_central", "-d", tmp_sample_project])
    output = capsys.readouterr()

    out_str = output.out
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


def test_fixture_missing_results(capsys, tmp_sample_project):
    """Test cli for listing missing results
    """
    main(["missing_results", "energy_central", "-d", tmp_sample_project])
    output = capsys.readouterr()

    out_str = output.out
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no missing results') == 0
    assert out_str.count('results missing for:') == 2

    # Run energy_central and re-check output with optional flag for completed results
    main(["run", "energy_central", "-d", tmp_sample_project])
    main(["missing_results", "energy_central", "-d", tmp_sample_project])
    output = capsys.readouterr()

    out_str = output.out
    assert out_str.count('model run: energy_central') == 1
    assert out_str.count('sos model: energy') == 1
    assert out_str.count('sector model:') == 1
    assert out_str.count('output:') == 2
    assert out_str.count('output: cost') == 1
    assert out_str.count('output: water_demand') == 1
    assert out_str.count('no missing results') == 2
    assert out_str.count('results missing for:') == 0


def test_fixture_prepare_model_runs(capsys, tmp_sample_project):
    """Test cli for preparing model runs from template
    referencing scenario with 1 or more variants
    """
    pop_variants = ['low', 'med', 'high']
    nb_variants = len(pop_variants)

    clear_model_runs(tmp_sample_project)

    main(["prepare-run", "population", "energy_central", "-d", tmp_sample_project])

    for suffix in pop_variants:
        filename = 'energy_central_population_' + suffix + '.yml'
        assert os.path.isfile(os.path.join(tmp_sample_project, 'config/model_runs', filename))

    variant_range = range(0, nb_variants)
    for s, e in product(variant_range, variant_range):
        if e < s:
            # skip variant ranges where end is less than start
            continue
        clear_model_runs(tmp_sample_project)
        main(["prepare-run", "population", "energy_central", "-s", str(s), "-e", str(e),
              "-d", tmp_sample_project])
        for suffix in pop_variants[s:e + 1]:
            filename = 'energy_central_population_' + suffix + '.yml'
            assert os.path.isfile(
                os.path.join(tmp_sample_project, 'config/model_runs', filename))
        for suffix in pop_variants[0:s]:
            filename = 'energy_central_population_' + suffix + '.yml'
            assert not os.path.isfile(
                os.path.join(tmp_sample_project, 'config/model_runs', filename))
        if e < variant_range[-1]:
            for suffix in pop_variants[e + 1:]:
                filename = 'energy_central_population_' + suffix + '.yml'
                assert not os.path.isfile(
                    os.path.join(tmp_sample_project, 'config/model_runs', filename))


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
    # clean up
    # r=root, d=directories, f = files
    path = os.path.join(tmp_sample_project, 'data')
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

    main(["prepare-convert", "energy_central", "-d", tmp_sample_project, "-i", "local_csv"])
    # assert that correct files have been generated
    for folder in list_of_files.keys():
        for filename in list_of_files[folder]:
            path = os.path.join(tmp_sample_project, 'data', folder, filename)
            path = "{}.parquet".format(path)
            assert os.path.isfile(path)

    sleep(2)

    # Now call prepare-convert with the --noclobber option
    # all previously generated parquet files should not be modified
    main(["prepare-convert", "energy_central", "--noclobber", "-d", tmp_sample_project,
          "-i", "local_csv"])
    # assert that files have not been modified
    for folder in list_of_files.keys():
        for filename in list_of_files[folder]:
            path = os.path.join(tmp_sample_project, 'data', folder, filename)
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


@patch('sys.exit')
def test_help(_, capsys):
    """Expect help from `smif` or `smif -h`
    """
    msg = "Command line tools for smif"
    main([])
    output = capsys.readouterr()
    assert msg in output.out

    main(['-h'])
    output = capsys.readouterr()
    assert msg in output.out


@patch('sys.exit')
def test_version_display(_, capsys):
    """Expect version number from `smif -V`
    """
    main(['-V'])
    output = capsys.readouterr()
    assert smif.__version__ in output.out


def test_verbose_debug(capsys, tmp_sample_project):
    """Expect debug message from `smif -vv`
    """
    main(['list', '-vv', '-d', tmp_sample_project])
    output = capsys.readouterr()
    assert 'DEBUG' in output.err


def test_verbose_debug_alt(capsys, tmp_sample_project):
    """Expect debug message from `smif --verbose --verbose`
    """
    main(['list', '--verbose', '--verbose', '-d', tmp_sample_project])
    output = capsys.readouterr()
    assert 'DEBUG' in output.err
