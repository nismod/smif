"""Test command line interface
"""

import os
from tempfile import TemporaryDirectory
from smif.cli import parse_arguments, setup_project_folder


def test_parse_arguments():
    """Setup a project folder argument parsing
    """
    with TemporaryDirectory() as project_folder:
        parser = parse_arguments()
        commands = ['setup', project_folder]
        # Project folder setup here
        args = parser.parse_args(commands)
        expected = project_folder
        actual = args.path
        assert actual == expected

        # Ensure that the `setup_configuration` function is called when `setup`
        # command is passed to the cli
        assert args.func.__name__ == 'setup_configuration'


def test_setup_project_folder():
    """Test contents of the setup project folder
    """
    with TemporaryDirectory() as project_folder:
        setup_project_folder(project_folder)

        assert os.path.exists(project_folder)

        folder_list = ['data']
        for folder in folder_list:
            folder_path = os.path.join(project_folder, folder)
            print(folder_path)
            assert os.path.exists(folder_path)


def test_run_sector_model(setup_folder_structure):
    """Run a sector model in the list
    """
    parser = parse_arguments()
    config_file = os.path.join(str(setup_folder_structure), "model.yaml")
    commands = ['run', '--model', 'water_supply', config_file]
    args = parser.parse_args(commands)
    expected = 'water_supply'
    actual = args.model
    assert actual == expected


def test_dont_run_invalid_sector_model(setup_folder_structure,
                                       setup_config_file):
    """Don't try to run a sector model which is not in the list
    """
    model_name = 'invalid_model_name'
    parser = parse_arguments()
    config_file = os.path.join(str(setup_folder_structure), "model.yaml")

    commands = ['run', '-m', model_name, config_file]
    args = parser.parse_args(commands)
    assert args.model == model_name
    assert args.path == config_file


def test_validation(setup_folder_structure,
                    setup_config_file):
    """Ensure configuration file is valid
    """
    project_folder = str(setup_folder_structure)
    config_file = os.path.join(project_folder, "model.yaml")
    parser = parse_arguments()
    commands = ['validate', config_file]
    # Project folder setup here
    args = parser.parse_args(commands)
    expected = config_file
    actual = args.path
    assert actual == expected
    assert args.func.__name__ == 'validate_config'
