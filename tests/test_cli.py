import os
from tempfile import TemporaryDirectory

from pytest import raises
from smif.cli import parse_arguments, setup_project_folder


def test_parse_arguments():
    """Setup a project folder argument parsing
    """
    with TemporaryDirectory() as project_folder:
        parser = parse_arguments(['water_supply'])
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

        folder_list = ['assets', 'config', 'planning']
        for folder in folder_list:
            folder_path = os.path.join(project_folder, folder)
            print(folder_path)
            assert os.path.exists(folder_path)


def test_run_sector_model():
    """Run a sector model in the list
    """
    model_name = 'water_supply'
    parser = parse_arguments([model_name])
    commands = ['run', model_name]
    args = parser.parse_args(commands)
    expected = model_name
    actual = args.model
    assert actual == expected


def test_dont_run_invalid_sector_model():
    """Don't try to run a sector model which is not in the list
    """
    model_name = 'water_supply'
    parser = parse_arguments([model_name])
    commands = ['run', 'energy_supply']
    with raises(SystemExit):
        parser.parse_args(commands)
