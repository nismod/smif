from tempfile import TemporaryDirectory

from pytest import raises
from smif.cli import parse_arguments


def test_parse_arguments():
    """Setup a project folder
    """
    project_folder = TemporaryDirectory().name
    parser = parse_arguments(['water_supply'])
    commands = ['setup', project_folder]
    args = parser.parse_args(commands)
    expected = project_folder
    actual = args.path
    assert actual == expected


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
