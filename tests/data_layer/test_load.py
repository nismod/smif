"""Test custom pieces of YAML input/output
"""
from smif import SpaceTimeValue
from smif.data_layer.load import dump, load


def test_dump_space_time_value(setup_folder_structure):
    """Test custom YAML representation
    """
    test_file = str(setup_folder_structure.join("test.yaml"))
    data = SpaceTimeValue("Middlesborough", 0, 33, "GW")
    dump(data, test_file)
    with open(test_file, 'r') as fh:
        line = fh.readline()
        assert line.strip() == "!<SpaceTimeValue> [Middlesborough, 0, 33, GW]"


def test_load_space_time_value(setup_folder_structure):
    test_file = str(setup_folder_structure.join("test.yaml"))
    with open(test_file, 'w') as fh:
        fh.write("!<SpaceTimeValue> [Middlesborough, 0, 33, GW]")
    data = load(test_file)
    assert data == SpaceTimeValue("Middlesborough", 0, 33, "GW")
