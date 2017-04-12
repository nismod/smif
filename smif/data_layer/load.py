# -*- coding: utf-8 -*-
"""Parse yaml config files, to construct sector models
"""
import yaml
from smif import SpaceTimeValue

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load(file_path):
    """Parse yaml config file into plain data (lists, dicts and simple values)

    Parameters
    ----------
    file_path : str
        The path of the configuration file to parse
    """
    with open(file_path, 'r') as file_handle:
        return yaml.load(file_handle, Loader=Loader)


def dump(data, file_path):
    """Write plain data to a file as yaml

    Parameters
    ----------
    file_path : str
        The path of the configuration file to write
    data
        Data to write (should be lists, dicts and simple values)
    """
    with open(file_path, 'w') as file_handle:
        return yaml.dump(data, file_handle, Dumper=Dumper)


def space_time_value_representer(dumper, data):
    """Dump custom yaml representation of SpaceTimeValue
    """
    return dumper.represent_sequence(
        "SpaceTimeValue", [
            data.region,
            data.interval,
            data.value,
            data.units
        ]
    )


yaml.add_representer(SpaceTimeValue, space_time_value_representer, Dumper=Dumper)


def space_time_value_constructor(loader, node):
    """Load ustom yaml representation of SpaceTimeValue
    """
    value = loader.construct_sequence(node)
    return SpaceTimeValue(
        value[0],
        value[1],
        value[2],
        value[3]
    )


yaml.add_constructor("SpaceTimeValue", space_time_value_constructor, Loader=Loader)
