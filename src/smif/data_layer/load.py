# -*- coding: utf-8 -*-
"""Parse yaml config files, to construct sector models
"""
from ruamel.yaml import YAML


def load(file_path):
    """Parse yaml config file into plain data (lists, dicts and simple values)

    Parameters
    ----------
    file_path : str
        The path of the configuration file to parse
    """
    with open(file_path, 'r') as file_handle:
        return YAML().load(file_handle)


def dump(data, file_path):
    """Write plain data to a file as yaml

    Parameters
    ----------
    data
        Data to write (should be lists, dicts and simple values)
    file_path : str
        The path of the configuration file to write
    """
    with open(file_path, 'w') as file_handle:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.allow_unicode = True
        return yaml.dump(data, file_handle)
