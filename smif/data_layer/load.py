# -*- coding: utf-8 -*-
"""Parse yaml config files, to construct sector models
"""
import yaml


def load(file_path):
    """Parse yaml config file into plain data (lists, dicts and simple values)

    Parameters
    ----------
    file_path : str
        The path of the configuration file to parse
    """
    with open(file_path, 'r') as file_handle:
        return yaml.load(file_handle)
