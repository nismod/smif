# -*- coding: utf-8 -*-
"""Read and parse the config for sector models
"""
import logging
import os
from glob import glob
from . parse_config import ConfigParser


class SectorModelReader(object):
    """Parses the configuration and input data for a sector model

    Arguments
    =========
    model_name : str
        The name of the model
    model_path : str
        The path to the python module file that contains an implementation of SectorModel
    model_classname : str
        The name of the class that implements SectorModel
    model_config_dir : str
        The root path of model config/data to use

    """
    def __init__(self, model_name, model_path, model_classname, model_config_dir):
        self.model_name = model_name
        self.model_path = model_path
        self.model_classname = model_classname
        self.model_config_dir = model_config_dir

        self.inputs = None
        self.outputs = None
        self.asset_types = None

    def load(self):
        """Load and check all config
        """
        self.inputs = self._load_inputs()
        self.outputs = self._load_outputs()

    @property
    def data(self):
        """Expose all loaded config data
        """
        return {
            "name": self.model_name,
            "path": self.model_path,
            "classname": self.model_classname,
            "inputs": self.inputs,
            "outputs": self.outputs
        }

    def _load_inputs(self):
        """Input spec is located in the ``data/<sectormodel>/inputs.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'inputs.yaml')
        return ConfigParser(path).data

    def _load_outputs(self):
        """Output spec is located in ``data/<sectormodel>/output.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'outputs.yaml')
        return ConfigParser(path).data
