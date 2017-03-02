# -*- coding: utf-8 -*-
"""Read and parse the config files for the system-of-systems model
"""
import logging
import os

from .load import load
from .validate import validate_sos_model_config, validate_timesteps


class SosModelReader(object):
    """Encapsulates the parsing of the system-of-systems configuration

    Parameters
    ----------
    config_file_path : str
        A path to the master config file

    """
    def __init__(self, config_file_path):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Getting config file from %s", config_file_path)

        self._config_file_path = config_file_path
        self._config_file_dir = os.path.dirname(config_file_path)

        self._config = None
        self.timesteps = None
        self.scenario_data = None
        self.sector_model_data = None
        self.planning = None

        self.names = None

    def load(self):
        """Load and check all config
        """
        self._config = self.load_sos_config()
        self.timesteps = self.load_timesteps()
        self.scenario_data = self.load_scenario_data()
        self.sector_model_data = self.load_sector_model_data()
        self.planning = self.load_planning()

    @property
    def data(self):
        """Expose all model configuration data
        """
        return {
            "timesteps": self.timesteps,
            "sector_model_config": self.sector_model_data,
            "scenario_data": self.scenario_data,
            "planning": self.planning
        }

    def load_sos_config(self):
        """Parse model master config

        - configures run mode
        - points to timesteps file
        - points to shared data files
        - points to sector models and sector model data files
        """
        msg = "Looking for configuration data in {}".format(self._config_file_path)
        self.logger.info(msg)

        data = load(self._config_file_path)
        validate_sos_model_config(data)
        return data

    def load_timesteps(self):
        """Parse model timesteps
        """
        file_path = self._get_path_from_config(self._config['timesteps'])
        os.path.exists(file_path)

        data = load(file_path)
        validate_timesteps(data, file_path)

        return data

    def load_sector_model_data(self):
        """Parse list of sector models to run

        Model details include:
        - model name
        - model config directory
        - SectorModel class name to call
        """
        return self._config['sector_models']

    def load_scenario_data(self):
        """Load scenario data from list in sos model config

        Working assumptions:

        - scenario data is list of dicts, each like::

            {
                'parameter': 'parameter_name',
                'file': 'relative file path',
                'spatial_resolution': 'national'
                'temporal_resolution': 'annual'
            }

        - data in file is list of dicts, each like::

            {
                'value': 100,
                'units': 'kg',
                # optional, depending on parameter type:
                'region': 'UK',
                'year': 2015
            }
        """
        scenario_data = {}
        if 'scenario_data' in self._config:
            for data_type in self._config['scenario_data']:
                file_path = self._get_path_from_config(data_type['file'])
                self.logger.debug("Loading scenario data from %s", file_path)
                data = load(file_path)
                scenario_data[data_type["parameter"]] = data

        return scenario_data

    def load_planning(self):
        """Loads the set of build instructions for planning
        """
        if self._config['planning']['pre_specified']['use']:
            planning_relative_paths = self._config['planning']['pre_specified']['files']
            planning_instructions = []

            for data in self._data_from_relative_paths(planning_relative_paths):
                planning_instructions.extend(data)

            return planning_instructions
        else:
            return []

    def _data_from_relative_paths(self, paths):
        for rel_path in paths:
            file_path = self._get_path_from_config(rel_path)
            yield load(file_path)

    def _get_path_from_config(self, path):
        """Return an absolute path, given a path provided from a config file

        If the provided path is relative, join it to the config file directory
        """
        if os.path.isabs(path):
            return os.path.normpath(path)
        else:
            return os.path.normpath(os.path.join(self._config_file_dir, path))
