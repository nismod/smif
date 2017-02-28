# -*- coding: utf-8 -*-
"""Read and parse the config files for the system-of-systems model
"""
import logging
import os
from . parse_config import ConfigParser


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
        self._config = self._load_sos_config()
        self.timesteps = self._load_timesteps()
        self.scenario_data = self._load_scenario_data()
        self.sector_model_data = self._load_sector_model_data()
        self.names = self._load_names()
        self.planning = self._load_planning()

    @property
    def data(self):
        """Expose all model configuration data
        """
        return {
            "timesteps": self.timesteps,
            "sector_model_config": self.sector_model_data,
            "scenario_data": self.scenario_data,
            "planning": self.planning,
            "names": self.names
        }

    def _load_sos_config(self):
        """Parse model master config

        - configures run mode
        - points to timesteps file
        - points to shared data files
        - points to sector models and sector model data files
        """
        msg = "Looking for configuration data in {}".format(self._config_file_path)
        self.logger.info(msg)

        config_parser = ConfigParser(self._config_file_path)
        config_parser.validate_as_modelrun_config()
        self.logger.debug(config_parser.data)

        return config_parser.data

    def _load_timesteps(self):
        """Parse model timesteps
        """
        file_path = self._get_path_from_config(self._config['timesteps'])
        os.path.exists(file_path)

        config_parser = ConfigParser(file_path)
        config_parser.validate_as_timesteps()

        return config_parser.data

    def _load_sector_model_data(self):
        """Parse list of sector models to run

        Model details include:
        - model name
        - model config directory
        - SectorModel class name to call
        """
        return self._config['sector_models']


    def _load_scenario_data(self):
        """Load scenario data from list in sos model config

        Working assumptions:
        - scenario data is list of dicts, each like:
            {
                'parameter': 'parameter_name',
                'file': 'relative file path',
                'spatial_resolution': 'national'
                'temporal_resolution': 'annual'
            }
        - data in file is list of dicts, each like:
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
                parser = ConfigParser(file_path)
                scenario_data[data_type["parameter"]] = parser.data

        return scenario_data


    def _load_planning(self):
        """Loads the set of build instructions for planning
        """
        if self._config['planning']['pre_specified']['use']:
            planning_relative_paths = self._config['planning']['pre_specified']['files']
            planning_instructions = []

            for parser in self._parsers_from_relative_paths(planning_relative_paths):
                parser.validate_as_pre_specified_planning()
                planning_instructions.extend(parser.data)

            return planning_instructions
        else:
            return []

    def _load_names(self):
        names = []
        if 'names' in self._config:
            names_relative_paths = self._config['names']

            for parser in self._parsers_from_relative_paths(names_relative_paths):
                parser.validate_as_assets()
                names.extend(parser.data)

        return names

    def _parsers_from_relative_paths(self, paths):
        for rel_path in paths:
            file_path = self._get_path_from_config(rel_path)
            parser = ConfigParser(file_path)
            yield parser

    def _get_path_from_config(self, path):
        """Return an absolute path, given a path provided from a config file

        If the provided path is relative, join it to the config file directory
        """
        if os.path.isabs(path):
            return os.path.normpath(path)
        else:
            return os.path.normpath(os.path.join(self._config_file_dir, path))
