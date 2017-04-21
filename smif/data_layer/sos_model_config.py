# -*- coding: utf-8 -*-
"""Read and parse the config files for the system-of-systems model
"""

import logging
import os

import fiona

from .load import load
from .validate import (validate_scenario_data,
                       validate_sos_model_config,
                       validate_time_intervals,
                       validate_timesteps)


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

        self._config = {}
        self.timesteps = []
        self.max_iterations = 0
        self.scenario_data = {}
        self.sector_model_data = []
        self.planning = []
        self.time_intervals = []
        self.regions = []

        self.names = []

        self.resolution_mapping = {
            'scenario': {},
            'input': {},
            'output': {}
        }

    def load(self):
        """Load and check all config
        """
        self._config = self.load_sos_config()
        self.timesteps = self.load_timesteps()
        self.max_iterations = self.load_max_iterations()
        self.time_intervals = self.load_time_intervals()
        self.regions = self.load_regions()
        self.scenario_data = self.load_scenario_data()
        self.sector_model_data = self.load_sector_model_data()
        self.planning = self.load_planning()

    @property
    def data(self):
        """Expose all model configuration data

        Returns
        -------
        dict
            Returns a dictionary with the following keys:

            timesteps
                the sequence of years
            max_iterations
                limit iterations for solving interdependencies
            sector_model_config: list
                The list of sector model configuration data
            scenario_data: dict
                A dictionary of scenario data, with the parameter name
                as the key and the data as the value
            planning: list
                A list of dicts of planning instructions
            region_sets: dict
                A dictionary of region set data, with the name as the key
                and the data as the value
            interval_sets: dict
                A dictionary of interval set data, with the name as the key
                and the data as the value
            resolution_mapping: dict
                The mapping of spatial and temporal resolutions to
                inputs, outputs and scenarios
        """
        return {
            "timesteps": self.timesteps,
            "max_iterations": self.max_iterations,
            "sector_model_config": self.sector_model_data,
            "scenario_data": self.scenario_data,
            "planning": self.planning,
            "region_sets": self.regions,
            "interval_sets": self.time_intervals,
            "resolution_mapping": self.resolution_mapping
        }

    def load_sos_config(self):
        """Parse model master config

        - configures run mode
        - sets max iterations for solving interdependencies
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

    def load_max_iterations(self):
        """Parse max_iterations setting
        """
        if "max_iterations" in self._config \
                and isinstance(self._config["max_iterations"], int) \
                and self._config["max_iterations"] > 0:
            self.max_iterations = self._config["max_iterations"]

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

        Returns
        -------
        dict
            A dictionary where keys are parameters names and values are the file contents,
            so a list of dicts

        """
        scenario_data = {}
        if 'scenario_data' in self._config:
            for data_type in self._config['scenario_data']:

                name = data_type['parameter']

                spatial_res = data_type['spatial_resolution']
                temporal_res = data_type['temporal_resolution']

                self.resolution_mapping['scenario'][name] = \
                    {'spatial_resolution': spatial_res,
                     'temporal_resolution': temporal_res}

                file_path = self._get_path_from_config(data_type['file'])
                self.logger.debug("Loading scenario data from %s with %s and %s",
                                  file_path, spatial_res, temporal_res)
                data = load(file_path)
                validate_scenario_data(data, file_path)

                scenario_data[name] = data

        return scenario_data

    def load_planning(self):
        """Loads the set of build instructions for planning

        Returns
        -------
        list
            A list of planning instructions loaded from the planning file
        """
        if self._config['planning']['pre_specified']['use']:
            planning_relative_paths = self._config['planning']['pre_specified']['files']
            planning_instructions = []

            for rel_path in planning_relative_paths:
                file_path = self._get_path_from_config(rel_path)
                if os.path.exists(file_path):
                    data = load(file_path)
                    if isinstance(data, list):
                        planning_instructions.extend(data)
                    else:
                        self.logger.warning("Invalid planning instructions in %s", file_path)
                else:
                    self.logger.warning("Planning file %s not found", file_path)

            return planning_instructions
        else:
            return []

    def _get_path_from_config(self, path):
        """Return an absolute path, given a path provided from a config file

        If the provided path is relative, join it to the config file directory
        """
        if os.path.isabs(path):
            return os.path.normpath(path)
        else:
            return os.path.normpath(os.path.join(self._config_file_dir, path))

    def load_time_intervals(self):
        """Within-year time intervals are specified in ``data/<sectormodel>/time_intervals.yaml``

        These specify the mapping of model timesteps to durations within a year
        (assume modelling 365 days: no extra day in leap years, no leap seconds)

        Each time interval must have
        - start (period since beginning of year)
        - end (period since beginning of year)
        - id (label to use when passing between integration layer and sector model)

        use ISO 8601[1]_ duration format to specify periods::

            P[n]Y[n]M[n]DT[n]H[n]M[n]S

        For example::

            P1Y == 1 year
            P3M == 3 months
            PT168H == 168 hours

        So to specify a period from the beginning of March to the end of May::

            start: P2M
            end: P5M
            id: spring

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations

        """
        time_interval_data = {}
        if 'interval_sets' in self._config:
            for interval_set in self._config['interval_sets']:
                file_path = self._get_path_from_config(interval_set['file'])
                self.logger.debug("Loading time interval data from %s", file_path)
                data = load(file_path)
                validate_time_intervals(data, file_path)
                time_interval_data[interval_set["name"]] = data

        return time_interval_data

    def load_regions(self):
        """Model regions are specified in ``data/<sectormodel>/regions.*``

        The file format must be possible to parse with GDAL, and must contain
        an attribute "name" to use as an identifier for the region.
        """
        region_set_data = {}
        assert isinstance(self._config, dict)
        if 'region_sets' in self._config:
            for region_set in self._config['region_sets']:
                file_path = self._get_path_from_config(region_set['file'])
                self.logger.debug("Loading region set data from %s", file_path)
                data = self._parse_region_data(file_path)
                region_set_data[region_set["name"]] = data

        return region_set_data

    def _parse_region_data(self, path):
        """
        Read regions from GDAL-readable files

        Arguments
        ---------
        path: str
            Path to regions file

        Returns
        -------
        data: list
            A list of Fiona feature collections

        """
        with fiona.drivers():
            with fiona.open(path) as src:
                data = [f for f in src]

        return data
