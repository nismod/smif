# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""
from abc import ABCMeta, abstractmethod
import os
import yaml


class DataInterface(metaclass=ABCMeta):

    @abstractmethod
    def read_sos_model_runs(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model_run(self, sos_model_run_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_models(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model(self, sos_model_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    @abstractmethod
    def read_sector_models(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sector_model(self, sector_model):
        raise NotImplementedError()

    @abstractmethod
    def read_region_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_region_set_data(self, region_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_set_data(self, interval_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_units(self):
        raise NotImplementedError()

    @abstractmethod
    def write_region_set(self, data):
        raise NotImplementedError()

    @abstractmethod
    def write_interval_set(self, data):
        raise NotImplementedError()

    @abstractmethod
    def write_units(self, data):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_scenarios(self, scenario_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_data(self, scenario_name):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario_data(self, scenario_name, data):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_narratives(self, narrative_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative(self, narrative):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_data(self, narrative_set_name, data):
        raise NotImplementedError()


class YamlInterface(DataInterface):
    """ Read and write interface to YAML configuration files
    """
    def __init__(self, config_path):
        """Initialize file paths
        """
        config_path = str(config_path)
        self.filepath = {
            'sos_model_runs': os.path.join(config_path, 'sos_model_runs'),
            'sos_models': os.path.join(config_path, 'sos_models'),
            'sector_models': os.path.join(config_path, 'sector_models')
        }

    def read_sos_model_runs(self):
        """Returns a list of excisting sos_model_runs
        """
        return self._read_yaml_files(self.filepath['sos_model_runs'])

    def read_sos_model_run(self, sos_model_run_name):
        """Read a sos_model_run dictionary from a Yaml file
        raises an exception when the file does not excists

        Arguments
        ---------
        name : sos_model_run_name
            String containing sos_model_run['name']
        """
        return self._read_yaml_file(self.filepath['sos_model_runs'],
                                    sos_model_run_name)

    def write_sos_model_run(self, sos_model_run):
        """Write sos_model_run dictionary to Yaml file
        Existing configuration will be overwritten without warning

        Arguments
        ---------
        name : sos_model_run
            Dictionary containing sos_model_run
        """
        self._write_yaml_file(self.filepath['sos_model_runs'],
                              sos_model_run['name'], sos_model_run)

    def read_sos_models(self):
        """Returns a list of excisting sos_models
        """
        return self._read_yaml_files(self.filepath['sos_models'])

    def read_sos_model(self, sos_model_name):
        """Read a sos_model dictionary from a Yaml file
        raises an exception when the file does not excists

        Arguments
        ---------
        name : sos_model_name
            String containing sos_model['name']
        """
        return self._read_yaml_file(self.filepath['sos_models'], sos_model_name)

    def write_sos_model(self, sos_model):
        """Write sos_model dictionary to Yaml file
        Existing configuration will be overwritten without warning

        Arguments
        ---------
        name : sos_model
            Dictionary containing sos_model
        """
        self._write_yaml_file(self.filepath['sos_models'], sos_model['name'], sos_model)

    def read_sector_models(self):
        """Returns a list of excisting sector_models
        """
        return self._read_yaml_files(self.filepath['sector_models'])

    def read_sector_model(self, sector_model_name):
        """Read a sector_model dictionary from a Yaml file
        raises an exception when the file does not excists

        Arguments
        ---------
        name : sector_model_name
            String containing sector_model['name']
        """
        return self._read_yaml_file(self.filepath['sector_models'], sector_model_name)

    def write_sector_model(self, sector_model):
        """Write sos_model dictionary to Yaml file
        Existing configuration will be overwritten without warning

        Arguments
        ---------
        name : sector_model
            Dictionary containing sector_model
        """
        self._write_yaml_file(self.filepath['sector_models'], sector_model['name'],
                              sector_model)

    def read_region_sets(self):
        raise NotImplementedError()

    def read_region_set_data(self, region_set_name):
        raise NotImplementedError()

    def read_interval_sets(self):
        raise NotImplementedError()

    def read_interval_set_data(self, interval_set_name):
        raise NotImplementedError()

    def read_units(self):
        raise NotImplementedError()

    def write_region_set(self, data):
        raise NotImplementedError()

    def write_interval_set(self, data):
        raise NotImplementedError()

    def write_units(self, data):
        raise NotImplementedError()

    def read_scenario_sets(self):
        raise NotImplementedError()

    def read_scenarios(self, scenario_set_name):
        raise NotImplementedError()

    def read_scenario_data(self, scenario_name):
        raise NotImplementedError()

    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    def write_scenario(self, scenario):
        raise NotImplementedError()

    def write_scenario_data(self, scenario_name, data):
        raise NotImplementedError()

    def read_narrative_sets(self):
        raise NotImplementedError()

    def read_narratives(self, narrative_set_name):
        raise NotImplementedError()

    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    def write_narrative(self, narrative):
        raise NotImplementedError()

    def write_narrative_data(self, narrative_set_name, data):
        raise NotImplementedError()

    def _read_yaml_files(self, path):
        """Returns the name of the Yaml files in a certain directory

        Arguments
        ---------
        path : string
            Path to directory
        """
        files = list()
        for file in os.listdir(path):
            if file.endswith('.yml'):
                files.append(os.path.splitext(file)[0])
        return files

    def _read_yaml_file(self, path, filename):
        """Returns the contents of a Yaml file in a Dict

        Arguments
        ---------
        path : string
            Path to directory
        name : string
            Name of file
        """
        filename = filename + '.yml'
        with open(os.path.join(path, filename), 'r') as stream:
            return yaml.load(stream)

    def _write_yaml_file(self, path, filename, contents):
        """Writes a Dict to a Yaml file

        Arguments
        ---------
        path : string
            Path to directory
        name : string
            Name of file
        contents: dics
            Contents to be written to the file
        """
        filename = filename + '.yml'
        with open(os.path.join(path, filename), 'w') as outfile:
            yaml.dump(contents, outfile, default_flow_style=False)
