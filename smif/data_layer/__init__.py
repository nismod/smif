# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""
from abc import ABCMeta, abstractmethod
import yaml
import os


class DataInterface(metaclass=ABCMeta):

    @abstractmethod
    def read_sos_model_runs(self):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, model_run):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_models(self):
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
        self.config_path = config_path

    def read_sos_model_runs(self):
        raise NotImplementedError()

    def write_sos_model_run(self, sos_model_run):
        """Write sos_model_run dictionary to Yaml file
        Existing configuration will be overwritten without warning

        Arguments
        ---------
        name : sos_model_run
            Dictionary containing sos_model_run configuration
        """
        filename = sos_model_run['name'] + '.yml'
        filepath = str(self.config_path.join('sos_model_runs'))
        with open(os.path.join(filepath, filename), 'w') as outfile:
            yaml.dump(sos_model_run, outfile, default_flow_style=False)

    def read_sos_models(self):
        raise NotImplementedError()

    def write_sos_model(self, sos_model):
        """Write sos_model dictionary to Yaml file
        Existing configuration will be overwritten without warning

        Arguments
        ---------
        name : sos_model
            Dictionary containing sos_model configuration
        """
        filename = sos_model['name'] + '.yml'
        filepath = str(self.config_path.join('sos_models'))
        with open(os.path.join(filepath, filename), 'w') as outfile:
            yaml.dump(sos_model, outfile, default_flow_style=False)

    def read_sector_models(self):
        raise NotImplementedError()

    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    def write_sector_model(self, sector_model):
        """Write sos_model dictionary to Yaml file
        Existing configuration will be overwritten without warning

        Arguments
        ---------
        name : sector_model
            Dictionary containing sector_model configuration
        """
        filename = sector_model['name'] + '.yml'
        filepath = str(self.config_path.join('sector_models'))
        with open(os.path.join(filepath, filename), 'w') as outfile:
            yaml.dump(sector_model, outfile, default_flow_style=False)

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
