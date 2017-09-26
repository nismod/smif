# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""
from abc import ABCMeta, abstractmethod
import os
from smif.data_layer.load import load, dump


class DataInterface(metaclass=ABCMeta):

    @abstractmethod
    def read_sos_model_runs(self):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, sos_model_run):
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


class DatafileInterface(DataInterface):
    """Read and write interface to YAML / CSV configuration files

    Arguments
    ---------
    config_path : str
        The path to the configuration folder
    """
    def __init__(self, config_path):
        self.filepath = {
            'sos_model_runs': os.path.join(config_path, 'sos_model_runs'),
            'sos_models': os.path.join(config_path, 'sos_models'),
            'sector_models': os.path.join(config_path, 'sector_models')
        }

    def read_sos_model_runs(self):
        """Read all system-of-system model runs from Yaml files

        Returns
        -------
        list
            A list of sos_model_run dicts
        """
        sos_model_runs = []

        sos_model_run_names = self._read_yaml_filenames_in_dir(self.filepath['sos_model_runs'])
        for sos_model_run_name in sos_model_run_names:
            sos_model_runs.append(self._read_yaml_file(self.filepath['sos_model_runs'],
                                                       sos_model_run_name))

        return sos_model_runs

    def write_sos_model_run(self, sos_model_run):
        """Write system-of-system model run to Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sos_model_run : dict
            A sos_model_run dictionary
        """
        self._write_yaml_file(self.filepath['sos_model_runs'],
                              sos_model_run['name'], sos_model_run)

    def read_sos_models(self):
        """Read all system-of-system models from Yaml files

        Returns
        -------
        list
            A list of sos_models dicts
        """
        sos_models = []

        sos_model_names = self._read_yaml_filenames_in_dir(self.filepath['sos_models'])
        for sos_model_name in sos_model_names:
            sos_models.append(self._read_yaml_file(self.filepath['sos_models'],
                                                   sos_model_name))
        return sos_models

    def write_sos_model(self, sos_model):
        """Write system-of-system model to Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sos_model : dict
            A sos_model dictionary
        """
        self._write_yaml_file(self.filepath['sos_models'], sos_model['name'], sos_model)

    def read_sector_models(self):
        """Read all sector models from Yaml files

        Returns
        -------
        list
            A list of sector_model dicts
        """
        return self._read_yaml_filenames_in_dir(self.filepath['sector_models'])

    def read_sector_model(self, sector_model_name):
        """Read a sector model from a Yaml file

        Raises an exception when the file does not exists

        Arguments
        ---------
        sector_model_name : str
            Name of the sector_model (sector_model['name'])
        """
        return self._read_yaml_file(self.filepath['sector_models'], sector_model_name)

    def write_sector_model(self, sector_model):
        """Write sector model to a Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sector_model : dict
            A sector_model dictionary
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

    def _read_yaml_filenames_in_dir(self, path):
        """Returns the name of the Yaml files in a certain directory

        Arguments
        ---------
        path : str
            Path to directory

        Returns
        -------
        list
            The list of Yaml files in `path`
        """
        files = []
        for filename in os.listdir(path):
            if filename.endswith('.yml'):
                files.append(os.path.splitext(filename)[0])
        return files

    def _read_yaml_file(self, path, filename):
        """
        Arguments
        ---------
        path : str
            Path to directory
        name : str
            Name of file

        Returns
        -------
        dict
            The contents of the Yaml file `name` in `path`
        """
        filename = filename + '.yml'
        filepath = os.path.join(path, filename)
        return load(filepath)

    def _write_yaml_file(self, path, filename, contents):
        """Writes a Dict to a Yaml file

        Arguments
        ---------
        path : str
            Path to directory
        name : str
            Name of file
        contents: dict
            Contents to be written to the file
        """
        filename = filename + '.yml'
        filepath = os.path.join(path, filename)
        dump(contents, filepath)


class DatabaseInterface(DataInterface):
    """ Read and write interface to Database
    """
    def __init__(self, config_path):
        raise NotImplementedError()

    def read_sos_model_runs(self):
        raise NotImplementedError()

    def write_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    def read_sos_models(self):
        raise NotImplementedError()

    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    def read_sector_models(self):
        raise NotImplementedError()

    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    def write_sector_model(self, sector_model):
        raise NotImplementedError()

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
