# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""
from abc import ABCMeta, abstractmethod
import os
import fiona
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
    def read_region_set_data(self, region_set_data_file):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_set_data(self, interval_set_data_file):
        raise NotImplementedError()

    @abstractmethod
    def read_units(self):
        raise NotImplementedError()

    @abstractmethod
    def write_region_sets(self, data):
        raise NotImplementedError()

    @abstractmethod
    def write_interval_sets(self, data):
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

    Project.yml

    Arguments
    ---------
    base_folder: str
        The path to the configuration and data files
    """
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.file_dir = {}
        self.file_dir['project'] = os.path.join(base_folder, 'config')

        config_folders = {
            'sos_model_runs': 'config',
            'sos_models': 'config',
            'sector_models': 'config',
            'initial_conditions': 'data',
            'intervals': 'data',
            'interventions': 'data',
            'narratives': 'data',
            'regions': 'data',
            'scenarios': 'data'
        }

        for config_file in config_folders:
            self.file_dir[config_file] = os.path.join(base_folder, config_folders[config_file],
                                                      config_file)

    def read_sos_model_runs(self):
        """Read all system-of-system model runs from Yaml files

        sos_model_runs.yml

        Returns
        -------
        list
            A list of sos_model_run dicts
        """
        sos_model_runs = []

        sos_model_run_names = self._read_filenames_in_dir(self.file_dir['sos_model_runs'],
                                                          '.yml')
        for sos_model_run_name in sos_model_run_names:
            sos_model_runs.append(self._read_yaml_file(self.file_dir['sos_model_runs'],
                                                       sos_model_run_name))

        return sos_model_runs

    def write_sos_model_run(self, sos_model_run):
        """Write system-of-system model run to Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sos_model_run: dict
            A sos_model_run dictionary
        """
        self._write_yaml_file(self.file_dir['sos_model_runs'],
                              sos_model_run['name'], sos_model_run)

    def read_sos_models(self):
        """Read all system-of-system models from Yaml files

        Returns
        -------
        list
            A list of sos_models dicts
        """
        sos_models = []

        sos_model_names = self._read_filenames_in_dir(self.file_dir['sos_models'], '.yml')
        for sos_model_name in sos_model_names:
            sos_models.append(self._read_yaml_file(self.file_dir['sos_models'],
                                                   sos_model_name))
        return sos_models

    def write_sos_model(self, sos_model):
        """Write system-of-system model to Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sos_model: dict
            A sos_model dictionary
        """
        self._write_yaml_file(self.file_dir['sos_models'], sos_model['name'], sos_model)

    def read_sector_models(self):
        """Read all sector models from Yaml files

        Returns
        -------
        list
            A list of sector_model dicts
        """
        return self._read_filenames_in_dir(self.file_dir['sector_models'], '.yml')

    def read_sector_model(self, sector_model_name):
        """Read a sector model from a Yaml file

        Raises an exception when the file does not exists

        Arguments
        ---------
        sector_model_name: str
            Name of the sector_model (sector_model['name'])
        """
        return self._read_yaml_file(self.file_dir['sector_models'], sector_model_name)

    def write_sector_model(self, sector_model):
        """Write sector model to a Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sector_model: dict
            A sector_model dictionary
        """
        self._write_yaml_file(self.file_dir['sector_models'], sector_model['name'],
                              sector_model)

    def read_region_sets(self):
        """Read region sets from project configuration

        Returns
        -------
        list
            A list of region set dicts
        """
        project_config = self._read_yaml_file(self.file_dir['project'], 'project')
        return project_config['region_sets']

    def read_region_set_data(self, region_set_data_file):
        """Read region_set_data file into a Fiona feature collection

        The file format must be possible to parse with GDAL, and must contain
        an attribute "name" to use as an identifier for the region.

        Arguments
        ---------
        region_set_data_file: str
            Filename of a GDAL-readable region file

        Returns
        -------
        list
            A list of data from the specified file in a fiona formatted dict
        """
        filepath = os.path.join(self.file_dir['regions'], region_set_data_file)

        with fiona.drivers():
            with fiona.open(filepath) as src:
                data = [f for f in src]

        return data

    def read_interval_sets(self):
        """Read interval sets from project configuration

        Returns
        -------
        list
            A list of interval set dicts
        """
        project_config = self._read_yaml_file(self.file_dir['project'], 'project')
        return project_config['interval_sets']

    def read_interval_set_data(self, interval_set_data_file):
        raise NotImplementedError()

    def read_units(self):
        raise NotImplementedError()

    def write_region_sets(self, data):
        """Write region sets to project configuration

        Arguments
        ---------
        data: list
            A list of region set dicts
        """
        project_config = self._read_project_config()
        project_config['region_sets'] = data
        self._write_project_config(project_config)

    def write_interval_sets(self, data):
        """Write interval sets to project configuration

        Arguments
        ---------
        data: list
            A list of interval set dicts
        """
        project_config = self._read_project_config()
        project_config['interval_sets'] = data
        self._write_project_config(project_config)

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

    def _read_project_config(self):
        """Read the project configuration

        Returns
        -------
        dict
            The project configuration
        """

        return self._read_yaml_file(self.file_dir['project'], 'project')

    def _write_project_config(self, data):
        """Write the project configuration

        Argument
        --------
        data: dict
            The project configuration
        """
        self._write_yaml_file(self.file_dir['project'], 'project', data)

    def _read_filenames_in_dir(self, path, extension):
        """Returns the name of the Yaml files in a certain directory

        Arguments
        ---------
        path: str
            Path to directory
        extension: str
            Extension of files (such as: '.yml' or '.csv')

        Returns
        -------
        list
            The list of files in `path` with extension
        """
        files = []
        for filename in os.listdir(path):
            if filename.endswith(extension):
                files.append(os.path.splitext(filename)[0])
        return files

    def _read_yaml_file(self, path, filename):
        """Read a Data dict from a Yaml file

        Arguments
        ---------
        path: str
            Path to directory
        name: str
            Name of file

        Returns
        -------
        dict
            The data of the Yaml file `name` in `path`
        """
        filename = filename + '.yml'
        filepath = os.path.join(path, filename)
        return load(filepath)

    def _write_yaml_file(self, path, filename, data):
        """Write a data dict to a Yaml file

        Arguments
        ---------
        path: str
            Path to directory
        name: str
            Name of file
        data: dict
            Data to be written to the file
        """
        filename = filename + '.yml'
        filepath = os.path.join(path, filename)
        dump(data, filepath)


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

    def read_region_set_data(self, region_set_data_file):
        raise NotImplementedError()

    def read_interval_sets(self):
        raise NotImplementedError()

    def read_interval_set_data(self, interval_set_data_file):
        raise NotImplementedError()

    def read_units(self):
        raise NotImplementedError()

    def write_region_sets(self, data):
        raise NotImplementedError()

    def write_interval_sets(self, data):
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
