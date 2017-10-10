# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""
from abc import ABCMeta, abstractmethod
import csv
import os
import fiona
from smif.data_layer.load import load, dump
from csv import DictReader


class DataInterface(metaclass=ABCMeta):

    @abstractmethod
    def read_sos_model_runs(self):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model_run(self, sos_model_run_name, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_models(self):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model(self, sos_model_name, sos_model):
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
    def update_sector_model(self, sector_model_name, sector_model):
        raise NotImplementedError()

    @abstractmethod
    def read_region_definitions(self):
        raise NotImplementedError()

    @abstractmethod
    def read_region_definition_data(self, region_definition_name):
        raise NotImplementedError()

    @abstractmethod
    def write_region_definition(self, region_definition):
        raise NotImplementedError()

    @abstractmethod
    def update_region_definition(self, region_definition):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definitions(self):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definition_data(self, interval_definition_name):
        raise NotImplementedError()

    @abstractmethod
    def write_interval_definition(self, interval_definition):
        raise NotImplementedError()

    @abstractmethod
    def update_interval_definition(self, interval_definition):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_set(self, scenario_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_data(self, scenario_name):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_set(self, narrative_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def update_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative(self, narrative):
        raise NotImplementedError()

    @abstractmethod
    def update_narrative(self, narrative):
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
            'interval_definitions': 'data',
            'interventions': 'data',
            'narratives': 'data',
            'region_definitions': 'data',
            'scenarios': 'data'
        }

        for category, folder in config_folders.items():
            self.file_dir[category] = os.path.join(base_folder, folder,
                                                   category)

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

        Arguments
        ---------
        sos_model_run: dict
            A sos_model_run dictionary
        """
        self._write_yaml_file(self.file_dir['sos_model_runs'],
                              sos_model_run['name'], sos_model_run)

    def update_sos_model_run(self, sos_model_run_name, sos_model_run):
        """Update system-of-system model run in Yaml file

        Arguments
        ---------
        sos_model_run_name: str
            A sos_model_run name
        sos_model_run: dict
            A sos_model_run dictionary
        """
        if sos_model_run_name != sos_model_run['name']:
            os.remove(os.path.join(self.file_dir['sos_model_runs'],
                                   sos_model_run_name + '.yml'))
        self.write_sos_model_run(sos_model_run)

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

    def read_sos_model(self, sos_model_name):
        """Read a specific system-of-system model

        Returns
        -------
        dict
            A sos model configuration dictionary
        """
        filename = sos_model_name
        return self._read_yaml_file(self.file_dir['sos_models'], filename)

    def write_sos_model(self, sos_model):
        """Write system-of-system model to Yaml file

        Existing configuration will be overwritten without warning

        Arguments
        ---------
        sos_model: dict
            A sos_model dictionary
        """
        self._write_yaml_file(self.file_dir['sos_models'],
                              sos_model['name'],
                              sos_model)

    def update_sos_model(self, sos_model_name, sos_model):
        """Update system-of-system model in Yaml file

        Arguments
        ---------
        sos_model_name: str
            A sos_model name
        sos_model: dict
            A sos_model dictionary
        """
        if sos_model_name != sos_model['name']:
            os.remove(os.path.join(self.file_dir['sos_models'],
                                   sos_model_name + '.yml'))
        self.write_sos_model(sos_model)

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
        sector_model_config = self._read_yaml_file(
            self.file_dir['sector_models'], sector_model_name)

        return sector_model_config

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

    def update_sector_model(self, sector_model_name, sector_model):
        """Update sector model in Yaml file

        Arguments
        ---------
        sector_model_name: str
            A sector_model name
        sector_model: dict
            A sector_model dictionary
        """
        if sector_model_name != sector_model['name']:
            os.remove(os.path.join(self.file_dir['sector_models'],
                                   sector_model_name + '.yml'))
        self.write_sector_model(sector_model)

    def read_interventions(self, filename):
        """Read the interventions from filename

        Arguments
        ---------
        filename: str
            The name of the intervention yml file to read in
        """
        filepath = self.file_dir['interventions']
        return self._read_yaml_file(filepath, filename, extension='')

    def read_initial_conditions(self, filename):
        """Read the initial conditions from filename

        Arguments
        ---------
        filename: str
            The name of the initial conditions yml file to read in
        """
        filepath = self.file_dir['initial_conditions']
        return self._read_yaml_file(filepath, filename, extension='')

    def read_region_definitions(self):
        """Read region_definitions from project configuration

        Returns
        -------
        list
            A list of region_definition dicts
        """
        project_config = self._read_project_config()
        return project_config['region_definitions']

    def read_region_definition_data(self, region_definition_name):
        """Read region_definition data file into a Fiona feature collection

        The file format must be possible to parse with GDAL, and must contain
        an attribute "name" to use as an identifier for the region_definition.

        Arguments
        ---------
        region_definition_name: str
            Name of the region_definition

        Returns
        -------
        list
            A list of data from the specified file in a fiona formatted dict
        """
        # Find filename for this region_definition_name
        filename = ''
        for region_definition in self.read_region_definitions():
            if region_definition['name'] == region_definition_name:
                filename = region_definition['filename']
                break

        # Read the region data from file
        filepath = os.path.join(self.file_dir['region_definitions'], filename)
        with fiona.drivers():
            with fiona.open(filepath) as src:
                data = [f for f in src]

        return data

    def write_region_definition(self, region_definition):
        """Write region_definition to project configuration

        Arguments
        ---------
        region_definition: dict
            A region_definition dict
        """
        project_config = self._read_project_config()

        project_config['region_definitions'].append(region_definition)
        self._write_project_config(project_config)

    def update_region_definition(self, region_definition_name, region_definition):
        """Update region_definition to project configuration

        Arguments
        ---------
        region_definition_name: str
            Name of the (original) entry
        region_definition: dict
            The updated region_definition dict
        """
        project_config = self._read_project_config()

        # Create updated list
        project_config['region_definitions'] = [
            entry for entry in project_config['region_definitions']
            if (entry['name'] != region_definition['name'] and
                entry['name'] != region_definition_name)
        ]
        project_config['region_definitions'].append(region_definition)

        self._write_project_config(project_config)

    def read_interval_definitions(self):
        """Read interval_definition sets from project configuration

        Returns
        -------
        list
            A list of interval_definition set dicts
        """
        project_config = self._read_project_config()
        return project_config['interval_definitions']

    def read_interval_definition_data(self, interval_definition_name):
        """

        Arguments
        ---------
        interval_definition_name: str

        Returns
        -------
        dict
            Interval definition data

        Notes
        -----
        Expects csv file to contain headings of `year`, `start`, `end`
        """
        interval_defs = self.read_interval_definitions()
        filename = None
        while not filename:
            for interval_def in interval_defs:
                if interval_def['name'] == interval_definition_name:
                    filename = interval_def['filename']

        filepath = os.path.join(self.file_dir['interval_definitions'], filename)
        with open(filepath, 'r') as csvfile:
            reader = DictReader(csvfile)
            data = []
            for row in reader:
                data.append(row)
        return data

    def write_interval_definition(self, interval_definition):
        """Write interval_definition to project configuration

        Arguments
        ---------
        interval_definition: dict
            A interval_definition dict
        """
        project_config = self._read_project_config()

        project_config['interval_definitions'].append(interval_definition)
        self._write_project_config(project_config)

    def update_interval_definition(self, interval_definition_name, interval_definition):
        """Update interval_definition to project configuration

        Arguments
        ---------
        interval_definition_name: str
            Name of the (original) entry
        interval_definition: dict
            The updated interval_definition dict
        """
        project_config = self._read_project_config()

        # Create updated list
        project_config['interval_definitions'] = [
            entry for entry in project_config['interval_definitions']
            if (entry['name'] != interval_definition['name'] and
                entry['name'] != interval_definition_name)
        ]
        project_config['interval_definitions'].append(interval_definition)

        self._write_project_config(project_config)

    def read_scenario_sets(self):
        """Read scenario sets from project configuration

        Returns
        -------
        list
            A list of scenario set dicts
        """
        project_config = self._read_project_config()
        return project_config['scenario_sets']

    def read_scenario_set(self, scenario_set_name):
        """Read all scenarios from a certain scenario_set

        Arguments
        ---------
        scenario_set_name: str
            Name of the scenario_set

        Returns
        -------
        list
            A list of scenarios within the specified 'scenario_set_name'
        """
        project_config = self._read_project_config()

        # Filter only the scenarios of the selected scenario_set_name
        filtered_scenario_data = []
        for scenario_data in project_config['scenarios']:
            if scenario_data['scenario_set'] == scenario_set_name:
                filtered_scenario_data.append(scenario_data)

        return filtered_scenario_data

    def read_scenario_definition(self, scenario_name):
        """Read scenario definition data

        Arguments
        ---------
        scenario_name: str
            Name of the scenario

        Returns
        -------
        dict
            The scenario definition
        """
        project_config = self._read_project_config()
        for scenario_data in project_config['scenarios']:
            if scenario_data['name'] == scenario_name:
                return scenario_data

    def read_scenario_data(self, scenario_name):
        """Read scenario data file

        Arguments
        ---------
        scenario_name: str
            Name of the scenario

        Returns
        -------
        dict
            A dict of lists of dicts containing the contents of `scenario_name`
            data file(s) associated with the scenario parameters. The keys of
            the dict are the parameter names
        """
        data = {}
        # Find filenames for this scenario
        filename = None
        project_config = self._read_project_config()
        for scenario_data in project_config['scenarios']:
            if scenario_data['name'] == scenario_name:
                for param in scenario_data['parameters']:
                    filename = param['filename']
                    # Read the scenario data from file
                    filepath = os.path.join(self.file_dir['scenarios'], filename)
                    data[param['name']] = self._get_data_from_csv(filepath)

        return data

    def write_scenario_set(self, scenario_set):
        """Write scenario_set to project configuration

        Arguments
        ---------
        scenario_set: dict
            A scenario_set dict
        """
        project_config = self._read_project_config()

        project_config['scenario_sets'].append(scenario_set)
        self._write_project_config(project_config)

    def update_scenario_set(self, scenario_set_name, scenario_set):
        """Update scenario_set to project configuration

        Arguments
        ---------
        scenario_set_name: str
            Name of the (original) entry
        scenario_set: dict
            The updated scenario_set dict
        """
        project_config = self._read_project_config()

        # Create updated list
        project_config['scenario_sets'] = [
            entry for entry in project_config['scenario_sets']
            if (entry['name'] != scenario_set['name'] and
                entry['name'] != scenario_set_name)
        ]
        project_config['scenario_sets'].append(scenario_set)

        self._write_project_config(project_config)

    def write_scenario(self, scenario):
        """Write scenario to project configuration

        Arguments
        ---------
        scenario: dict
            A scenario dict
        """
        project_config = self._read_project_config()

        project_config['scenarios'].append(scenario)
        self._write_project_config(project_config)

    def update_scenario(self, scenario_name, scenario):
        """Update scenario to project configuration

        Arguments
        ---------
        scenario_name: str
            Name of the (original) entry
        scenario: dict
            The updated scenario dict
        """
        project_config = self._read_project_config()

        # Create updated list
        project_config['scenarios'] = [
            entry for entry in project_config['scenarios']
            if (entry['name'] != scenario['name'] and
                entry['name'] != scenario_name)
        ]
        project_config['scenarios'].append(scenario)

        self._write_project_config(project_config)

    def read_narrative_sets(self):
        """Read narrative sets from project configuration

        Returns
        -------
        list
            A list of narrative set dicts
        """
        project_config = self._read_project_config()
        return project_config['narrative_sets']

    def read_narrative_set(self, narrative_set_name):
        """Read all narratives from a certain narrative_set

        Arguments
        ---------
        narrative_set_name: str
            Name of the narrative_set

        Returns
        -------
        list
            A list of narratives within the specified 'narrative_set_name'
        """
        project_config = self._read_project_config()

        # Filter only the narratives of the selected narrative_set_name
        filtered_narrative_data = []
        for narrative_data in project_config['narratives']:
            if narrative_data['narrative_set'] == narrative_set_name:
                filtered_narrative_data.append(narrative_data)

        return filtered_narrative_data

    def read_narrative_data(self, narrative_name):
        """Read narrative data file

        Arguments
        ---------
        narrative_name: str
            Name of the narrative

        Returns
        -------
        list
            A list with dictionaries containing the contents of 'narrative_name' data file
        """
        # Find filename for this narrative
        filename = ''
        project_config = self._read_project_config()
        for narrative in project_config['narratives']:
            if narrative['name'] == narrative_name:
                filename = narrative['filename']
                break

        # Read the narrative data from file
        return load(os.path.join(self.file_dir['narratives'], filename))

    def read_narrative_definition(self, narrative_name):
        """Read the narrative definition

        Arguments
        ---------
        narrative_name: str
            Name of the narrative

        Returns
        -------
        dict

        """
        definition = None
        project_config = self._read_project_config()
        for narrative in project_config['narratives']:
            if narrative['name'] == narrative_name:
                definition = narrative
        return definition

    def write_narrative_set(self, narrative_set):
        """Write narrative_set to project configuration

        Arguments
        ---------
        narrative_set: dict
            A narrative_set dict
        """
        project_config = self._read_project_config()

        project_config['narrative_sets'].append(narrative_set)
        self._write_project_config(project_config)

    def update_narrative_set(self, narrative_set_name, narrative_set):
        """Update narrative_set to project configuration

        Arguments
        ---------
        narrative_set_name: str
            Name of the (original) entry
        narrative_set: dict
            The updated narrative_set dict
        """
        project_config = self._read_project_config()

        # Create updated list
        project_config['narrative_sets'] = [
            entry for entry in project_config['narrative_sets']
            if (entry['name'] != narrative_set['name'] and
                entry['name'] != narrative_set_name)
        ]
        project_config['narrative_sets'].append(narrative_set)

        self._write_project_config(project_config)

    def write_narrative(self, narrative):
        """Write narrative to project configuration

        Arguments
        ---------
        narrative: dict
            A narrative dict
        """
        project_config = self._read_project_config()

        project_config['narratives'].append(narrative)
        self._write_project_config(project_config)

    def update_narrative(self, narrative_name, narrative):
        """Update narrative to project configuration

        Arguments
        ---------
        narrative_name: str
            Name of the (original) entry
        narrative: dict
            The updated narrative dict
        """
        project_config = self._read_project_config()

        # Create updated list
        project_config['narratives'] = [
            entry for entry in project_config['narratives']
            if (entry['name'] != narrative['name'] and
                entry['name'] != narrative_name)
        ]
        project_config['narratives'].append(narrative)

        self._write_project_config(project_config)

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

    def _get_data_from_csv(self, filepath):
        scenario_data = []
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            scenario_data = []
            for row in reader:
                scenario_data.append(row)
        return scenario_data

    def _read_yaml_file(self, path, filename, extension='.yml'):
        """Read a Data dict from a Yaml file

        Arguments
        ---------
        path: str
            Path to directory
        name: str
            Name of file
        extension: str, default='.yml'
            The file extension

        Returns
        -------
        dict
            The data of the Yaml file `filename` in `path`
        """
        filename = filename + extension
        filepath = os.path.join(path, filename)
        return load(filepath)

    def _write_yaml_file(self, path, filename, data, extension='.yml'):
        """Write a data dict to a Yaml file

        Arguments
        ---------
        path: str
            Path to directory
        name: str
            Name of file
        data: dict
            Data to be written to the file
        extension: str, default='.yml'
            The file extension
        """
        filename = filename + extension
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

    def read_units(self):
        raise NotImplementedError()

    def write_unit(self, unit):
        raise NotImplementedError()

    def read_regions(self):
        raise NotImplementedError()

    def read_region_data(self, region_name):
        raise NotImplementedError()

    def write_region(self, region):
        raise NotImplementedError()

    def read_intervals(self):
        raise NotImplementedError()

    def read_interval_data(self, interval_name):
        raise NotImplementedError()

    def write_interval(self, interval):
        raise NotImplementedError()

    def read_scenario_sets(self):
        raise NotImplementedError()

    def read_scenario_set(self, scenario_set_name):
        raise NotImplementedError()

    def read_scenario_data(self, scenario_name):
        raise NotImplementedError()

    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    def write_scenario(self, scenario):
        raise NotImplementedError()

    def read_narrative_sets(self):
        raise NotImplementedError()

    def read_narrative_set(self, narrative_set_name):
        raise NotImplementedError()

    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    def write_narrative(self, narrative):
        raise NotImplementedError()
