"""File-backed data interface
"""
import csv
import os
import re
import shutil
from csv import DictReader

import glob

import fiona
import pyarrow as pa
import pyarrow.parquet as pq
from smif.data_layer.data_interface import (DataExistsError, DataInterface,
                                            DataMismatchError,
                                            DataNotFoundError)
from smif.data_layer.load import dump, load


class DatafileInterface(DataInterface):
    """Read and write interface to YAML / CSV configuration files
    and intermediate CSV / native-binary data storage.

    Project.yml

    Arguments
    ---------
    base_folder: str
        The path to the configuration and data files
    storage_format: str
        The format used to store intermediate data (local_csv, local_binary)
    timestamp: str
        The ISO-8601 timestamp that identifies the modelrun (%y%m%dT%H%M%S)
    """
    def __init__(self, base_folder, storage_format='local_binary', timestamp='yyyy_mm_dd_hhmm'):
        super().__init__()

        self.base_folder = base_folder
        self.timestamp = timestamp
        self.storage_format = storage_format

        self.file_dir = {}
        self.file_dir['project'] = os.path.join(base_folder, 'config')
        self.file_dir['results'] = os.path.join(base_folder, 'results')

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

    def read_units_file_name(self):
        project_config = self._read_project_config()
        filename = project_config['units']
        self.logger.debug("Units filename is %s", filename)
        if filename is not None:
            path = os.path.join(self.base_folder, 'data')
            units_file_path = os.path.join(path, filename)
            if os.path.isfile(units_file_path):
                return units_file_path
            else:
                return None
        else:
            return None

    def read_sos_model_runs(self):
        """Read all system-of-system model runs from Yaml files

        sos_model_runs.yml

        Returns
        -------
        list
            A list of sos_model_run dicts
        """
        sos_model_run_name_desc = []

        sos_model_run_names = self._read_filenames_in_dir(self.file_dir['sos_model_runs'],
                                                          '.yml')
        for sos_model_run_name in sos_model_run_names:
            sos_model_run = self._read_yaml_file(
                self.file_dir['sos_model_runs'], sos_model_run_name)
            sos_model_run_name_desc.append({
                'name': sos_model_run['name'],
                'description': sos_model_run['description']
            })

        return sos_model_run_name_desc

    def read_sos_model_run(self, sos_model_run_name):
        """Read a system-of-system model run

        Arguments
        ---------
        sos_model_run_name: str
            A sos_model_run name

        Returns
        -------
        sos_model_run: dict
            A sos_model_run dictionary
        """
        if not self._sos_model_run_exists(sos_model_run_name):
            raise DataNotFoundError("sos_model_run '%s' not found" % sos_model_run_name)
        return self._read_yaml_file(self.file_dir['sos_model_runs'], sos_model_run_name)

    def _sos_model_run_exists(self, name):
        return os.path.exists(
            os.path.join(self.file_dir['sos_model_runs'], name + '.yml'))

    def write_sos_model_run(self, sos_model_run):
        """Write system-of-system model run to Yaml file

        Arguments
        ---------
        sos_model_run: dict
            A sos_model_run dictionary
        """
        if self._sos_model_run_exists(sos_model_run['name']):
            raise DataExistsError("sos_model_run '%s' already exists" % sos_model_run['name'])
        else:
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
            raise DataMismatchError(
                "sos_model_run name '{}' must match '{}'".format(
                    sos_model_run_name,
                    sos_model_run['name']))

        if not self._sos_model_run_exists(sos_model_run_name):
            raise DataNotFoundError("sos_model_run '%s' not found" % sos_model_run_name)
        self._write_yaml_file(self.file_dir['sos_model_runs'],
                              sos_model_run['name'], sos_model_run)

    def delete_sos_model_run(self, sos_model_run_name):
        """Delete a system-of-system model run

        Arguments
        ---------
        sos_model_run_name: str
            A sos_model_run name
        """
        if not self._sos_model_run_exists(sos_model_run_name):
            raise DataNotFoundError("sos_model_run '%s' not found" % sos_model_run_name)

        os.remove(os.path.join(self.file_dir['sos_model_runs'], sos_model_run_name + '.yml'))

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

        Arguments
        ---------
        sos_model_name: str
            A sos_model name

        Returns
        -------
        sos_model: dict
            A sos_model dictionary
        """
        if not self._sos_model_exists(sos_model_name):
            raise DataNotFoundError("sos_model '%s' not found" % sos_model_name)
        return self._read_yaml_file(self.file_dir['sos_models'], sos_model_name)

    def _sos_model_exists(self, name):
        return os.path.exists(
            os.path.join(self.file_dir['sos_models'], name + '.yml'))

    def write_sos_model(self, sos_model):
        """Write system-of-system model to Yaml file

        Arguments
        ---------
        sos_model: dict
            A sos_model dictionary
        """
        if self._sos_model_exists(sos_model['name']):
            raise DataExistsError("sos_model '%s' already exists" % sos_model['name'])
        else:
            self._write_yaml_file(
                self.file_dir['sos_models'],
                sos_model['name'],
                sos_model
            )

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
            raise DataMismatchError(
                "sos_model name '{}' must match '{}'".format(
                    sos_model_name,
                    sos_model['name']))

        if not self._sos_model_exists(sos_model_name):
            raise DataNotFoundError("sos_model '%s' not found" % sos_model_name)
        self._write_yaml_file(self.file_dir['sos_models'],
                              sos_model['name'], sos_model)

    def delete_sos_model(self, sos_model_name):
        """Delete a system-of-system model run

        Arguments
        ---------
        sos_model_name: str
            A sos_model name
        """
        if not self._sos_model_exists(sos_model_name):
            raise DataNotFoundError("sos_model '%s' not found" % sos_model_name)

        os.remove(os.path.join(self.file_dir['sos_models'], sos_model_name + '.yml'))

    def read_sector_models(self):
        """Read all sector models from Yaml files

        sector_models.yml

        Returns
        -------
        list
            A list of sector_model dicts
        """
        sector_models = []

        sector_model_names = self._read_filenames_in_dir(
            self.file_dir['sector_models'], '.yml')
        for sector_model_name in sector_model_names:
            sector_models.append(
                self._read_yaml_file(
                    self.file_dir['sector_models'],
                    sector_model_name
                )
            )

        return sector_models

    def read_sector_model(self, sector_model_name):
        """Read a sector model

        Arguments
        ---------
        sector_model_name: str
            A sector_model name

        Returns
        -------
        sector_model: dict
            A sector_model dictionary
        """
        if not self._sector_model_exists(sector_model_name):
            raise DataNotFoundError("sector_model '%s' not found" % sector_model_name)
        return self._read_yaml_file(self.file_dir['sector_models'], sector_model_name)

    def _sector_model_exists(self, name):
        return os.path.exists(
            os.path.join(self.file_dir['sector_models'], name + '.yml'))

    def write_sector_model(self, sector_model):
        """Write sector model to Yaml file

        Arguments
        ---------
        sector_model: dict
            A sector_model dictionary
        """
        if self._sector_model_exists(sector_model['name']):
            raise DataExistsError("sector_model '%s' already exists" % sector_model['name'])
        else:
            self._write_yaml_file(self.file_dir['sector_models'],
                                  sector_model['name'], sector_model)

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
            raise DataMismatchError(
                "sector_model name '{}' must match '{}'".format(
                    sector_model_name,
                    sector_model['name']))

        if not self._sector_model_exists(sector_model_name):
            raise DataNotFoundError("sector_model '%s' not found" % sector_model_name)
        self._write_yaml_file(self.file_dir['sector_models'],
                              sector_model['name'], sector_model)

    def delete_sector_model(self, sector_model_name):
        """Delete a sector model

        Arguments
        ---------
        sector_model_name: str
            A sector_model name
        """
        if not self._sector_model_exists(sector_model_name):
            raise DataNotFoundError("sector_model '%s' not found" % sector_model_name)

        os.remove(os.path.join(self.file_dir['sector_models'], sector_model_name + '.yml'))

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
        region_definitions = self.read_region_definitions()
        try:
            filename = next(
                rdef['filename']
                for rdef in region_definitions
                if rdef['name'] == region_definition_name
            )
        except StopIteration:
            raise DataNotFoundError(
                "Region definition '{}' not found".format(region_definition_name))

        # Read the region data from file
        filepath = os.path.join(self.file_dir['region_definitions'], filename)
        with fiona.drivers():
            with fiona.open(filepath) as src:
                data = [f for f in src]

        return data

    def _read_region_names(self, region_definition_name):
        return [
            feature['properties']['name']
            for feature in self.read_region_definition_data(region_definition_name)
        ]

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
        Expects csv file to contain headings of `id`, `start`, `end`
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

    def _read_interval_names(self, interval_definition_name):
        return [
            interval['id']
            for interval in self.read_interval_definition_data(interval_definition_name)
        ]

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
        """Read a scenario_set

        Arguments
        ---------
        scenario_set_name: str
            Name of the scenario_set

        Returns
        -------
        dict
            Scenario set definition
        """
        project_config = self._read_project_config()

        try:
            return next(
                scenario_set
                for scenario_set in project_config['scenario_sets']
                if scenario_set['name'] == scenario_set_name
            )
        except StopIteration:
            raise DataNotFoundError(
                "Scenario set '{}' not found".format(scenario_set_name))

    def read_scenario_set_scenario_definitions(self, scenario_set_name):
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
        filtered_scenario_data = [
            data for data in project_config['scenarios']
            if data['scenario_set'] == scenario_set_name
        ]

        if not filtered_scenario_data:
            self.logger.warning(
                "Scenario set '{}' has no scenarios defined".format(scenario_set_name))

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
        try:
            return next(
                sdef
                for sdef in project_config['scenarios']
                if sdef['name'] == scenario_name
            )
        except StopIteration:
            raise DataNotFoundError(
                "Scenario definition '{}' not found".format(scenario_name))

    def _scenario_set_exists(self, scenario_set_name):
        project_config = self._read_project_config()
        for scenario_set in project_config['scenario_sets']:
            if scenario_set['name'] == scenario_set_name:
                return scenario_set

    def write_scenario_set(self, scenario_set):
        """Write scenario_set to project configuration

        Arguments
        ---------
        scenario_set: dict
            A scenario_set dict
        """
        if self._scenario_set_exists(scenario_set['name']):
            raise DataExistsError("scenario_set '%s' already exists" % scenario_set['name'])
        else:
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
        if not self._scenario_set_exists(scenario_set_name):
            raise DataNotFoundError("scenario_set '%s' not found" % scenario_set_name)

        project_config = self._read_project_config()

        project_config['scenario_sets'] = [
            entry for entry in project_config['scenario_sets']
            if (entry['name'] != scenario_set['name'] and
                entry['name'] != scenario_set_name)
        ]
        project_config['scenario_sets'].append(scenario_set)

        self._write_project_config(project_config)

    def delete_scenario_set(self, scenario_set_name):
        """Delete scenario_set from project configuration

        Arguments
        ---------
        scenario_set_name: str
            A scenario_set name
        """
        if not self._scenario_set_exists(scenario_set_name):
            raise DataNotFoundError("scenario_set '%s' not found" % scenario_set_name)

        project_config = self._read_project_config()

        project_config['scenario_sets'] = [
            entry for entry in project_config['scenario_sets']
            if (entry['name'] != scenario_set_name)
        ]

        self._write_project_config(project_config)

    def read_scenarios(self):
        """Read scenarios from project configuration

        Returns
        -------
        list
            A list of scenario dicts
        """
        project_config = self._read_project_config()
        return project_config['scenarios']

    def read_scenario(self, scenario_name):
        """Read a scenario

        Arguments
        ---------
        scenario_name: str
            Name of the scenario

        Returns
        -------
        dict
            A scenario dictionary
        """
        project_config = self._read_project_config()
        for scenario_data in project_config['scenarios']:
            if scenario_data['name'] == scenario_name:
                return scenario_data
        raise DataNotFoundError("scenario '%s' not found" % scenario_name)

    def _scenario_exists(self, scenario_name):
        project_config = self._read_project_config()
        for scenario in project_config['scenarios']:
            if scenario['name'] == scenario_name:
                return scenario

    def write_scenario(self, scenario):
        """Write scenario to project configuration

        Arguments
        ---------
        scenario: dict
            A scenario dict
        """
        if self._scenario_exists(scenario['name']):
            raise DataExistsError("scenario '%s' already exists" % scenario['name'])
        else:
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
        if not self._scenario_exists(scenario_name):
            raise DataNotFoundError("scenario '%s' not found" % scenario_name)

        project_config = self._read_project_config()

        project_config['scenarios'] = [
            entry for entry in project_config['scenarios']
            if (entry['name'] != scenario['name'] and
                entry['name'] != scenario_name)
        ]
        project_config['scenarios'].append(scenario)

        self._write_project_config(project_config)

    def delete_scenario(self, scenario_name):
        """Delete scenario from project configuration

        Arguments
        ---------
        scenario_name: str
            A scenario name
        """
        if not self._scenario_exists(scenario_name):
            raise DataNotFoundError("scenario '%s' not found" % scenario_name)

        project_config = self._read_project_config()

        project_config['scenarios'] = [
            entry for entry in project_config['scenarios']
            if (entry['name'] != scenario_name)
        ]

        self._write_project_config(project_config)

    def read_scenario_data(self, scenario_name, parameter_name,
                           spatial_resolution, temporal_resolution, timestep):
        """Read scenario data file

        Arguments
        ---------
        scenario_name: str
            Name of the scenario
        parameter_name: str
            Name of the scenario parameter to read
        spatial_resolution : str
        temporal_resolution : str
        timestep: int

        Returns
        -------
        data: numpy.ndarray

        """
        # Find filenames for this scenario
        filename = None
        project_config = self._read_project_config()
        for scenario_data in project_config['scenarios']:
            if scenario_data['name'] == scenario_name:
                for param in scenario_data['parameters']:
                    if param['name'] == parameter_name:
                        filename = param['filename']
                        break
                break

        if filename is None:
            raise DataNotFoundError(
                "Scenario '{}' with parameter '{}' not found".format(
                    scenario_name, parameter_name))

        # Read the scenario data from file
        filepath = os.path.join(self.file_dir['scenarios'], filename)
        data = [
            datum for datum in
            self._get_data_from_csv(filepath)
            if int(datum['year']) == timestep
        ]

        region_names = self._read_region_names(spatial_resolution)
        interval_names = self._read_interval_names(temporal_resolution)

        return self.data_list_to_ndarray(data, region_names, interval_names)

    def read_narrative_sets(self):
        """Read narrative sets from project configuration

        Returns
        -------
        list
            A list of narrative set dicts
        """
        # Find filename for this narrative
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
            A narrative_set dictionary
        """
        project_config = self._read_project_config()
        for narrative_data in project_config['narrative_sets']:
            if narrative_data['name'] == narrative_set_name:
                return narrative_data
        raise DataNotFoundError("narrative_set '%s' not found" % narrative_set_name)

    def _narrative_set_exists(self, narrative_set_name):
        project_config = self._read_project_config()
        for narrative_set in project_config['narrative_sets']:
            if narrative_set['name'] == narrative_set_name:
                return narrative_set

    def write_narrative_set(self, narrative_set):
        """Write narrative_set to project configuration

        Arguments
        ---------
        narrative_set: dict
            A narrative_set dict
        """
        if self._narrative_set_exists(narrative_set['name']):
            raise DataExistsError("narrative_set '%s' already exists" % narrative_set['name'])
        else:
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
        if not self._narrative_set_exists(narrative_set_name):
            raise DataNotFoundError("narrative_set '%s' not found" % narrative_set_name)

        project_config = self._read_project_config()

        project_config['narrative_sets'] = [
            entry for entry in project_config['narrative_sets']
            if (entry['name'] != narrative_set['name'] and
                entry['name'] != narrative_set_name)
        ]
        project_config['narrative_sets'].append(narrative_set)

        self._write_project_config(project_config)

    def delete_narrative_set(self, narrative_set_name):
        """Delete narrative_set from project configuration

        Arguments
        ---------
        narrative_set_name: str
            A narrative_set name
        """
        if not self._narrative_set_exists(narrative_set_name):
            raise DataNotFoundError("narrative_set '%s' not found" % narrative_set_name)

        project_config = self._read_project_config()

        project_config['narrative_sets'] = [
            entry for entry in project_config['narrative_sets']
            if (entry['name'] != narrative_set_name)
        ]

        self._write_project_config(project_config)

    def read_narratives(self):
        """Read narrative sets from project configuration

        Returns
        -------
        list
            A list of narrative set dicts
        """
        project_config = self._read_project_config()
        return project_config['narratives']

    def read_narrative(self, narrative_name):
        """Read all narratives from a certain narrative

        Arguments
        ---------
        narrative_name: str
            Name of the narrative

        Returns
        -------
        list
            A narrative dictionary
        """
        project_config = self._read_project_config()
        for narrative_data in project_config['narratives']:
            if narrative_data['name'] == narrative_name:
                return narrative_data
        raise DataNotFoundError("narrative '%s' not found" % narrative_name)

    def _narrative_exists(self, narrative_name):
        project_config = self._read_project_config()
        for narrative in project_config['narratives']:
            if narrative['name'] == narrative_name:
                return narrative

    def write_narrative(self, narrative):
        """Write narrative to project configuration

        Arguments
        ---------
        narrative: dict
            A narrative dict
        """
        if self._narrative_exists(narrative['name']):
            raise DataExistsError("narrative '%s' already exists" % narrative['name'])
        else:
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
        if not self._narrative_exists(narrative_name):
            raise DataNotFoundError("narrative '%s' not found" % narrative_name)

        project_config = self._read_project_config()

        project_config['narratives'] = [
            entry for entry in project_config['narratives']
            if (entry['name'] != narrative['name'] and
                entry['name'] != narrative_name)
        ]
        project_config['narratives'].append(narrative)

        self._write_project_config(project_config)

    def delete_narrative(self, narrative_name):
        """Delete narrative from project configuration

        Arguments
        ---------
        narrative_name: str
            A narrative name
        """
        if not self._narrative_exists(narrative_name):
            raise DataNotFoundError("narrative '%s' not found" % narrative_name)

        project_config = self._read_project_config()

        project_config['narratives'] = [
            entry for entry in project_config['narratives']
            if (entry['name'] != narrative_name)
        ]

        self._write_project_config(project_config)

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
        filename = None
        project_config = self._read_project_config()
        for narrative in project_config['narratives']:
            if narrative['name'] == narrative_name:
                filename = narrative['filename']
                break

        if filename is None:
            raise DataNotFoundError(
                'Narrative \'{}\' has no data defined'.format(narrative_name))

        # Read the narrative data from file
        try:
            narrative_data = load(os.path.join(self.file_dir['narratives'], filename))
        except FileNotFoundError:
            raise DataNotFoundError(
                'Narrative \'{}\' has no data defined'.format(narrative_name))

        return narrative_data

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
        project_config = self._read_project_config()
        for narrative in project_config['narratives']:
            if narrative['name'] == narrative_name:
                return narrative

        raise DataNotFoundError('Narrative \'{}\' not found'.format(narrative_name))

    def read_results(self, modelrun_id, model_name, output_name, spatial_resolution,
                     temporal_resolution, timestep=None, modelset_iteration=None,
                     decision_iteration=None):
        """Return path to text file for a given output

        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_name : str
        spatial_resolution : str
        temporal_resolution : str
        timestep : int, optional
        modelset_iteration : int, optional
        decision_iteration : int, optional

        Returns
        -------
        data: numpy.ndarray

        """
        if timestep is None:
            raise NotImplementedError

        results_path = self._get_results_path(
            modelrun_id, self.timestamp, model_name, output_name, spatial_resolution, temporal_resolution,
            timestep, modelset_iteration, decision_iteration)

        if self.storage_format == 'local_csv':
            csv_data = self._get_data_from_csv(results_path)
            region_names = self._read_region_names(spatial_resolution)
            interval_names = self._read_interval_names(temporal_resolution)
            return self.data_list_to_ndarray(csv_data, region_names, interval_names)
        elif self.storage_format == 'local_binary':
            return self._get_data_from_native_file(results_path)

    def write_results(self, modelrun_id, model_name, output_name, data, spatial_resolution,
                      temporal_resolution, timestep=None, modelset_iteration=None,
                      decision_iteration=None):
        """Return path to text file for a given output
        
        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_name : str
        data : numpy.ndarray
        spatial_resolution : str
        temporal_resolution : str
        timestep : int, optional
        modelset_iteration : int, optional
        decision_iteration : int, optional
        """
        if timestep is None:
            raise NotImplementedError

        results_path = self._get_results_path(
            modelrun_id, self.timestamp, model_name, output_name, spatial_resolution, temporal_resolution,
            timestep, modelset_iteration, decision_iteration)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        if data.ndim == 3:
            raise NotImplementedError
        elif data.ndim == 2:
            region_names = self._read_region_names(spatial_resolution)
            interval_names = self._read_interval_names(temporal_resolution)

            if self.storage_format == 'local_csv':
                csv_data = self.ndarray_to_data_list(data, region_names, interval_names)
                self._write_data_to_csv(results_path, csv_data)
            elif self.storage_format == 'local_binary':
                buffer = self.ndarray_to_buffer(data)
                self._write_data_to_native_file(results_path, buffer)
        else:
            raise DataMismatchError(
                "Expected to write either timestep x region x interval or " +
                "region x interval data"
            )

    def prepare_warm_start(self, modelrun_id):
        """Copy the results from the previous modelrun if available

        Parameters
        ----------
        modelrun_id: str

        Returns
        -------
        num: The timestep where the data store was recovered to
        """
        results_dir = os.path.join(self.file_dir['results'], modelrun_id)
        previous_results = sorted([
            name for name in os.listdir(results_dir) 
            if os.path.isdir(os.path.join(results_dir, name)) 
        ])

        # Check if warm start is possible
        previous_results_have_different_storage_format = False
        previous_results_dont_exist = False

        if len(previous_results) > 0:
            previous_results_dir = os.path.join(self.file_dir['results'], modelrun_id, previous_results[-1])
            
            # Check if the previous modelrun used the same storage format
            results = list(glob.iglob(os.path.join(previous_results_dir, '**/*.*'), recursive=True))

            if len(results) > 0:
                for filename in results:
                    if ((self.storage_format == 'local_csv' and not filename.endswith(".csv")) or
                    (self.storage_format == 'local_binary' and not filename.endswith(".dat"))):
                        previous_results_have_different_storage_format = True
            else:
                previous_results_dont_exist = True
        else:
            previous_results_dont_exist = True

        # Attempt warm start
        if previous_results_dont_exist:
            self.logger.info("Warm start not possible because there are no previous results")
            return None
        elif previous_results_have_different_storage_format:
            self.logger.info("Warm start not possible because a different storage mode was used in the previous run")
            return None
        else:
            self.logger.info("Warm start from timestamp", previous_results[-1])
        
            # Copy results from latest timestep from this modelrun_id
            current_results_dir = os.path.join(self.file_dir['results'], modelrun_id, self.timestamp)
            shutil.copytree(previous_results_dir, current_results_dir)
            
            # Get metadata for all results
            result_metadata = []
            for filename in glob.iglob(os.path.join(current_results_dir, '**/*.*'), recursive=True):
                result_metadata.append(self._parse_results_path(filename.replace(self.file_dir['results'], '')[1:]))

            # Find latest timestep
            result_metadata = sorted(result_metadata, key=lambda k: k['timestep'], reverse=True)
            latest_timestep = result_metadata[0]['timestep']

            # Remove all results with this timestep
            results_to_remove = [result for result in result_metadata if result['timestep'] == latest_timestep]

            for result in results_to_remove:
                os.remove(self._get_results_path(result['modelrun_id'], result['timestamp'], result['model_name'],
                        result['output_name'], result['spatial_resolution'], result['temporal_resolution'],
                        result['timestep'], result['modelset_iteration'], result['decision_iteration']))

            return latest_timestep

    def _get_results_path(self, modelrun_id, timestamp, model_name, output_name, spatial_resolution,
                          temporal_resolution, timestep, modelset_iteration=None,
                          decision_iteration=None):
        """Return path to filename for a given output without file extension

        On the pattern of:
            results/
            <modelrun_name>/
            <timestamp>/
            <model_name>/
            decision_<id>_modelset_<id>/ or decision_<id>/ or modelset_<id>/ or none
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv

        Parameters
        ----------
        modelrun_id : str
        timestamp : str
        model_name : str
        output_name : str
        spatial_resolution : str
        temporal_resolution : str
        timestep : str or int
        modelset_iteration : int, optional
        decision_iteration : int, optional

        Returns
        -------
        path : strs
        """
        results_dir = self.file_dir['results']
        if modelset_iteration is None and decision_iteration is None:
            path = os.path.join(
                results_dir,
                modelrun_id,
                timestamp,
                model_name,
                "output_{}_timestep_{}_regions_{}_intervals_{}".format(
                    output_name,
                    timestep,
                    spatial_resolution,
                    temporal_resolution
                )
            )
        elif modelset_iteration is None and decision_iteration is not None:
            path = os.path.join(
                results_dir,
                modelrun_id,
                timestamp,
                model_name,
                "decision_{}".format(decision_iteration),
                "output_{}_timestep_{}_regions_{}_intervals_{}".format(
                    output_name,
                    timestep,
                    spatial_resolution,
                    temporal_resolution
                )
            )
        elif modelset_iteration is not None and decision_iteration is None:
            path = os.path.join(
                results_dir,
                modelrun_id,
                timestamp,
                model_name,
                "modelset_{}".format(modelset_iteration),
                "output_{}_timestep_{}_regions_{}_intervals_{}".format(
                    output_name,
                    timestep,
                    spatial_resolution,
                    temporal_resolution
                )
            )
        else:
            path = os.path.join(
                results_dir,
                modelrun_id,
                timestamp,
                model_name,
                "decision_{}_modelset_{}".format(decision_iteration, modelset_iteration),
                "output_{}_timestep_{}_regions_{}_intervals_{}".format(
                    output_name,
                    timestep,
                    spatial_resolution,
                    temporal_resolution
                )
            )

        if self.storage_format == 'local_csv':
            path += '.csv'
        elif self.storage_format == 'local_binary':
            path += '.dat'

        return path

    def _parse_results_path(self, path):
        """Return result metadata for a given result path

        On the pattern of:
            results/
            <modelrun_name>/
            <timestamp>/
            <model_name>/
            decision_<id>_modelset_<id>/ or decision_<id>/ or modelset_<id>/ or none
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv

        Parameters
        ----------
        path : str

        Returns
        -------
        dict : A dict containing all of the metadata
        """
        modelset_iteration = None
        decision_iteration = None

        data = re.findall(r"[\w']+", path)

        for section in data[3:len(data)]:
            if section.startswith('modelset'):
                modelset_iteration = int(section.replace('modelset_', ''))
            elif section.startswith('decision'):
                decision_iteration = int(section.replace('decision_', ''))
            elif section.startswith('output'):
                result_elements = re.findall(r"[^_]+", section)
                results = {}
                for element in result_elements:
                    if element in ('output', 'timestep', 'regions', 'intervals'):
                        parse_element = element
                    elif parse_element == 'output':
                        results.setdefault('output', []).append(element)
                    elif parse_element == 'timestep':
                        results['timestep'] = int(element)
                    elif parse_element == 'regions':
                        results.setdefault('regions', []).append(element)
                    elif parse_element == 'intervals':
                        results.setdefault('intervals', []).append(element)
            elif section == 'csv':
                storage_format = 'local_csv'
            elif section == 'dat':
                storage_format = 'local_binary'

        return {
            'modelrun_id': data[0],
            'timestamp': data[1],
            'model_name': data[2],
            'output_name': '_'.join(results['output']),
            'spatial_resolution': '_'.join(results['regions']),
            'temporal_resolution': '_'.join(results['intervals']),
            'timestep': results['timestep'],
            'modelset_iteration': modelset_iteration,
            'decision_iteration': decision_iteration,
            'storage_format': storage_format
        }

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

    @staticmethod
    def _read_filenames_in_dir(path, extension):
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

    @staticmethod
    def _get_data_from_csv(filepath):
        scenario_data = []
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            scenario_data = []
            for row in reader:
                scenario_data.append(row)
        return scenario_data

    @staticmethod
    def _write_data_to_csv(filepath, data):
        with open(filepath, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=(
                'timestep',
                'region',
                'interval',
                'value'
            ))
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    @staticmethod
    def _get_data_from_native_file(filepath):
        with pa.memory_map(filepath, 'rb') as f:
            f.seek(0)
            buf = f.read_buffer()

            data = pa.deserialize(buf)
        return data

    @staticmethod
    def _write_data_to_native_file(filepath, data):
        with pa.OSFile(filepath, 'wb') as f:
            f.write(data)

    @staticmethod
    def _read_yaml_file(path, filename, extension='.yml'):
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

    @staticmethod
    def _write_yaml_file(path, filename, data, extension='.yml'):
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
