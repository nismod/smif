"""File-backed data interface
"""
import csv
import glob
import os
import re
from csv import DictReader

import fiona
import pyarrow as pa
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
    """
    def __init__(self, base_folder, storage_format='local_binary'):
        super().__init__()

        self.base_folder = base_folder
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
            'scenarios': 'data',
            'coefficients': 'data',
            'strategies': 'data'
        }

        for category, folder in config_folders.items():
            dirname = os.path.join(base_folder, folder, category)
            # ensure each directory exists
            os.makedirs(dirname, exist_ok=True)
            # store dirname
            self.file_dir[category] = dirname

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
        sos_model_runs = []

        sos_model_run_names = self._read_filenames_in_dir(self.file_dir['sos_model_runs'],
                                                          '.yml')
        for sos_model_run_name in sos_model_run_names:
            sos_model_runs.append(self._read_yaml_file(
                self.file_dir['sos_model_runs'], sos_model_run_name))

        return sos_model_runs

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
        file_dir = self.file_dir['sector_models']

        sector_model_names = self._read_filenames_in_dir(file_dir, '.yml')
        for sector_model_name in sector_model_names:
            sector_models.append(self._read_yaml_file(file_dir, sector_model_name))

        return sector_models

    def _get_sector_model_filepath(self, sector_model_name):
        file_name = '{}.yml'.format(sector_model_name)
        file_dir = self.file_dir['sector_models']
        return os.path.join(file_dir, file_name)

    def _sector_model_exists(self, sector_model_name):
        return os.path.exists(self._get_sector_model_filepath(sector_model_name))

    def _read_sector_model_file(self, sector_model_name):
        return self._read_yaml_file(self._get_sector_model_filepath(sector_model_name))

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

        sector_model = self._read_sector_model_file(sector_model_name)

        sector_model['interventions'] = \
            self.read_sector_model_interventions(sector_model_name)

        sector_model['initial_conditions'] = \
            self.read_sector_model_initial_conditions(sector_model_name)

        return sector_model

    def write_sector_model(self, sector_model):
        """Write sector model to Yaml file

        Arguments
        ---------
        sector_model: dict
            A sector_model dictionary
        """
        if self._sector_model_exists(sector_model['name']):
            raise DataExistsError("sector_model '%s' already exists" % sector_model['name'])

        if sector_model['interventions']:
            self.logger.warning("Ignoring interventions")
            sector_model['interventions'] = []

        self._write_yaml_file(
            self.file_dir['sector_models'], sector_model['name'], sector_model)

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

        # ignore interventions and initial conditions which the app doesn't handle
        if sector_model['interventions'] or sector_model['initial_conditions']:
            old_sector_model = self._read_sector_model_file(sector_model['name'])

        if sector_model['interventions']:
            self.logger.warning("Ignoring interventions write")
            sector_model['interventions'] = old_sector_model['interventions']

        if sector_model['initial_conditions']:
            self.logger.warning("Ignoring initial conditions write")
            sector_model['initial_conditions'] = old_sector_model['initial_conditions']

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
        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            data = self._read_state_file(os.path.join(filepath, filename))
            try:
                data = self._reshape_csv_interventions(data)
            except ValueError:
                raise ValueError("Error reshaping data for {}".format(filename))
        else:
            data = self._read_yaml_file(filepath, filename, extension='')

        return data

    def _reshape_csv_interventions(self, data):
        new_data = []
        for element in data:
            reshaped_data = {}
            for key, value in element.items():
                if key.endswith(('_value', '_unit')):
                    new_key, sub_key = key.rsplit(sep="_", maxsplit=1)
                    if new_key in reshaped_data:
                        if not isinstance(reshaped_data[new_key], dict):
                            msg = "Duplicate heading in csv data: {}"
                            raise ValueError(msg.format(new_key))
                        else:
                            reshaped_data[new_key].update({sub_key: value})
                    else:
                        reshaped_data[new_key] = {sub_key: value}
                else:
                    reshaped_data[key] = value
            new_data.append(reshaped_data)
        return new_data

    def read_sector_model_interventions(self, sector_model_name):
        """Read a SectorModel's interventions

        Arguments
        ---------
        sector_model_name: str
        """
        if not self._sector_model_exists(sector_model_name):
            raise DataNotFoundError("sector_model '%s' not found" % sector_model_name)

        sector_model = self._read_sector_model_file(sector_model_name)

        intervention_files = sector_model['interventions']
        intervention_list = []
        for intervention_file in intervention_files:
            interventions = self.read_interventions(intervention_file)
            intervention_list.extend(interventions)
        return intervention_list

    def read_strategies(self, filename):
        return self._read_planned_interventions(filename, 'strategies')

    def read_initial_conditions(self, filename):
        return self._read_planned_interventions(filename, 'initial_conditions')

    def _read_planned_interventions(self, filename, filedir):
        """Read the planned intervention data from a file

        Planned interventions are stored either a csv or yaml file. In the case
        of the former, the file should look like this::

            name,build_year
            asset_a,2010
            asset_b,2015

        In the case of a yaml, file, the format is as follows::

            - name: asset_a
              build_year: 2010
            - name: asset_b
              build_year: 2015

        Arguments
        ---------
        filename: str
            The name of the strategy yml file to read in
        filedir: str
            The key of the filedir e.g. ``strategies`` or ``initial_conditions``

        """
        filepath = self.file_dir[filedir]
        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            strategies = self._read_state_file(os.path.join(filepath, filename))
        else:
            strategies = self._read_yaml_file(filepath, filename, extension='')
        return strategies

    def read_sector_model_initial_conditions(self, sector_model_name):
        """Read a SectorModel's initial conditions

        Arguments
        ---------
        sector_model_name: str
        """
        if not self._sector_model_exists(sector_model_name):
            raise DataNotFoundError("sector_model '%s' not found" % sector_model_name)

        sector_model = self._read_sector_model_file(sector_model_name)

        initial_condition_files = sector_model['initial_conditions']
        initial_condition_list = []
        for initial_condition_file in initial_condition_files:
            initial_conditions = self._read_planned_interventions(
                initial_condition_file, 'initial_conditions')
            initial_condition_list.extend(initial_conditions)
        return initial_condition_list

    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        """Read list of (name, build_year) for a given modelrun, timestep,
        decision
        """
        fname = self._get_state_filename(modelrun_name, timestep, decision_iteration)
        if not os.path.exists(fname):
            msg = "State file does not exist for timestep {} and iteration {}"
            raise ValueError(msg.format(timestep, decision_iteration))
        state = self._read_state_file(fname)
        return state

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        """Write state, a list of decision tuples (name, build_year) to file
        """
        fname = self._get_state_filename(modelrun_name, timestep, decision_iteration)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w+') as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=(
                'name',
                'build_year'
            ))
            writer.writeheader()
            for row in state:
                writer.writerow(row)

    def _get_state_filename(self, modelrun_name, timestep=None, decision_iteration=None):
        """Compose a unique filename for state file:
                state_{timestep|0000}[_decision_{iteration}].csv
        """
        results_dir = self.file_dir['results']
        if timestep is None and decision_iteration is None:
            fname = os.path.join(
                results_dir, modelrun_name, 'state_0000.csv')
        elif timestep is not None and decision_iteration is None:
            fname = os.path.join(
                results_dir, modelrun_name, 'state_{}.csv'.format(timestep))
        elif timestep is None and decision_iteration is not None:
            fname = os.path.join(
                results_dir, modelrun_name,
                'state_0000_decision_{}.csv'.format(decision_iteration))
        else:
            fname = os.path.join(
                results_dir, modelrun_name,
                'state_{}_decision_{}.csv'.format(timestep, decision_iteration))

        return fname

    @staticmethod
    def _read_state_file(fname):
        """Read list of name, build_year from state file

        Returns
        -------
        dict
            Keys of dict are header names from csv file
        """
        with open(fname, 'r') as file_handle:
            reader = csv.DictReader(file_handle)
            state = list(reader)
        return state

    def read_region_definitions(self):
        """Read region_definitions from project configuration

        Returns
        -------
        list
            A list of region_definition dicts
        """
        project_config = self._read_project_config()
        return project_config['region_definitions']

    def _region_definition_exists(self, region_definition_name):
        for region_definition in self.read_region_definitions():
            if region_definition['name'] == region_definition_name:
                return region_definition

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
        region_definition = self._region_definition_exists(region_definition_name)

        if region_definition is None:
            raise DataNotFoundError(
                "Region definition '{}' not found".format(region_definition_name))
        else:
            filename = region_definition['filename']

        # Read the region data from file
        filepath = os.path.join(self.file_dir['region_definitions'], filename)
        with fiona.drivers():
            with fiona.open(filepath) as src:
                data = [f for f in src]

        return data

    def read_region_names(self, region_definition_name):
        """Return the set of unique region names in region set `region_definition_name`
        """
        names = []
        for feature in self.read_region_definition_data(region_definition_name):
            if isinstance(feature['properties']['name'], str):
                if feature['properties']['name'].isdigit():
                    names.append(int(feature['properties']['name']))
                else:
                    names.append(feature['properties']['name'])
            else:
                names.append(feature['properties']['name'])

        return names

    def write_region_definition(self, region_definition):
        """Write region_definition to project configuration

        Arguments
        ---------
        region_definition: dict
            A region_definition dict
        """

        if self._region_definition_exists(region_definition['name']):
            raise DataExistsError(
                "region_definition '%s' already exists" % region_definition['name'])

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
        previous_region_definition = self._region_definition_exists(region_definition_name)
        if previous_region_definition is None:
            raise DataNotFoundError(
                "region_definition '%s' does not exist" % region_definition_name)

        # Update
        project_config = self._read_project_config()

        idx = None
        for i, existing_region_definition in enumerate(project_config['region_definitions']):
            if existing_region_definition['name'] == region_definition_name:
                # Guaranteed to match thanks to existence check above
                idx = i
                break

        project_config['region_definitions'][idx] = region_definition

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

    def _interval_definition_exists(self, interval_definition_name):
        for interval_definition in self.read_interval_definitions():
            if interval_definition['name'] == interval_definition_name:
                return interval_definition

    def read_interval_definition_data(self, interval_definition_name):
        """Read data for an interval definition

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
        interval_definition = self._interval_definition_exists(interval_definition_name)

        if interval_definition is None:
            raise DataNotFoundError("Interval set definition '{}' does not exist".format(
                interval_definition_name))

        filename = interval_definition['filename']
        filepath = os.path.join(self.file_dir['interval_definitions'], filename)

        names = {}

        with open(filepath, 'r') as csvfile:
            reader = DictReader(csvfile)
            data = []
            for interval in reader:

                if interval['id'].isdigit():
                    name = int(interval['id'])
                else:
                    name = interval['id']
                interval_tuple = (interval['start'], interval['end'])
                if name in names:
                    # Append duration to existing entry
                    data[names[name]][1].append(interval_tuple)
                else:
                    # Make a new entry
                    data.append((name, [interval_tuple]))
                    names[name] = len(data) - 1

        return data

    def read_interval_names(self, interval_definition_name):
        return [
            interval[0]
            for interval
            in self.read_interval_definition_data(interval_definition_name)
        ]

    def write_interval_definition(self, interval_definition):
        """Write interval_definition to project configuration

        Arguments
        ---------
        interval_definition: dict
            A interval_definition dict
        """
        if self._interval_definition_exists(interval_definition['name']):
            raise DataExistsError(
                "Interval definition '%s' already exists" % interval_definition['name'])

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
        if not self._interval_definition_exists(interval_definition_name):
            raise DataNotFoundError(
                "Interval definition '%s' does not exist" % interval_definition_name)

        project_config = self._read_project_config()

        # Create updated list
        idx = None
        for i, existing_interval_def in enumerate(project_config['interval_definitions']):
            if existing_interval_def['name'] == interval_definition_name:
                # Guaranteed to match thanks to existence check above
                idx = i
                break

        project_config['interval_definitions'][idx] = interval_definition

        self._write_project_config(project_config)

    def read_scenario_definitions(self):
        project_config = self._read_project_config()
        return project_config['scenarios']

    def _scenario_definition_exists(self, scenario_name):
        for scenario in self.read_scenario_definitions():
            if scenario['name'] == scenario_name:
                return scenario

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
        # Filter only the scenarios of the selected scenario_set_name
        filtered_scenario_data = [
            data for data in self.read_scenario_definitions()
            if data['scenario_set'] == scenario_set_name
        ]

        if not filtered_scenario_data:
            self.logger.warning(
                "Scenario set '%s' has no scenarios defined", scenario_set_name)

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
        scenario_definition = self._scenario_definition_exists(scenario_name)

        if scenario_definition is None:
            raise DataNotFoundError(
                "Scenario definition '{}' not found".format(scenario_name))

        return scenario_definition

    def read_scenario_sets(self):
        """Read scenario sets from project configuration

        Returns
        -------
        list
            A list of scenario set dicts
        """
        project_config = self._read_project_config()
        return project_config['scenario_sets']

    def _scenario_set_exists(self, scenario_set_name):
        for scenario_set in self.read_scenario_sets():
            if scenario_set['name'] == scenario_set_name:
                return scenario_set

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
        scenario_set = self._scenario_set_exists(scenario_set_name)

        if scenario_set is None:
            raise DataNotFoundError(
                "Scenario set '{}' not found".format(scenario_set_name))

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

        idx = None
        for i, existing_scenario_set in enumerate(project_config['scenario_sets']):
            if existing_scenario_set['name'] == scenario_set_name:
                # Guaranteed to find a match thanks to existence assertion above
                idx = i
                break

        project_config['scenario_sets'][idx] = scenario_set

        self._write_project_config(project_config)

    def delete_scenario_set(self, scenario_set_name):
        """Delete scenario_set from project configuration
        and all scenarios within scenario_set

        Arguments
        ---------
        scenario_set_name: str
            A scenario_set name
        """
        if not self._scenario_set_exists(scenario_set_name):
            raise DataNotFoundError("scenario_set '%s' not found" % scenario_set_name)

        project_config = self._read_project_config()

        project_config['scenarios'] = [
            entry for entry in project_config['scenarios']
            if entry['scenario_set'] != scenario_set_name
        ]

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

        idx = None
        for i, existing_scenario in enumerate(project_config['scenarios']):
            if existing_scenario['name'] == scenario_name:
                # Guaranteed to match given existence check above
                idx = i
                break

        project_config['scenarios'][idx] = scenario

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

    def read_scenario_data(self, scenario_name, facet_name,
                           spatial_resolution, temporal_resolution, timestep):
        """Read scenario data file

        Arguments
        ---------
        scenario_name: str
            Name of the scenario
        facet_name: str
            Name of the scenario facet to read
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
                for facet in scenario_data['facets']:
                    if facet['name'] == facet_name:
                        filename = facet['filename']
                        break
                break

        if filename is None:
            raise DataNotFoundError(
                "Scenario '{}' with facet '{}' not found".format(
                    scenario_name, facet_name))

        # Read the scenario data from file
        filepath = os.path.join(self.file_dir['scenarios'], filename)
        data = [
            datum for datum in
            self._get_data_from_csv(filepath)
            if int(datum['year']) == timestep
        ]

        # Position of names in these lists dictates position of
        # data in data array
        region_names = self.read_region_names(spatial_resolution)
        interval_names = self.read_interval_names(temporal_resolution)

        try:
            array = self.data_list_to_ndarray(data, region_names, interval_names)
        except DataMismatchError:
            msg = "DataMismatch in scenario: '{}' and facet:'{}'"
            raise DataMismatchError(msg.format(scenario_name, facet_name))
        else:
            return array

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

        idx = None
        for i, existing_narrative_set in enumerate(project_config['narrative_sets']):
            if existing_narrative_set['name'] == narrative_set_name:
                # Guaranteed to match thanks to existence check above
                idx = i
                break

        project_config['narrative_sets'][idx] = narrative_set

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

        idx = None
        for i, existing_narrative in enumerate(project_config['narratives']):
            if existing_narrative['name'] == narrative_name:
                # Guaranteed to match given existence check
                idx = i
                break

        project_config['narratives'][idx] = narrative

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

    def read_coefficients(self, source_name, destination_name):
        """Reads coefficients from file on disk

        Coefficients are uniquely identified by their source/destination names

        """
        results_path = self._get_coefficients_path(source_name, destination_name)
        if os.path.isfile(results_path):
            return self._get_data_from_native_file(results_path)
        else:
            msg = "Could not find the coefficients file for %s to %s"
            self.logger.warning(msg, source_name, destination_name)
            return None

    def write_coefficients(self, source_name, destination_name, data):
        """Writes coefficients to file on disk

        Coefficients are uniquely identified by their source/destination names

        """
        results_path = self._get_coefficients_path(source_name, destination_name)
        self._write_data_to_native_file(results_path, data)

    def _get_coefficients_path(self, source_name, destination_name):

        results_dir = self.file_dir['coefficients']
        path = os.path.join(results_dir, source_name + '_' + destination_name)
        return path + '.dat'

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
            modelrun_id, model_name, output_name, spatial_resolution,
            temporal_resolution,
            timestep, modelset_iteration, decision_iteration)

        if self.storage_format == 'local_csv':
            csv_data = self._get_data_from_csv(results_path)
            region_names = self.read_region_names(spatial_resolution)
            interval_names = self.read_interval_names(temporal_resolution)
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
            modelrun_id, model_name, output_name, spatial_resolution,
            temporal_resolution,
            timestep, modelset_iteration, decision_iteration)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        if data.ndim == 3:
            raise NotImplementedError
        elif data.ndim == 2:
            region_names = self.read_region_names(spatial_resolution)
            interval_names = self.read_interval_names(temporal_resolution)
            assert data.shape == (len(region_names), len(interval_names))

            if self.storage_format == 'local_csv':
                csv_data = self.ndarray_to_data_list(
                    data, region_names, interval_names, timestep=timestep)
                self._write_data_to_csv(results_path, csv_data)
            elif self.storage_format == 'local_binary':
                self._write_data_to_native_file(results_path, data)
        else:
            raise DataMismatchError(
                "Expected to write either timestep x region x interval or " +
                "region x interval data"
            )

    def results_exist(self, modelrun_name):
        """Checks whether modelrun results exists on the filesystem
        for a particular modelrun_name

        Parameters
        ----------
        modelrun_name: str

        Returns
        -------
        bool: True when results exist for this modelrun_name
        """
        previous_results_dir = os.path.join(self.file_dir['results'],
                                            modelrun_name)
        results = list(glob.iglob(os.path.join(previous_results_dir, '**/*.*'),
                                  recursive=True))

        return len(results) > 0

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

        # Return if path to previous modelruns does not exist
        if not os.path.isdir(results_dir):
            self.logger.info("Warm start not possible because modelrun has "
                             "no previous results (path does not exist)")
            return None

        # Return if no results exist in last modelrun
        if not self.results_exist(modelrun_id):
            self.logger.info("Warm start not possible because the "
                             "modelrun does not have any results")
            return None

        # Return if previous results were stored in a different format
        previous_results_dir = os.path.join(self.file_dir['results'], modelrun_id)
        results = list(glob.iglob(os.path.join(previous_results_dir, '**/*.*'),
                                  recursive=True))
        for filename in results:
            if (
                    (self.storage_format == 'local_csv' and
                        not filename.endswith(".csv")) or
                    (self.storage_format == 'local_binary' and
                        not filename.endswith(".dat"))
                        ):
                self.logger.info("Warm start not possible because a different "
                                 "storage mode was used in the previous run")
                return None

        # Perform warm start
        self.logger.info("Warm start " + modelrun_id)

        # Get metadata for all results
        result_metadata = []
        for filename in glob.iglob(os.path.join(results_dir, '**/*.*'),
                                   recursive=True):
            result_metadata.append(self._parse_results_path(
                filename.replace(self.file_dir['results'], '')[1:]))

        # Find latest timestep
        result_metadata = sorted(result_metadata, key=lambda k: k['timestep'],
                                 reverse=True)
        latest_timestep = result_metadata[0]['timestep']

        # Remove all results with this timestep
        results_to_remove = \
            [result for result in result_metadata
             if result['timestep'] == latest_timestep]

        for result in results_to_remove:
            os.remove(self._get_results_path(result['modelrun_id'],
                      result['model_name'],
                      result['output_name'],
                      result['spatial_resolution'],
                      result['temporal_resolution'],
                      result['timestep'],
                      result['modelset_iteration'],
                      result['decision_iteration']))

        self.logger.info("Warm start will resume at timestep %s",
                         latest_timestep)
        return latest_timestep

    def _get_results_path(self, modelrun_id, model_name, output_name,
                          spatial_resolution,
                          temporal_resolution, timestep, modelset_iteration=None,
                          decision_iteration=None):
        """Return path to filename for a given output without file extension

        On the pattern of:
            results/
            <modelrun_name>/
            <model_name>/
            decision_<id>_modelset_<id>/ or decision_<id>/ or modelset_<id>/ or none
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv

        Parameters
        ----------
        modelrun_id : str
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

        for section in data[2:len(data)]:
            if 'modelset' in section or 'decision' in section:
                regex_decision = re.findall(r"decision_(\d{1,})", section)
                regex_modelset = re.findall(r"modelset_(\d{1,})", section)
                if regex_decision:
                    decision_iteration = int(regex_decision[0])
                if regex_decision:
                    modelset_iteration = int(regex_modelset[0])
            elif section.startswith('output'):
                results = self._parse_output_section(section)
            elif section == 'csv':
                storage_format = 'local_csv'
            elif section == 'dat':
                storage_format = 'local_binary'

        return {
            'modelrun_id': data[0],
            'model_name': data[1],
            'output_name': '_'.join(results['output']),
            'spatial_resolution': '_'.join(results['regions']),
            'temporal_resolution': '_'.join(results['intervals']),
            'timestep': results['timestep'],
            'modelset_iteration': modelset_iteration,
            'decision_iteration': decision_iteration,
            'storage_format': storage_format
        }

    def _parse_output_section(self, section):
        result_elements = re.findall(r"[^_]+", section)
        results = {}
        parse_element = ""
        for element in result_elements:
            if element in ('output', 'timestep', 'regions', 'intervals') and \
                    parse_element != element:
                parse_element = element
            elif parse_element == 'output':
                results.setdefault('output', []).append(element)
            elif parse_element == 'timestep':
                results['timestep'] = int(element)
            elif parse_element == 'regions':
                results.setdefault('regions', []).append(element)
            elif parse_element == 'intervals':
                results.setdefault('intervals', []).append(element)
        return results

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
                converted_row = {}

                converted_row['region'] = DatafileInterface._cast_str_to_int(row['region'])
                converted_row['interval'] = DatafileInterface._cast_str_to_int(row['interval'])

                if 'year' in row.keys():
                    converted_row['year'] = row['year']
                converted_row['value'] = row['value']

                scenario_data.append(converted_row)

        return scenario_data

    @staticmethod
    def _cast_str_to_int(value):
        if value.isdigit():
            return int(value)
        else:
            return value

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
            f.write(
                pa.serialize(data).to_buffer()
            )

    @staticmethod
    def _read_yaml_file(path, filename=None, extension='.yml'):
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
        if filename is not None:
            filename = filename + extension
            filepath = os.path.join(path, filename)
        else:
            filepath = path
        return load(filepath)

    @staticmethod
    def _write_yaml_file(path, filename=None, data=None, extension='.yml'):
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
        if filename is not None:
            filename = filename + extension
            filepath = os.path.join(path, filename)
        else:
            filepath = path
        dump(data, filepath)
