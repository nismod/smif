"""File-backed data interface
"""
import copy
import csv
import glob
import os
import re
from functools import lru_cache, reduce
from logging import getLogger

import numpy as np
import pyarrow as pa
from ruamel.yaml import YAML
from smif.data_layer.abstract_config_store import ConfigStore
from smif.data_layer.abstract_data_store import DataStore
from smif.data_layer.abstract_metadata_store import MetadataStore
from smif.data_layer.data_array import DataArray
from smif.data_layer.validate import (validate_sos_model_config,
                                      validate_sos_model_format)
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError, SmifDataReadError,
                            SmifValidationError)

# Import fiona if available (optional dependency)
try:
    import fiona
except ImportError:
    pass


class YamlConfigStore(ConfigStore):
    """Config backend saving to YAML configuration files.

    Arguments
    ---------
    base_folder: str
        The path to the configuration and data files
    """
    def __init__(self, base_folder, validation=False):
        super().__init__()
        self.logger = getLogger(__name__)
        self.validation = validation

        self.base_folder = str(base_folder)
        self.config_folder = str(os.path.join(self.base_folder, 'config'))
        self.config_folders = {}
        config_folders = [
            'dimensions',
            'model_runs',
            'scenarios',
            'sector_models',
            'sos_models',
        ]
        for folder in config_folders:
            dirname = os.path.join(self.config_folder, folder)
            # ensure each directory exists
            if not os.path.exists(dirname):
                msg = "Expected configuration folder at '{}' but it does not exist"
                abs_path = os.path.abspath(dirname)
                raise SmifDataNotFoundError(msg.format(abs_path))

            self.config_folders[folder] = dirname

        # cache results of reading project_config (invalidate on write)
        self._project_config_cache_invalid = True
        # MUST ONLY access through self.read_project_config()
        self._project_config_cache = None

        # ensure project config file exists
        try:
            self.read_project_config()
        except FileNotFoundError:
            # write empty config if none found
            self._write_project_config({})

    def read_project_config(self):
        """Read the project configuration

        Returns
        -------
        dict
            The project configuration
        """
        if self._project_config_cache_invalid:
            self._project_config_cache = _read_yaml_file(
                self.base_folder, 'project')
            self._project_config_cache_invalid = False
        return copy.deepcopy(self._project_config_cache)

    def _write_project_config(self, data):
        """Write the project configuration

        Argument
        --------
        data: dict
            The project configuration
        """
        self._project_config_cache_invalid = True
        self._project_config_cache = None
        _write_yaml_file(self.base_folder, 'project', data)

    def _read_config(self, config_type, config_name):
        """Read config item - used by decorators for existence/consistency checks
        """
        if config_type == 'scenario':
            return self.read_scenario(config_name)
        else:
            raise NotImplementedError(
                "Cannot read %s:%s through generic method." % (config_type, config_name))

    # region Model runs
    def read_model_runs(self):
        names = _read_filenames_in_dir(self.config_folders['model_runs'], '.yml')
        sorted_names = sorted(names)
        model_runs = [self.read_model_run(name) for name in sorted_names]
        return model_runs

    def read_model_run(self, model_run_name):
        _assert_file_exists(self.config_folders, 'model_run', model_run_name)
        modelrun_config = self._read_model_run(model_run_name)
        del modelrun_config['strategies']
        return modelrun_config

    def _read_model_run(self, model_run_name):
        return _read_yaml_file(self.config_folders['model_runs'], model_run_name)

    def _overwrite_model_run(self, model_run_name, model_run):
        _write_yaml_file(self.config_folders['model_runs'], model_run_name, model_run)

    def write_model_run(self, model_run):
        _assert_file_not_exists(self.config_folders, 'model_run', model_run['name'])
        config = copy.copy(model_run)
        config['strategies'] = []
        _write_yaml_file(self.config_folders['model_runs'], config['name'], config)

    def update_model_run(self, model_run_name, model_run):
        _assert_file_exists(self.config_folders, 'model_run', model_run_name)
        prev = self._read_model_run(model_run_name)
        config = copy.copy(model_run)
        config['strategies'] = prev['strategies']
        self._overwrite_model_run(model_run_name, config)

    def delete_model_run(self, model_run_name):
        _assert_file_exists(self.config_folders, 'model_run', model_run_name)
        os.remove(os.path.join(self.config_folders['model_runs'], model_run_name + '.yml'))
    # endregion

    # region System-of-system models
    def read_sos_models(self):
        names = _read_filenames_in_dir(self.config_folders['sos_models'], '.yml')
        sos_models = [self.read_sos_model(name) for name in names]
        return sos_models

    def read_sos_model(self, sos_model_name):
        _assert_file_exists(self.config_folders, 'sos_model', sos_model_name)
        data = _read_yaml_file(self.config_folders['sos_models'], sos_model_name)
        if self.validation:
            validate_sos_model_format(data)
        return data

    def write_sos_model(self, sos_model):
        _assert_file_not_exists(self.config_folders, 'sos_model', sos_model['name'])
        _write_yaml_file(self.config_folders['sos_models'], sos_model['name'], sos_model)

    def update_sos_model(self, sos_model_name, sos_model):
        _assert_file_exists(self.config_folders, 'sos_model', sos_model_name)
        if self.validation:
            validate_sos_model_config(
                sos_model,
                self.read_models(),
                self.read_scenarios(),
            )
        _write_yaml_file(self.config_folders['sos_models'], sos_model['name'], sos_model)

    def delete_sos_model(self, sos_model_name):
        _assert_file_exists(self.config_folders, 'sos_model', sos_model_name)
        os.remove(os.path.join(self.config_folders['sos_models'], sos_model_name + '.yml'))
    # endregion

    # region Models
    def read_models(self):
        names = _read_filenames_in_dir(self.config_folders['sector_models'], '.yml')
        models = [self.read_model(name) for name in names]
        return models

    def read_model(self, model_name):
        _assert_file_exists(self.config_folders, 'sector_model', model_name)
        model = _read_yaml_file(self.config_folders['sector_models'], model_name)
        return model

    def write_model(self, model):
        _assert_file_not_exists(self.config_folders, 'sector_model', model['name'])
        model = copy.deepcopy(model)
        if model['interventions']:
            self.logger.warning("Ignoring interventions")
            model['interventions'] = []

        model = _skip_coords(model, ('inputs', 'outputs', 'parameters'))
        _write_yaml_file(
            self.config_folders['sector_models'], model['name'], model)

    def update_model(self, model_name, model):
        _assert_file_exists(self.config_folders, 'sector_model', model_name)
        model = copy.deepcopy(model)
        # ignore interventions and initial conditions which the app doesn't handle
        if model['interventions'] or model['initial_conditions']:
            old_model = _read_yaml_file(
                self.config_folders['models'], model['name'])

        if model['interventions']:
            self.logger.warning("Ignoring interventions write")
            model['interventions'] = old_model['interventions']

        if model['initial_conditions']:
            self.logger.warning("Ignoring initial conditions write")
            model['initial_conditions'] = old_model['initial_conditions']

        model = _skip_coords(model, ('inputs', 'outputs', 'parameters'))

        _write_yaml_file(
            self.config_folders['sector_models'], model['name'], model)

    def delete_model(self, model_name):
        _assert_file_exists(self.config_folders, 'sector_model', model_name)
        os.remove(
            os.path.join(self.config_folders['sector_models'], model_name + '.yml'))
    # endregion

    # region Scenarios
    def read_scenarios(self):
        scenario_names = _read_filenames_in_dir(self.config_folders['scenarios'], '.yml')
        return [self.read_scenario(name) for name in scenario_names]

    def read_scenario(self, scenario_name):
        _assert_file_exists(self.config_folders, 'scenario', scenario_name)
        scenario = _read_yaml_file(self.config_folders['scenarios'], scenario_name)
        return scenario

    def write_scenario(self, scenario):
        _assert_file_not_exists(self.config_folders, 'scenario', scenario['name'])
        scenario = _skip_coords(scenario, ['provides'])
        _write_yaml_file(self.config_folders['scenarios'], scenario['name'], scenario)

    def update_scenario(self, scenario_name, scenario):
        _assert_file_exists(self.config_folders, 'scenario', scenario_name)
        scenario = _skip_coords(scenario, ['provides'])
        _write_yaml_file(self.config_folders['scenarios'], scenario['name'], scenario)

    def delete_scenario(self, scenario_name):
        _assert_file_exists(self.config_folders, 'scenario', scenario_name)
        os.remove(
            os.path.join(self.config_folders['scenarios'], "{}.yml".format(scenario_name)))
    # endregion

    # region Scenario Variants
    def read_scenario_variants(self, scenario_name):
        scenario = self.read_scenario(scenario_name)
        return scenario['variants']

    def read_scenario_variant(self, scenario_name, variant_name):
        variants = self.read_scenario_variants(scenario_name)
        return _pick_from_list(variants, variant_name)

    def write_scenario_variant(self, scenario_name, variant):
        scenario = self.read_scenario(scenario_name)
        scenario['variants'].append(variant)
        self.update_scenario(scenario_name, scenario)

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        scenario = self.read_scenario(scenario_name)
        v_idx = _idx_in_list(scenario['variants'], variant_name)
        scenario['variants'][v_idx] = variant
        self.update_scenario(scenario_name, scenario)

    def delete_scenario_variant(self, scenario_name, variant_name):
        scenario = self.read_scenario(scenario_name)
        v_idx = _idx_in_list(scenario['variants'], variant_name)
        del scenario['variants'][v_idx]
        self.update_scenario(scenario_name, scenario)
    # endregion

    # region Narratives
    def read_narrative(self, sos_model_name, narrative_name):
        sos_model = self.read_sos_model(sos_model_name)
        narrative = _pick_from_list(sos_model['narratives'], narrative_name)
        if not narrative:
            msg = "Narrative '{}' not found in '{}'"
            raise SmifDataNotFoundError(msg.format(narrative_name, sos_model_name))
        return narrative
    # endregion

    # region Strategies
    def read_strategies(self, modelrun_name):
        model_run_config = self._read_model_run(modelrun_name)
        return model_run_config['strategies']

    def write_strategies(self, modelrun_name, strategies):
        model_run = self._read_model_run(modelrun_name)
        model_run['strategies'] = strategies
        self._overwrite_model_run(modelrun_name, model_run)
    # endregion


class FileMetadataStore(MetadataStore):
    """Various file-based metadata store (YAML/GDAL-compatible)
    """
    def __init__(self, base_folder):
        super().__init__()
        self.logger = getLogger(__name__)

        self.units_path = os.path.join(base_folder, 'data', 'user-defined-units.txt')
        self.data_folder = os.path.join(base_folder, 'data', 'dimensions')
        self.config_folder = os.path.join(base_folder, 'config', 'dimensions')

    # region Units
    def read_unit_definitions(self):
        try:
            with open(self.units_path, 'r') as units_fh:
                return [line.strip() for line in units_fh]
        except FileNotFoundError:
            self.logger.warn('Units file not found, expected at %s', str(self.units_path))
            return []

    def write_unit_definitions(self, units):
        with open(self.units_path, 'w') as units_fh:
            units_fh.writelines(units)
    # endregion

    # region Dimensions
    def read_dimensions(self):
        dim_names = _read_filenames_in_dir(self.config_folder, '.yml')
        return [self.read_dimension(name) for name in dim_names]

    def read_dimension(self, dimension_name):
        dim = _read_yaml_file(self.config_folder, dimension_name)
        dim['elements'] = self._read_dimension_file(dim['elements'])
        return dim

    def write_dimension(self, dimension):
        # write elements to yml file (by default, can handle any nested data)
        elements_filename = "{}.yml".format(dimension['name'])
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and add to config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename
        _write_yaml_file(
            self.config_folder, dimension['name'], dimension_with_ref)

    def update_dimension(self, dimension_name, dimension):
        # look up elements filename and write elements
        old_dim = _read_yaml_file(self.config_folder, dimension_name)
        elements_filename = old_dim['elements']
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and update config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename
        _write_yaml_file(
            self.config_folder, dimension_name, dimension_with_ref)

    def delete_dimension(self, dimension_name):
        # read to find filename
        old_dim = _read_yaml_file(self.config_folder, dimension_name)
        elements_filename = old_dim['elements']
        # remove elements data
        os.remove(os.path.join(self.data_folder, elements_filename))
        # remove description
        os.remove(
            os.path.join(self.config_folder, "{}.yml".format(dimension_name)))

    @lru_cache(maxsize=32)
    def _read_dimension_file(self, filename):
        filepath = os.path.join(self.data_folder, filename)
        _, ext = os.path.splitext(filename)
        if ext in ('.yml', '.yaml'):
            data = _read_yaml_file(filepath)
        elif ext == '.csv':
            data = _get_data_from_csv(filepath)
        elif ext in ('.geojson', '.shp'):
            data = self._read_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', '.yml', '.yaml', "
            msg += "'.geojson', '.shp') when reading {}"
            raise SmifDataReadError(msg.format(ext, filepath))
        return data

    def _write_dimension_file(self, filename, data):
        # lru_cache may now be invalid, so clear it
        self._read_dimension_file.cache_clear()
        filepath = os.path.join(self.data_folder, filename)
        _, ext = os.path.splitext(filename)
        if ext in ('.yml', '.yaml'):
            _write_yaml_file(filepath, data=data)
        elif ext == '.csv':
            _write_data_to_csv(filepath, data)
        elif ext in ('.geojson', '.shp'):
            raise NotImplementedError("Writing spatial dimensions not yet supported")
            # self._write_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', '.yml', '.yaml', "
            msg += "'.geojson', '.shp') when writing {}"
            raise SmifDataReadError(msg.format(ext, filepath))
        return data
    # endregion

    @staticmethod
    def _read_spatial_file(filepath):
        try:
            with fiona.drivers():
                with fiona.open(filepath) as src:
                    data = []
                    for f in src:
                        element = {
                            'name': f['properties']['name'],
                            'feature': f
                        }
                        data.append(element)
            return data
        except NameError as ex:
            msg = "Could not read spatial dimension definition. Please install fiona to read"
            msg += "geographic data files. Try running: \n"
            msg += "    pip install smif[spatial]\n"
            msg += "or:\n"
            msg += "    conda install fiona shapely rtree\n"
            raise SmifDataReadError(msg) from ex
        except IOError as ex:
            msg = "Could not read spatial dimension definition. '%s'" % (filepath)
            msg += "Please verify that the path is correct and "
            msg += "that the file is present on this location."
            raise SmifDataNotFoundError(msg) from ex


class CSVDataStore(DataStore):
    """CSV text file data store
    """
    def __init__(self, base_folder):
        super().__init__()
        self.logger = getLogger(__name__)
        self.base_folder = str(base_folder)
        self.data_folder = str(os.path.join(self.base_folder, 'data'))
        self.data_folders = {}
        self.results_folder = str(os.path.join(self.base_folder, 'results'))
        data_folders = [
            'coefficients',
            'strategies',
            'initial_conditions',
            'initial_inputs',
            'interventions',
            'narratives',
            'scenarios',
            'strategies',
            'parameters'
        ]
        for folder in data_folders:
            dirname = os.path.join(self.data_folder, folder)
            # ensure each directory exists
            if not os.path.exists(dirname):
                msg = "Expected data folder at '{}' but it does does not exist"
                abs_path = os.path.abspath(dirname)
                raise SmifDataNotFoundError(msg.format(abs_path))
            self.data_folders[folder] = dirname

    # region Data Array
    def read_scenario_variant_data(self, key, spec, timestep=None):
        path = os.path.join(self.data_folder, 'scenarios', key)
        return self._read_data_array(path, spec, timestep)

    def write_scenario_variant_data(self, key, data, timestep=None):
        path = os.path.join(self.data_folder, 'scenarios', key)
        self._write_data_array(path, data, timestep)

    def read_narrative_variant_data(self, key, spec, timestep=None):
        path = os.path.join(self.data_folder, 'narratives', key)
        return self._read_data_array(path, spec, timestep)

    def write_narrative_variant_data(self, key, data, timestep=None):
        path = os.path.join(self.data_folder, 'narratives', key)
        self._write_data_array(path, data, timestep)

    def _read_data_array(self, key, spec, timestep=None):
        try:
            data = _get_data_from_csv(key)
        except FileNotFoundError:
            raise SmifDataNotFoundError
        if timestep:
            if 'timestep' not in data[0].keys():
                msg = "Header in '{}' missing 'timestep' key. Found {}"
                raise SmifDataMismatchError(msg.format(key, list(data[0].keys())))
            data = [datum for datum in data if int(datum['timestep']) == timestep]

        try:
            da = data_list_to_ndarray(data, spec)
        except SmifDataMismatchError as ex:
            msg = "DataMismatch in key: {}, from {}"
            raise SmifDataMismatchError(
                msg.format(key, str(ex))
            ) from ex

        return da

    def _write_data_array(self, key, data_array, timestep=None):
        spec = data_array.spec
        data = ndarray_to_data_list(data_array, timestep=timestep)

        if timestep:
            fieldnames = ('timestep', ) + tuple(spec.dims) + (spec.name, )
            self.logger.debug("%s, %s", fieldnames, data)
            _write_data_to_csv(key, data, fieldnames=fieldnames)
        else:
            _write_data_to_csv(key, data, spec=spec)
    # endregion

    # region Interventions
    def read_interventions(self, keys):
        all_interventions = {}
        interventions = self._read_files(keys, os.path.join(self.data_folder, 'interventions'))
        for entry in interventions:
            name = entry.pop('name')
            if name in all_interventions:
                msg = "An entry for intervention {} already exists"
                raise ValueError(msg.format(name))
            else:
                all_interventions[name] = entry
        return all_interventions

    def write_interventions(self, key, interventions):
        data = [interventions[intervention] for intervention in interventions.keys()]
        _write_data_to_csv(os.path.join(self.data_folder, 'interventions', key), data)

    def read_initial_conditions(self, keys):
        return self._read_file(keys, os.path.join(self.data_folder, 'initial_conditions'))

    def write_initial_conditions(self, key, initial_conditions):
        data = initial_conditions
        _write_data_to_csv(os.path.join(self.data_folder, 'initial_conditions', key), data)

    def _read_files(self, keys, dirname):
        data_list = []
        for key in keys:
            interventions = self._read_file(key, dirname)
            data_list.extend(interventions)
        return data_list

    def _read_file(self, filename, dirname):
        """Read data from a file

        Arguments
        ---------
        filename: str
            The name of the csv file to read in
        dirname: str
            The key of the dirname e.g. ``strategies`` or ``initial_conditions``

        Returns
        -------
        dict of dict
            Dict of intervention attribute dicts, keyed by intervention name
        """
        data = _get_data_from_csv(os.path.join(dirname, filename))
        try:
            data = self._reshape_csv_interventions(data)
        except ValueError:
            raise ValueError("Error reshaping data for {}".format(filename))

        return data

    def _reshape_csv_interventions(self, data):
        """

        Arguments
        ---------
        data : list of dict
            A list of dicts containing intervention data

        Returns
        -------
        dict of dicts
        """
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
                    if key in reshaped_data:
                        msg = "Duplicate heading in csv data: {}"
                        raise ValueError(msg.format(new_key))
                    else:
                        reshaped_data[key] = value
            new_data.append(reshaped_data)
        return new_data
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        fname = self._get_state_filename(modelrun_name, timestep, decision_iteration)
        if not os.path.exists(fname):
            msg = "State file does not exist for timestep {} and iteration {}"
            raise SmifDataNotFoundError(msg.format(timestep, decision_iteration))
        state = self._read_state_file(fname)
        return state

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        fname = self._get_state_filename(modelrun_name, timestep, decision_iteration)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w+') as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=('name', 'build_year'))
            writer.writeheader()
            for row in state:
                writer.writerow(row)

    def _get_state_filename(self, modelrun_name, timestep=None, decision_iteration=None):
        """Compose a unique filename for state file:
                state_{timestep|0000}[_decision_{iteration}].csv
        """
        if timestep is None:
            timestep = '0000'

        if decision_iteration is None:
            separator = ''
            decision_iteration = ''
        else:
            separator = '_decision_'

        filename = 'state_{}{}{}.csv'.format(timestep, separator, decision_iteration)
        path = os.path.join(self.results_folder, modelrun_name, filename)

        return path

    def _read_state_file(self, fname):
        """Read list of {name, build_year} dicts from state file
        """
        with open(fname, 'r') as file_handle:
            reader = csv.DictReader(file_handle)
            state = []
            for line in reader:
                try:
                    item = {
                        'name': line['name'],
                        'build_year': int(line['build_year'])
                    }
                except KeyError:
                    msg = "Interventions must have name and build year, got {} in {}"
                    raise SmifDataReadError(msg.format(line, fname))
                state.append(item)

        return state
    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        results_path = self._get_coefficients_path(source_spec, destination_spec)
        try:
            return _get_data_from_native_file(results_path)
        except (FileNotFoundError, pa.lib.ArrowIOError):
            msg = "Could not find the coefficients file for %s to %s"
            self.logger.warning(msg, source_spec, destination_spec)
            raise SmifDataNotFoundError(msg.format(source_spec, destination_spec))

    def write_coefficients(self, source_spec, destination_spec, data):
        results_path = self._get_coefficients_path(source_spec, destination_spec)
        _write_data_to_native_file(results_path, data)

    def _get_coefficients_path(self, source_spec, destination_spec):
        path = os.path.join(
            self.data_folders['coefficients'],
            "{}_{}.{}_{}.dat".format(
                source_spec.name, "-".join(source_spec.dims),
                destination_spec.name, "-".join(destination_spec.dims)
            )
        )
        return path
    # endregion

    # region Results
    def read_results(self, modelrun_id, model_name, output_spec, timestep,
                     decision_iteration=None):
        if timestep is None:
            raise ValueError("You must pass a timestep argument")

        results_path = self._get_results_path(
            modelrun_id, model_name, output_spec.name,
            timestep, decision_iteration
        )

        try:
            data = _get_data_from_csv(results_path)
            return data_list_to_ndarray(data, output_spec)
        except (FileNotFoundError):
            key = str([modelrun_id, model_name, output_spec.name, timestep,
                       decision_iteration])
            raise SmifDataNotFoundError("Could not find results for {}".format(key))

    def write_results(self, data_array, modelrun_id, model_name, timestep=None,
                      decision_iteration=None):
        if timestep is None:
            raise NotImplementedError()

        if timestep:
            assert isinstance(timestep, int)
        if decision_iteration:
            assert isinstance(decision_iteration, int)

        spec = data_array.spec

        results_path = self._get_results_path(
            modelrun_id, model_name, data_array.name,
            timestep, decision_iteration
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        _data = ndarray_to_data_list(data_array)
        _write_data_to_csv(results_path, _data, spec=spec)

    def available_results(self, modelrun_name):
        return None

    def _results_exist(self, modelrun_name):
        """Checks whether modelrun results exists on the filesystem
        for a particular modelrun_name

        Parameters
        ----------
        modelrun_name: str

        Returns
        -------
        bool: True when results exist for this modelrun_name
        """
        previous_results_dir = os.path.join(self.results_folder, modelrun_name)
        return list(
            glob.iglob(os.path.join(previous_results_dir, '**/*.*'), recursive=True))

    def prepare_warm_start(self, modelrun_id):
        results_dir = os.path.join(self.results_folder, modelrun_id)

        # Return if path to previous modelruns does not exist
        if not os.path.isdir(results_dir):
            self.logger.info("Warm start not possible because modelrun has "
                             "no previous results (path does not exist)")
            return None

        # Return if no results exist in last modelrun
        if not self._results_exist(modelrun_id):
            self.logger.info("Warm start not possible because the "
                             "modelrun does not have any results")
            return None

        # Return if previous results were stored in a different format
        previous_results_dir = os.path.join(self.results_folder, modelrun_id)
        results = list(glob.iglob(os.path.join(previous_results_dir, '**/*.*'),
                                  recursive=True))
        for filename in results:
            if not filename.endswith(".csv"):
                self.logger.info("Warm start not possible because a different "
                                 "storage mode was used in the previous run")
                return None

        # Perform warm start
        self.logger.info("Warm start %s", modelrun_id)

        # Get metadata for all results
        result_metadata = []
        for filename in glob.iglob(os.path.join(results_dir, '**/*.*'), recursive=True):
            result_metadata.append(self._parse_results_path(
                filename.replace(self.results_folder, '')[1:]))

        # Find latest timestep
        result_metadata = sorted(result_metadata, key=lambda k: k['timestep'], reverse=True)
        latest_timestep = result_metadata[0]['timestep']

        # Remove all results with this timestep
        results_to_remove = [
            result for result in result_metadata
            if result['timestep'] == latest_timestep
        ]

        for result in results_to_remove:
            os.remove(
                self._get_results_path(
                    result['modelrun_id'],
                    result['model_name'],
                    result['output_name'],
                    result['timestep'],
                    result['decision_iteration']))

        self.logger.info("Warm start will resume at timestep %s", latest_timestep)
        return latest_timestep

    def _get_results_path(self, modelrun_id, model_name, output_name, timestep,
                          decision_iteration=None):
        """Return path to filename for a given output without file extension

        On the pattern of:
            results/<modelrun_name>/<model_name>/
            decision_<id>/
            output_<output_name>_timestep_<timestep>.csv

        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_name : str
        timestep : str or int
        decision_iteration : int, optional

        Returns
        -------
        path : strs
        """
        if decision_iteration is None:
            decision_iteration = 'none'

        path = os.path.join(
            self.results_folder, modelrun_id, model_name,
            "decision_{}".format(decision_iteration),
            "output_{}_timestep_{}.{}".format(output_name, timestep, 'csv')
        )
        return path

    def _parse_results_path(self, path):
        """Return result metadata for a given result path

        On the pattern of:
            results/<modelrun_name>/<model_name>/
            decision_<id>/
            output_<output_name>_timestep_<timestep>.csv

        Parameters
        ----------
        path : str

        Returns
        -------
        dict : A dict containing all of the metadata
        """
        decision_iteration = None

        data = re.findall(r"[\w']+", path)

        for section in data[2:len(data)]:
            if 'decision' in section:
                regex_decision = re.findall(r"decision_(\d{1,})", section)
                if regex_decision:
                    decision_iteration = int(regex_decision[0])
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
            'timestep': results['timestep'],
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
    # endregion


def ndarray_to_data_list(data_array, timestep=None):
    """Convert :class:`numpy.ndarray` to list of observations

    Parameters
    ----------
    data_array : ~smif.data_layer.data_array.DataArray
    timestep : int, default=None

    Returns
    -------
    observations : list of dict
        Each dict has keys: one for the variable name, one for each dimension in spec.dims,
        and optionally one for the given timestep
    """
    observations = []

    data = data_array.as_ndarray()
    spec = data_array.spec

    for indices, value in np.ndenumerate(data):
        obs = {}
        obs[spec.name] = value
        for dim, idx in zip(spec.dims, indices):
            obs[dim] = spec.dim_coords(dim).elements[idx]['name']
            if timestep:
                obs['timestep'] = timestep
        observations.append(obs)

    if data.shape == () and timestep:
        observations[0]['timestep'] = timestep

    return observations


def data_list_to_ndarray(observations, spec):
    """Convert list of observations to a ``DataArray``

    Parameters
    ----------
    observations : list[dict]
        Required keys for each dict are:
        - one key to match spec.name
        - one key per dimension in spec.dims
    spec : ~smif.metadata.spec.Spec

    Returns
    -------
    ~smif.data_layer.data_array.DataArray

    Raises
    ------
    KeyError
        If an observation is missing a required key
    ValueError
        If an observation region or interval is not in region_names or
        interval_names
    SmifDataNotFoundError
        If the observations don't include data for any dimension
        combination
    SmifDataMismatchError
        If the dimension coordinate ids do not
        match the observations
    """
    _validate_observations(observations, spec)

    data = np.full(spec.shape, np.nan, dtype=spec.dtype)

    for obs in observations:
        indices = []
        for dim in spec.dims:
            key = obs[dim]  # name (id/label) of coordinate element along dimension
            idx = spec.dim_coords(dim).ids.index(key)  # index of name in dim elements
            indices.append(idx)
        data[tuple(indices)] = obs[spec.name]

    return DataArray(spec, data)


def _validate_observations(observations, spec):
    if len(observations) != reduce(lambda x, y: x * y, spec.shape, 1):
        msg = "Number of observations ({}) is not equal to product of {}"
        raise SmifDataMismatchError(
            msg.format(len(observations), spec.shape)
        )
    _validate_observation_keys(observations, spec)
    for dim in spec.dims:
        _validate_observation_meta(
            observations,
            spec.dim_coords(dim).ids,
            dim
        )


def _validate_observation_keys(observations, spec):
    for obs in observations:
        if spec.name not in obs:
            raise KeyError(
                "Observation missing variable key ({}): {}".format(spec.name, obs))
        for dim in spec.dims:
            if dim not in obs:
                raise KeyError(
                    "Observation missing dimension key ({}): {}".format(dim, obs))


def _validate_observation_meta(observations, meta_list, meta_name):
    observed = set()
    for line, obs in enumerate(observations):
        if obs[meta_name] not in meta_list:
            raise ValueError("Unknown {} '{}' in row {}".format(
                meta_name, obs[meta_name], line))
        else:
            observed.add(obs[meta_name])
    missing = set(meta_list) - observed
    if missing:
        raise SmifDataNotFoundError(
            "Missing values for {}s: {}".format(meta_name, list(missing)))


def _skip_coords(config, keys):
    """Given a config dict and list of top-level keys for lists of specs,
    delete coords from each spec in each list.
    """
    config = copy.deepcopy(config)
    for key in keys:
        for spec in config[key]:
            try:
                del spec['coords']
            except KeyError:
                pass
    return config


def _pick_from_list(list_of_dicts, name):
    for item in list_of_dicts:
        if 'name' in item and item['name'] == name:
            return item
    return None


def _key_from_list(name, dict_of_lists):
    for key, items in dict_of_lists.items():
        if name in items:
            return key
    return None


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


def _get_data_from_csv(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        scenario_data = list(reader)
    return scenario_data


def _write_data_to_csv(filepath, data, spec=None, fieldnames=None):
    if fieldnames is not None:
        pass
    elif spec is not None:
        fieldnames = tuple(spec.dims) + (spec.name, )
    else:
        fieldnames = tuple(data[0].keys())

    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def _get_data_from_native_file(filepath):
    with pa.memory_map(filepath, 'rb') as native_file:
        native_file.seek(0)
        buf = native_file.read_buffer()
        data = pa.deserialize(buf)
    return data


def _write_data_to_native_file(filepath, data):
    with pa.OSFile(filepath, 'wb') as native_file:
        native_file.write(pa.serialize(data).to_buffer())


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
    """
    if filename is not None:
        filename = filename + extension
        filepath = os.path.join(path, filename)
    else:
        filepath = path
    return load_yaml(filepath)


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
    dump_yaml(data, filepath)


def load_yaml(file_path):
    """Parse yaml config file into plain data (lists, dicts and simple values)

    Parameters
    ----------
    file_path : str
        The path of the configuration file to parse
    """
    with open(file_path, 'r') as file_handle:
        return YAML().load(file_handle)


def dump_yaml(data, file_path):
    """Write plain data to a file as yaml

    Parameters
    ----------
    data
        Data to write (should be lists, dicts and simple values)
    file_path : str
        The path of the configuration file to write
    """
    with open(file_path, 'w') as file_handle:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.allow_unicode = True
        return yaml.dump(data, file_handle)


def _assert_no_mismatch(dtype, name, obj):
    try:
        if name != obj['name']:
            raise SmifDataMismatchError(
                "%s name '%s' must match '%s'" % (dtype.capitalize(), name, obj['name']))
    except KeyError:
        raise SmifValidationError("%s must have name defined" % dtype)
    except TypeError:
        pass


def _file_exists(file_dir, dtype, name):
    dir_key = "%ss" % dtype
    try:
        return os.path.exists(os.path.join(file_dir[dir_key], name + '.yml'))
    except TypeError:
        msg = "Could not parse file name {} and dtype {}"
        raise SmifDataNotFoundError(msg.format(name, dtype))


def _assert_file_exists(file_dir, dtype, name):
    if not _file_exists(file_dir, dtype, name):
        raise SmifDataNotFoundError("%s '%s' not found" % (dtype.capitalize(), name))


def _assert_file_not_exists(file_dir, dtype, name):
    if _file_exists(file_dir, dtype, name):
        raise SmifDataExistsError("%s '%s' already exists" % (dtype.capitalize(), name))


def _config_item_exists(config, dtype, name):
    key = "%ss" % dtype
    return key in config and _name_in_list(config[key], name)


def _name_in_list(list_of_dicts, name):
    for item in list_of_dicts:
        if 'name' in item and item['name'] == name:
            return True
    return False


def _idx_in_list(list_of_dicts, name):
    for i, item in enumerate(list_of_dicts):
        if 'name' in item and item['name'] == name:
            return i
    return None


def _assert_config_item_exists(config, dtype, name):
    if not _config_item_exists(config, dtype, name):
        raise SmifDataNotFoundError(
            "%s '%s' not found in '%s'" % (str(dtype).capitalize(), name, config['name']))


def _assert_config_item_not_exists(config, dtype, name):
    if _config_item_exists(config, dtype, name):
        raise SmifDataExistsError(
            "%s '%s' already exists in '%s'" % (str(dtype).capitalize(), name, config['name']))
