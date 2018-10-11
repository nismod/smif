"""File-backed data interface
"""
import copy
import csv
import glob
import os
import re
from functools import lru_cache, wraps

import pyarrow as pa
from smif.data_layer.data_interface import DataInterface
from smif.data_layer.load import dump, load
from smif.data_layer.validate import (validate_sos_model_config,
                                      validate_sos_model_format)
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError, SmifDataReadError,
                            SmifValidationError)
from smif.metadata import Spec

# Import fiona if available (optional dependency)
try:
    import fiona
except ImportError:
    pass


# Note: these decorators must be defined before being used below
def check_exists(dtype):
    """Decorator to check an item of dtype exists in a config file
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, name, item=None, *func_args, **func_kwargs):
            if item is not None:
                _assert_no_mismatch(dtype, name, item)
            _assert_file_exists(self.config_folders, dtype, name)
            if item is not None:
                return func(self, name, item, *func_args, **func_kwargs)
            return func(self, name, *func_args, **func_kwargs)

        return wrapped
    return wrapper


def check_not_exists(dtype):
    """Decorator creator to check an item of dtype does not exist
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, item, *func_args, **func_kwargs):
            name = item['name']
            _assert_file_not_exists(self.config_folders, dtype, name)
            return func(self, item, *func_args, **func_kwargs)
        return wrapped
    return wrapper


def check_exists_as_child(parent_dtype, child_dtype):
    """Decorator to check an item of dtype exists in a list in a config file
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, parent_name, name, item=None, *func_args, **func_kwargs):
            if item is not None:
                _assert_no_mismatch("{} {}".format(parent_dtype, child_dtype), name, item)
            config = self._read_config(parent_dtype, parent_name)
            _assert_config_item_exists(config, child_dtype, name)
            if item is not None:
                return func(self, parent_name, name, item, *func_args, **func_kwargs)
            return func(self, parent_name, name, *func_args, **func_kwargs)

        return wrapped
    return wrapper


def check_not_exists_as_child(parent_dtype, child_dtype):
    """Decorator creator to check an item of dtype does not exist in a list in a config file
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, parent_name, item, *func_args, **func_kwargs):
            name = item['name']
            config = self._read_config(parent_dtype, parent_name)
            _assert_config_item_not_exists(config, child_dtype, name)
            return func(self, parent_name, item, *func_args, **func_kwargs)
        return wrapped
    return wrapper


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
    def __init__(self, base_folder, storage_format='local_binary', validation=True):
        super().__init__()

        self.base_folder = str(base_folder)
        self.config_folder = str(os.path.join(self.base_folder, 'config'))
        self.config_folders = {}
        self.data_folder = str(os.path.join(self.base_folder, 'data'))
        self.data_folders = {}
        self.results_folder = str(os.path.join(self.base_folder, 'results'))

        self.storage_format = storage_format
        self.validation = validation

        # cache results of reading project_config (invalidate on write)
        self._project_config_cache_invalid = True
        # MUST ONLY access through self.read_project_config()
        self._project_config_cache = None

        config_folders = [
            'dimensions',
            'model_runs',
            'narratives',
            'scenarios',
            'sector_models',
            'sos_models',
        ]
        for folder in config_folders:
            dirname = os.path.join(self.config_folder, folder)
            # ensure each directory exists
            os.makedirs(dirname, exist_ok=True)
            self.config_folders[folder] = dirname

        data_folders = [
            'coefficients',
            'dimensions',
            'strategies',
            'initial_conditions',
            'initial_inputs',
            'interventions',
            'narratives',
            'scenarios',
            'strategies',
        ]
        for folder in data_folders:
            dirname = os.path.join(self.data_folder, folder)
            # ensure each directory exists
            os.makedirs(dirname, exist_ok=True)
            self.data_folders[folder] = dirname

        # ensure project config file exists
        try:
            self.read_project_config()
        except FileNotFoundError:
            # write empty config if none found
            self._write_project_config({})

    def _read_config(self, config_type, config_name):
        """Read config item - used by decorators for existence/consistency checks
        """
        if config_type == 'scenario':
            return self.read_scenario(config_name)
        elif config_type == 'narrative':
            return self.read_narrative(config_name)
        else:
            raise NotImplementedError(
                "Cannot read %s:%s through generic method." % (config_type, config_name))

    # region Model runs
    def read_model_runs(self):
        names = self._read_filenames_in_dir(self.config_folders['model_runs'], '.yml')
        model_runs = [self.read_model_run(name) for name in names]
        return model_runs

    def read_model_run(self, model_run_name):
        modelrun_config = self._read_model_run(model_run_name)
        del modelrun_config['strategies']
        return modelrun_config

    @check_exists('model_run')
    def _read_model_run(self, model_run_name):
        return self._read_yaml_file(self.config_folders['model_runs'], model_run_name)

    def _overwrite_model_run(self, model_run_name, model_run):
        self._write_yaml_file(self.config_folders['model_runs'], model_run_name, model_run)

    @check_not_exists('model_run')
    def write_model_run(self, model_run):
        config = copy.copy(model_run)
        config['strategies'] = []
        self._write_yaml_file(self.config_folders['model_runs'], config['name'], config)

    @check_exists('model_run')
    def update_model_run(self, model_run_name, model_run):
        prev = self._read_model_run(model_run_name)
        config = copy.copy(model_run)
        config['strategies'] = prev['strategies']
        self._overwrite_model_run(model_run_name, config)

    @check_exists('model_run')
    def delete_model_run(self, model_run_name):
        os.remove(os.path.join(self.config_folders['model_runs'], model_run_name + '.yml'))
    # endregion

    # region System-of-system models
    def read_sos_models(self):
        names = self._read_filenames_in_dir(self.config_folders['sos_models'], '.yml')
        sos_models = [self.read_sos_model(name) for name in names]
        return sos_models

    @check_exists('sos_model')
    def read_sos_model(self, sos_model_name):
        data = self._read_yaml_file(self.config_folders['sos_models'], sos_model_name)
        if self.validation:
            validate_sos_model_format(data)
        return data

    @check_not_exists('sos_model')
    def write_sos_model(self, sos_model):
        if self.validation:
            validate_sos_model_config(
                sos_model,
                self.read_sector_models(skip_coords=True),
                self.read_scenarios(skip_coords=True),
                self.read_narratives(skip_coords=True)
            )
        self._write_yaml_file(self.config_folders['sos_models'], sos_model['name'], sos_model)

    @check_exists('sos_model')
    def update_sos_model(self, sos_model_name, sos_model):
        if self.validation:
            validate_sos_model_config(
                sos_model,
                self.read_sector_models(skip_coords=True),
                self.read_scenarios(skip_coords=True),
                self.read_narratives(skip_coords=True)
            )
        self._write_yaml_file(self.config_folders['sos_models'], sos_model['name'], sos_model)

    @check_exists('sos_model')
    def delete_sos_model(self, sos_model_name):
        os.remove(os.path.join(self.config_folders['sos_models'], sos_model_name + '.yml'))
    # endregion

    # region Sector models
    def read_sector_models(self, skip_coords=False):
        names = self._read_filenames_in_dir(self.config_folders['sector_models'], '.yml')
        sector_models = [self.read_sector_model(name, skip_coords) for name in names]
        return sector_models

    @check_exists('sector_model')
    def read_sector_model(self, sector_model_name, skip_coords=False):
        sector_model = self._read_yaml_file(
            self.config_folders['sector_models'], sector_model_name)
        if not skip_coords:
            self._set_list_coords(sector_model['inputs'])
            self._set_list_coords(sector_model['outputs'])
            self._set_list_coords(sector_model['parameters'])
        return sector_model

    @check_not_exists('sector_model')
    def write_sector_model(self, sector_model):
        sector_model = copy.deepcopy(sector_model)
        if sector_model['interventions']:
            self.logger.warning("Ignoring interventions")
            sector_model['interventions'] = []

        sector_model = self._skip_coords(sector_model, ('inputs', 'outputs', 'parameters'))

        self._write_yaml_file(
            self.config_folders['sector_models'], sector_model['name'], sector_model)

    @check_exists('sector_model')
    def update_sector_model(self, sector_model_name, sector_model):
        sector_model = copy.deepcopy(sector_model)
        # ignore interventions and initial conditions which the app doesn't handle
        if sector_model['interventions'] or sector_model['initial_conditions']:
            old_sector_model = self._read_yaml_file(
                self.config_folders['sector_models'], sector_model['name'])

        if sector_model['interventions']:
            self.logger.warning("Ignoring interventions write")
            sector_model['interventions'] = old_sector_model['interventions']

        if sector_model['initial_conditions']:
            self.logger.warning("Ignoring initial conditions write")
            sector_model['initial_conditions'] = old_sector_model['initial_conditions']

        sector_model = self._skip_coords(sector_model, ('inputs', 'outputs', 'parameters'))

        self._write_yaml_file(
            self.config_folders['sector_models'], sector_model['name'], sector_model)

    @check_exists('sector_model')
    def delete_sector_model(self, sector_model_name):
        os.remove(
            os.path.join(self.config_folders['sector_models'], sector_model_name + '.yml'))

    @check_exists('sector_model')
    def read_interventions(self, sector_model_name):
        all_interventions = {}
        sector_model = self._read_yaml_file(
            self.config_folders['sector_models'], sector_model_name)
        interventions = self._read_interventions_files(
            sector_model['interventions'], self.data_folders['interventions'])
        for entry in interventions:
            name = entry.pop('name')
            if name in all_interventions:
                msg = "An entry for intervention {} already exists"
                raise ValueError(msg.format(name))
            else:
                all_interventions[name] = entry
        return all_interventions

    @check_exists('sector_model')
    def read_initial_conditions(self, sector_model_name):
        sector_model = self._read_yaml_file(
            self.config_folders['sector_models'], sector_model_name)
        return self._read_interventions_files(
            sector_model['initial_conditions'], self.data_folders['initial_conditions'])

    def _read_interventions_files(self, filenames, dirname):
        intervention_list = []
        for filename in filenames:
            interventions = self._read_interventions_file(filename, dirname)
            intervention_list.extend(interventions)
        return intervention_list

    def _read_interventions_file(self, filename, dirname):
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
            The name of the strategy yml or csv file to read in
        dirname: str
            The key of the dirname e.g. ``strategies`` or ``initial_conditions``

        Returns
        -------
        dict of dict
            Dict of intervention attribute dicts, keyed by intervention name
        """
        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            data = self._get_data_from_csv(os.path.join(dirname, filename))
            try:
                data = self._reshape_csv_interventions(data)
            except ValueError:
                raise ValueError("Error reshaping data for {}".format(filename))
        else:
            data = self._read_yaml_file(dirname, filename, extension='')

        return data

    def _write_interventions_file(self, filename, dirname, data):
        self._write_yaml_file(dirname, filename=filename, data=data, extension='')

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

    # region Strategies
    def read_strategies(self, modelrun_name):
        output = []
        model_run_config = self._read_model_run(modelrun_name)
        strategies = copy.deepcopy(model_run_config['strategies'])

        for strategy in strategies:
            if strategy['type'] == 'pre-specified-planning':
                decisions = self._read_interventions_file(
                    strategy['filename'], self.data_folders['strategies'])
                if decisions is None:
                    decisions = []
                del strategy['filename']
                strategy['interventions'] = decisions
                self.logger.info("Added %s pre-specified planning interventions to %s",
                                 len(decisions), strategy['model_name'])
                output.append(strategy)
            else:
                output.append(strategy)
        return output

    def write_strategies(self, modelrun_name, strategies):
        strategies = copy.deepcopy(strategies)
        model_run = self._read_model_run(modelrun_name)
        model_run['strategies'] = []
        for i, strategy in enumerate(strategies):
            if strategy['type'] == 'pre-specified-planning':
                decisions = strategy['interventions']
                del strategy['interventions']
                filename = 'strategy-{}.yml'.format(i)
                strategy['filename'] = filename
                self._write_interventions_file(
                    filename, self.data_folders['strategies'], decisions)

            model_run['strategies'].append(strategy)
        self._overwrite_model_run(modelrun_name, model_run)
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

    @staticmethod
    def _read_state_file(fname):
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

    # region Units
    def write_unit_definitions(self, units):
        project_config = self.read_project_config()
        filename = 'user-defined-units.txt'
        path = os.path.join(self.base_folder, 'data', filename)
        with open(path, 'w') as units_fh:
            units_fh.writelines(units)
        project_config['units'] = filename
        self._write_project_config(project_config)

    def read_unit_definitions(self):
        project_config = self.read_project_config()
        try:
            filename = project_config['units']
            if filename is None:
                return []
            path = os.path.join(self.base_folder, 'data', filename)
            try:
                with open(path, 'r') as units_fh:
                    return [line.strip() for line in units_fh]
            except FileNotFoundError as ex:
                raise SmifDataNotFoundError('Units file not found:' + str(ex)) from ex
        except KeyError:
            return []
    # endregion

    # region Dimensions
    def read_dimensions(self):
        dim_names = self._read_filenames_in_dir(self.config_folders['dimensions'], '.yml')
        return [self.read_dimension(name) for name in dim_names]

    @check_exists('dimension')
    def read_dimension(self, dimension_name):
        dim = self._read_yaml_file(self.config_folders['dimensions'], dimension_name)
        dim['elements'] = self._read_dimension_file(dim['elements'])
        return dim

    @check_not_exists('dimension')
    def write_dimension(self, dimension):
        # write elements to yml file (by default, can handle any nested data)
        elements_filename = "{}.yml".format(dimension['name'])
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and add to config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename
        self._write_yaml_file(
            self.config_folders['dimensions'], dimension['name'], dimension_with_ref)

    @check_exists('dimension')
    def update_dimension(self, dimension_name, dimension):
        # look up elements filename and write elements
        old_dim = self._read_yaml_file(self.config_folders['dimensions'], dimension_name)
        elements_filename = old_dim['elements']
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and update config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename
        self._write_yaml_file(
            self.config_folders['dimensions'], dimension_name, dimension_with_ref)

    @check_exists('dimension')
    def delete_dimension(self, dimension_name):
        # read to find filename
        old_dim = self._read_yaml_file(self.config_folders['dimensions'], dimension_name)
        elements_filename = old_dim['elements']
        # remove elements data
        os.remove(os.path.join(self.data_folders['dimensions'], elements_filename))
        # remove description
        os.remove(
            os.path.join(self.config_folders['dimensions'], "{}.yml".format(dimension_name)))

    def _set_list_coords(self, list_):
        for item in list_:
            self._set_item_coords(item)

    def _set_item_coords(self, item):
        """If dims exists and is not empty
        """
        if 'dims' in item and item['dims']:
            item['coords'] = {
                dim: self.read_dimension(dim)['elements']
                for dim in item['dims']
            }

    @lru_cache(maxsize=32)
    def _read_dimension_file(self, filename):
        filepath = os.path.join(self.data_folders['dimensions'], filename)
        _, ext = os.path.splitext(filename)
        if ext in ('.yml', '.yaml'):
            data = self._read_yaml_file(filepath)
        elif ext == '.csv':
            data = self._get_data_from_csv(filepath)
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
        filepath = os.path.join(self.data_folders['dimensions'], filename)
        _, ext = os.path.splitext(filename)
        if ext in ('.yml', '.yaml'):
            self._write_yaml_file(filepath, data=data)
        elif ext == '.csv':
            self._write_data_to_csv(filepath, data)
        elif ext in ('.geojson', '.shp'):
            raise NotImplementedError("Writing spatial dimensions not yet supported")
            # self._write_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', '.yml', '.yaml', "
            msg += "'.geojson', '.shp') when writing {}"
            raise SmifDataReadError(msg.format(ext, filepath))
        return data
    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        results_path = self._get_coefficients_path(source_spec, destination_spec)
        try:
            return self._get_data_from_native_file(results_path)
        except (FileNotFoundError, pa.lib.ArrowIOError):
            msg = "Could not find the coefficients file for %s to %s"
            self.logger.warning(msg, source_spec, destination_spec)
            raise SmifDataNotFoundError(msg.format(source_spec, destination_spec))

    def write_coefficients(self, source_spec, destination_spec, data):
        results_path = self._get_coefficients_path(source_spec, destination_spec)
        self._write_data_to_native_file(results_path, data)

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

    # region Scenarios
    def read_scenarios(self, skip_coords=False):
        scenario_names = self._read_filenames_in_dir(self.config_folders['scenarios'], '.yml')
        return [self.read_scenario(name, skip_coords) for name in scenario_names]

    @check_exists('scenario')
    def read_scenario(self, scenario_name, skip_coords=False):
        scenario = self._read_yaml_file(self.config_folders['scenarios'], scenario_name)
        if not skip_coords:
            self._set_list_coords(scenario['provides'])
        return scenario

    @check_not_exists('scenario')
    def write_scenario(self, scenario):
        scenario = self._skip_coords(scenario, ['provides'])
        self._write_yaml_file(self.config_folders['scenarios'], scenario['name'], scenario)

    @check_exists('scenario')
    def update_scenario(self, scenario_name, scenario):
        scenario = self._skip_coords(scenario, ['provides'])
        self._write_yaml_file(self.config_folders['scenarios'], scenario['name'], scenario)

    @check_exists('scenario')
    def delete_scenario(self, scenario_name):
        os.remove(
            os.path.join(self.config_folders['scenarios'], "{}.yml".format(scenario_name)))

    def read_scenario_variants(self, scenario_name):
        scenario = self.read_scenario(scenario_name, skip_coords=True)
        return scenario['variants']

    @check_exists_as_child('scenario', 'variant')
    def read_scenario_variant(self, scenario_name, variant_name):
        variants = self.read_scenario_variants(scenario_name)
        return _pick_from_list(variants, variant_name)

    @check_not_exists_as_child('scenario', 'variant')
    def write_scenario_variant(self, scenario_name, variant):
        scenario = self.read_scenario(scenario_name, skip_coords=True)
        scenario['variants'].append(variant)
        self.update_scenario(scenario_name, scenario)

    @check_exists_as_child('scenario', 'variant')
    def update_scenario_variant(self, scenario_name, variant_name, variant):
        scenario = self.read_scenario(scenario_name, skip_coords=True)
        v_idx = _idx_in_list(scenario['variants'], variant_name)
        scenario['variants'][v_idx] = variant
        self.update_scenario(scenario_name, scenario)

    @check_exists_as_child('scenario', 'variant')
    def delete_scenario_variant(self, scenario_name, variant_name):
        scenario = self.read_scenario(scenario_name, skip_coords=True)
        v_idx = _idx_in_list(scenario['variants'], variant_name)
        del scenario['variants'][v_idx]
        self.update_scenario(scenario_name, scenario)

    @check_exists_as_child('scenario', 'variant')
    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        spec = self._read_scenario_variable_spec(scenario_name, variable)
        filepath = self._get_scenario_variant_filepath(scenario_name, variant_name, variable)
        data = self._get_data_from_csv(filepath)

        if 'timestep' not in data[0].keys():
            msg = "Could not read data for scenario variant {} for scenario {}. " + \
                  "Header in '{}' missing 'timestep' key. Found {}"
            raise SmifDataMismatchError(msg.format(variant_name, scenario_name,
                                                   filepath, list(data[0].keys())))

        if timestep is not None:
            data = [datum for datum in data if int(datum['timestep']) == timestep]

        try:
            array = self.data_list_to_ndarray(data, spec)
        except SmifDataMismatchError as ex:
            msg = "DataMismatch in scenario: {}:{}.{}, from {}"
            raise SmifDataMismatchError(
                msg.format(scenario_name, variant_name, variable, str(ex))
            ) from ex

        return array

    @check_exists_as_child('scenario', 'variant')
    def write_scenario_variant_data(self, scenario_name, variant_name, variable, data):
        spec = self._read_scenario_variable_spec(scenario_name, variable)
        data = self.ndarray_to_data_list(data, spec)
        filepath = self._get_scenario_variant_filepath(scenario_name, variant_name, variable)
        self._write_data_to_csv(filepath, data, spec=spec)

    def _get_scenario_variant_filepath(self, scenario_name, variant_name, variable):
        variant = self.read_scenario_variant(scenario_name, variant_name)
        if 'data' not in variant or variable not in variant['data']:
            raise SmifDataNotFoundError(
                "Scenario data file not defined for {}:{}, {}".format(
                    scenario_name, variant_name, variable)
            )
        filename = variant['data'][variable]
        return os.path.join(self.data_folders['scenarios'], filename)

    def _read_scenario_variable_spec(self, scenario_name, variable):
        # Read spec from scenario->provides->variable
        scenario = self.read_scenario(scenario_name)
        spec = _pick_from_list(scenario['provides'], variable)
        if spec is not None:
            self._set_item_coords(spec)
            return Spec.from_dict(spec)
        else:
            msg = "Could not find spec definition for scenario '{}' " + \
                  "and variable '{}'"
            raise SmifDataNotFoundError(msg.format(scenario_name, variable))
    # endregion

    # region Narratives
    def read_narratives(self, skip_coords=False):
        narr_names = self._read_filenames_in_dir(self.config_folders['narratives'], '.yml')
        return [self.read_narrative(name, skip_coords) for name in narr_names]

    @check_exists('narrative')
    def read_narrative(self, narrative_name, skip_coords=False):
        narrative = self._read_yaml_file(self.config_folders['narratives'], narrative_name)
        if not skip_coords:
            self._set_list_coords(narrative['provides'])
        return narrative

    @check_not_exists('narrative')
    def write_narrative(self, narrative):
        narrative = self._skip_coords(narrative, ['provides'])
        self._write_yaml_file(
            self.config_folders['narratives'], narrative['name'], narrative)

    @check_exists('narrative')
    def update_narrative(self, narrative_name, narrative):
        narrative = self._skip_coords(narrative, ['provides'])
        self._write_yaml_file(
            self.config_folders['narratives'], narrative_name, narrative)

    @check_exists('narrative')
    def delete_narrative(self, narrative_name):
        os.remove(
            os.path.join(self.config_folders['narratives'], "{}.yml".format(narrative_name)))

    def read_narrative_variants(self, narrative_name):
        narrative = self.read_narrative(narrative_name, skip_coords=True)
        return narrative['variants']

    @check_exists_as_child('narrative', 'variant')
    def read_narrative_variant(self, narrative_name, variant_name):
        variants = self.read_narrative_variants(narrative_name)
        return _pick_from_list(variants, variant_name)

    @check_not_exists_as_child('narrative', 'variant')
    def write_narrative_variant(self, narrative_name, variant):
        narrative = self.read_narrative(narrative_name)
        narrative['variants'].append(variant)
        self.update_narrative(narrative_name, narrative)

    @check_exists_as_child('narrative', 'variant')
    def update_narrative_variant(self, narrative_name, variant_name, variant):
        narrative = self.read_narrative(narrative_name)
        v_idx = _idx_in_list(narrative['variants'], variant_name)
        narrative['variants'][v_idx] = variant
        self.update_narrative(narrative_name, narrative)

    @check_exists_as_child('narrative', 'variant')
    def delete_narrative_variant(self, narrative_name, variant_name):
        narrative = self.read_narrative(narrative_name)
        v_idx = _idx_in_list(narrative['variants'], variant_name)
        del narrative['variants'][v_idx]
        self.update_narrative(narrative_name, narrative)

    @check_exists_as_child('narrative', 'variant')
    def read_narrative_variant_data(self, narrative_name, variant_name, variable,
                                    timestep=None):
        spec = self._read_narrative_variable_spec(narrative_name, variable)
        filepath = self._get_narrative_variant_filepath(narrative_name, variant_name, variable)
        data = self._get_data_from_csv(filepath)

        if timestep is not None:
            data = [datum for datum in data if int(datum['timestep']) == timestep]

        try:
            array = self.data_list_to_ndarray(data, spec)
        except SmifDataMismatchError as ex:
            msg = "DataMismatch in narrative: {}:{}, {}, from {}"
            raise SmifDataMismatchError(
                msg.format(narrative_name, variant_name, variable, str(ex))
            ) from ex

        return array

    @check_exists_as_child('narrative', 'variant')
    def write_narrative_variant_data(self, narrative_name, variant_name, variable, data,
                                     timestep=None):
        spec = self._read_narrative_variable_spec(narrative_name, variable)
        data = self.ndarray_to_data_list(data, spec)
        filepath = self._get_narrative_variant_filepath(narrative_name, variant_name, variable)
        self._write_data_to_csv(filepath, data, spec=spec)

    def _get_narrative_variant_filepath(self, narrative_name, variant_name, variable):
        variant = self.read_narrative_variant(narrative_name, variant_name)
        if 'data' not in variant or variable not in variant['data']:
            raise SmifDataNotFoundError(
                "narrative data file not defined for {}:{}, {}".format(
                    narrative_name, variant_name, variable)
            )
        filename = variant['data'][variable]
        return os.path.join(self.data_folders['narratives'], filename)

    def _read_narrative_variable_spec(self, narrative_name, variable):
        # Read spec from narrative->provides->variable
        narrative = self.read_narrative(narrative_name)
        spec = _pick_from_list(narrative['provides'], variable)
        self._set_item_coords(spec)
        return Spec.from_dict(spec)
    # endregion

    # region Results
    def read_results(self, modelrun_id, model_name, output_spec, timestep=None,
                     decision_iteration=None):
        if timestep is None:
            raise NotImplementedError()

        results_path = self._get_results_path(
            modelrun_id, model_name, output_spec.name,
            timestep, decision_iteration
        )

        try:
            if self.storage_format == 'local_csv':
                data = self._get_data_from_csv(results_path)
                return self.data_list_to_ndarray(data, output_spec)
            elif self.storage_format == 'local_binary':
                return self._get_data_from_native_file(results_path)
            else:
                msg = "Unrecognised storage format: %s"
                raise NotImplementedError(msg % self.storage_format)
        except (FileNotFoundError, pa.lib.ArrowIOError):
            key = str([modelrun_id, model_name, output_spec.name, timestep,
                       decision_iteration])
            raise SmifDataNotFoundError("Could not find results for {}".format(key))

    def write_results(self, data, modelrun_id, model_name, output_spec, timestep=None,
                      decision_iteration=None):
        if timestep is None:
            raise NotImplementedError()

        results_path = self._get_results_path(
            modelrun_id, model_name, output_spec.name,
            timestep, decision_iteration
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        if self.storage_format == 'local_csv':
            data = self.ndarray_to_data_list(data, output_spec)
            self._write_data_to_csv(results_path, data, spec=output_spec)
        elif self.storage_format == 'local_binary':
            self._write_data_to_native_file(results_path, data)
        else:
            raise NotImplementedError("Unrecognised storage format: %s" % self.storage_format)

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
            warn = (self.storage_format == 'local_csv' and not filename.endswith(".csv")) or \
                   (self.storage_format == 'local_binary' and not filename.endswith(".dat"))
            if warn:
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

        if self.storage_format == 'local_csv':
            ext = 'csv'
        elif self.storage_format == 'local_binary':
            ext = 'dat'
        else:
            ext = 'unknown'

        path = os.path.join(
            self.results_folder, modelrun_id, model_name,
            "decision_{}".format(decision_iteration),
            "output_{}_timestep_{}.{}".format(output_name, timestep, ext)
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

    # region Common methods
    def read_project_config(self):
        """Read the project configuration

        Returns
        -------
        dict
            The project configuration
        """
        if self._project_config_cache_invalid:
            self._project_config_cache = self._read_yaml_file(
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
        self._write_yaml_file(self.base_folder, 'project', data)

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
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            scenario_data = list(reader)
        return scenario_data

    @staticmethod
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

    @staticmethod
    def _get_data_from_native_file(filepath):
        with pa.memory_map(filepath, 'rb') as native_file:
            native_file.seek(0)
            buf = native_file.read_buffer()
            data = pa.deserialize(buf)
        return data

    @staticmethod
    def _write_data_to_native_file(filepath, data):
        with pa.OSFile(filepath, 'wb') as native_file:
            native_file.write(pa.serialize(data).to_buffer())

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
    # endregion


def _assert_no_mismatch(dtype, name, obj):
    try:
        if name != obj['name']:
            raise SmifDataMismatchError(
                "%s name '%s' must match '%s'" % (dtype, name, obj['name']))
    except KeyError:
        raise SmifValidationError("%s must have name defined" % dtype)
    except TypeError:
        pass


def _file_exists(file_dir, dtype, name):
    dir_key = "%ss" % dtype
    return os.path.exists(os.path.join(file_dir[dir_key], name + '.yml'))


def _assert_file_exists(file_dir, dtype, name):
    if not _file_exists(file_dir, dtype, name):
        raise SmifDataNotFoundError("%s '%s' not found" % (dtype, name))


def _assert_file_not_exists(file_dir, dtype, name):
    if _file_exists(file_dir, dtype, name):
        raise SmifDataExistsError("%s '%s' already exists" % (dtype, name))


def _config_item_exists(config, dtype, name):
    key = "%ss" % dtype
    return key in config and _name_in_list(config[key], name)


def _name_in_list(list_of_dicts, name):
    for item in list_of_dicts:
        if 'name' in item and item['name'] == name:
            return True
    return False


def _pick_from_list(list_of_dicts, name):
    for item in list_of_dicts:
        if 'name' in item and item['name'] == name:
            return item
    return None


def _idx_in_list(list_of_dicts, name):
    for i, item in enumerate(list_of_dicts):
        if 'name' in item and item['name'] == name:
            return i
    return None


def _assert_config_item_exists(config, dtype, name):
    if not _config_item_exists(config, dtype, name):
        raise SmifDataNotFoundError(
            "%s '%s' not found in '%s'" % (str(dtype).title(), name, config['name']))


def _assert_config_item_not_exists(config, dtype, name):
    if _config_item_exists(config, dtype, name):
        raise SmifDataExistsError(
            "%s '%s' already exists in '%s'" % (str(dtype).title(), name, config['name']))
