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
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError, SmifDataReadError)
from smif.metadata import Spec

# Import fiona if available (optional dependency)
try:
    import fiona
except ImportError:
    pass


# Note: these decorators must be defined before being used below
def check_exists(dtype):
    """Decorator to check an item of dtype exists
    """
    def wrapper(func):
        """Decorator specialised by dtype (class/item type)
        """
        @wraps(func)
        def wrapped(self, name, primary=None, secondary=None, *func_args, **func_kwargs):
            """Wrapper to implement error checking
            """
            _assert_no_mismatch(dtype, name, primary, secondary)
            if dtype in _file_dtypes():
                _assert_file_exists(self.file_dir, dtype, name)
            if dtype in _config_dtypes():
                config = self.read_project_config()
                _assert_config_item_exists(config, dtype, name)
            if dtype in _nested_config_dtypes():
                config = self.read_project_config()
                _assert_nested_config_item_exists(config, dtype, name, primary)

            if primary is None:
                return func(self, name, *func_args, **func_kwargs)
            elif secondary is None:
                return func(self, name, primary, *func_args, **func_kwargs)
            return func(self, name, primary, secondary, *func_args, **func_kwargs)

        return wrapped
    return wrapper


def check_not_exists(dtype):
    """Decorator creator to check an item of dtype does not exist
    """
    def wrapper(func):
        """Decorator specialised by dtype (class/item type)
        """
        @wraps(func)
        def wrapped(self, primary, secondary=None, *func_args, **func_kwargs):
            """Wrapper to implement error checking
            """
            if dtype in _file_dtypes():
                _assert_file_not_exists(self.file_dir, dtype, primary['name'])
            if dtype in _config_dtypes():
                config = self.read_project_config()
                _assert_config_item_not_exists(config, dtype, primary['name'])
            if dtype in _nested_config_dtypes():
                config = self.read_project_config()
                _assert_nested_config_item_not_exists(config, dtype, primary, secondary)

            if secondary is None:
                return func(self, primary, *func_args, **func_kwargs)
            return func(self, primary, secondary, *func_args, **func_kwargs)
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
    def __init__(self, base_folder, storage_format='local_binary'):
        super().__init__()

        self.base_folder = str(base_folder)
        self.storage_format = storage_format

        self.file_dir = {}
        self.file_dir['project'] = os.path.join(self.base_folder, 'config')
        self.file_dir['results'] = os.path.join(self.base_folder, 'results')

        # cache results of reading project_config (invalidate on write)
        self._project_config_cache_invalid = True
        # MUST ONLY access through self.read_project_config()
        self._project_config_cache = None

        config_folders = {
            'model_runs': 'config',
            'sos_models': 'config',
            'sector_models': 'config',
            'strategies': 'data',
            'interventions': 'data',
            'initial_conditions': 'data',
            'dimensions': 'data',
            'coefficients': 'data',
            'scenarios': 'data',
            'narratives': 'data',
        }

        for category, folder in config_folders.items():
            dirname = os.path.join(self.base_folder, folder, category)
            # ensure each directory exists
            os.makedirs(dirname, exist_ok=True)
            # store dirname
            self.file_dir[category] = dirname

        # ensure project config file exists
        try:
            self.read_project_config()
        except FileNotFoundError:
            # write empty config if none found
            self._write_project_config({})

    # region Model runs
    def read_model_runs(self):
        names = self._read_filenames_in_dir(self.file_dir['model_runs'], '.yml')
        model_runs = [self.read_model_run(name) for name in names]
        return model_runs

    def read_model_run(self, model_run_name):
        modelrun_config = self._read_model_run(model_run_name)
        del modelrun_config['strategies']
        return modelrun_config

    @check_exists(dtype='model_run')
    def _read_model_run(self, model_run_name):
        return self._read_yaml_file(self.file_dir['model_runs'], model_run_name)

    def _overwrite_model_run(self, model_run_name, model_run):
        self._write_yaml_file(self.file_dir['model_runs'], model_run_name, model_run)

    @check_not_exists(dtype='model_run')
    def write_model_run(self, model_run):
        config = copy.copy(model_run)
        config['strategies'] = []
        self._write_yaml_file(self.file_dir['model_runs'], config['name'], config)

    @check_exists(dtype='model_run')
    def update_model_run(self, model_run_name, model_run):
        prev = self._read_model_run(model_run_name)
        config = copy.copy(model_run)
        config['strategies'] = prev['strategies']
        self._overwrite_model_run(model_run_name, config)

    @check_exists(dtype='model_run')
    def delete_model_run(self, model_run_name):
        os.remove(os.path.join(self.file_dir['model_runs'], model_run_name + '.yml'))
    # endregion

    # region System-of-system models
    def read_sos_models(self):
        names = self._read_filenames_in_dir(self.file_dir['sos_models'], '.yml')
        sos_models = [self.read_sos_model(name) for name in names]
        return sos_models

    @check_exists(dtype='sos_model')
    def read_sos_model(self, sos_model_name):
        data = self._read_yaml_file(self.file_dir['sos_models'], sos_model_name)
        return data

    @check_not_exists(dtype='sos_model')
    def write_sos_model(self, sos_model):
        self._write_yaml_file(self.file_dir['sos_models'], sos_model['name'], sos_model)

    @check_exists(dtype='sos_model')
    def update_sos_model(self, sos_model_name, sos_model):
        self._write_yaml_file(self.file_dir['sos_models'], sos_model['name'], sos_model)

    @check_exists(dtype='sos_model')
    def delete_sos_model(self, sos_model_name):
        os.remove(os.path.join(self.file_dir['sos_models'], sos_model_name + '.yml'))
    # endregion

    # region Sector models
    def read_sector_models(self, skip_coords=False):
        names = self._read_filenames_in_dir(self.file_dir['sector_models'], '.yml')
        sector_models = [self.read_sector_model(name, skip_coords) for name in names]
        return sector_models

    @check_exists(dtype='sector_model')
    def read_sector_model(self, sector_model_name, skip_coords=False):
        sector_model = self._read_yaml_file(self.file_dir['sector_models'], sector_model_name)
        if not skip_coords:
            self._set_list_coords(sector_model['inputs'])
            self._set_list_coords(sector_model['outputs'])
            self._set_list_coords(sector_model['parameters'])
        return sector_model

    @check_not_exists(dtype='sector_model')
    def write_sector_model(self, sector_model):
        sector_model = copy.deepcopy(sector_model)
        if sector_model['interventions']:
            self.logger.warning("Ignoring interventions")
            sector_model['interventions'] = []

        sector_model = self._skip_coords(sector_model, ('inputs', 'outputs', 'parameters'))

        self._write_yaml_file(
            self.file_dir['sector_models'], sector_model['name'], sector_model)

    @check_exists(dtype='sector_model')
    def update_sector_model(self, sector_model_name, sector_model):
        sector_model = copy.deepcopy(sector_model)
        # ignore interventions and initial conditions which the app doesn't handle
        if sector_model['interventions'] or sector_model['initial_conditions']:
            old_sector_model = self._read_yaml_file(
                self.file_dir['sector_models'], sector_model['name'])

        if sector_model['interventions']:
            self.logger.warning("Ignoring interventions write")
            sector_model['interventions'] = old_sector_model['interventions']

        if sector_model['initial_conditions']:
            self.logger.warning("Ignoring initial conditions write")
            sector_model['initial_conditions'] = old_sector_model['initial_conditions']

        sector_model = self._skip_coords(sector_model, ('inputs', 'outputs', 'parameters'))

        self._write_yaml_file(
            self.file_dir['sector_models'], sector_model['name'], sector_model)

    @check_exists(dtype='sector_model')
    def delete_sector_model(self, sector_model_name):
        os.remove(os.path.join(self.file_dir['sector_models'], sector_model_name + '.yml'))

    @check_exists(dtype='sector_model')
    def read_interventions(self, sector_model_name):
        all_interventions = {}
        sector_model = self._read_yaml_file(self.file_dir['sector_models'], sector_model_name)
        interventions = self._read_interventions_files(
            sector_model['interventions'], 'interventions')
        for entry in interventions:
            name = entry.pop('name')
            if name in all_interventions:
                msg = "An entry for intervention {} already exists"
                raise ValueError(msg.format(name))
            else:
                all_interventions[name] = entry
        return all_interventions

    @check_exists(dtype='sector_model')
    def read_initial_conditions(self, sector_model_name):
        sector_model = self._read_yaml_file(self.file_dir['sector_models'], sector_model_name)
        return self._read_interventions_files(
            sector_model['initial_conditions'], 'initial_conditions')

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
        filepath = self.file_dir[dirname]
        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            data = self._get_data_from_csv(os.path.join(filepath, filename))
            try:
                data = self._reshape_csv_interventions(data)
            except ValueError:
                raise ValueError("Error reshaping data for {}".format(filename))
        else:
            data = self._read_yaml_file(filepath, filename, extension='')

        return data

    def _write_interventions_file(self, filename, dirname, data):
        filepath = self.file_dir[dirname]
        self._write_yaml_file(filepath, filename=filename, data=data, extension='')

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
    def read_strategies(self, model_run_name):
        output = []
        model_run_config = self._read_model_run(model_run_name)
        strategies = copy.deepcopy(model_run_config['strategies'])

        for strategy in strategies:
            if strategy['type'] == 'pre-specified-planning':
                decisions = self._read_interventions_file(strategy['filename'], 'strategies')
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

    def write_strategies(self, model_run_name, strategies):
        strategies = copy.deepcopy(strategies)
        model_run = self._read_model_run(model_run_name)
        model_run['strategies'] = []
        for i, strategy in enumerate(strategies):
            if strategy['type'] == 'pre-specified-planning':
                decisions = strategy['interventions']
                del strategy['interventions']
                filename = 'strategy-{}.yml'.format(i)
                strategy['filename'] = filename
                self._write_interventions_file(filename, 'strategies', decisions)

            model_run['strategies'].append(strategy)
        self._overwrite_model_run(model_run_name, model_run)
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
        results_dir = self.file_dir['results']

        if timestep is None:
            timestep = '0000'

        if decision_iteration is None:
            separator = ''
            decision_iteration = ''
        else:
            separator = '_decision_'

        fmt = 'state_{}{}{}.csv'
        fname = os.path.join(
            results_dir, modelrun_name, fmt.format(timestep, separator, decision_iteration))

        return fname

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
        project_config = self.read_project_config()
        dimensions = []
        for dim_with_ref in project_config['dimensions']:
            dim = copy.copy(dim_with_ref)
            dim['elements'] = self._read_dimension_file(dim_with_ref['elements'])
            dimensions.append(dim)
        return dimensions

    @check_exists(dtype='dimension')
    def read_dimension(self, dimension_name):
        project_config = self.read_project_config()
        dim_with_ref = _pick_from_list(project_config['dimensions'], dimension_name)
        dim = copy.copy(dim_with_ref)
        dim['elements'] = self._read_dimension_file(dim_with_ref['elements'])
        return dim

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
        filepath = os.path.join(self.file_dir['dimensions'], filename)
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
        filepath = os.path.join(self.file_dir['dimensions'], filename)
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

    def _delete_dimension_file(self, filename):
        os.remove(os.path.join(self.file_dir['dimensions'], filename))

    @check_not_exists(dtype='dimension')
    def write_dimension(self, dimension):
        project_config = self.read_project_config()

        # write elements to yml file (by default, can handle any nested data)
        filename = "{}.yml".format(dimension['name'])
        elements = dimension['elements']
        self._write_dimension_file(filename, elements)

        # refer to elements by filename and add to config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = filename
        try:
            project_config['dimensions'].append(dimension_with_ref)
        except KeyError:
            project_config['dimensions'] = [dimension_with_ref]
        self._write_project_config(project_config)

    @check_exists(dtype='dimension')
    def update_dimension(self, dimension_name, dimension):
        project_config = self.read_project_config()
        idx = _idx_in_list(project_config['dimensions'], dimension_name)

        # look up project-config filename and write elements
        filename = project_config['dimensions'][idx]['elements']
        elements = dimension['elements']
        self._write_dimension_file(filename, elements)

        # refer to elements by filename and update config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = filename
        project_config['dimensions'][idx] = dimension_with_ref
        self._write_project_config(project_config)

    @check_exists(dtype='dimension')
    def delete_dimension(self, dimension_name):
        project_config = self.read_project_config()
        idx = _idx_in_list(project_config['dimensions'], dimension_name)

        # look up project-config filename and delete file
        filename = project_config['dimensions'][idx]['elements']
        self._delete_dimension_file(filename)

        # delete from config
        del project_config['dimensions'][idx]
        self._write_project_config(project_config)
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
        results_dir = self.file_dir['coefficients']
        path = os.path.join(
            results_dir,
            "{}_{}.{}_{}.dat".format(
                source_spec.name, "-".join(source_spec.dims),
                destination_spec.name, "-".join(destination_spec.dims)
            )
        )
        return path
    # endregion

    # region Scenarios
    def read_scenarios(self, skip_coords=False):
        project_config = self.read_project_config()
        scenarios = copy.deepcopy(project_config['scenarios'])
        if not skip_coords:
            for scenario in scenarios:
                self._set_list_coords(scenario['provides'])
        return scenarios

    @check_exists(dtype='scenario')
    def read_scenario(self, scenario_name, skip_coords=False):
        project_config = self.read_project_config()
        scenario = copy.deepcopy(_pick_from_list(project_config['scenarios'], scenario_name))
        if not skip_coords:
            self._set_list_coords(scenario['provides'])
        return scenario

    @check_not_exists(dtype='scenario')
    def write_scenario(self, scenario):
        project_config = self.read_project_config()
        scenario = copy.deepcopy(scenario)
        scenario = self._skip_coords(scenario, ['provides'])
        try:
            project_config['scenarios'].append(scenario)
        except KeyError:
            project_config['scenarios'] = [scenario]
        self._write_project_config(project_config)

    @check_exists(dtype='scenario')
    def update_scenario(self, scenario_name, scenario):
        project_config = self.read_project_config()
        scenario = copy.deepcopy(scenario)
        scenario = self._skip_coords(scenario, ['provides'])
        idx = _idx_in_list(project_config['scenarios'], scenario_name)
        project_config['scenarios'][idx] = scenario
        self._write_project_config(project_config)

    @check_exists(dtype='scenario')
    def delete_scenario(self, scenario_name):
        project_config = self.read_project_config()
        idx = _idx_in_list(project_config['scenarios'], scenario_name)
        del project_config['scenarios'][idx]
        self._write_project_config(project_config)

    @check_exists(dtype='scenario')
    def read_scenario_variants(self, scenario_name):
        project_config = self.read_project_config()
        scenario = _pick_from_list(project_config['scenarios'], scenario_name)
        return scenario['variants']

    @check_exists(dtype='scenario_variant')
    def read_scenario_variant(self, scenario_name, variant_name):
        variants = self.read_scenario_variants(scenario_name)
        return _pick_from_list(variants, variant_name)

    @check_not_exists(dtype='scenario_variant')
    def write_scenario_variant(self, scenario_name, variant):
        project_config = self.read_project_config()
        s_idx = _idx_in_list(project_config['scenarios'], scenario_name)
        project_config['scenarios'][s_idx]['variants'].append(variant)
        self._write_project_config(project_config)

    @check_exists(dtype='scenario_variant')
    def update_scenario_variant(self, scenario_name, variant_name, variant):
        project_config = self.read_project_config()
        s_idx = _idx_in_list(project_config['scenarios'], scenario_name)
        v_idx = _idx_in_list(project_config['scenarios'][s_idx]['variants'], variant_name)
        project_config['scenarios'][s_idx]['variants'][v_idx] = variant
        self._write_project_config(project_config)

    @check_exists(dtype='scenario_variant')
    def delete_scenario_variant(self, scenario_name, variant_name):
        project_config = self.read_project_config()
        s_idx = _idx_in_list(project_config['scenarios'], scenario_name)
        v_idx = _idx_in_list(project_config['scenarios'][s_idx]['variants'], variant_name)
        del project_config['scenarios'][s_idx]['variants'][v_idx]
        self._write_project_config(project_config)

    @check_exists(dtype='scenario_variant')
    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        spec = self._read_scenario_variable_spec(scenario_name, variable)
        filepath = self._get_scenario_variant_filepath(scenario_name, variant_name, variable)
        data = self._get_data_from_csv(filepath)

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

    @check_exists(dtype='scenario_variant')
    def write_scenario_variant_data(self, data, scenario_name, variant_name, variable,
                                    timestep=None):
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
        return os.path.join(self.file_dir['scenarios'], filename)

    def _read_scenario_variable_spec(self, scenario_name, variable):
        # Read spec from scenario->provides->variable
        scenario = self.read_scenario(scenario_name)
        spec = _pick_from_list(scenario['provides'], variable)
        self._set_item_coords(spec)
        return Spec.from_dict(spec)
    # endregion

    # region Narratives
    def read_narratives(self, skip_coords=False):
        # Find filename for this narrative
        project_config = self.read_project_config()
        narratives = copy.deepcopy(project_config['narratives'])
        if not skip_coords:
            for narrative in narratives:
                self._set_list_coords(narrative['provides'])
        return narratives

    @check_exists(dtype='narrative')
    def read_narrative(self, narrative_name, skip_coords=False):
        project_config = self.read_project_config()
        narrative = copy.deepcopy(
            _pick_from_list(project_config['narratives'], narrative_name))
        if not skip_coords:
            self._set_list_coords(narrative['provides'])
        return narrative

    @check_not_exists(dtype='narrative')
    def write_narrative(self, narrative):
        project_config = self.read_project_config()
        narrative = copy.deepcopy(narrative)
        narrative = self._skip_coords(narrative, ['provides'])
        try:
            project_config['narratives'].append(narrative)
        except KeyError:
            project_config['narratives'] = [narrative]
        self._write_project_config(project_config)

    @check_exists(dtype='narrative')
    def update_narrative(self, narrative_name, narrative):
        project_config = self.read_project_config()
        narrative = copy.deepcopy(narrative)
        narrative = self._skip_coords(narrative, ['provides'])
        idx = _idx_in_list(project_config['narratives'], narrative_name)
        project_config['narratives'][idx] = narrative
        self._write_project_config(project_config)

    @check_exists(dtype='narrative')
    def delete_narrative(self, narrative_name):
        project_config = self.read_project_config()
        idx = _idx_in_list(project_config['narratives'], narrative_name)
        del project_config['narratives'][idx]
        self._write_project_config(project_config)

    @check_exists(dtype='narrative')
    def read_narrative_variants(self, narrative_name):
        project_config = self.read_project_config()
        n_idx = _idx_in_list(project_config['narratives'], narrative_name)
        return project_config['narratives'][n_idx]['variants']

    @check_exists(dtype='narrative_variant')
    def read_narrative_variant(self, narrative_name, variant_name):
        project_config = self.read_project_config()
        n_idx = _idx_in_list(project_config['narratives'], narrative_name)
        variants = project_config['narratives'][n_idx]['variants']
        return _pick_from_list(variants, variant_name)

    @check_not_exists(dtype='narrative_variant')
    def write_narrative_variant(self, narrative_name, variant):
        project_config = self.read_project_config()
        n_idx = _idx_in_list(project_config['narratives'], narrative_name)
        project_config['narratives'][n_idx]['variants'].append(variant)
        self._write_project_config(project_config)

    @check_exists(dtype='narrative_variant')
    def update_narrative_variant(self, narrative_name, variant_name, variant):
        project_config = self.read_project_config()
        n_idx = _idx_in_list(project_config['narratives'], narrative_name)
        v_idx = _idx_in_list(project_config['narratives'][n_idx]['variants'], variant_name)
        project_config['narratives'][n_idx]['variants'][v_idx] = variant
        self._write_project_config(project_config)

    @check_exists(dtype='narrative_variant')
    def delete_narrative_variant(self, narrative_name, variant_name):
        project_config = self.read_project_config()
        n_idx = _idx_in_list(project_config['narratives'], narrative_name)
        v_idx = _idx_in_list(project_config['narratives'][n_idx]['variants'], variant_name)
        del project_config['narratives'][n_idx]['variants'][v_idx]
        self._write_project_config(project_config)

    @check_exists(dtype='narrative_variant')
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

    @check_exists(dtype='narrative_variant')
    def write_narrative_variant_data(self, data, narrative_name, variant_name, variable,
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
        return os.path.join(self.file_dir['narratives'], filename)

    def _read_narrative_variable_spec(self, narrative_name, variable):
        # Read spec from narrative->provides->variable
        narrative = self.read_narrative(narrative_name)
        spec = _pick_from_list(narrative['provides'], variable)
        self._set_item_coords(spec)
        return Spec.from_dict(spec)
    # endregion

    # region Results
    def read_results(self, modelrun_id, model_name, output_spec, timestep=None,
                     modelset_iteration=None, decision_iteration=None):
        if timestep is None:
            raise NotImplementedError()

        results_path = self._get_results_path(
            modelrun_id, model_name, output_spec.name,
            timestep, modelset_iteration, decision_iteration
        )

        if self.storage_format == 'local_csv':
            data = self._get_data_from_csv(results_path)
            return self.data_list_to_ndarray(data, output_spec)
        if self.storage_format == 'local_binary':
            return self._get_data_from_native_file(results_path)

    def write_results(self, data, modelrun_id, model_name, output_spec, timestep=None,
                      modelset_iteration=None, decision_iteration=None):
        if timestep is None:
            raise NotImplementedError()

        results_path = self._get_results_path(
            modelrun_id, model_name, output_spec.name,
            timestep, modelset_iteration, decision_iteration
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        if self.storage_format == 'local_csv':
            data = self.ndarray_to_data_list(data, output_spec)
            self._write_data_to_csv(results_path, data, spec=output_spec)
        if self.storage_format == 'local_binary':
            self._write_data_to_native_file(results_path, data)

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
        previous_results_dir = os.path.join(self.file_dir['results'], modelrun_name)
        return list(
            glob.iglob(os.path.join(previous_results_dir, '**/*.*'), recursive=True))

    def prepare_warm_start(self, modelrun_id):
        results_dir = os.path.join(self.file_dir['results'], modelrun_id)

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
        previous_results_dir = os.path.join(self.file_dir['results'], modelrun_id)
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
                filename.replace(self.file_dir['results'], '')[1:]))

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
                    result['modelset_iteration'],
                    result['decision_iteration']))

        self.logger.info("Warm start will resume at timestep %s", latest_timestep)
        return latest_timestep

    def _get_results_path(self, modelrun_id, model_name, output_name, timestep,
                          modelset_iteration=None, decision_iteration=None):
        """Return path to filename for a given output without file extension

        On the pattern of:
            results/<modelrun_name>/<model_name>/
            decision_<id>_modelset_<id>/
            output_<output_name>_timestep_<timestep>.csv

        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_name : str
        timestep : str or int
        modelset_iteration : int, optional
        decision_iteration : int, optional

        Returns
        -------
        path : strs
        """
        results_dir = self.file_dir['results']

        if modelset_iteration is None:
            modelset_iteration = 'none'
        if decision_iteration is None:
            decision_iteration = 'none'

        if self.storage_format == 'local_csv':
            ext = 'csv'
        elif self.storage_format == 'local_binary':
            ext = 'dat'
        else:
            ext = 'unknown'

        path = os.path.join(
            results_dir, modelrun_id, model_name,
            "decision_{}_modelset_{}".format(decision_iteration, modelset_iteration),
            "output_{}_timestep_{}.{}".format(output_name, timestep, ext)
        )
        return path

    def _parse_results_path(self, path):
        """Return result metadata for a given result path

        On the pattern of:
            results/<modelrun_name>/<model_name>/
            decision_<id>_modelset_<id>/
            output_<output_name>_timestep_<timestep>.csv

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
                self.file_dir['project'], 'project')
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
    # endregion


def _file_dtypes():
    return ('model_run', 'sos_model', 'sector_model')


def _config_dtypes():
    return ('dimension', 'narrative', 'scenario')


def _nested_config_dtypes():
    return ('narrative_variant', 'scenario_variant')


def _assert_no_mismatch(dtype, name, obj, secondary=None):
    try:
        iter(obj)
    except TypeError:
        return

    if 'name' in obj and name != obj['name']:
        raise SmifDataMismatchError("%s name '%s' must match '%s'" %
                                    (dtype, name, obj['name']))


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


def _nested_config_item_exists(config, dtype, parent_name, child_name):
    keys = dtype.split("_")
    parent_key = "%ss" % keys[0]
    child_key = "%ss" % keys[1]
    if parent_key not in config:
        return False
    parent_idx = _idx_in_list(config[parent_key], parent_name)
    if parent_idx is None:
        return False
    if child_key not in config[parent_key][parent_idx]:
        return False
    return _name_in_list(config[parent_key][parent_idx][child_key], child_name)


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
        raise SmifDataNotFoundError("%s '%s' not found" % (dtype, name))


def _assert_config_item_not_exists(config, dtype, name):
    if _config_item_exists(config, dtype, name):
        raise SmifDataExistsError("%s '%s' already exists" % (dtype, name))


def _assert_nested_config_item_exists(config, dtype, primary, secondary):
    if not _nested_config_item_exists(config, dtype, primary, secondary):
        raise SmifDataNotFoundError("%s '%s:%s' not found" % (dtype, primary, secondary))


def _assert_nested_config_item_not_exists(config, dtype, primary, secondary):
    if _nested_config_item_exists(config, dtype, primary, secondary):
        raise SmifDataExistsError("%s '%s:%s' already exists" % (dtype, primary, secondary))
