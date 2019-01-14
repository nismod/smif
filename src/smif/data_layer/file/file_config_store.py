"""File-backed config store
"""
import copy
import os
from logging import getLogger

from ruamel.yaml import YAML
from smif.data_layer.abstract_config_store import ConfigStore
from smif.data_layer.validate import (validate_sos_model_config,
                                      validate_sos_model_format)
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError)


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

            self._project_config_cache = _read_yaml_file(self.base_folder, 'project')
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
        model_runs = [self.read_model_run(name) for name in names]
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
        if model_run['name'] != model_run_name:
            raise SmifDataMismatchError(
                "Model run name '%s' must match '%s'" % (model_run_name, model_run['name']))
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
        if sos_model['name'] != sos_model_name:
            raise SmifDataMismatchError(
                "SoSModel name '%s' must match '%s'" % (sos_model_name, sos_model['name']))
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
        _write_yaml_file(self.config_folders['sector_models'], model['name'], model)

    def update_model(self, model_name, model):
        if model['name'] != model_name:
            raise SmifDataMismatchError(
                "Model name '%s' must match '%s'" % (model_name, model['name']))
        _assert_file_exists(self.config_folders, 'sector_model', model_name)
        model = copy.deepcopy(model)

        # ignore interventions and initial conditions which the app doesn't handle
        if model['interventions'] or model['initial_conditions']:

            old_model = _read_yaml_file(self.config_folders['sector_models'], model['name'])

        if model['interventions']:
            self.logger.warning("Ignoring interventions write")
            model['interventions'] = old_model['interventions']

        if model['initial_conditions']:
            self.logger.warning("Ignoring initial conditions write")
            model['initial_conditions'] = old_model['initial_conditions']

        model = _skip_coords(model, ('inputs', 'outputs', 'parameters'))

        _write_yaml_file(self.config_folders['sector_models'], model['name'], model)

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


def _read_yaml_file(directory, name):
    """Read yaml config file into plain data (lists, dicts and simple values)

    Parameters
    ----------
    directory : str
    name : str
    """
    path = os.path.join(directory, "{}.yml".format(name))
    with open(path, 'r') as file_handle:
        return YAML().load(file_handle)


def _write_yaml_file(directory, name, data):
    """Write plain data to a file as yaml

    Arguments
    ---------
    directory: str
        Path to directory
    name: str
        Name of config item (filename without .yml extension)
    data
        Data to be written to the file
    """
    path = os.path.join(directory, "{}.yml".format(name))
    with open(path, 'w') as file_handle:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.allow_unicode = True
        return yaml.dump(data, file_handle)


def _assert_file_exists(file_dir, dtype, name):
    if not _file_exists(file_dir, dtype, name):
        raise SmifDataNotFoundError("%s '%s' not found" % (dtype.capitalize(), name))


def _assert_file_not_exists(file_dir, dtype, name):
    if _file_exists(file_dir, dtype, name):
        raise SmifDataExistsError("%s '%s' already exists" % (dtype.capitalize(), name))


def _file_exists(file_dir, dtype, name):
    dir_key = "%ss" % dtype
    try:
        return os.path.exists(os.path.join(file_dir[dir_key], name + '.yml'))
    except TypeError:
        msg = "Could not parse file name {} and dtype {}"
        raise SmifDataNotFoundError(msg.format(name, dtype))


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


def _idx_in_list(list_of_dicts, name):
    for i, item in enumerate(list_of_dicts):
        if 'name' in item and item['name'] == name:
            return i
    return None
