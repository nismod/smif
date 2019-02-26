"""Memory-backed store implementations
"""
from collections import OrderedDict
from copy import copy, deepcopy

from smif.data_layer.abstract_config_store import ConfigStore
from smif.data_layer.abstract_data_store import DataStore
from smif.data_layer.abstract_metadata_store import MetadataStore
from smif.data_layer.data_array import DataArray
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError)


class MemoryConfigStore(ConfigStore):
    """Config store in memory
    """
    def __init__(self):
        super().__init__()
        self._model_runs = OrderedDict()
        self._sos_models = OrderedDict()
        self._models = OrderedDict()
        self._scenarios = OrderedDict()
        self._narratives = OrderedDict()
        self._strategies = OrderedDict()

    # region Model runs
    def read_model_runs(self):
        return list(self._model_runs.values())

    def read_model_run(self, model_run_name):
        try:
            return self._model_runs[model_run_name]
        except KeyError:
            raise SmifDataNotFoundError("sos_model_run '%s' not found" % (model_run_name))

    def write_model_run(self, model_run):
        if model_run['name'] not in self._model_runs:
            self._model_runs[model_run['name']] = model_run
        else:
            raise SmifDataExistsError("model_run '%s' already exists" % (model_run['name']))

    def update_model_run(self, model_run_name, model_run):
        if model_run_name in self._model_runs:
            self._model_runs[model_run_name] = model_run
        else:
            raise SmifDataNotFoundError("model_run '%s' not found" % (model_run_name))

    def delete_model_run(self, model_run_name):
        try:
            del self._model_runs[model_run_name]
        except KeyError:
            raise SmifDataNotFoundError("model_run '%s' not found" % (model_run_name))
    # endregion

    # region System-of-systems models
    def read_sos_models(self):
        return list(self._sos_models.values())

    def read_sos_model(self, sos_model_name):
        try:
            return self._sos_models[sos_model_name]
        except KeyError:
            raise SmifDataNotFoundError("sos_model '%s' not found" % (sos_model_name))

    def write_sos_model(self, sos_model):
        if sos_model['name'] not in self._sos_models:
            self._sos_models[sos_model['name']] = sos_model
        else:
            raise SmifDataExistsError("sos_model '%s' already exists" % (sos_model['name']))

    def update_sos_model(self, sos_model_name, sos_model):
        if sos_model_name in self._sos_models:
            self._sos_models[sos_model_name] = sos_model
        else:
            raise SmifDataNotFoundError("sos_model '%s' not found" % (sos_model_name))

    def delete_sos_model(self, sos_model_name):
        try:
            del self._sos_models[sos_model_name]
        except KeyError:
            raise SmifDataNotFoundError("sos_model '%s' not found" % (sos_model_name))
    # endregion

    # region Models
    def read_models(self):
        return list(self._models.values())

    def read_model(self, model_name):
        try:
            return self._models[model_name]
        except KeyError:
            raise SmifDataNotFoundError("model '%s' not found" % (model_name))

    def write_model(self, model):
        if model['name'] not in self._models:
            model = _skip_coords(model, ('inputs', 'outputs', 'parameters'))
            self._models[model['name']] = model
        else:
            raise SmifDataExistsError("model '%s' already exists" % (model['name']))

    def update_model(self, model_name, model):
        if model_name in self._models:
            model = _skip_coords(model, ('inputs', 'outputs', 'parameters'))
            self._models[model_name] = model
        else:
            raise SmifDataNotFoundError("model '%s' not found" % (model_name))

    def delete_model(self, model_name):
        try:
            del self._models[model_name]
        except KeyError:
            raise SmifDataNotFoundError("model '%s' not found" % (model_name))
    # endregion

    # region Scenarios
    def read_scenarios(self):
        scenarios = self._scenarios.values()
        return [_variant_dict_to_list(s) for s in scenarios]

    def read_scenario(self, scenario_name):
        try:
            scenario = self._scenarios[scenario_name]
            return _variant_dict_to_list(scenario)
        except KeyError:
            raise SmifDataNotFoundError("scenario '%s' not found" % (scenario_name))

    def write_scenario(self, scenario):
        if scenario['name'] not in self._scenarios:
            scenario = _variant_list_to_dict(scenario)
            scenario = _skip_coords(scenario, ['provides'])
            self._scenarios[scenario['name']] = scenario
        else:
            raise SmifDataExistsError("scenario '%s' already exists" % (scenario['name']))

    def update_scenario(self, scenario_name, scenario):
        if scenario_name in self._scenarios:
            scenario = _variant_list_to_dict(scenario)
            scenario = _skip_coords(scenario, ['provides'])
            self._scenarios[scenario_name] = scenario
        else:
            raise SmifDataNotFoundError("scenario '%s' not found" % (scenario_name))

    def delete_scenario(self, scenario_name):
        try:
            del self._scenarios[scenario_name]
        except KeyError:
            raise SmifDataNotFoundError("scenario '%s' not found" % (scenario_name))
    # endregion

    # region Scenario Variants
    def read_scenario_variants(self, scenario_name):
        return list(self._scenarios[scenario_name]['variants'].values())

    def read_scenario_variant(self, scenario_name, variant_name):
        try:
            return self._scenarios[scenario_name]['variants'][variant_name]
        except KeyError:
            raise SmifDataNotFoundError("scenario '%s' variant '%s' not found"
                                        % (scenario_name, variant_name))

    def write_scenario_variant(self, scenario_name, variant):
        self._scenarios[scenario_name]['variants'][variant['name']] = variant

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        self._scenarios[scenario_name]['variants'][variant_name] = variant

    def delete_scenario_variant(self, scenario_name, variant_name):
        del self._scenarios[scenario_name]['variants'][variant_name]
    # endregion

    # region Narratives
    def _read_narratives(self, sos_model_name):
        return self._sos_models[sos_model_name]['narratives']

    def read_narrative(self, sos_model_name, narrative_name):
        try:
            narrative = [x for x in self._read_narratives(sos_model_name)
                         if x['name'] == narrative_name][0]
        except IndexError:
            msg = "Narrative '{}' not found in '{}'"
            raise SmifDataNotFoundError(msg.format(narrative_name, sos_model_name))
        return narrative

    def _read_narrative_variant(self, sos_model_name, narrative_name, variant_name):
        narrative = self.read_narrative(sos_model_name, narrative_name)
        try:
            variant = [x for x in narrative['variants'] if x['name'] == variant_name][0]
        except IndexError:
            msg = "Variant '{}' not found in '{}'"
            raise SmifDataNotFoundError(msg.format(variant_name, narrative_name))
        return variant
    # endregion

    # region Strategies
    def read_strategies(self, modelrun_name):
        try:
            return self._strategies[modelrun_name]
        except KeyError:
            raise SmifDataNotFoundError("strategies in modelrun '%s' not found"
                                        % (modelrun_name))

    def write_strategies(self, modelrun_name, strategies):
        self._strategies[modelrun_name] = strategies
    # endregion


class MemoryMetadataStore(MetadataStore):
    """Store metadata in-memory
    """
    def __init__(self):
        super().__init__()
        self._units = []  # list[str] of pint definitions
        self._dimensions = OrderedDict()

    # region Units
    def write_unit_definitions(self, units):
        self._units = units

    def read_unit_definitions(self):
        return self._units
    # endregion

    # region Dimensions
    def read_dimensions(self, skip_coords=False):
        return [self.read_dimension(k, skip_coords) for k in self._dimensions]

    def read_dimension(self, dimension_name, skip_coords=False):
        dim = self._dimensions[dimension_name]
        if skip_coords:
            dim = {
                'name': dim['name'],
                'description': dim['description']
            }
        return dim

    def write_dimension(self, dimension):
        self._dimensions[dimension['name']] = dimension

    def update_dimension(self, dimension_name, dimension):
        self._dimensions[dimension['name']] = dimension

    def delete_dimension(self, dimension_name):
        del self._dimensions[dimension_name]
    # endregion


class MemoryDataStore(DataStore):
    """Store data in-memory
    """
    def __init__(self):
        super().__init__()
        self._data_array = OrderedDict()
        self._interventions = OrderedDict()
        self._initial_conditions = OrderedDict()
        self._state = OrderedDict()
        self._model_parameter_defaults = OrderedDict()
        self._coefficients = OrderedDict()
        self._results = OrderedDict()

    # region Data Array
    def read_scenario_variant_data(self, key, spec, timestep=None):
        return self._read_data_array(key, spec, timestep)

    def write_scenario_variant_data(self, key, data, timestep=None):
        self._write_data_array(key, data, timestep)

    def read_narrative_variant_data(self, key, spec, timestep=None):
        return self._read_data_array(key, spec, timestep)

    def write_narrative_variant_data(self, key, data, timestep=None):
        self._write_data_array(key, data, timestep)

    def _read_data_array(self, key, spec, timestep=None):
        if timestep:
            try:
                data = self._data_array[key, timestep]
            except KeyError:
                try:
                    data = self._filter_timestep(self._data_array[key], spec, timestep)
                except KeyError:
                    raise SmifDataNotFoundError(
                        "Data for {} not found for timestep {}".format(spec.name, timestep))
        else:
            try:
                data = self._data_array[key]
            except KeyError:
                raise SmifDataNotFoundError(
                    "Data for {} not found".format(spec.name))

        if data.spec != spec:
            raise SmifDataMismatchError(
                "Spec did not match reading {}, requested {}, got {}".format(
                    spec.name, spec, data.spec))
        return data

    def _filter_timestep(self, data, read_spec, timestep):
        dataframe = data.as_df().reset_index()
        if 'timestep' not in dataframe.columns:
            msg = "Missing 'timestep' key, found {} in {}"
            raise SmifDataMismatchError(msg.format(list(dataframe.columns), data.name))
        dataframe = dataframe[dataframe.timestep == timestep]
        if dataframe.empty:
            raise SmifDataNotFoundError(
                "Data for {} not found for timestep {}".format(data.name, timestep))
        dataframe.drop('timestep', axis=1, inplace=True)
        return DataArray.from_df(read_spec, dataframe)

    def _write_data_array(self, key, data, timestep=None):
        if timestep:
            self._data_array[key, timestep] = data
        else:
            self._data_array[key] = data
    # endregion

    # region Model parameters
    def read_model_parameter_default(self, key, spec):
        data = self._model_parameter_defaults[key]
        if data.spec != spec:
            raise SmifDataMismatchError(
                "Spec did not match reading {}, requested {}, got {}".format(
                    spec.name, spec, data.spec))
        return data

    def write_model_parameter_default(self, key, data):
        self._model_parameter_defaults[key] = data
    # endregion

    # region Interventions
    def read_interventions(self, keys):
        all_interventions = {}
        interventions = [list(self._interventions[key].values()) for key in keys][0]

        for entry in interventions:
            name = entry.pop('name')
            if name in all_interventions:
                msg = "An entry for intervention {} already exists"
                raise ValueError(msg.format(name))
            else:
                all_interventions[name] = entry

        return all_interventions

    def write_interventions(self, key, interventions):
        self._interventions[key] = interventions

    def read_strategy_interventions(self, strategy):
        return strategy['interventions']

    def read_initial_conditions(self, keys):
        return [self._initial_conditions[key] for key in keys][0]

    def write_initial_conditions(self, key, initial_conditions):
        self._initial_conditions[key] = initial_conditions
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep=None, decision_iteration=None):
        return self._state[(modelrun_name, timestep, decision_iteration)]

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        self._state[(modelrun_name, timestep, decision_iteration)] = state
    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        spec = (source_spec.name, destination_spec.name)
        try:
            return self._coefficients[(source_spec.name, destination_spec.name)]
        except KeyError:
            msg = "Could not find coefficients for spec pair {}.{}"
            raise SmifDataNotFoundError(msg.format(spec[0], spec[1]))

    def write_coefficients(self, source_spec, destination_spec, data):
        self._coefficients[(source_spec.name, destination_spec.name)] = data
    # endregion

    # region Results
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     decision_iteration=None):
        key = (modelrun_name, model_name, output_spec.name, timestep, decision_iteration)

        try:
            results = self._results[key]
        except KeyError:
            raise SmifDataNotFoundError("Cannot find results for {}".format(key))

        return DataArray(output_spec, results)

    def write_results(self, data_array, modelrun_name, model_name, timestep=None,
                      decision_iteration=None):
        key = (modelrun_name, model_name, data_array.spec.name, timestep, decision_iteration)
        self._results[key] = data_array.as_ndarray()

    def available_results(self, model_run_name):
        results_keys = [
            (timestep, decision_iteration, model_name, output_name)
            for (result_modelrun_name, model_name, output_name, timestep, decision_iteration)
            in self._results.keys()
            if model_run_name == result_modelrun_name
        ]
        return results_keys
    # endregion


def _variant_list_to_dict(config):
    config = copy(config)
    try:
        list_ = config['variants']
    except KeyError:
        list_ = []
    config['variants'] = {variant['name']: variant for variant in list_}
    return config


def _variant_dict_to_list(config):
    config = copy(config)
    try:
        dict_ = config['variants']
    except KeyError:
        dict_ = {}
    config['variants'] = list(dict_.values())
    return config


def _skip_coords(config, keys):
    """Given a config dict and list of top-level keys for lists of specs,
    delete coords from each spec in each list.
    """
    config = deepcopy(config)
    for key in keys:
        for spec in config[key]:
            try:
                del spec['coords']
            except KeyError:
                pass
    return config
