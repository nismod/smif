"""Memory-backed store implementations
"""
from collections import OrderedDict
from copy import copy

from smif.data_layer.abstract_config_store import ConfigStore
from smif.data_layer.abstract_data_store import DataStore
from smif.data_layer.abstract_metadata_store import MetadataStore
from smif.data_layer.data_array import DataArray
from smif.exception import SmifDataExistsError, SmifDataNotFoundError


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
        model_runs = self._model_runs.items()
        sorted_model_runs = OrderedDict(sorted(model_runs,
                                        key=lambda t: t[0]))
        return list(sorted_model_runs.values())

    def read_model_run(self, model_run_name):
        return self._model_runs[model_run_name]

    def write_model_run(self, model_run):
        self._model_runs[model_run['name']] = model_run

    def update_model_run(self, model_run_name, model_run):
        self._model_runs[model_run_name] = model_run

    def delete_model_run(self, model_run_name):
        del self._model_runs[model_run_name]
    # endregion

    # region System-of-systems models
    def read_sos_models(self):
        return list(self._sos_models.values())

    def read_sos_model(self, sos_model_name):
        return self._sos_models[sos_model_name]

    def write_sos_model(self, sos_model):
        if sos_model['name'] in self._sos_models:
            raise SmifDataExistsError()
        self._sos_models[sos_model['name']] = sos_model

    def update_sos_model(self, sos_model_name, sos_model):
        self._sos_models[sos_model_name] = sos_model

    def delete_sos_model(self, sos_model_name):
        del self._sos_models[sos_model_name]
    # endregion

    # region Models
    def read_models(self):
        return list(self._models.values())

    def read_model(self, model_name):
        m = self._models[model_name]
        return m

    def write_model(self, model):
        self._models[model['name']] = model

    def update_model(self, model_name, model):
        self._models[model_name] = model

    def delete_model(self, model_name):
        del self._models[model_name]
    # endregion

    # region Scenarios
    def read_scenarios(self):
        scenarios = self._scenarios.values()
        return [_variant_dict_to_list(s) for s in scenarios]

    def read_scenario(self, scenario_name):
        scenario = self._scenarios[scenario_name]
        return _variant_dict_to_list(scenario)

    def write_scenario(self, scenario):
        scenario = _variant_list_to_dict(scenario)
        self._scenarios[scenario['name']] = scenario

    def update_scenario(self, scenario_name, scenario):
        scenario = _variant_list_to_dict(scenario)
        self._scenarios[scenario_name] = scenario

    def delete_scenario(self, scenario_name):
        del self._scenarios[scenario_name]
    # endregion

    # region Scenario Variants
    def read_scenario_variants(self, scenario_name):
        return list(self._scenarios[scenario_name]['variants'].values())

    def read_scenario_variant(self, scenario_name, variant_name):
        return self._scenarios[scenario_name]['variants'][variant_name]

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
        return self._strategies[modelrun_name]

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
    def read_dimensions(self):
        return list(self._dimensions.values())

    def read_dimension(self, dimension_name):
        return self._dimensions[dimension_name]

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
        self._scenario_data = OrderedDict()
        self._narrative_data = OrderedDict()
        self._interventions = OrderedDict()
        self._initial_conditions = []
        self._state = OrderedDict()
        self._model_parameter_defaults = OrderedDict()
        self._coefficients = OrderedDict()
        self._results = OrderedDict()

    # region Scenario Variant Data
    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        return self._scenario_data[(scenario_name, variant_name, variable, timestep)]

    def write_scenario_variant_data(self, scenario_name, variant_name, data, timestep=None):
        self._scenario_data[(scenario_name, variant_name, data.name, timestep)] = data
    # endregion

    # region Narrative Data
    def read_narrative_variant_data(self, sos_model_name, narrative_name, variant_name,
                                    parameter_name, timestep=None):
        key = (sos_model_name, narrative_name, variant_name, parameter_name, timestep)
        try:
            return self._narrative_data[key]
        except KeyError:
            raise SmifDataNotFoundError

    def write_narrative_variant_data(self, sos_model_name, narrative_name, variant_name,
                                     data, timestep=None):
        key = (sos_model_name, narrative_name, variant_name, data.name, timestep)
        self._narrative_data[key] = data

    def read_model_parameter_default(self, model_name, parameter_name):
        return self._model_parameter_defaults[(model_name, parameter_name)]

    def write_model_parameter_default(self, model_name, parameter_name, data):
        self._model_parameter_defaults[(model_name, parameter_name)] = data
    # endregion

    # region Interventions
    def read_interventions(self, model_name):
        return self._interventions[model_name]

    def read_initial_conditions(self, model_name):
        return self._initial_conditions[model_name]
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep=None, decision_iteration=None):
        return self._state[(modelrun_name, timestep, decision_iteration)]

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        self._state[(modelrun_name, timestep, decision_iteration)] = state
    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        return self._coefficients[(source_spec, destination_spec)]

    def write_coefficients(self, source_spec, destination_spec, data):
        self._coefficients[(source_spec, destination_spec)] = data
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

    def prepare_warm_start(self, modelrun_name):
        results_keys = [k for k in self._results.keys() if k[0] == modelrun_name]
        if results_keys:
            return max(k[3] for k in results_keys)  # max timestep reached
        return None
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
