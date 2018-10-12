"""Memory-backed data interface
"""
from collections import OrderedDict
from copy import copy

from smif.data_layer.data_array import DataArray
from smif.data_layer.data_interface import DataInterface
from smif.exception import SmifDataExistsError, SmifDataNotFoundError


class MemoryInterface(DataInterface):
    """ Read and write interface to main memory
    """
    def __init__(self):
        super().__init__()
        self._model_runs = OrderedDict()
        self._sos_models = OrderedDict()
        self._sector_models = OrderedDict()
        self._strategies = OrderedDict()
        self._state = OrderedDict()
        self._units = []  # list[str] of pint definitions
        self._dimensions = OrderedDict()
        self._coefficients = OrderedDict()
        self._scenarios = OrderedDict()
        self._scenario_data = OrderedDict()
        self._narratives = OrderedDict()
        self._narrative_data = OrderedDict()
        self._results = OrderedDict()
        self._interventions = OrderedDict()
        self._initial_conditions = []

    # region Model runs
    def read_model_runs(self):
        return list(self._model_runs.values())

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

    # region Sector models
    def read_sector_models(self, skip_coords=False):
        if skip_coords:
            return [
                self._skip_coords(m, ('inputs', 'outputs', 'parameters'))
                for m in self._sector_models.values()
            ]
        return list(self._sector_models.values())

    def read_sector_model(self, sector_model_name, skip_coords=False):
        m = self._sector_models[sector_model_name]
        if skip_coords:
            return self._skip_coords(m, ('inputs', 'outputs', 'parameters'))
        return m

    def write_sector_model(self, sector_model):
        self._sector_models[sector_model['name']] = sector_model

    def update_sector_model(self, sector_model_name, sector_model):
        self._sector_models[sector_model_name] = sector_model

    def delete_sector_model(self, sector_model_name):
        del self._sector_models[sector_model_name]
    # endregion

    # region Interventions
    def read_interventions(self, sector_model_name):
        return self._interventions[sector_model_name]
    # endregion

    # region Strategies
    def read_strategies(self, modelrun_name):
        return self._strategies[modelrun_name]

    def write_strategies(self, modelrun_name, strategies):
        self._strategies[modelrun_name] = strategies

    def read_initial_conditions(self, sector_model_name):
        return self._initial_conditions[sector_model_name]
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep=None, decision_iteration=None):
        return self._state[(modelrun_name, timestep, decision_iteration)]

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        self._state[(modelrun_name, timestep, decision_iteration)] = state
    # endregion

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

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        return self._coefficients[(source_spec, destination_spec)]

    def write_coefficients(self, source_spec, destination_spec, data):
        self._coefficients[(source_spec, destination_spec)] = data
    # endregion

    # region Scenarios
    def read_scenarios(self, skip_coords=False):
        scenarios = self._scenarios.values()
        if skip_coords:
            scenarios = [
                self._skip_coords(s, ['provides'])
                for s in scenarios
            ]
        return [_variant_dict_to_list(s) for s in scenarios]

    def read_scenario(self, scenario_name, skip_coords=False):
        scenario = self._scenarios[scenario_name]
        if skip_coords:
            scenario = self._skip_coords(scenario, ['provides'])
        return _variant_dict_to_list(scenario)

    def write_scenario(self, scenario):
        scenario = _variant_list_to_dict(scenario)
        self._scenarios[scenario['name']] = scenario

    def update_scenario(self, scenario_name, scenario):
        scenario = _variant_list_to_dict(scenario)
        self._scenarios[scenario_name] = scenario

    def delete_scenario(self, scenario_name):
        del self._scenarios[scenario_name]

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

    def read_scenario_variant_data(self, scenario_name, variant_name, variable,
                                   timestep=None):
        scenario = self.read_scenario(scenario_name)
        spec = self._get_spec_from_provider(scenario['provides'], variable)
        data = self._scenario_data[(scenario_name, variant_name, variable, timestep)]
        return DataArray(spec, data)

    def write_scenario_variant_data(self, scenario_name, variant_name, data_array,
                                    timestep=None):

        self._scenario_data[(
            scenario_name,
            variant_name,
            data_array.name,
            timestep)] = data_array.as_ndarray()
    # endregion

    # region Narratives
    def _read_narratives(self, sos_model_name):
        return self._sos_models[sos_model_name]['narratives']

    def _read_narrative(self, sos_model_name, narrative_name):
        try:
            narrative = [x for x in self._read_narratives(sos_model_name)
                         if x['name'] == narrative_name][0]
        except IndexError:
            msg = "Narrative '{}' not found in '{}'"
            raise SmifDataNotFoundError(msg.format(narrative_name, sos_model_name))
        return narrative

    def _read_narrative_variant(self, sos_model_name, narrative_name, variant_name):
        narrative = self._read_narrative(sos_model_name, narrative_name)
        try:
            variant = [x for x in narrative['variants'] if x['name'] == variant_name][0]
        except IndexError:
            msg = "Variant '{}' not found in '{}'"
            raise SmifDataNotFoundError(msg.format(variant_name, narrative_name))
        return variant

    def read_narrative_variant_data(self, sos_model_name, narrative_name,
                                    variant_name, variable, timestep=None):
        variant = self._read_narrative_variant(sos_model_name, narrative_name, variant_name)
        if variable not in variant['data'].keys():
            self.logger.debug(variant['data'])
            msg = "Variable '{}' not found in '{}'"
            raise SmifDataNotFoundError(msg.format(variable, variant_name))
        else:
            data = self._narrative_data[(
                sos_model_name, narrative_name, variant_name, variable, timestep)]
            spec = self._read_narrative_variable_spec(sos_model_name, narrative_name, variable)
            return DataArray(spec, data)

    def write_narrative_variant_data(self, sos_model_name, narrative_name,
                                     variant_name, data_array, timestep=None):
        variable = data_array.name
        data = data_array.as_ndarray()

        self._narrative_data[(
            sos_model_name, narrative_name, variant_name, variable, timestep)] = data
    # endregion

    # region Results
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     decision_iteration=None):
        key = (
            modelrun_name, model_name, output_spec.name, timestep, decision_iteration
        )
        self.logger.debug("Get %s", key)

        try:
            results = self._results[key]
        except KeyError:
            raise SmifDataNotFoundError("Cannot find results for {}".format(key))

        return DataArray(output_spec, results)

    def write_results(self, data_array, modelrun_name, model_name, timestep=None,
                      decision_iteration=None):
        key = (
            modelrun_name, model_name, data_array.spec.name, timestep, decision_iteration
        )
        self.logger.debug("Set %s", key)
        self.logger.debug("Writing data %s", data_array.as_ndarray())
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
