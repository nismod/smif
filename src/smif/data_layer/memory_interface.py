"""Memory-backed data interface
"""
from smif.data_layer.data_interface import DataExistsError, DataInterface


class MemoryInterface(DataInterface):
    """ Read and write interface to main memory
    """
    def __init__(self):
        super().__init__()
        self._model_runs = {}
        self._sos_models = {}
        self._sector_models = {}
        self._strategies = {}
        self._state = {}
        self._units = {}
        self._dimensions = {}
        self._coefficients = {}
        self._scenarios = {}
        self._scenario_data = {}
        self._narratives = {}
        self._narrative_data = {}
        self._results = {}

    # region Model runs
    def read_model_runs(self):
        return [x for x in self._model_runs.values()]

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
        return [x for x in self._sos_models.values()]

    def read_sos_model(self, sos_model_name):
        return self._sos_models[sos_model_name]

    def write_sos_model(self, sos_model):
        if sos_model['name'] in self._sos_models:
            raise DataExistsError()
        self._sos_models[sos_model['name']] = sos_model

    def update_sos_model(self, sos_model_name, sos_model):
        self._sos_models[sos_model_name] = sos_model

    def delete_sos_model(self, sos_model_name):
        del self._sos_models[sos_model_name]
    # endregion

    # region Sector models
    def read_sector_models(self):
        return self._sector_models.values()

    def read_sector_model(self, sector_model_name):
        return self._sector_models[sector_model_name]

    def write_sector_model(self, sector_model):
        self._sector_models[sector_model['name']] = sector_model

    def update_sector_model(self, sector_model_name, sector_model):
        self._sector_models[sector_model_name] = sector_model

    def delete_sector_model(self, sector_model_name):
        del self._sector_models[sector_model_name]
    # endregion

    # region Strategies
    def read_strategies(self):
        return self._strategies.values()
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep=None, decision_iteration=None):
        return self._state[(modelrun_name, timestep, decision_iteration)]

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        self._state[(modelrun_name, timestep, decision_iteration)] = state
    # endregion

    # region Units
    def read_unit_definitions(self):
        return self._units.values()
    # endregion

    # region Dimensions
    def read_dimensions(self):
        return self._dimensions.values()

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
    def read_scenarios(self):
        return self._scenarios.values()

    def read_scenario(self, scenario_name):
        return self._scenarios[scenario_name]

    def write_scenario(self, scenario):
        self._scenarios[scenario['name']] = scenario

    def update_scenario(self, scenario_name, scenario):
        self._scenarios[scenario_name] = scenario

    def delete_scenario(self, scenario_name):
        del self._scenarios[scenario_name]

    def read_scenario_variants(self, scenario_name):
        return self._scenarios.values()

    def read_scenario_variant(self, scenario_name, variant_name):
        return self._scenarios[scenario_name]

    def write_scenario_variant(self, scenario_name, variant):
        self._scenarios[scenario_name]['variants'][variant['name']] = variant

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        self._scenarios[scenario_name]['variants'][variant_name] = variant

    def delete_scenario_variant(self, scenario_name, variant_name):
        del self._scenarios[scenario_name]['variants'][variant_name]

    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        return self._scenario_data[(scenario_name, variant_name, variable, timestep)]

    def write_scenario_variant_data(self, data, scenario_name, variant_name, variable,
                                    timestep=None):
        self._scenario_data[(scenario_name, variant_name, variable, timestep)] = data
    # endregion

    # region Narratives
    def read_narratives(self):
        return self._narratives.values()

    def read_narrative(self, narrative_name):
        return self._narratives[narrative_name]

    def write_narrative(self, narrative):
        self._narratives[narrative['name']] = narrative

    def update_narrative(self, narrative_name, narrative):
        self._narratives[narrative_name] = narrative

    def delete_narrative(self, narrative_name):
        del self._narratives[narrative_name]

    def read_narrative_variants(self, narrative_name):
        return self._narratives[narrative_name]['variants'].values()

    def read_narrative_variant(self, narrative_name, variant_name):
        return self._narratives[narrative_name]['variants'][variant_name]

    def write_narrative_variant(self, narrative_name, variant):
        self._narratives[narrative_name]['variants'][variant['name']] = variant

    def update_narrative_variant(self, narrative_name, variant_name, variant):
        self._narratives[narrative_name]['variants'][variant_name] = variant

    def delete_narrative_variant(self, narrative_name, variant_name):
        del self._narratives[narrative_name]['variants'][variant_name]

    def read_narrative_variant_data(self, narrative_name, variant_name, variable,
                                    timestep=None):
        return self._narrative_data[(narrative_name, variant_name, variable, timestep)]

    def write_narrative_variant_data(self, data, narrative_name, variant_name, variable,
                                     timestep=None):
        self._narrative_data[(narrative_name, variant_name, variable, timestep)] = data
    # endregion

    # region Results
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     modelset_iteration=None, decision_iteration=None):
        key = (
            modelrun_name, model_name, output_spec, timestep, modelset_iteration,
            decision_iteration
        )
        self.logger.debug("Get %s", key)
        return self._results[key]

    def write_results(self, data, modelrun_name, model_name, output_spec, timestep=None,
                      modelset_iteration=None, decision_iteration=None):
        key = (
            modelrun_name, model_name, output_spec, timestep, modelset_iteration,
            decision_iteration
        )
        self.logger.debug("Set %s", key)
        self._results[key] = data

    def prepare_warm_start(self, modelrun_id):
        return self._model_runs[modelrun_id]['timesteps'][0]
    # endregion
