"""Memory-backed data interface
"""
from smif.data_layer.data_interface import DataInterface


class MemoryInterface(DataInterface):
    """ Read and write interface to main memory
    """
    def __init__(self):
        self._sos_model_runs = {}
        self._sos_models = {}
        self._sector_models = {}
        self._units = {}
        self._regions = {}
        self._intervals = {}
        self._scenario_sets = {}
        self._scenarios = {}
        self._narrative_sets = {}
        self._narratives = {}
        self._results = {}
        self._coefficients = {}
        self._strategies = {}
        self._state = {}

    def read_strategies(self):
        return self._strategies.items()

    def read_units_file_name(self):
        return self._units.values()

    def read_sos_model_runs(self):
        return self._sos_model_runs.values()

    def read_sos_model_run(self, sos_model_run_name):
        return self._sos_model_runs[sos_model_run_name]

    def write_sos_model_run(self, sos_model_run):
        self._sos_model_runs[sos_model_run['name']] = sos_model_run

    def update_sos_model_run(self, sos_model_run_name, sos_model_run):
        self._sos_model_runs[sos_model_run_name] = sos_model_run

    def delete_sos_model_run(self, sos_model_run):
        del self._sos_model_runs[sos_model_run['name']]

    def read_sos_models(self):
        return self._sos_models.values()

    def write_sos_model(self, sos_model):
        self._sos_models[sos_model['name']] = sos_model

    def update_sos_model(self, sos_model_name, sos_model):
        self._sos_models[sos_model_name] = sos_model

    def read_sector_models(self):
        return self._sector_models.values()

    def read_sector_model(self, sector_model_name):
        return self._sector_models[sector_model_name]

    def write_sector_model(self, sector_model):
        self._sector_models[sector_model['name']] = sector_model

    def update_sector_model(self, sector_model_name, sector_model):
        self._sector_models[sector_model_name] = sector_model

    def read_region_definitions(self):
        return self._regions.values()

    def read_region_definition_data(self, region_name):
        return self._regions[region_name]

    def write_region_definition(self, region):
        self._regions[region['name']] = region

    def update_region_definition(self, region):
        self._regions[region['name']] = region

    def read_interval_definitions(self):
        return self._intervals.values()

    def read_interval_definition_data(self, interval_name):
        return self._intervals[interval_name]

    def write_interval_definition(self, interval):
        self._intervals[interval['name']] = interval

    def update_interval_definition(self, interval):
        self._intervals[interval['name']] = interval

    def read_scenario_sets(self):
        return self._scenario_sets.values()

    def read_scenario_set(self, scenario_set_name):
        return self._scenario_sets[scenario_set_name]

    def write_scenario_set(self, scenario_set):
        self._scenario_sets[scenario_set['name']] = scenario_set

    def update_scenario_set(self, scenario_set):
        self._scenario_sets[scenario_set['name']] = scenario_set

    def read_scenario_data(self, scenario_name, parameter_name,
                           spatial_resolution, temporal_resolution, timestep):
        return self._scenarios[(
                scenario_name, parameter_name, spatial_resolution,
                temporal_resolution, timestep
            )]

    def write_scenario_data(self, scenario_name, parameter_name, data,
                            spatial_resolution, temporal_resolution, timestep):
        self._scenarios[(
            scenario_name, parameter_name, spatial_resolution,
            temporal_resolution, timestep
        )] = data

    def write_scenario(self, scenario):
        self._scenarios[scenario['name']] = scenario

    def update_scenario(self, scenario):
        self._scenarios[scenario['name']] = scenario

    def read_narrative_sets(self):
        return self._narrative_sets.values()

    def read_narrative_set(self, narrative_set_name):
        return self._narrative_sets[narrative_set_name]

    def write_narrative_set(self, narrative_set):
        self._narrative_sets[narrative_set['name']] = narrative_set

    def update_narrative_set(self, narrative_set):
        self._narrative_sets[narrative_set['name']] = narrative_set

    def read_narrative_data(self, narrative_name):
        return self._narratives[narrative_name]

    def write_narrative(self, narrative):
        self._narratives[narrative['name']] = narrative

    def update_narrative(self, narrative):
        self._narratives[narrative['name']] = narrative

    def read_state(self, modelrun_name, timestep=None, decision_iteration=None):
        """state is a list of (intervention_name, build_year), output of decision module/s
        """
        return self._state[(modelrun_name, timestep, decision_iteration)]

    def write_state(self, state, modelrun_name, timestep=None, decision_iteration=None):
        """state is a list of (intervention_name, build_year), output of decision module/s
        """
        self._state[(modelrun_name, timestep, decision_iteration)] = state

    def read_results(self, modelrun_name, model_name, output_name, spatial_resolution,
                     temporal_resolution, timestep=None, modelset_iteration=None,
                     decision_iteration=None):
        return self._results[
            (
                modelrun_name, model_name, output_name, spatial_resolution,
                temporal_resolution, timestep, modelset_iteration,
                decision_iteration
            )]

    def write_results(self, modelrun_name, model_name, output_name, data, spatial_resolution,
                      temporal_resolution, timestep=None, modelset_iteration=None,
                      decision_iteration=None):
        self._results[
            (
                modelrun_name, model_name, output_name, spatial_resolution,
                temporal_resolution, timestep, modelset_iteration,
                decision_iteration
            )] = data

    def read_coefficients(self, source_name, destination_name):
        return self._coefficients[(source_name, destination_name)]

    def write_coefficients(self, source_name, destination_name, data):
        self._coefficients[(source_name, destination_name)] = data
