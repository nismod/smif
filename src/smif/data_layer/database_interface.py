
from smif.data_layer.data_interface import DataInterface


class DatabaseInterface(DataInterface):
    def __init__(self, config_path):
        raise NotImplementedError()

    # region Model runs
    def read_model_runs(self):
        raise NotImplementedError()

    def read_model_run(self, model_run_name):
        raise NotImplementedError()

    def write_model_run(self, model_run):
        raise NotImplementedError()

    def update_model_run(self, model_run_name, model_run):
        raise NotImplementedError()

    def delete_model_run(self, model_run_name):
        raise NotImplementedError()
    # endregion

    # region System-of-systems models
    def read_sos_models(self):
        raise NotImplementedError()

    def read_sos_model(self, sos_model_name):
        raise NotImplementedError()

    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    def update_sos_model(self, sos_model_name, sos_model):
        raise NotImplementedError()

    def delete_sos_model(self, sos_model_name):
        raise NotImplementedError()
    # endregion

    # region Sector models
    def read_sector_models(self):
        raise NotImplementedError()

    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    def write_sector_model(self, sector_model):
        raise NotImplementedError()

    def update_sector_model(self, sector_model_name, sector_model):
        raise NotImplementedError()

    def delete_sector_model(self, sector_model_name):
        raise NotImplementedError()
    # endregion

    # region Strategies
    def read_strategies(self, model_run_name):
        raise NotImplementedError()

    def write_strategies(self, model_run_name, strategies):
        raise NotImplementedError()
    # endregion

    # region Interventions
    def read_interventions(self, sector_model_name):
        raise NotImplementedError()

    def read_initial_conditions(self, sector_model_name):
        raise NotImplementedError()
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        raise NotImplementedError()

    def write_state(self, state, modelrun_name, timestep, decision_iteration=None):
        raise NotImplementedError()
    # endregion

    # region Units
    def read_unit_definitions(self):
        raise NotImplementedError()
    # endregion

    # region Dimensions
    def read_dimensions(self):
        raise NotImplementedError()

    def read_dimension(self, dimension_name):
        raise NotImplementedError()

    def write_dimension(self, dimension):
        raise NotImplementedError()

    def update_dimension(self, dimension_name, dimension):
        raise NotImplementedError()

    def delete_dimension(self, dimension_name):
        raise NotImplementedError()
    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        raise NotImplementedError

    def write_coefficients(self, source_spec, destination_spec, data):
        raise NotImplementedError()
    # endregion

    # region Scenarios
    def read_scenarios(self):
        raise NotImplementedError()

    def read_scenario(self, scenario_name):
        raise NotImplementedError()

    def write_scenario(self, scenario):
        raise NotImplementedError()

    def update_scenario(self, scenario_name, scenario):
        raise NotImplementedError()

    def delete_scenario(self, scenario_name):
        raise NotImplementedError()

    def read_scenario_variants(self, scenario_name):
        raise NotImplementedError()

    def read_scenario_variant(self, scenario_name, variant_name):
        raise NotImplementedError()

    def write_scenario_variant(self, scenario_name, variant):
        raise NotImplementedError()

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        raise NotImplementedError()

    def delete_scenario_variant(self, scenario_name, variant_name):
        raise NotImplementedError()

    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        raise NotImplementedError()

    def write_scenario_variant_data(self, data, scenario_name, variant_name, variable,
                                    timestep=None):
        raise NotImplementedError()
    # endregion

    # region Narratives
    def read_narratives(self):
        raise NotImplementedError()

    def read_narrative(self, narrative_name):
        raise NotImplementedError()

    def write_narrative(self, narrative):
        raise NotImplementedError()

    def update_narrative(self, narrative_name, narrative):
        raise NotImplementedError()

    def delete_narrative(self, narrative_name):
        raise NotImplementedError()

    def read_narrative_variants(self, narrative_name):
        raise NotImplementedError()

    def read_narrative_variant(self, narrative_name, variant_name):
        raise NotImplementedError()

    def write_narrative_variant(self, narrative_name, variant):
        raise NotImplementedError()

    def update_narrative_variant(self, narrative_name, variant_name, variant):
        raise NotImplementedError()

    def delete_narrative_variant(self, narrative_name, variant_name):
        raise NotImplementedError()

    def read_narrative_variant_data(self, narrative_name, variant_name, variable,
                                    timestep=None):
        raise NotImplementedError()

    def write_narrative_variant_data(self, data, narrative_name, variant_name, variable,
                                     timestep=None):
        raise NotImplementedError()
    # endregion

    # region Results
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     modelset_iteration=None, decision_iteration=None):
        raise NotImplementedError()

    def write_results(self, data, modelrun_name, model_name, output_spec, timestep=None,
                      modelset_iteration=None, decision_iteration=None):
        raise NotImplementedError()

    def prepare_warm_start(self, modelrun_id):
        raise NotImplementedError()
    # endregion
