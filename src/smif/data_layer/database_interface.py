"""Database-backed data interface
"""

from smif.data_layer.data_interface import DataInterface


class DatabaseInterface(DataInterface):
    """ Read and write interface to Database
    """
    def __init__(self, config_path):
        raise NotImplementedError()

    def read_sos_model_runs(self):
        raise NotImplementedError()

    def write_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    def read_sos_models(self):
        raise NotImplementedError()

    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    def read_sector_models(self):
        raise NotImplementedError()

    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    def write_sector_model(self, sector_model):
        raise NotImplementedError()

    def read_units(self):
        raise NotImplementedError()

    def write_unit(self, unit):
        raise NotImplementedError()

    def read_regions(self):
        raise NotImplementedError()

    def read_region_data(self, region_name):
        raise NotImplementedError()

    def write_region(self, region):
        raise NotImplementedError()

    def read_intervals(self):
        raise NotImplementedError()

    def read_interval_data(self, interval_name):
        raise NotImplementedError()

    def write_interval(self, interval):
        raise NotImplementedError()

    def read_scenario_sets(self):
        raise NotImplementedError()

    def read_scenario_set(self, scenario_set_name):
        raise NotImplementedError()

    def read_scenario_data(self, scenario_name):
        raise NotImplementedError()

    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    def write_scenario(self, scenario):
        raise NotImplementedError()

    def read_narrative_sets(self):
        raise NotImplementedError()

    def read_narrative_set(self, narrative_set_name):
        raise NotImplementedError()

    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    def write_narrative(self, narrative):
        raise NotImplementedError()
