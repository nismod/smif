"""Common data interface
"""

from abc import ABCMeta, abstractmethod


class DataInterface(metaclass=ABCMeta):
    """Abstract base class to define common data interface
    """
    @abstractmethod
    def read_sos_model_runs(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model_run(self, sos_model_run_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model_run(self, sos_model_run_name, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def delete_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_models(self):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model(self, sos_model_name, sos_model):
        raise NotImplementedError()

    @abstractmethod
    def read_sector_models(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sector_model(self, sector_model):
        raise NotImplementedError()

    @abstractmethod
    def update_sector_model(self, sector_model_name, sector_model):
        raise NotImplementedError()

    @abstractmethod
    def read_region_definitions(self):
        raise NotImplementedError()

    @abstractmethod
    def read_region_definition_data(self, region_definition_name):
        raise NotImplementedError()

    @abstractmethod
    def write_region_definition(self, region_definition):
        raise NotImplementedError()

    @abstractmethod
    def update_region_definition(self, region_definition):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definitions(self):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definition_data(self, interval_definition_name):
        raise NotImplementedError()

    @abstractmethod
    def write_interval_definition(self, interval_definition):
        raise NotImplementedError()

    @abstractmethod
    def update_interval_definition(self, interval_definition):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_set(self, scenario_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_data(self, scenario_name):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_set(self, narrative_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def update_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative(self, narrative):
        raise NotImplementedError()

    @abstractmethod
    def update_narrative(self, narrative):
        raise NotImplementedError()
