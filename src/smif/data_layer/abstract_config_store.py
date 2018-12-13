"""A config store holds the configuration data for running system-of-systems models with smif:
- model runs
- system-of-systems models
- model definitions
- strategies
- scenarios and scenario variants
- narratives
"""
from abc import ABCMeta, abstractmethod


class ConfigStore(metaclass=ABCMeta):
    """A ConfigStore must implement each of the abstract methods defined in this interface
    """
    # region Model runs
    @abstractmethod
    def read_model_runs(self):
        """Read all system-of-system model runs

        Returns
        -------
        list[~smif.controller.modelrun.ModelRun]
        """

    @abstractmethod
    def read_model_run(self, model_run_name):
        """Read a system-of-system model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        ~smif.controller.modelrun.ModelRun
        """

    @abstractmethod
    def write_model_run(self, model_run):
        """Write system-of-system model run

        Parameters
        ----------
        model_run : ~smif.controller.modelrun.ModelRun
        """

    @abstractmethod
    def update_model_run(self, model_run_name, model_run):
        """Update system-of-system model run

        Parameters
        ----------
        model_run_name : str
        model_run : ~smif.controller.modelrun.ModelRun
        """

    @abstractmethod
    def delete_model_run(self, model_run_name):
        """Delete a system-of-system model run

        Parameters
        ----------
        model_run_name : str
        """
    # endregion

    # region System-of-systems models
    @abstractmethod
    def read_sos_models(self):
        """Read all system-of-system models

        Returns
        -------
        list[~smif.model.sos_model.SosModel]
        """

    @abstractmethod
    def read_sos_model(self, sos_model_name):
        """Read a specific system-of-system model

        Parameters
        ----------
        sos_model_name : str

        Returns
        -------
        ~smif.model.sos_model.SosModel
        """

    @abstractmethod
    def write_sos_model(self, sos_model):
        """Write system-of-system model

        Parameters
        ----------
        sos_model : ~smif.model.sos_model.SosModel
        """

    @abstractmethod
    def update_sos_model(self, sos_model_name, sos_model):
        """Update system-of-system model

        Parameters
        ----------
        sos_model_name : str
        sos_model : ~smif.model.sos_model.SosModel
        """

    @abstractmethod
    def delete_sos_model(self, sos_model_name):
        """Delete a system-of-system model

        Parameters
        ----------
        sos_model_name : str
        """
    # endregion

    # region Models
    @abstractmethod
    def read_models(self):
        """Read all models

        Returns
        -------
        list[~smif.model.Model]
        """

    @abstractmethod
    def read_model(self, model_name):
        """Read a model

        Parameters
        ----------
        model_name : str

        Returns
        -------
        ~smif.model.Model
        """

    @abstractmethod
    def write_model(self, model):
        """Write a model

        Parameters
        ----------
        model : ~smif.model.Model
        """

    @abstractmethod
    def update_model(self, model_name, model):
        """Update a model

        Parameters
        ----------
        model_name : str
        model : ~smif.model.Model
        """

    @abstractmethod
    def delete_model(self, model_name):
        """Delete a model

        Parameters
        ----------
        model_name : str
        """
    # endregion

    # region Scenarios
    @abstractmethod
    def read_scenarios(self):
        """Read scenarios

        Returns
        -------
        list[~smif.model.ScenarioModel]
        """

    @abstractmethod
    def read_scenario(self, scenario_name):
        """Read a scenario

        Parameters
        ----------
        scenario_name : str

        Returns
        -------
        ~smif.model.ScenarioModel
        """

    @abstractmethod
    def write_scenario(self, scenario):
        """Write scenario

        Parameters
        ----------
        scenario : ~smif.model.ScenarioModel
        """

    @abstractmethod
    def update_scenario(self, scenario_name, scenario):
        """Update scenario

        Parameters
        ----------
        scenario_name : str
        scenario : ~smif.model.ScenarioModel
        """

    @abstractmethod
    def delete_scenario(self, scenario_name):
        """Delete scenario from project configuration

        Parameters
        ----------
        scenario_name : str
        """
    # endregion

    # region Scenario Variants
    @abstractmethod
    def read_scenario_variants(self, scenario_name):
        """Read variants of a given scenario

        Parameters
        ----------
        scenario_name : str

        Returns
        -------
        list[dict]
        """

    @abstractmethod
    def read_scenario_variant(self, scenario_name, variant_name):
        """Read a scenario variant

        Parameters
        ----------
        scenario_name : str
        variant_name : str

        Returns
        -------
        dict
        """

    @abstractmethod
    def write_scenario_variant(self, scenario_name, variant):
        """Write scenario to project configuration

        Parameters
        ----------
        scenario_name : str
        variant : dict
        """

    @abstractmethod
    def update_scenario_variant(self, scenario_name, variant_name, variant):
        """Update scenario to project configuration

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        variant : dict
        """

    @abstractmethod
    def delete_scenario_variant(self, scenario_name, variant_name):
        """Delete scenario from project configuration

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        """
    # endregion

    # region Narratives
    @abstractmethod
    def read_narrative(self, sos_model_name, narrative_name):
        """Read narrative from sos_model

        Parameters
        ----------
        sos_model_name : str
        narrative_name : str
        """
    # endregion

    # region Strategies
    @abstractmethod
    def read_strategies(self, modelrun_name):
        """Read strategies for a given model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        list[dict]
        """

    @abstractmethod
    def write_strategies(self, modelrun_name, strategies):
        """Write strategies for a given model_run

        Parameters
        ----------
        model_run_name : str
        strategies : list[dict]
        """
    # endregion
