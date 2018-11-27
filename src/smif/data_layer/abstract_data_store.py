"""A data store holds the bulk of model setup, intermediate and output data:
- scenario variant and narrative data (including parameter defaults)
- model interventions, initial conditions and state
- conversion coefficients
- results
"""
from abc import ABCMeta, abstractmethod


class DataStore(metaclass=ABCMeta):
    """A DataStore must implement each of the abstract methods defined in this interface
    """
    # region Scenario Variant Data
    @abstractmethod
    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        """Read scenario data file

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        variable : str
        timestep : int (optional)
            If None, read data for all timesteps

        Returns
        -------
        data : ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_scenario_variant_data(self, scenario_name, variant_name, data, timestep=None):
        """Write scenario data file

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        data : ~smif.data_layer.data_array.DataArray
        timestep : int (optional)
            If None, write data for all timesteps
        """
    # endregion

    # region Narrative Data
    @abstractmethod
    def read_narrative_variant_data(self, sos_model_name, narrative_name, variant_name,
                                    parameter_name, timestep=None):
        """Read narrative data file

        Parameters
        ----------
        sos_model_name : str
        narrative_name : str
        variant_name : str
        parameter_name : str
        timestep : int (optional)
            If None, read data for all timesteps

        Returns
        -------
        ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_narrative_variant_data(self, sos_model_name, narrative_name, variant_name,
                                     data, timestep=None):
        """Read narrative data file

        Parameters
        ----------
        sos_model_name : str
        narrative_name : str
        variant_name : str
        data : ~smif.data_layer.data_array.DataArray
        timestep : int (optional)
            If None, write data for all timesteps
        """

    @abstractmethod
    def read_model_parameter_default(self, model_name, parameter_name):
        """Read default data for a sector model parameter

        Parameters
        ----------
        model_name : str
        parameter_name : str

        Returns
        -------
        ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_model_parameter_default(self, model_name, parameter_name, data):
        """Write default data for a sector model parameter

        Parameters
        ----------
        model_name : str
        parameter_name : str
        data : ~smif.data_layer.data_array.DataArray
        """
    # endregion

    # region Interventions
    @abstractmethod
    def read_interventions(self, model_name):
        """Read interventions data for `model_name`

        Returns
        -------
        dict[str, dict]
            A dict of intervention dictionaries containing intervention
            attributes keyed by intervention name
        """

    @abstractmethod
    def read_initial_conditions(self, model_name):
        """Read historical interventions for `model_name`

        Returns
        -------
        list[dict]
        """
    # endregion

    # region State
    @abstractmethod
    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        """Read list of (name, build_year) for a given model_run, timestep,
        decision

        Parameters
        ----------
        model_run_name : str
        timestep : int
        decision_iteration : int, optional

        Returns
        -------
        list[dict]
        """

    @abstractmethod
    def write_state(self, state, modelrun_name, timestep, decision_iteration=None):
        """State is a list of decisions with name and build_year.

        State is output from the DecisionManager

        Parameters
        ----------
        state : list[dict]
        model_run_name : str
        timestep : int
        decision_iteration : int, optional
        """
    # endregion

    # region Conversion coefficients
    @abstractmethod
    def read_coefficients(self, source_spec, destination_spec):
        """Reads coefficients from the store

        Coefficients are uniquely identified by their source/destination specs.
        This method and `write_coefficients` implement caching of conversion
        coefficients between dimensions.

        Parameters
        ----------
        source_spec : ~smif.metadata.spec.Spec
        destination_spec : ~smif.metadata.spec.Spec

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """

    @abstractmethod
    def write_coefficients(self, source_spec, destination_spec, data):
        """Writes coefficients to the store

        Coefficients are uniquely identified by their source/destination specs.
        This method and `read_coefficients` implement caching of conversion
        coefficients between dimensions.

        Parameters
        ----------
        source_spec : ~smif.metadata.spec.Spec
        destination_spec : ~smif.metadata.spec.Spec
        data : numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """
    # endregion

    # region Results
    @abstractmethod
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     decision_iteration=None):
        """Return results of a `model_name` in `model_run_name` for a given `output_name`

        Parameters
        ----------
        model_run_id : str
        model_name : str
        output_spec : ~smif.metadata.spec.Spec
        timestep : int, default=None
        decision_iteration : int, default=None

        Returns
        -------
        ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_results(self, data, modelrun_name, model_name, timestep=None,
                      decision_iteration=None):
        """Write results of a `model_name` in `model_run_name` for a given `output_name`

        Parameters
        ----------
        data_array : ~smif.data_layer.data_array.DataArray
        model_run_id : str
        model_name : str
        timestep : int, optional
        decision_iteration : int, optional
        """
    # endregion
