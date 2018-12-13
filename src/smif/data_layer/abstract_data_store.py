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
    # region DataArray
    @abstractmethod
    def read_scenario_variant_data(self, key, spec, timestep=None):
        """Read data array

        Parameters
        ----------
        key : str
        spec : ~smif.metadata.spec.Spec
        timestep : int (optional)
            If None, read data for all timesteps

        Returns
        -------
        data_array : ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_scenario_variant_data(self, key, data_array, timestep=None):
        """Write data array

        Parameters
        ----------
        key : str
        data_array : ~smif.data_layer.data_array.DataArray
        timestep : int (optional)
            If None, write data for all timesteps
        """

    @abstractmethod
    def read_narrative_variant_data(self, key, spec, timestep=None):
        """Read data array

        Parameters
        ----------
        key : str
        spec : ~smif.metadata.spec.Spec
        timestep : int (optional)
            If None, read data for all timesteps

        Returns
        -------
        data_array : ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_narrative_variant_data(self, key, data_array, timestep=None):
        """Write data array

        Parameters
        ----------
        key : str
        data_array : ~smif.data_layer.data_array.DataArray
        timestep : int (optional)
            If None, write data for all timesteps
        """

    @abstractmethod
    def read_model_parameter_default(self, key, spec):
        """Read data array

        Parameters
        ----------
        key : str
        spec : ~smif.metadata.spec.Spec

        Returns
        -------
        data_array : ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def write_model_parameter_default(self, key, data_array):
        """Read data array

        Parameters
        ----------
        key : str
        data_array : ~smif.data_layer.data_array.DataArray

        Returns
        -------
        data_array : ~smif.data_layer.data_array.DataArray
        """
    # endregion

    # region Interventions
    @abstractmethod
    def read_interventions(self, key):
        """Read interventions data for `key`

        Parameters
        ----------
        key : str

        Returns
        -------
        dict[str, dict]
            A dict of intervention dictionaries containing intervention
            attributes keyed by intervention name
        """

    @abstractmethod
    def write_interventions(self, key, interventions):
        """Write interventions data for `key`

        Parameters
        ----------
        key : str
        interventions : dict[str, dict]
        """

    @abstractmethod
    def read_initial_conditions(self, key):
        """Read historical interventions for `key`

        Parameters
        ----------
        key : str

        Returns
        -------
        list[dict]
        """

    @abstractmethod
    def write_initial_conditions(self, key, initial_conditions):
        """Write historical interventions for `key`

        Parameters
        ----------
        key : str
        initial_conditions: list[dict]
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
        """Return results of a model from a model_run for a given output at a timestep and
        decision iteration

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

    @abstractmethod
    def available_results(self, modelrun_name):
        """List available results from a model run

        Returns
        -------
        list[tuple]
             Each tuple is (timestep, decision_iteration, model_name, output_name)
        """
    # endregion
