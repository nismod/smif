"""A data store holds the bulk of model setup, intermediate and output data:
- scenario variant and narrative data (including parameter defaults)
- model interventions, initial conditions and state
- conversion coefficients
- results
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from smif.data_layer.data_array import DataArray
from smif.exception import SmifDataMismatchError, SmifDataNotFoundError
from smif.metadata import Spec


class DataStore(metaclass=ABCMeta):
    """A DataStore must implement each of the abstract methods defined in this interface
    """
    # region DataArray
    @abstractmethod
    def read_scenario_variant_data(
            self, key, spec, timestep=None, timesteps=None) -> DataArray:
        """Read scenario variant data array.

        If a single timestep is specified, the spec MAY include 'timestep' as a dimension,
        which should match the timestep specified.

        If multiple timesteps are specified, the spec MUST include 'timestep' as a dimension,
        which should match the timesteps specified.

        If timestep and timesteps are None, read all available timesteps. Whether or not the
        spec includes 'timestep' as a dimension, the returned DataArray will include a
        'timestep' dimension with all available timesteps included.

        Parameters
        ----------
        key : str
        spec : ~smif.metadata.spec.Spec
        timestep : int (optional)
            If set, read data for single timestep
        timesteps : list[int] (optional)
            If set, read data for specified timesteps

        Returns
        -------
        data_array : ~smif.data_layer.data_array.DataArray
        """

    @abstractmethod
    def scenario_variant_data_exists(self, key) -> bool:
        """Test if scenario variant data exists

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """

    @abstractmethod
    def write_scenario_variant_data(self, key, data_array):
        """Write data array

        Parameters
        ----------
        key : str
        data_array : ~smif.data_layer.data_array.DataArray
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
    def write_narrative_variant_data(self, key, data_array):
        """Write data array

        Parameters
        ----------
        key : str
        data_array : ~smif.data_layer.data_array.DataArray
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
    def read_initial_conditions(self, key) -> List[Dict]:
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
    def read_state(self, modelrun_name, timestep, decision_iteration=None) -> List[Dict]:
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
    def write_state(self, state: List[Dict],
                    modelrun_name: str,
                    timestep: int,
                    decision_iteration=None):
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
    def read_coefficients(self, source_dim, destination_dim):
        """Reads coefficients from the store

        Coefficients are uniquely identified by their source/destination dimensions.
        This method and `write_coefficients` implement caching of conversion
        coefficients between a single pair of dimensions.

        Parameters
        ----------
        source_dim : str
            dimension name
        destination_dim : str
            dimension name

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """

    @abstractmethod
    def write_coefficients(self, source_dim, destination_dim, data):
        """Writes coefficients to the store

        Coefficients are uniquely identified by their source/destination dimensions.
        This method and `read_coefficients` implement caching of conversion
        coefficients between a single pair of dimensions.

        Parameters
        ----------
        source_dim : str
            dimension name
        destination_dim : str
            dimension name
        data : numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """
    # endregion

    # region Results
    @abstractmethod
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     decision_iteration=None) -> DataArray:
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
    def delete_results(self, model_run_name, model_name, output_name, timestep=None,
                       decision_iteration=None):
        """Delete results for a single timestep/iteration of a model output in a model run

        Parameters
        ----------
        model_run_name : str
        model_name : str
        output_name : str
        timestep : int, default=None
        decision_iteration : int, default=None
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

    @classmethod
    def filter_on_timesteps(cls, dataframe, spec, path, timestep=None, timesteps=None):
        """Filter dataframe by timestep

        The 'timestep' dimension is treated as follows:

        If a single timestep is specified, the spec MAY include 'timestep' as a dimension. If
        so, the returned DataArray's spec will match the timestep requested. Otherwise, the
        DataArray will not include timestep as a dimension.

        If multiple timesteps are specified, the returned DataArray's spec will include a
        'timestep' dimension to match the timesteps requested.

        If timestep and timesteps are None, and the stored data has a timestep column, read all
        available timesteps. The returned DataArray's spec 'timestep' dimension will match the
        timesteps requested. If the stored data does not have a timestep column, ignore and
        pass through unchanged.
        """
        if timestep is not None:
            dataframe = cls._check_timestep_column_exists(dataframe, spec, path)
            dataframe = dataframe[dataframe.timestep == timestep]
            if 'timestep' in spec.dims:
                spec = cls._set_spec_timesteps(spec, [timestep])
            else:
                dataframe = dataframe.drop('timestep', axis=1)
        elif timesteps is not None:
            dataframe = cls._check_timestep_column_exists(dataframe, spec, path)
            dataframe = dataframe[dataframe.timestep.isin(timesteps)]
            spec = cls._set_spec_timesteps(spec, timesteps)
        elif timestep is None and timesteps is None:
            try:
                dataframe = cls._check_timestep_column_exists(dataframe, spec, path)
                spec = cls._set_spec_timesteps(spec, sorted(list(dataframe.timestep.unique())))
            except SmifDataMismatchError:
                pass

        if dataframe.empty:
            raise SmifDataNotFoundError(
                "Data for '{}' not found for timestep {}".format(spec.name, timestep))

        return dataframe, spec

    @staticmethod
    def dataframe_to_data_array(dataframe, spec, path):
        if spec.dims:
            data_array = DataArray.from_df(spec, dataframe)
        else:
            # zero-dimensional case (scalar)
            data = dataframe[spec.name]
            if data.shape != (1,):
                msg = "Data for '{}' should contain a single value, instead got {} while " + \
                        "reading from {}"
                raise SmifDataMismatchError(msg.format(spec.name, len(data), path))
            data_array = DataArray(spec, data.iloc[0])

        return data_array

    @staticmethod
    def _check_timestep_column_exists(dataframe, spec, path):
        if 'timestep' not in dataframe.columns:
            if 'timestep' in dataframe.index.names:
                dataframe = dataframe.reset_index()
            else:
                msg = "Data for '{name}' expected a column called 'timestep', instead " + \
                        "got data columns {data_columns} and index names {index_names} " + \
                        "while reading from {path}"
                raise SmifDataMismatchError(msg.format(
                    data_columns=dataframe.columns.values.tolist(),
                    index_names=dataframe.index.names,
                    name=spec.name,
                    path=path))
        return dataframe

    @staticmethod
    def _set_spec_timesteps(spec, timesteps):
        spec_config = spec.as_dict()
        if 'timestep' not in spec_config['dims']:
            spec_config['dims'] = ['timestep'] + spec_config['dims']
        spec_config['coords']['timestep'] = timesteps
        return Spec.from_dict(spec_config)
