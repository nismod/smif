"""Provide an interface to data, parameters and results

A :class:`DataHandle` is passed in to a :class:`Model` at runtime, to provide
transparent access to the relevant data and parameters for the current
:class:`ModelRun` and iteration. It gives read access to parameters and input
data (at any computed or pre-computed timestep) and write access to output data
(at the current timestep).
"""
from logging import getLogger
from types import MappingProxyType

from smif.data_layer.data_array import DataArray
from smif.exception import SmifDataError
from smif.metadata import RelativeTimestep


class DataHandle(object):
    """Get/set model parameters and data
    """
    def __init__(self, store, modelrun_name, current_timestep, timesteps, model,
                 decision_iteration=None):
        """Create a DataHandle for a Model to access data, parameters and state, and to
        communicate results.

        Parameters
        ----------
        store : Store
            Backing store for inputs, parameters, results
        modelrun_name : str
            Name of the current modelrun
        model : Model
            Model which will use this DataHandle
        decision_iteration : int, default=None
            ID of the current Decision iteration
        state : list, default=None
        """
        self.logger = getLogger(__name__)
        self._store = store
        self._modelrun_name = modelrun_name
        self._current_timestep = current_timestep
        self._timesteps = timesteps
        self._decision_iteration = decision_iteration

        self._model_name = model.name
        self._inputs = model.inputs
        self._outputs = model.outputs
        self._model = model

        modelrun = self._store.read_model_run(self._modelrun_name)
        sos_model = self._store.read_sos_model(modelrun['sos_model'])

        self._scenario_dependencies = {}
        self._model_dependencies = {}
        scenario_variants = modelrun['scenarios']
        self._load_dependencies(sos_model, scenario_variants)
        self.logger.debug(
            "Create with %s model, %s scenario dependencies",
            len(self._scenario_dependencies),
            len(self._model_dependencies))

        self._parameters = {}
        self._load_parameters(sos_model, modelrun['narratives'])

    def _load_dependencies(self, sos_model, scenario_variants):
        """Load Model dependencies as a dict with {input_name: list[Dependency]}
        """
        for dep in sos_model['model_dependencies']:
            if dep['sink'] == self._model_name:
                input_name = dep['sink_input']
                self._model_dependencies[input_name] = {
                    'source_model_name': dep['source'],
                    'source_output_name': dep['source_output'],
                    'type': 'model'
                }

        for dep in sos_model['scenario_dependencies']:
            if dep['sink'] == self._model_name:
                input_name = dep['sink_input']
                self._scenario_dependencies[input_name] = {
                    'source_model_name': dep['source'],
                    'source_output_name': dep['source_output'],
                    'type': 'scenario',
                    'variant': scenario_variants[dep['source']]
                }

    def _load_parameters(self, sos_model, concrete_narratives):
        """Load parameter values for model run

        Parameters for each of the contained sector models are loaded
        into memory as a data_handle is initialised.

        Firstly, default values for the parameters are loaded from the parameter
        specs contained within each of the sector models

        Then, the data from the list of narrative variants linked to the current
        model run are loaded into the parameters contained within the

        Arguments
        ---------
        sos_model : dict
            A configuration dictionary of a system-of-systems model
        concrete_narratives: dict of list
            Links narrative names to a list of variants to furnish parameters
            with values {narrative_name: [variant_name, ...]}
        """
        # Populate the parameters with their default values
        for parameter in self._model.parameters.values():
            self._parameters[parameter.name] = \
                self._store.read_model_parameter_default(self._model.name, parameter.name)

        # Load in the concrete narrative and selected variants from the model run
        for narrative_name, variant_names in concrete_narratives.items():
            # Load the narrative
            try:
                narrative = [x for x in sos_model['narratives']
                             if x['name'] == narrative_name][0]
            except IndexError:
                msg = "Couldn't find a match for {} in {}"
                raise IndexError(msg.format(narrative_name, sos_model['name']))
            self.logger.debug("Loaded narrative: %s", narrative)
            self.logger.debug("Considering variants: %s", variant_names)

            # Read parameter data from each variant, later variants overriding
            # previous parameter values
            for variant_name in variant_names:
                try:
                    parameter_list = narrative['provides'][self._model.name]
                except KeyError:
                    parameter_list = []

                for parameter in parameter_list:
                    da = self._store.read_narrative_variant_data(
                        sos_model['name'],
                        narrative_name, variant_name, parameter
                    )
                    self._parameters[parameter].update(da)

    def derive_for(self, model):
        """Derive a new DataHandle configured for the given Model

        Parameters
        ----------
        model : Model
            Model which will use this DataHandle
        """
        return DataHandle(
            store=self._store,
            modelrun_name=self._modelrun_name,
            current_timestep=self._current_timestep,
            timesteps=list(self.timesteps),
            model=model,
            decision_iteration=self._decision_iteration
        )

    def __getitem__(self, key):
        if key in self._parameters:
            return self.get_parameter(key)
        elif key in self._inputs:
            return self.get_data(key)
        elif key in self._outputs:
            return self.get_results(key)
        else:
            raise KeyError(
                "'%s' not recognised as input, output or parameter for '%s'" %
                (key, self._model_name)
                )

    def __setitem__(self, key, value):
        if hasattr(value, 'as_ndarray'):
            raise TypeError("Pass in a numpy array")
        self.set_results(key, value)

    @property
    def current_timestep(self):
        """Current timestep
        """
        return self._current_timestep

    @property
    def previous_timestep(self):
        """Previous timestep
        """
        return RelativeTimestep.PREVIOUS.resolve_relative_to(
            self._current_timestep,
            self._timesteps
        )

    @property
    def base_timestep(self):
        """Base timestep
        """
        return RelativeTimestep.BASE.resolve_relative_to(
            self._current_timestep,
            self._timesteps
        )

    @property
    def timesteps(self):
        """All timesteps (as tuple)
        """
        return tuple(self._timesteps)

    @property
    def decision_iteration(self):
        return self._decision_iteration

    def get_state(self):
        """The current state of the model

        If the DataHandle instance has a timestep, then state is
        established from the state file.

        Returns
        -------
        list of tuple
            A list of (intervention name, build_year) installed in the current timestep

        Raises
        ------
        ValueError
            If self._current_timestep is None an error is raised.
        """
        if self._current_timestep is None:
            raise ValueError("You must pass a timestep value to get state")
        else:

            sos_state = self._store.read_state(
                self._modelrun_name,
                self._current_timestep,
                self._decision_iteration
            )

        return sos_state

    def get_current_interventions(self):
        """Get the interventions that exist in the current state

        Returns
        -------
        dict of dicts
            A dict of intervention dicts with build_year attribute keyed by name
        """
        state = self.get_state()

        current_interventions = {}
        all_interventions = self._store.read_interventions(self._model_name)

        for decision in state:
            name = decision['name']
            build_year = decision['build_year']
            try:
                serialised = all_interventions[name]
                serialised['build_year'] = build_year
                current_interventions[name] = serialised
            except KeyError:
                # ignore if intervention is not in current set
                pass

        msg = "State matched with %s interventions"
        self.logger.info(msg, len(current_interventions))

        return current_interventions

    def get_data(self, input_name: str, timestep=None) -> DataArray:
        """Get data required for model inputs

        Parameters
        ----------
        input_name : str
        timestep : RelativeTimestep or int, optional
            defaults to RelativeTimestep.CURRENT

        Returns
        -------
        smif.data_layer.data_array.DataArray
            Contains data annotated with the metadata and provides utility methods
            to access the data in different ways

        Raises
        ------
        SmifDataError
            If any data reading error occurs below this method, the error is
            handled and reraised within the context of the current call
        """
        if input_name not in self._inputs:
            raise KeyError(
                "'{}' not recognised as input for '{}'".format(input_name, self._model_name))

        timestep = self._resolve_timestep(timestep)

        dep = self._resolve_source(input_name)

        self.logger.debug(
            "Read %s %s %s", dep['source_model_name'], dep['source_output_name'],
            timestep)

        if dep['type'] == 'scenario':
            data = self._get_scenario(dep, timestep)
        else:
            spec = self._inputs[input_name]
            data = self._get_result(dep, timestep, spec)

        return data

    def _resolve_timestep(self, timestep):
        """Resolves a relative timestep to an absolute timestep

        Arguments
        ---------
        timestep : RelativeTimestep or int

        Returns
        -------
        int
        """
        if self._current_timestep is None:
            if timestep is None:
                raise ValueError("You must provide a timestep to obtain data")
            elif hasattr(timestep, "resolve_relative_to"):
                timestep = timestep.resolve_relative_to(self._timesteps[0], self._timesteps)
            else:
                assert isinstance(timestep, int) and timestep in self._timesteps
        else:
            if timestep is None:
                timestep = self._current_timestep
            elif hasattr(timestep, "resolve_relative_to"):
                timestep = timestep.resolve_relative_to(self._current_timestep,
                                                        self._timesteps)
            else:
                assert isinstance(timestep, int) and timestep <= self._current_timestep
        return timestep

    def _get_result(self, dep, timestep, spec):
        """Retrieves a model result for a dependency
        """
        try:
            data = self._store.read_results(
                self._modelrun_name,
                dep['source_model_name'],  # read from source model
                spec,  # using source model output spec
                timestep,
                self._decision_iteration
            )
        except SmifDataError as ex:
            msg = "Could not read data for output '{}' from '{}' in {}, iteration {}"
            raise SmifDataError(msg.format(
                spec.name,
                dep['source_model_name'],
                timestep,
                self._decision_iteration
            )) from ex
        return data

    def _get_scenario(self, dep, timestep):
        """Retrieves data from a scenario

        Arguments
        ---------
        dep : dict
            A scenario dependency
        timestep : int

        Returns
        -------
        DataArray
        """
        try:
            data = self._store.read_scenario_variant_data(
                dep['source_model_name'],  # read from a given scenario model
                dep['variant'],  # with given scenario variant
                dep['source_output_name'],  # using output (variable) name
                timestep
            )
        except SmifDataError as ex:
            msg = "Could not read data for output '{}' from '{}.{}' in {}"
            raise SmifDataError(msg.format(
                dep['source_output_name'],
                dep['source_model_name'],
                dep['variant'],
                timestep
            )) from ex
        return data

    def _resolve_source(self, input_name) -> dict:
        """Find best dependency to provide input data

        Returns
        -------
        dep : dict
            A scenario or model dependency dictionary
        """
        try:
            scenario_dep = self._scenario_dependencies[input_name]
        except KeyError:
            scenario_dep = None
        try:
            model_dep = self._model_dependencies[input_name]
        except KeyError:
            model_dep = None

        if scenario_dep is not None and model_dep is not None:
            # if multiple dependencies, use scenario for timestep 0, model for
            # subsequent timesteps
            if self._current_timestep == self._timesteps[0]:
                dep = scenario_dep
            else:
                dep = model_dep
        elif scenario_dep is not None:
            # else assume single dependency per input
            dep = scenario_dep
        elif model_dep is not None:
            dep = model_dep
        else:
            raise SmifDataError("Dependency not defined for input '{}' in model '{}'".format(
                input_name, self._model_name
            ))
        return dep

    def get_base_timestep_data(self, input_name):
        """Get data from the base timestep as required for model inputs

        Parameters
        ----------
        input_name : str

        Returns
        -------
        smif.data_layer.data_array.DataArray
        """
        return self.get_data(input_name, RelativeTimestep.BASE)

    def get_previous_timestep_data(self, input_name):
        """Get data from the previous timestep as required for model inputs

        Parameters
        ----------
        input_name : str

        Returns
        -------
        smif.data_layer.data_array.DataArray
        """
        return self.get_data(input_name, RelativeTimestep.PREVIOUS)

    def get_parameter(self, parameter_name):
        """Get the value for a  parameter

        Parameters
        ----------
        parameter_name : str

        Returns
        -------
        smif.data_layer.data_array.DataArray
            Contains data annotated with the metadata and provides utility methods
            to access the data in different ways
        """
        if parameter_name not in self._parameters:
            raise KeyError(
                "'{}' not recognised as parameter for '{}'".format(
                    parameter_name, self._model_name))

        return self._parameters[parameter_name]

    def get_parameters(self):
        """Get all parameter values

        Returns
        -------
        parameters : MappingProxyType
            Read-only view of parameters (like a read-only dict)
        """
        return MappingProxyType(self._parameters)

    def set_results(self, output_name, data):
        """Set results values for model outputs

        Parameters
        ----------
        output_name : str
        data : numpy.ndarray
        """
        if hasattr(data, 'as_ndarray'):
            raise TypeError("Pass in a numpy array")

        if output_name not in self._outputs:
            raise KeyError(
                "'{}' not recognised as output for '{}'".format(output_name, self._model_name))

        self.logger.debug(
            "Write %s %s %s", self._model_name, output_name, self._current_timestep)

        spec = self._outputs[output_name]

        da = DataArray(spec, data)

        self._store.write_results(
            da,
            self._modelrun_name,
            self._model_name,
            self._current_timestep,
            self._decision_iteration
        )

    def get_results(self, output_name, decision_iteration=None,
                    timestep=None):
        """Get results values for model outputs

        Parameters
        ----------
        output_name : str
            The name of an output for `model_name`
        decision_iteration : int, default=None
        timestep : int or RelativeTimestep, default=None

        Returns
        -------
        smif.data_layer.data_array.DataArray
            Contains data annotated with the metadata and provides utility methods
            to access the data in different ways

        Notes
        -----
        Access to model results is only granted to models contained
        within self._model if self._model is a smif.model.model.CompositeModel
        """
        model_name = self._model.name

        # resolve timestep
        if timestep is None:
            timestep = self._current_timestep
        elif isinstance(timestep, RelativeTimestep):
            timestep = timestep.resolve_relative_to(self._current_timestep, self._timesteps)
        else:
            assert isinstance(timestep, int) and timestep <= self._current_timestep

        # find output spec
        try:
            spec = self._model.outputs[output_name]
        except KeyError:
            msg = "'{}' not recognised as output for '{}'"
            raise KeyError(msg.format(output_name, model_name))

        if decision_iteration is None:
            decision_iteration = self._decision_iteration

        self.logger.debug(
            "Read %s %s %s", model_name, output_name, timestep)

        return self._store.read_results(
            self._modelrun_name,
            model_name,
            spec,
            timestep,
            decision_iteration
        )

    def read_coefficients(self, source_spec, destination_spec):
        data = self._store.read_coefficients(source_spec, destination_spec)
        return data

    def write_coefficients(self, source_spec, destination_spec, data):
        data = self._store.write_coefficients(source_spec, destination_spec, data)
        return data


class ResultsHandle(object):
    """Results access for decision modules
    """
    def __init__(self, store, modelrun_name, sos_model, current_timestep, timesteps=None,
                 decision_iteration=None):
        self._store = store
        self._modelrun_name = modelrun_name
        self._sos_model = sos_model

        self._current_timestep = current_timestep
        self._timesteps = timesteps
        self._decision_iteration = decision_iteration

    @property
    def base_timestep(self):
        return self._timesteps[0]

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def previous_timestep(self):
        rel = RelativeTimestep.PREVIOUS
        return rel.resolve_relative_to(self._current_timestep, self._timesteps)

    @property
    def decision_iteration(self):
        return self._decision_iteration

    def get_results(self, model_name, output_name, timestep, decision_iteration):
        """Access model results

        Parameters
        ----------
        model_name : str
        output_name : str
        timestep : int
        decision_iteration : int

        Returns
        -------
        smif.data_layer.data_array.DataArray
            Contains data annotated with the metadata and provides utility methods
            to access the data in different ways
        """
        # resolve timestep
        if hasattr(timestep, 'resolve_relative_to'):
            timestep = timestep.resolve_relative_to(self._current_timestep, self._timesteps)
        else:
            assert isinstance(timestep, int) and timestep <= self._current_timestep

        if model_name in [model.name for model in self._sos_model.models]:
            results_model = self._sos_model.get_model(model_name)
        else:
            msg = "Model '{}' is not contained in SosModel '{}'. Found {}."
            raise KeyError(msg.format(model_name, self._sos_model.name,
                                      self._sos_model.models)
                           )

        try:
            spec = results_model.outputs[output_name]
        except KeyError:
            msg = "'{}' not recognised as output for '{}'"
            raise KeyError(msg.format(output_name, model_name))

        results = self._store.read_results(self._modelrun_name,
                                           model_name,
                                           spec,
                                           timestep,
                                           decision_iteration)

        return results
