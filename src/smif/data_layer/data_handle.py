"""Provide an interface to data, parameters and results

A :class:`DataHandle` is passed in to a :class:`Model` at runtime, to provide
transparent access to the relevant data and parameters for the current
:class:`ModelRun` and iteration. It gives read access to parameters and input
data (at any computed or pre-computed timestep) and write access to output data
(at the current timestep).
"""

from enum import Enum
from logging import getLogger
from types import MappingProxyType

from smif.metadata import Spec
from smif.model.scenario_model import ScenarioModel


class DataHandle(object):
    """Get/set model parameters and data
    """
    def __init__(self, store, modelrun_name, current_timestep, timesteps, model,
                 modelset_iteration=None, decision_iteration=None):
        """Create a DataHandle for a Model to access data, parameters and state, and to
        communicate results.

        Parameters
        ----------
        store : DataInterface
            Backing store for inputs, parameters, results
        modelrun_name : str
            Name of the current modelrun
        model : Model
            Model which will use this DataHandle
        modelset_iteration : int, default=None
            ID of the current ModelSet iteration
        decision_iteration : int, default=None
            ID of the current Decision iteration
        state : list, default=None
        """
        self.logger = getLogger(__name__)
        self._store = store
        self._modelrun_name = modelrun_name
        self._current_timestep = current_timestep
        self._timesteps = timesteps
        self._modelset_iteration = modelset_iteration
        self._decision_iteration = decision_iteration

        self._model_name = model.name
        self._inputs = model.inputs
        self._outputs = model.outputs
        self._dependencies = model.deps
        self._model = model

        self._parameters = {}
        for parameter in model.parameters.values():
            self._parameters[parameter.name] = parameter.default

        modelrun = self._store.read_model_run(self._modelrun_name)
        # modelrun['narratives'] is a dict of lists: {narrative_name: [variant_name, ...]}
        # e.g. { 'technology': ['high_tech_dsm'] }
        for narrative_name, variants in modelrun['narratives'].items():
            narrative = self._store.read_narrative(narrative_name)
            variable_specs = [Spec.from_dict(v) for v in narrative['provides']]
            for variant_name in variants:
                for variable in variable_specs:
                    data = self._store.read_narrative_variant_data(
                        narrative_name, variant_name, variable.name
                    )
                    self._parameters[variable.name] = data

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
            modelset_iteration=self._modelset_iteration,
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
                "'%s' not recognised as input or parameter for '%s'", key, self._model_name)

    def __setitem__(self, key, value):
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

    def get_data(self, input_name, timestep=None):
        """Get data required for model inputs

        Parameters
        ----------
        input_name : str
        timestep : RelativeTimestep or int, optional
            defaults to RelativeTimestep.CURRENT

        Returns
        -------
        data : numpy.ndarray
            Two-dimensional array with shape (len(regions), len(intervals))
        """
        if input_name not in self._inputs:
            raise KeyError(
                "'{}' not recognised as input for '{}'".format(input_name, self._model_name))

        # resolve timestep
        if timestep is None:
            timestep = self._current_timestep
        elif isinstance(timestep, RelativeTimestep):
            timestep = timestep.resolve_relative_to(self._current_timestep, self._timesteps)
        else:
            assert isinstance(timestep, int) and timestep <= self._current_timestep

        # resolve source
        source_model = self._dependencies[input_name].source_model
        source_model_name = source_model.name
        source_output_name = self._dependencies[input_name].source.name
        if self._modelset_iteration is not None:
            i = self._modelset_iteration - 1  # read from previous
        else:
            i = self._modelset_iteration

        self.logger.debug(
            "Read %s %s %s %s", source_model_name, source_output_name, timestep, i)

        spec = self._inputs[input_name]

        if isinstance(source_model, ScenarioModel):
            data = self._store.read_scenario_variant_data(
                source_model_name,  # read from a given scenario model
                source_model.scenario,  # with given scenario variant
                source_output_name,  # using output (variable) name
                timestep
            )
        else:
            data = self._store.read_results(
                self._modelrun_name,
                source_model_name,  # read from source model
                spec,  # using source model output spec
                timestep,
                i,
                self._decision_iteration
            )

        return data

    def get_base_timestep_data(self, input_name):
        """Get data from the base timestep as required for model inputs

        Parameters
        ----------
        input_name : str

        Returns
        -------
        data : numpy.ndarray
            Two-dimensional array with shape (len(regions), len(intervals))
        """
        return self.get_data(input_name, RelativeTimestep.BASE)

    def get_previous_timestep_data(self, input_name):
        """Get data from the previous timestep as required for model inputs

        Parameters
        ----------
        input_name : str

        Returns
        -------
        data : numpy.ndarray
            Two-dimensional array with shape (len(regions), len(intervals))
        """
        return self.get_data(input_name, RelativeTimestep.PREVIOUS)

    def get_region_names(self, spatial_resolution):
        return self._store.read_region_names(spatial_resolution)

    def get_interval_names(self, temporal_resolution):
        return self._store.read_interval_names(temporal_resolution)

    def get_parameter(self, parameter_name):
        """Get the value for a  parameter

        Parameters
        ----------
        parameter_name : str

        Returns
        -------
        parameter_value
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
        if output_name not in self._outputs:
            raise KeyError(
                "'{}' not recognised as output for '{}'".format(output_name, self._model_name))

        self.logger.debug(
            "Write %s %s %s %s", self._model_name, output_name, self._current_timestep,
            self._modelset_iteration)

        spec = self._outputs[output_name]

        if data.shape != spec.shape:
            raise ValueError(
                "Tried to set results with shape {}, expected {} for {}:{}".format(
                    data.shape,
                    spec.shape,
                    self._model_name,
                    output_name
                )
            )

        self._store.write_results(
            data,
            self._modelrun_name,
            self._model_name,
            spec,
            self._current_timestep,
            self._modelset_iteration,
            self._decision_iteration
        )

    def get_results(self, output_name, model_name=None, modelset_iteration=None,
                    decision_iteration=None, timestep=None):
        """Get results values for model outputs

        Parameters
        ----------
        output_name : str
            The name of an output for `model_name`
        model_name : str, default=None
            The name of a model contained in the composite model,
            or ``None`` if accessing results in the current model
        modelset_iteration : int, default=None
        decision_iteration : int, default=None
        timestep : int or RelativeTimestep, default=None

        Notes
        -----
        Access to model results is only granted to models contained
        within self._model if self._model is a  smif.model.model.CompositeModel
        """

        # resolve timestep
        if timestep is None:
            timestep = self._current_timestep
        elif isinstance(timestep, RelativeTimestep):
            timestep = timestep.resolve_relative_to(self._current_timestep, self._timesteps)
        else:
            assert isinstance(timestep, int) and timestep <= self._current_timestep

        if model_name is None:  # Accessing results in the current model
            model_name = self._model_name
            results_model = self._model
        elif model_name in self._model.models:  # Accessing a contained model
            results_model = self._model.models[model_name]
        else:
            raise KeyError(
                '{} is not contained in the current model'.format(
                    model_name
                )
            )

        try:
            spec = results_model.outputs[output_name]
        except KeyError:
            msg = "'{}' not recognised as output for '{}'"
            raise KeyError(msg.format(output_name, model_name))

        if modelset_iteration is None:
            modelset_iteration = self._modelset_iteration
        if decision_iteration is None:
            decision_iteration = self._decision_iteration

        self.logger.debug(
            "Read %s %s %s %s", model_name, output_name, timestep,
            modelset_iteration)

        return self._store.read_results(
            self._modelrun_name,
            model_name,
            spec,
            timestep,
            modelset_iteration,
            decision_iteration
        )


class RelativeTimestep(Enum):
    """Specify current, previous or base year timestep
    """
    CURRENT = 1  # current planning timestep
    PREVIOUS = 2  # previous planning timestep
    BASE = 3  # base year planning timestep
    ALL = 4  # all planning timesteps

    def resolve_relative_to(self, timestep, timesteps):
        """Resolve a relative timestep with respect to a given timestep and
        sequence of timesteps.

        Parameters
        ----------
        timestep : int
        timesteps : list of int
        """
        if self.name == 'CURRENT':
            return timestep

        if self.name == 'PREVIOUS':
            try:
                return element_before(timestep, timesteps)
            except ValueError:
                raise TimestepResolutionError(
                    "{} has no previous timestep in {}".format(timestep, timesteps))

        if self.name == 'BASE':
            return timesteps[0]

        if self.name == 'ALL':
            return None


class TimestepResolutionError(Exception):
    """Raise when timestep cannot be resolved
    """
    pass


class DataError(Exception):
    """Raise on attempts at invalid data access (get/set)
    """
    pass


def element_before(element, list_):
    """Return the element before a given element in a list, or None if the
    given element is first or not in the list.
    """
    if element not in list_ or element == list_[0]:
        raise ValueError("No element before {} in {}".format(element, list_))
    else:
        index = list_.index(element)
        return list_[index - 1]
