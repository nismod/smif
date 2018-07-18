"""The decision module handles the three planning levels

Currently, only pre-specified planning is implemented.

The choices made in the three planning levels influence the set of interventions
and assets available within a model run.

The interventions available in a model run are stored in the
:class:`~smif.intervention.InterventionRegister`.

When pre-specified planning are declared, each of the corresponding
interventions in the InterventionRegister are moved to the BuiltInterventionRegister.

Once pre-specified planning is instantiated, the action space for rule-based and
optimisation approaches can be generated from the remaining Interventions in the
InterventionRegister.

"""
__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"

from abc import ABCMeta, abstractmethod
from logging import getLogger

from smif.intervention import InterventionRegister


class DecisionManager(object):
    """A DecisionManager is initialised with one or more model run strategies that refer to
    DecisionModules such as pre-specified planning, a rule-based models or multi-objective
    optimisation. These implementations influence the combination and ordering of decision
    iterations and model timesteps that need to be performed by the model runner.

    The DecisionManager presents a simple decision loop interface to the model runner, in the
    form of a generator which allows the model runner to iterate over the collection of
    independent simulations required at each step.

    (Not yet implemented.) A DecisionManager collates the output of the decision algorithms and
    writes the post-decision state through a DataHandle. This allows Models to access a given
    decision state (identified uniquely by timestep and decision iteration id).

    (Not yet implemented.) A DecisionManager may also pass a DataHandle down to a
    DecisionModule, allowing the DecisionModule to access model results from previous timesteps
    and decision iterations when making decisions.

    Arguments
    ---------
    timesteps: list
    strategies: list
    """

    def __init__(self, timesteps, strategies, interventions):

        self.logger = getLogger(__name__)

        self._timesteps = timesteps
        self._strategies = strategies
        self._decision_modules = []

        self._set_up_decision_modules()

        self.register = InterventionRegister()
        for intervention in interventions:
            self.register.register(intervention)

    def _set_up_decision_modules(self):

        self.logger.info("%s strategies found", len(self._strategies))
        interventions = []

        for index, strategy in enumerate(self._strategies):
            if strategy['strategy'] == 'pre-specified-planning':

                msg = "Adding %s interventions to pre-specified-planning %s"
                self.logger.info(msg, len(strategy['interventions']), index)

                interventions.extend(strategy['interventions'])

            else:
                msg = "Only pre-specified planning strategies are implemented"
                raise NotImplementedError(msg)

        if interventions:
            self._decision_modules.append(
                PreSpecified(self._timesteps, interventions)
                )

    def decision_loop(self):
        """Generate bundles of simulation steps to run.

        Each iteration returns a dict: {decision_iteration (int) => list of timesteps (int)}

        With only pre-specified planning, there is a single step in the loop, with a single
        decision iteration with timesteps covering the entire model horizon.

        With a rule based approach, there might be many steps in the loop, each with a single
        decision iteration and single timestep, moving on once some threshold is satisfied.

        With a genetic algorithm, there might be a configurable number of steps in the loop,
        each with multiple decision iterations (one for each member of the algorithm's
        population) and timesteps covering the entire model horizon.

        Implicitly, if the bundle returned in an iteration contains multiple decision
        iterations, they can be performed in parallel. If each decision iteration contains
        multiple timesteps, they can also be parallelised, so long as there are no temporal
        dependencies.
        """
        if len(self._decision_modules) > 0:
            for module in self._decision_modules:
                yield module._get_next_decision_iteration()
        else:
            yield {0: [x for x in self._timesteps]}

    def get_decision(self, timestep, iteration):
        """Return all interventions built in the given timestep

        for the given decision
        iteration.

        Arguments
        ---------
        timestep : int
            A timestep (planning year)
        iteration : int
            A decision iteration

        """
        decisions = []
        for module in self._decision_modules:
            decisions.extend(module.get_decision(timestep, iteration))
        self.logger.debug(
            "Retrieved %s decisions from %s",
            len(decisions), str(self._decision_modules))
        return decisions


class DecisionModule(metaclass=ABCMeta):
    """Abstract class which provides the interface to decision mechanisms.

    These mechanisms including Pre-Specified Planning, a Rule-based Approach and
    Multi-objective Optimisation.

    This class provides two main public methods, ``__next__`` which is normally
    called implicitly as a call to the class as an iterator, and ``get_decision()``
    which takes as arguments a smif.model.Model object, and ``timestep`` and
    ``decision_iteration`` integers. The first of these returns a dict of
    decision_iterations and timesteps over which a SosModel should be iterated.
    The latter provides a means to furnish the structure of contained Model
    objects through a list of historical and recent interventions.

    Arguments
    ---------
    timesteps : list
        A list of planning timesteps

    """
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def __next__(self):
        return self._get_next_decision_iteration()

    @abstractmethod
    def _get_next_decision_iteration(self):
        """Implement to return the next decision iteration

        Within a list of decision-iteration/timestep pairs, the assumption is
        that all decision iterations can be run in parallel
        (otherwise only one will be returned) and within a decision interation,
        all timesteps may be run in parallel as long as there are no
        inter-timestep state transitions required (e.g. reservoir level)

        Returns
        -------
        dict
            Yields a dictionary keyed by decision iteration (int) whose values contain
            a list of timesteps
        """
        raise NotImplementedError

    @abstractmethod
    def get_decision(self, timestep, iteration):
        """Return decisions for a given timestep and decision iteration
        """
        raise NotImplementedError

    def _get_previous_state(self, timestep, decision_iteration):
        """Gets state of the previous `timestep` for `decision_iteration`

        Arguments
        ---------
        timestep : int
        decision_iteration : int

        Returns
        -------
        numpy.ndarray
        """
        return self.get_decision(timestep.PREVIOUS, decision_iteration)

    def _set_post_decision_state(self, timestep, decision_iteration, decision):
        """Sets the post-decision state

        Arguments
        ---------
        timestep : int
        decision_iteration : int
        decision : numpy.ndarray

        Notes
        -----
        `decision` should contain only the newly decided interventions
        """
        state = self._get_previous_state(timestep, decision_iteration)
        post_decision_state = state.bitwise_or(decision)
        self.register.set_state(timestep, decision_iteration, post_decision_state)

    @abstractmethod
    def _set_state(self, timestep, decision_iteration):
        """Implement to set the current state of a Model

        Arguments
        ---------
        timestep : int
        decision_iteration : int

        Notes
        -----
        1. Get state at previous timestep
        2. Compute decisions (interventions at current timestep)
        3. Accumulate to create current state
        4. Write this via the DataHandle to DataInterface

        This is a candidate for memoization

        """
        raise NotImplementedError


class PreSpecified(DecisionModule):
    """Pre-specified planning

    Arguments
    ---------
    timesteps : list
    planned_interventions : list
        A list of dicts ``{'name': 'intervention_name', 'build_year': 2010}``
        representing historical or planned interventions
    """

    def __init__(self, timesteps, planned_interventions):
        super().__init__(timesteps)

        self._planned = planned_interventions

    def _get_next_decision_iteration(self):
        return {0: [x for x in self.timesteps]}

    def _set_state(self, timestep, decision_iteration):
        """Pre-specified planning interventions are loaded during initialisation

        Pre-specified planning interventions are loaded during initialisation
        of a model run. This method just needs to copy the existing system state
        to the correct decision iteration reference.

        Arguments
        ---------
        timestep : int
            A timestep (planning year)
        iteration : int
            A decision iteration
        """
        pass

    def get_decision(self, timestep, iteration=None):
        """Return a dict of intervention names built in timestep

        Arguments
        ---------
        timestep : int
            A timestep (planning year)
        iteration : int
            A decision iteration

        Returns
        -------
        list of tuples

        Examples
        --------
        >>> dm = PreSpecified([2010, 2015], [{'name': 'intervention_a', 'build_year': 2010}])
        >>> dm.get_decision(2010)
        [{'name': intervention_a', 'build_year': 2010}]
        """
        decisions = []

        assert isinstance(self._planned, list)

        for intervention in self._planned:
            build_year = int(intervention['build_year'])
            if self.buildable(build_year, timestep):
                decisions.append(intervention)
        return decisions

    def buildable(self, build_year, timestep):
        """Interventions are deemed available if build_year is less than next timestep

        For example, if `a` is built in 2011 and timesteps are
        [2005, 2010, 2015, 2020] then buildable returns True for timesteps
        2010, 2015 and 2020 and False for 2005.
        """
        if not isinstance(build_year, (int, float)):
            msg = "Build Year should be an integer but is a {}"
            raise TypeError(msg.format(type(build_year)))
        if timestep not in self.timesteps:
            raise ValueError("Timestep not in model timesteps")
        index = self.timesteps.index(timestep)
        if index == len(self.timesteps) - 1:
            next_year = timestep + 1
        else:
            next_year = self.timesteps[index + 1]

        if int(build_year) < next_year:
            return True
        else:
            return False


class RuleBased(DecisionModule):
    """Rule-base decision modules
    """

    def __init__(self, timesteps):
        super().__init__(timesteps)
        self.satisfied = False
        self.current_timestep_index = 0
        self.current_iteration = 0

    def _get_next_decision_iteration(self):
            if self.satisfied and self.current_timestep_index == len(self.timesteps) - 1:
                return None
            elif self.satisfied and self.current_timestep_index <= len(self.timesteps):
                self.satisfied = False
                self.current_timestep_index += 1
                self.current_iteration += 1
                return {self.current_iteration: [self.timesteps[self.current_timestep_index]]}
            else:
                self.current_iteration += 1
                return {self.current_iteration: [self.timesteps[self.current_timestep_index]]}

    def get_decision(self, timestep, iteration):
        return []

    def _set_state(self, timestep, decision_iteration):
        raise NotImplementedError
