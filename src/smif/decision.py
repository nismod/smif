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


class DecisionFactory(object):
    """Returns a DecisionModule implementation

    Arguments
    ---------
    horizon: list
    strategies: list
    """

    def __init__(self, horizon, strategies):
        self._horizon = horizon
        self._strategies = strategies

    def get_managers(self):

        decision_maker = None

        for strategy in self._strategies:
            if strategy['strategy'] == 'pre-specified-planning':
                decision_maker = PreSpecified(self._horizon,
                                              strategy['interventions'])
            else:
                msg = "Only pre-specified planning strategies are implemented"
                raise NotImplementedError(msg)

        if decision_maker is None:
            decision_maker = PreSpecified(self._horizon, [])

        return decision_maker


class DecisionModule(metaclass=ABCMeta):
    """Abstract class which provides the interface to decision mechanisms.

    These mechanisms including Pre-Specified Planning, a Rule-based Approach and
    Multi-objective Optimisation.

    This class provides two main public methods, ``__next__`` which is normally
    called implicitly as a call to the class as an iterator, and ``get_state()``
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
        self.horizon = timesteps

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
            A dictionary keyed by decision iteration (int) whose values contain
            a list of timesteps
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
        return self.get_state(timestep.PREVIOUS,
                              decision_iteration)

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

    def __init__(self, timesteps, register):
        super().__init__(timesteps)
        self.register = register

    def _get_next_decision_iteration(self):
        return {1: [year for year in self.horizon]}

    def _set_state(self, timestep, decision_iteration):
        """Pre-specified planning interventions are loaded during initialisation

        Pre-specified planning interventions are loaded during initialisation
        of a model run. This method just needs to copy the existing system state
        to the correct decision iteration reference.
        """
        pass

    def get_state(self, timestep):
        """Return a dict of intervention names built in timestep

        Returns
        -------
        dict

        Examples
        --------
        >>> dm = PreSpecified([2010, 2015], [{'name': 'intervention_a', 'build_year': 2010}])
        >>> dm.get_state(2010)
        {2010: ['intervention_a']}

        """
        state = {}

        for intervention in self.register:
            build_year = intervention['build_year']
            name = intervention['name']
            if self.buildable(build_year, timestep):
                if build_year in state:
                    state[build_year].append(name)
                else:
                    state[build_year] = [name]

        return state

    def buildable(self, build_year, timestep):
        """Interventions are deemed available if build_year is less than next timestep

        For example, if `a` is built in 2011 and timesteps are
        [2005, 2010, 2015, 2020] then buildable returns True for timesteps
        2010, 2015 and 2020 and False for 2005.
        """
        if timestep not in self.horizon:
            raise ValueError("Timestep not in model horizon")
        index = self.horizon.index(timestep)
        if index == len(self.horizon) - 1:
            next_year = 99999
        else:
            next_year = self.horizon[index + 1]

        if build_year < next_year:
            return True
        else:
            return False


class RuleBased(DecisionModule):

    def __init__(self, timesteps):
        super().__init__(timesteps)
        self.satisfied = False
        self.current_timestep_index = 0
        self.current_iteration = 0

    def _get_next_decision_iteration(self):
            if self.satisfied and self.current_timestep_index == len(self.horizon) - 1:
                return None
            elif self.satisfied and self.current_timestep_index <= len(self.horizon):
                self.satisfied = False
                self.current_timestep_index += 1
                self.current_iteration += 1
                return {self.current_iteration: [self.horizon[self.current_timestep_index]]}
            else:
                self.current_iteration += 1
                return {self.current_iteration: [self.horizon[self.current_timestep_index]]}

    def _set_state(self, model, timestep, decision_iteration):
        pass
