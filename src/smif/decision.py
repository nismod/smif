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
    intervention_register : smif.intervention.Register
        Reference to an intervention register object populated with available
        interventions for the current SosModel

    """
    def __init__(self, timesteps, intervention_register=None):
        self.horizon = timesteps
        self.register = intervention_register

    def __next__(self):
        return self._get_next_decision_iteration()

    @abstractmethod
    def _get_next_decision_iteration(self):
        """Implement to return the next decision iteration

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
        return self.register.get_state(timestep.PREVIOUS,
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

    def _get_next_decision_iteration(self):
        return {1: [year for year in self.horizon]}

    def _set_state(self, timestep, decision_iteration):
        """Pre-specified planning interventions are loaded during initialisation

        Pre-specified planning interventions are loaded during initialisation
        of a model run. This method just needs to copy the existing system state
        to the correct decision iteration reference.
        """
        pass


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


class Planning:
    """
    Holds the list of planned interventions, where a single planned intervention
    is an intervention with a build date after which it will be included in the
    modelled systems.

    For example, a small pumping station might be built in
    Oxford in 2045::

            {
                'name': 'small_pumping_station',
                'build_date': 2045
            }

    Attributes
    ----------
    planned_interventions : list
        A list of pre-specified planned interventions

    """

    def __init__(self, planned_interventions=None):

        if planned_interventions is not None:
            self.planned_interventions = planned_interventions
        else:
            self.planned_interventions = []

    @property
    def names(self):
        """Returns the set of assets defined in the planned interventions
        """
        return {plan['name'] for plan in self.planned_interventions}

    @property
    def timeperiods(self):
        """Returns the set of build dates defined in the planned interventions
        """
        return {plan['build_date'] for plan in self.planned_interventions}
