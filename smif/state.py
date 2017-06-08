"""This module manages the state, action space and registers associated with
interventions, pre-specified planning and built interventions.

"""

from .decision import Built


class State(object):
    """Builds state and actions from set of interventions, plus planned and built interventions.

    Arguments
    ---------
    planned_interventions : smif.decisions.Planning
        The list of pre-specified planning interventions
    intervention_register : smif.interventions.InterventionRegister
        The intervention register
    """
    def __init__(self, planned_interventions, intervention_register):
        self._planned = planned_interventions
        self._interventions = intervention_register

        self._built = Built()
        self._state = dict()
        self._action_space = self.get_initial_action_space()

    @property
    def action_space(self):
        """The set of available interventions
        """
        return self._action_space

    def reset(self):
        """Resets the state ready for a new iteration
        """
        self._reset_action_space()
        self._reset_state()
        self._reset_built()

    def _reset_built(self):
        self._built = Built()

    def _reset_action_space(self):
        """
        """
        self._action_space = set()

    def _reset_state(self):
        """Empties state
        """
        self._state = dict()

    def get_initial_action_space(self):
        """Initial action space is the difference between set of all interventions
        and all planned interventions
        """
        set_of_interventions = set(self._interventions._names)
        # Get the names of ALL planned interventions
        # These should not be available to the optimisation or rule-based approaches
        planned = self._planned.names
        self._action_space = set_of_interventions.difference(planned)

    def update_action_space(self, timeperiod):
        """The action space for the current timeperiod excludes planned
        interventions and interventions built previously
        """
        old_action_space = self._action_space
        current_state = self.get_current_state(timeperiod)
        _ = old_action_space.difference(current_state)
        new_action_space = _.difference(self._planned.current_interventions(timeperiod))
        self._action_space = new_action_space

    def get_initial_state(self, timeperiod):
        """Empties state and initialises with planned interventions
        """
        self._reset_state()
        self._state[timeperiod] = self._planned.current_interventions(timeperiod)

    def get_current_state(self, timeperiod):
        """The current state is the union of built and planned interventions upto
        the `timeperiod`
        """
        built = self._built.current_interventions(timeperiod)
        planned = self._planned.current_interventions(timeperiod)

        return built.union(planned)

    def build(self, name, timeperiod):
        """Adds an intervention available in action space to the built register
        """
        assert name in self._action_space
        self._built.add_intervention(name, timeperiod)
