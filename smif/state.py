"""This module manages the state, action space and registers associated with
interventions, pre-specified planning and built interventions.

"""
import logging
from collections import defaultdict

import numpy as np
from smif import StateData

from .decision import Built
from .intervention import Asset


class State(object):
    """Builds state and actions from set of interventions,
    planned and built interventions
    and intervention-attribute stateful information

    The latter is defined in intervention attributes using the `is_state`
    flag. For example::

        - name: reservoir
          location: Oxford
          capacity:
              value: 500
              units: ML
          operational_lifetime:
              value: 300
              units: years
          economic_lifetime:
              value: 150
              units: years
          capital_cost:
              value: 15
              units: million £
          current_level:
              value: 3
              units: ML
              is_state: True

    Arguments
    ---------
    planned_interventions : smif.decisions.Planning
        The list of pre-specified planning interventions
    intervention_register : smif.interverntions.InterventionRegister
        The intervention register
    """
    def __init__(self, planned_interventions,
                 intervention_register):

        self.logger = logging.getLogger(__name__)

        self._planned = planned_interventions
        self._interventions = intervention_register

        self._built = Built()
        self._state_data = defaultdict(dict)

        self._state = dict()
        self._action_space = self.get_initial_action_space()

    @property
    def action_space(self):
        """The set of available interventions

        Returns
        -------
        set
        """
        return self._action_space

    @property
    def action_list(self):
        """Immutible version of the action space

        Returns
        -------
        list
        """
        return sorted(self._action_space)

    def reset(self):
        """Resets the state ready for a new iteration
        """
        self.logger.debug("Resetting state instance")
        self._reset_action_space()
        self._reset_state()
        self._reset_built()
        self.get_initial_action_space()

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

        Returns
        -------
        set

        """
        set_of_interventions = self._interventions.names
        # Get the names of ALL planned interventions
        # These should not be available to the optimisation or rule-based approaches
        planned = self._planned.names
        return set_of_interventions.difference(planned)

    def update_action_space(self, timeperiod):
        """The action space for the current timeperiod excludes planned
        interventions and interventions built previously
        """
        self.logger.debug("Updating action space")
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
        """The current state is the union of built and planned
        interventions upto the `timeperiod`

        Returns
        -------
        set
            The set of built and planned interventions
        """
        built = self._built.current_interventions(timeperiod)
        planned = self._planned.current_interventions(timeperiod)

        return built.union(planned)

    def get_all_state(self, timeperiod, sector=None):
        """Returns all state filtered by sector model

        Arguments
        ---------
        timeperiod : int
        sector : str, default=None

        Returns
        -------
        state_ data : list
            A list of :class:`smif.StateData`
        build_interventions : list
            A list of :class:`smif.intervention.Asset`

        """
        intervention_names = self.get_current_state(timeperiod)
        built_interventions = []

        for name in intervention_names:
            data = self._interventions.get_intervention(name).data

            if sector is None:

                if name in self._built.names:
                    data['build_date'] = self._built.get_build_date(name)
                    built_interventions.append(Asset(data=data))
                elif name in self._planned.names:
                    data['build_date'] = self._planned.get_build_date(name)
                    built_interventions.append(Asset(data=data))

            else:

                if name in self._built.names and data['sector'] == sector:
                    data['build_date'] = self._built.get_build_date(name)
                    built_interventions.append(Asset(data=data))
                elif name in self._planned.names and data['sector'] == sector:
                    data['build_date'] = self._planned.get_build_date(name)
                    built_interventions.append(Asset(data=data))

        self.logger.debug("Current built interventions: %s",
                          self._built.names)

        if sector:
            state_data = self.get_state(timeperiod, sector)
        else:
            state_data = self.state_data[timeperiod]
        self.logger.debug("Current state data: %s", state_data)
        return state_data, built_interventions

    def build(self, name, timeperiod):
        """Adds an intervention available in action space to the built register

        Arguments
        ---------
        name : str
            The name ofintervention
        timeperiod : int
        """
        assert name in self._action_space
        self._built.add_intervention(name, timeperiod)

    def get_action_dimension(self):
        """The dimension of the current action space

        Returns
        -------
        int
            The number of dimensions in the action space
        """
        return len(self._action_space)

    def get_decision_vector(self):
        """Helper function to generate a numpy array for actions

        Returns
        -------
        numpy.ndarray
            An array of length of the action space
        """
        return np.zeros(self.get_action_dimension())

    @property
    def state_data(self):
        """Returns the state data

        Returns
        -------
        dict
            A nested dictionary of [timestep][model_name] = [:class:`smif.StateData`]
        """
        return self._state_data

    def get_state(self, timestep, model_name):
        """Gets the state data

        Arguments
        ---------
        timestep : int
        model_name : str

        Returns
        -------
        list
            A list of :class:`smif.StateData`

        """
        if model_name not in self.state_data[timestep]:
            self.logger.warning("Found no state for %s in timestep %s", model_name, timestep)
            return []
        return self.state_data[timestep][model_name]

    @state_data.setter
    def state_data(self, timestep_sector_data_tuple):
        assert isinstance(timestep_sector_data_tuple, tuple)
        timestep, sector, data = timestep_sector_data_tuple
        assert isinstance(sector, str)
        assert isinstance(timestep, int)
        assert isinstance(data, list)
        for entry in data:
            assert isinstance(entry, StateData)
        self.logger.debug("Added %s to %s in %s", data, sector, timestep)
        self._state_data[timestep][sector] = data

    def parse_decisions(self, decision_vector):
        """Returns intervention in list of array element is 1

        Arguments
        ---------
        decision_vector : numpy.ndarray
            A numpy array of length `self.action_space`

        Returns
        -------
        list
            List of Interventions
        """
        interventions = []

        indexes = np.where(decision_vector == 1)[0]
        for index in list(indexes):
            name = self.action_list[index]
            interventions.append(self._interventions.get_intervention(name))
        return interventions
