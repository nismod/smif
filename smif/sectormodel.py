import logging

import numpy as np
from scipy.optimize import minimize
from smif.inputs import ModelInputs

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class SectorModel(object):
    """An abstract representation of the sector model with inputs and outputs

    Parameters
    ==========
    schema : dict
        A dictionary of parameter, asset and exogenous data names with expected
        types. Used for validating presented data.

    Attributes
    ==========
    model : :class:`smif.abstract.AbstractModelWrapper`
        An instance of a wrapped simulation model

    """
    def __init__(self, model):
        self.model = model
        self._inputs = None
        self._schema = None

    @property
    def inputs(self):
        """The inputs to the model

        Returns
        =======
        :class:`smif.abstract.ModelInputs`

        """
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """The inputs to the model

        Arguments
        =========
        value : dict
            A dictionary of inputs to the model. This may include parameters,
            assets and exogenous data.

        """
        self._inputs = ModelInputs(value)

    def optimise(self):
        """Performs a static optimisation for a particular model instance

        Uses an off-the-shelf optimisation algorithm from the scipy library

        Returns
        =======
        dict
            A set of optimised simulation results

        """
        assert self.inputs, "Inputs to the model not yet specified"

        v_names = self.inputs.decision_variables.names
        v_initial = self.inputs.decision_variables.values
        v_bounds = self.inputs.decision_variables.bounds

        cons = self.model.constraints(self.inputs.parameters.values)

        opts = {'disp': True}
        res = minimize(self._simulate_optimised,
                       v_initial,
                       options=opts,
                       method='SLSQP',
                       bounds=v_bounds,
                       constraints=cons
                       )

        # results = {x: y for x, y in zip(v_names, res.x)}
        results = self.simulate(res.x)

        if res.success:
            logger.debug("Solver exited successfully with obj: {}".format(
                res.fun))
            logger.debug("and with solution: {}".format(res.x))
            logger.debug("and bounds: {}".format(v_bounds))
            logger.debug("from initial values: {}".format(v_initial))
            logger.debug("for variables: {}".format(v_names))
        else:
            logger.debug("Solver failed")

        return results

    def _simulate_optimised(self, decision_variables):
        results = self.simulate(decision_variables)
        obj = self.model.extract_obj(results)
        return obj

    def simulate(self, decision_variables):
        """Performs an operational simulation of the sector model

        Arguments
        =========
        decision_variables : :class:`numpy.ndarray`

        Note
        ====
        The term simulation may refer to operational optimisation, rather than
        simulation-only. This process is described as simulation to distinguish
        from the definition of investments in capacity, versus operation using
        the given capacity
        """

        assert self.inputs, "Inputs to the model not yet specified"

        static_inputs = self.inputs.parameters.values
        results = self.model.simulate(static_inputs, decision_variables)
        return results

    def sequential_simulation(self, timesteps, decisions):
        """Perform a sequential simulation on an initialised model

        Arguments
        =========
        timesteps : list
            List of timesteps over which to perform a sequential simulation
        decisions : :class:`numpy.ndarray`
            A vector of decisions of size `timesteps`.`decisions`

        """
        assert self.inputs, "Inputs to the model not yet specified"
        self.inputs.parameters.update_value('existing capacity', 0)

        results = []
        for index in range(len(timesteps)):
            # Update the state from the previous year
            if index > 0:
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                logger.debug("Updating {} with {}".format(state_var,
                                                          state_res))
                self.inputs.parameters.update_value(state_var,
                                                    state_res)

            # Run the simulation
            decision = decisions[:, index]
            results.append(self.simulate(decision))
        return results

    def _optimise_over_timesteps(self, decisions):
        """
        """
        self.inputs.parameters.update_value('raininess', 3)
        self.inputs.parameters.update_value('existing capacity', 0)
        assert decisions.shape == (3,)
        results = []
        years = [2010, 2015, 2020]
        for index in range(3):
            logger.debug("Running simulation for year {}".format(years[index]))
            # Update the state from the previous year
            if index > 0:
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                logger.debug("Updating {} with {}".format(state_var,
                                                          state_res))
                self.inputs.parameters.update_value(state_var,
                                                    state_res)
            # Run the simulation
            decision = np.array([decisions[index], ])
            assert decision.shape == (1, )
            results.append(self.simulate(decision))
        return results

    def seq_opt_obj(self, decisions):
        assert decisions.shape == (3,)
        results = self._optimise_over_timesteps(decisions)
        logger.debug("Decisions: {}".format(decisions))
        return self.get_objective(results, discount_rate=0.05)

    def get_objective(self, results, discount_rate=0.05):
        discount_factor = [(1 - discount_rate)**n for n in range(0, 15, 5)]
        costs = sum([x['cost']
                     * discount_factor[ix] for ix, x in enumerate(results)])
        logger.debug("Objective function: £{:2}".format(float(costs)))
        return costs

    def sequential_optimisation(self, timesteps):

        assert self.inputs, "Inputs to the model not yet specified"

        number_of_steps = len(timesteps)

        v_names = self.inputs.decision_variables.names
        v_initial = self.inputs.decision_variables.values
        v_bounds = self.inputs.decision_variables.bounds

        t_v_initial = np.tile(v_initial, (1, number_of_steps))
        t_v_bounds = np.tile(v_bounds, (number_of_steps, 1))
        logger.debug("Flat bounds: {}".format(v_bounds))
        logger.debug("Tiled Bounds: {}".format(t_v_bounds))
        logger.debug("Flat Bounds: {}".format(t_v_bounds.flatten()))
        logger.debug("DecVar: {}".format(t_v_initial))

        annual_rainfall = 5
        demand = [3, 4, 5]

        cons = ({'type': 'ineq',
                 'fun': lambda x: min(sum(x[0:1]),
                                      annual_rainfall) - demand[0]},
                {'type': 'ineq',
                 'fun': lambda x: min(sum(x[0:2]),
                                      annual_rainfall) - demand[1]},
                {'type': 'ineq',
                 'fun': lambda x: min(sum(x[0:3]),
                                      annual_rainfall) - demand[2]})

        opts = {'disp': True}
        res = minimize(self.seq_opt_obj,
                       t_v_initial,
                       options=opts,
                       method='SLSQP',
                       bounds=t_v_bounds,
                       constraints=cons
                       )

        results = self.sequential_simulation(timesteps, np.array([res.x]))

        if res.success:
            logger.debug("Solver exited successfully with obj: {}".format(
                res.fun))
            logger.debug("and with solution: {}".format(res.x))
            logger.debug("and bounds: {}".format(v_bounds))
            logger.debug("from initial values: {}".format(v_initial))
            logger.debug("for variables: {}".format(v_names))
        else:
            logger.debug("Solver failed")

        return results
