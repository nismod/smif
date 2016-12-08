"""This module acts as a bridge to the sector models from the controller

 The :class:`SectorModel` exposes several key methods for running wrapped
 sector models.  To add a sector model to an instance of the framework,
 first implement :class:`ModelWrapper`


"""
import logging
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum
from importlib import import_module

import numpy as np
from scipy.optimize import minimize

from smif.inputs import ModelInputs
from smif.outputs import ModelOutputs

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

LOGGER = logging.getLogger(__name__)


class SectorModel(ABC):
    """A representation of the sector model with inputs and outputs

    Attributes
    ==========
    model : :class:`smif.abstract.AbstractModelWrapper`
        An instance of a wrapped simulation model

    """
    def __init__(self):
        self._model_name = None
        self._assets = {}
        self._schema = None

        self._inputs = ModelInputs({})
        self._outputs = ModelOutputs({})


    def validate(self):
        """Validate that this SectorModel has been set up with sufficient data
        to run
        """
        pass

    @property
    def name(self):
        """The name of the sector model

        Returns
        =======
        str
            The name of the sector model

        Note
        ====
        The name corresponds to the name of the folder in which the
        configuration is expected to be found

        """
        return self._model_name

    @name.setter
    def name(self, value):
        self._model_name = value

    @property
    def inputs(self):
        """The inputs to the model

        Returns
        =======
        :class:`smif.inputs.ModelInputs`

        """
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """The inputs to the model

        The inputs should be specified in a dictionary, with the keys first to
        declare ``decision variables`` and ``parameters`` and corresponding
        lists.  The remainder of the dictionary should contain the bounds,
        index and values of the names decision variables and parameters.

        Arguments
        =========
        value : dict
            A dictionary of inputs to the model. This may include parameters,
            assets and exogenous data.

        """
        assert isinstance(value, dict)
        self._inputs = ModelInputs(value)

    @property
    def outputs(self):
        """The outputs from the model

        Returns
        =======
        :class:`smif.outputs.ModelOutputs`

        """
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        """
        Arguments
        =========
        value : dict
            A dictionary of outputs from the model. This may include results
            and metrics
        """
        self._outputs = ModelOutputs(value)

    @property
    def asset_names(self):
        """The names of the assets

        Returns
        =======
        list
            A list of the names of the assets
        """
        return self._assets.keys()

    @property
    def assets(self):
        """The collection of assets, with all attributes

        Returns
        =======
        dict
            The collection of asset attributes
        """
        return self._assets

    @assets.setter
    def assets(self, value):
        self._assets = value


    def constraints(self, parameters):
        """Express constraints for the optimisation

        Use the form outlined in :class:`scipy.optimise.minimize`, namely::

            constraints = ({'type': 'ineq',
                            'fun': lambda x: x - 3})

        Arguments
        =========
        parameters : :class:`numpy.ndarray`
            An array of parameter values passed in from the
            `SectorModel.optimise` method

        """
        constraints = ()
        return constraints

    def optimise(self):
        """Performs a static optimisation for a particular model instance

        Uses an off-the-shelf optimisation algorithm from the scipy library

        Returns
        =======
        dict
            A set of optimised simulation results

        """
        assert len(self.inputs) > 0, "Inputs to the model not yet specified"

        v_names = self.inputs.decision_variables.names
        v_initial = self.inputs.decision_variables.values
        v_bounds = self.inputs.decision_variables.bounds

        cons = self.constraints(self.inputs.parameters.values)

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
            LOGGER.debug("Solver exited successfully with obj: %s", res.fun)
            LOGGER.debug("and with solution: %s", res.x)
            LOGGER.debug("and bounds: %s", v_bounds)
            LOGGER.debug("from initial values: %s", v_initial)
            LOGGER.debug("for variables: %s", v_names)
        else:
            LOGGER.debug("Solver failed")

        return results

    def _simulate_optimised(self, decision_variables):
        results = self.simulate(decision_variables)
        obj = self.extract_obj(results)
        return obj

    @abstractmethod
    def simulate(self, decision_variables):
        """This method should allow run model with inputs and outputs as arrays

        Arguments
        =========
        decision_variables : x-by-1 :class:`numpy.ndarray`
        """
        pass

    @abstractmethod
    def extract_obj(self, results):
        """Implement this method to return a scalar value objective function

        This method should take the results from the output of the `simulate`
        method, process the results, and return a scalar value which can be
        used as the objective function

        Arguments
        =========
        results : :class:`dict`
            The results from the `simulate` method

        Returns
        =======
        float
            A scalar component generated from the simulation model results
        """
        pass

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
                # TODO move this to water_supply implementation
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                LOGGER.debug("Updating %s with %s", state_var, state_res)
                self.inputs.parameters.update_value(state_var, state_res)

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
            LOGGER.debug("Running simulation for year %s", years[index])
            # Update the state from the previous year
            if index > 0:
                # TODO move this to water_supply implementation
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                LOGGER.debug("Updating %s with %s", state_var, state_res)
                self.inputs.parameters.update_value(state_var, state_res)
            # Run the simulation
            decision = np.array([decisions[index], ])
            assert decision.shape == (1, )
            results.append(self.simulate(decision))
        return results

    def seq_opt_obj(self, decisions):
        assert decisions.shape == (3,)
        results = self._optimise_over_timesteps(decisions)
        LOGGER.debug("Decisions: {}".format(decisions))
        return self.get_objective(results, discount_rate=0.05)

    @staticmethod
    def get_objective(results, discount_rate=0.05):
        discount_factor = [(1 - discount_rate)**n for n in range(0, 15, 5)]
        costs = sum([x['cost']
                     * discount_factor[ix] for ix, x in enumerate(results)])
        LOGGER.debug("Objective function: £%.2f", float(costs))
        return costs

    def sequential_optimisation(self, timesteps):

        assert self.inputs, "Inputs to the model not yet specified"

        number_of_steps = len(timesteps)

        v_names = self.inputs.decision_variables.names
        v_initial = self.inputs.decision_variables.values
        v_bounds = self.inputs.decision_variables.bounds

        t_v_initial = np.tile(v_initial, (1, number_of_steps))
        t_v_bounds = np.tile(v_bounds, (number_of_steps, 1))
        LOGGER.debug("Flat bounds: %s", v_bounds)
        LOGGER.debug("Tiled Bounds: %s", t_v_bounds)
        LOGGER.debug("Flat Bounds: %s", t_v_bounds.flatten())
        LOGGER.debug("DecVar: %s", t_v_initial)

        # TODO move this to water_supply implementation
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
            LOGGER.debug("Solver exited successfully with obj: %s", res.fun)
            LOGGER.debug("and with solution: %s", res.x)
            LOGGER.debug("and bounds: %s", v_bounds)
            LOGGER.debug("from initial values: %s", v_initial)
            LOGGER.debug("for variables: %s", v_names)
        else:
            LOGGER.debug("Solver failed")

        return results


class SectorModelBuilder(object):
    """Build the components that make up a sectormodel from the configuration

    """

    def __init__(self, module_name):
        self.module_name = module_name
        self._sectormodel = None

    def load_model(self, model_path, classname):
        """Dynamically load model module

        """
        if os.path.exists(model_path):
            LOGGER.info("Importing run module from %s", model_path)

            model_dirname, model_filename = os.path.split(model_path)
            sys.path.append(model_dirname)

            module = import_module(model_filename)
            klass = module.__dict__[classname]

            self._sectormodel = klass()
            self._sectormodel.name = self.module_name

        else:
            msg = "Cannot find {} for the {} model".format(model_path, self.module_name)
            raise Exception(msg)

    def add_inputs(self, input_dict):
        """Add inputs to the sector model
        """
        msg = "Sector model must be loaded before adding inputs"
        assert self._sectormodel is not None, msg

        self._sectormodel.inputs = input_dict

    def add_outputs(self, output_dict):
        """Add outputs to the sector model
        """
        msg = "Sector model must be loaded before adding outputs"
        assert self._sectormodel is not None, msg

        self._sectormodel.outputs = output_dict

    def add_assets(self, dict_of_assets):
        """Add assets to the sector model
        """
        msg = "Sector model must be loaded before adding assets"
        assert self._sectormodel is not None, msg

        self._sectormodel.attributes = dict_of_assets

    def validate(self):
        """Check and/or assert that the sector model is correctly set up
        """
        assert self._sectormodel is not None
        self._sectormodel.validate()

    def finish(self):
        """Validate and return the sector model
        """
        self.validate()
        return self._sectormodel


class SectorModelMode(Enum):
    """Enumerates the operating modes of a sector model
    """
    static_simulation = 0
    sequential_simulation = 1
    static_optimisation = 2
    dynamic_optimisation = 3
