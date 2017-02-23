# -*- coding: utf-8 -*-
"""This module acts as a bridge to the sector models from the controller

The :class:`SectorModel` exposes several key methods for running wrapped
sector models.  To add a sector model to an instance of the framework,
first implement :class:`SectorModel`.

Data Required to Specify a Sectoral Model
=========================================

To integrate an infrastructure simulation model within the system-of-systems
modelling framework, it is necessary to provide the following configuration
data, alongside the implementation of the :class:`SectorModel` class.

Geographies
-----------
Define the set of unique regions which are used within the model as polygons.
Inputs and outputs are assigned a model-specific geography from this list
allowing automatic conversion from and to these geographies.

Model regions are specified in ``data/<sectormodel>/regions.*``

The file format must be possible to parse with GDAL, and must contain
an attribute "name" to use as an identifier for the region.

Temporal Resolution
-------------------
The attribution of hours in a year to the temporal resolution used
in the sectoral model.

Within-year time intervals are specified
in ``data/<sectormodel>/time_intervals.yaml``

These specify the mapping of model timesteps to durations within a year
(assume modelling 365 days: no extra day in leap years, no leap seconds)

Each time interval must have

- start (period since beginning of year)
- end (period since beginning of year)
- id (label to use when passing between integration layer and sector model)

use ISO 8601 [1]_ duration format to specify periods::

    P[n]Y[n]M[n]DT[n]H[n]M[n]S

References
----------
.. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations

Inputs
------
The collection of inputs required when defined as a dependency, for example
"electricity demand (kWh, <region>, <hour>)".
Inputs are defined with a region and temporal-resolution and a unit.

Only those inputs required as dependencies are defined here, although
dependencies are activated when configured in the system-of-systems model.

Outputs
-------
The collection of outputs used as metrics, for the purpose of optimisation or
rule-based planning approaches (so normally a cost-function), and those
outputs required for accounting purposes, such as operational cost, and
emissions.

Units
-----
The set of units used to define Inputs, Outputs and Interventions.

Interventions
-------------
An Intervention is an investment which has a name (or name),
other attributes (such as capital cost and economic lifetime), and location,
but no build date.

An Intervention is a possible investment, normally an infrastructure asset,
the timing of which can be decided by the logic-layer.

An exhaustive list of the Interventions (normally infrastructure assets)
should be defined.
These are represented internally in the system-of-systems model,
collected into a gazateer and allow the framework to reason on
infrastructure assets across all sectors.
Interventions are instances of :class:`~smif.asset.Intervention` and are
held in :class:`~smif.asset.InterventionRegister`.
Interventions include investments in assets,
supply side efficiency improvements, but not demand side management (these
are incorporated in the strategies).

Existing Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~
Existing infrastructure is specified in a
``<sectormodel>/assets/*.yaml`` file.  This uses the following
format::
   -
    name: CCGT
    description: Existing roll out of gas-fired power stations
    timeperiod: 1990 # 2010 is the first year in the model horizon
    location: "oxford"
    new_capacity:
        value: 6
        unit: GW
    lifetime:
        value: 20
        unit: years

Pre-Specified Planning
~~~~~~~~~~~~~~~~~~~~~~

A fixed pipeline of investments can be specified using the same format as for
existing infrastructure, in the ``<sectormodel>/planning/*.yaml`` files.

The only difference is that pre-specified planning investments occur in the
future (in comparison to the initial modelling date), whereas existing
infrastructure occur in the past. This difference is semantic at best, but a
warning is raised if future investments are included in the existing
infrastructure files in the situation where the initial model timeperiod is
altered.

State Parameters
----------------
Some simulation models require that state is passed between years, for example
reservoir level in the water-supply model.
These are treated as self-dependencies with a temporal offset. For example,
the sector model depends on the result of running the model for a previous
timeperiod.

Key Functions
=============
This class performs several key functions which ease the integration of sector
models into the system-of-systems framework.

The user must implement the various abstract functions throughout the class to
provide an interface to the sector model, which can be called upon by the
framework. From the model's perspective, :class:`SectorModel` provides a bridge
from the sector-specific problem representation to the general representation
which allows reasoning across infrastructure systems.

The key functions include

* converting input/outputs to/from geographies/temporal resolutions
* converting control vectors from the decision layer of the framework, to
  asset Interventions specific to the sector model
* returning scaler/vector values to the framework to enable measurements of
  performance, particularly for the purposes of optimisation and rule-based
  approaches


"""
import importlib
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from smif.inputs import ModelInputs
from smif.outputs import ModelOutputs

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


class SectorModel(ABC):
    """A representation of the sector model with inputs and outputs

    """
    def __init__(self):
        self._model_name = None
        self._schema = None

        self.interventions = []

        self._inputs = ModelInputs({})
        self._outputs = ModelOutputs({})

        self.logger = logging.getLogger(__name__)

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
    def intervention_names(self):
        """The names of the interventions

        Returns
        =======
        list
            A list of the names of the interventions
        """
        return [intervention['name'] for intervention in self.interventions]

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
                       constraints=cons)

        # results = {x: y for x, y in zip(v_names, res.x)}
        # TODO wire in state and data
        state = []
        data = []
        results = self.simulate(res.x, state, data)

        if res.success:
            self.logger.debug("Solver exited successfully with obj: %s", res.fun)
            self.logger.debug("and with solution: %s", res.x)
            self.logger.debug("and bounds: %s", v_bounds)
            self.logger.debug("from initial values: %s", v_initial)
            self.logger.debug("for variables: %s", v_names)
        else:
            self.logger.debug("Solver failed")

        return results

    def _simulate_optimised(self, decision_variables):
        # TODO wire in state and data
        state = []
        data = []
        results = self.simulate(decision_variables, state, data)
        obj = self.extract_obj(results)
        return obj

    @abstractmethod
    def simulate(self, decisions, state, data):
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
                self.logger.debug("Updating %s with %s", state_var, state_res)
                self.inputs.parameters.update_value(state_var, state_res)

            # Run the simulation
            decision = decisions[:, index]
            # TODO wire in state and data
            state = []
            data = []
            results.append(self.simulate(decision, state, data))
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
            self.logger.debug("Running simulation for year %s", years[index])
            # Update the state from the previous year
            if index > 0:
                # TODO move this to water_supply implementation
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                self.logger.debug("Updating %s with %s", state_var, state_res)
                self.inputs.parameters.update_value(state_var, state_res)
            # Run the simulation
            decision = np.array([decisions[index], ])
            assert decision.shape == (1, )
            results.append(self.simulate(decision))
        return results

    def seq_opt_obj(self, decisions):
        assert decisions.shape == (3,)
        results = self._optimise_over_timesteps(decisions)
        self.logger.debug("Decisions: {}".format(decisions))
        return self.get_objective(results, discount_rate=0.05)

    def get_objective(self, results, discount_rate=0.05):
        discount_factor = [(1 - discount_rate)**n for n in range(0, 15, 5)]
        costs = sum([x['cost']
                     * discount_factor[ix] for ix, x in enumerate(results)])
        self.logger.debug("Objective function: £%.2f", float(costs))
        return costs

    def sequential_optimisation(self, timesteps):

        assert self.inputs, "Inputs to the model not yet specified"

        number_of_steps = len(timesteps)

        v_names = self.inputs.decision_variables.names
        v_initial = self.inputs.decision_variables.values
        v_bounds = self.inputs.decision_variables.bounds

        t_v_initial = np.tile(v_initial, (1, number_of_steps))
        t_v_bounds = np.tile(v_bounds, (number_of_steps, 1))
        self.logger.debug("Flat bounds: %s", v_bounds)
        self.logger.debug("Tiled Bounds: %s", t_v_bounds)
        self.logger.debug("Flat Bounds: %s", t_v_bounds.flatten())
        self.logger.debug("DecVar: %s", t_v_initial)

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
                       constraints=cons)

        results = self.sequential_simulation(timesteps, np.array([res.x]))

        if res.success:
            self.logger.debug("Solver exited successfully with obj: %s", res.fun)
            self.logger.debug("and with solution: %s", res.x)
            self.logger.debug("and bounds: %s", v_bounds)
            self.logger.debug("from initial values: %s", v_initial)
            self.logger.debug("for variables: %s", v_names)
        else:
            self.logger.debug("Solver failed")

        return results


class SectorModelBuilder(object):
    """Build the components that make up a sectormodel from the configuration

    Parameters
    ----------
    name : str
        The name of the sector model

    """

    def __init__(self, name):
        self._sector_model_name = name
        self._sector_model = None
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path, classname):
        """Dynamically load model module

        """
        if os.path.exists(model_path):
            self.logger.info("Importing run module from %s", model_path)

            spec = importlib.util.spec_from_file_location(
                self._sector_model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            klass = module.__dict__[classname]

            self._sector_model = klass()
            self._sector_model.name = self._sector_model_name

        else:
            msg = "Cannot find {} for the {} model".format(
                model_path, self._sector_model_name)
            raise Exception(msg)

    def add_inputs(self, input_dict):
        """Add inputs to the sector model
        """
        msg = "Sector model must be loaded before adding inputs"
        assert self._sector_model is not None, msg

        self._sector_model.inputs = input_dict

    def add_outputs(self, output_dict):
        """Add outputs to the sector model
        """
        msg = "Sector model must be loaded before adding outputs"
        assert self._sector_model is not None, msg

        self._sector_model.outputs = output_dict

    def add_assets(self, asset_list):
        """Add assets to the sector model

        Parameters
        ----------
        asset_list : list
            A list of dicts of assets
        """
        msg = "Sector model must be loaded before adding assets"
        assert self._sector_model is not None, msg

        self._sector_model.assets = asset_list

    def add_interventions(self, intervention_list):
        """Add interventions to the sector model

        Parameters
        ----------
        intervention_list : list
            A list of dicts of interventions

        """
        msg = "Sector model must be loaded before adding interventions"
        assert self._sector_model is not None, msg

        self._sector_model.interventions = intervention_list

    def validate(self):
        """Check and/or assert that the sector model is correctly set up
        - should raise errors if invalid
        """
        assert self._sector_model is not None, "Sector model not loaded"
        self._sector_model.validate()
        return True

    def finish(self):
        """Validate and return the sector model
        """
        self.validate()
        return self._sector_model
