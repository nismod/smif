"""Solve the optimal planning problem for a system of systems model

"""

from pyomo.environ import (AbstractModel, Binary, Constraint,
                           NonNegativeIntegers, Objective, Param, RangeSet,
                           Set, Var, minimize, summation)
from pyomo.opt import SolverFactory


def _define_basic_model(assets, availability_constraint, asset_costs):
    """Define the binary integer planning problem

    Arguments
    =========
    assets : list
        The list of assets
    availability_constraint : dict
        A dictionary of binary constraints on whether you can build
        each asset in `assets`
    asset_costs : dict
        The investment cost of each asset
    asset_value : dict
        The value function approximation of each asset

    Returns
    =======
    model : pyomo.environ.ConcreteModel
        An abstract instance of the incomplete model with no objective function
    """
    model = AbstractModel()

    number_states = 2 ** len(assets)

    model.p = Param(default=number_states,
                    doc='The number of states',
                    within=NonNegativeIntegers)

    model.I = Set(initialize=assets,
                  doc='The set of infrastructure assets `I`')

    model.b = Param(model.I,
                    initialize=availability_constraint,
                    doc='Constraint on availability of asset `i`')

    model.c = Param(model.I,
                    initialize=asset_costs,
                    doc='Cost of asset `i`')

    model.x = Var(model.I,
                  doc='Decision to invest in asset `i`',
                  domain=Binary)

    def ax_constraint_rule(model, i):
        """Implements the action constraint

        Defines which assets can be built
        """
        return model.x[i] <= model.b[i]

    model.BuildConstraint = Constraint(model.I, rule=ax_constraint_rule)

    return model


def state_vfa_model(assets, availability_constraint, asset_costs, asset_value,
                    states):
    """Define the value function approximation

    Here we assume that the value function approximation is a function
    of the state, rather than individual assets

    Unfortunately, the number of states becomes very large,
    growing exponentially in the number of assets, and so representing the
    approximate value function like this is very inefficient as soon as the
    number of assets increases above 32 (about 4 GB).

    Arguments
    =========
    assets : list
        The list of assets
    availability_constraint : dict
        A dictionary of binary constraints on whether you can build
        each asset in `assets`
    asset_costs : dict
        The investment cost of each asset
    asset_value : dict
        The value function approximation of each asset
    states : dict
        A dictionary where the keys are tuples of entries in `assets` and an
        index of states (``2 ** len(assets)``) and the value is a binary
        indicator showing the possible combinations

    Returns
    =======
    model : pyomo.environ.ConcreteModel
        A concrete instance of the model
    """
    model = _define_basic_model(assets, availability_constraint, asset_costs)

    model.J = RangeSet(1, model.p,
                       doc='The set of states')

    model.d = Param(model.J,
                    initialize=asset_value,
                    doc='Value of being in state `j`')

    model.y = Var(model.J,
                  doc='Decision to NOT invest in state `j`',
                  domain=Binary)

    model.e = Param(model.I, model.J,
                    initialize=states,
                    doc='The combination to match')

    def left_constraint(model, i, j):
        """Forces state `j` to be chosen if x[i] matches e[i,j]
        """
        return (model.x[i] - model.e[i, j]) <= \
            0 + (model.p + 1) * model.y[j]

    model.LeftConstraint = Constraint(model.I, model.J, rule=left_constraint)

    def right_constraint(model, i, j):
        """Forces state `j` to be chosen if x[i] matches e[i,j]
        """
        return 0 <= \
            (model.x[i] - model.e[i, j]) + (model.p + 1) * model.y[j]

    model.RightConstraint = Constraint(model.I, model.J, rule=right_constraint)

    def only_one_constraint(model):
        """Only allow investment in one state

        Right hand side of constraint is N-k (number of assets minus one).

        """
        return summation(model.y) == (model.p - 1)

    model.UniqueConstraint = Constraint(rule=only_one_constraint)

    def objective_function(model):
        """Sum of investment cost of asset `i` and value of being in state `j`
        """
        return summation(model.c, model.x) + \
            sum(model.d[j] * (1 - model.y[j]) for j in model.J)

    model.OBJ = Objective(rule=objective_function,
                          sense=minimize)

    return model


def linear_vfa_model(assets, availability_constraint, asset_costs,
                     asset_value):
    """Define the value function approximation

    Here we assume a linear relationship (in the asset)

    Arguments
    =========
    assets : list
        The list of assets
    availability_constraint : dict
        A dictionary of binary constraints on whether you can build
        each asset in `assets`
    asset_costs : dict
        The investment cost of each asset
    asset_value : dict
        The value function approximation of each asset

    Returns
    =======
    model : pyomo.environ.ConcreteModel
        A concrete instance of the model
    """

    model = _define_basic_model(assets, availability_constraint, asset_costs)

    model.d = Param(model.I,
                    initialize=asset_value,
                    doc='Value of asset `i` (assumes linear relationship)')

    def obj_expression(model):
        """Total cost
        """
        return summation(model.c, model.x) + summation(model.d, model.x)

    model.OBJ = Objective(rule=obj_expression,
                          sense=minimize)

    return model


def solve_model(model):
    """Solves the model using glpk

    Arguments
    =========
    model : pyomo.environ.ConcreteModel
        A concrete instance of the model

    returns : pyomo.core.Model.Instance
        An instance of the model populated with results
    """

    opt = SolverFactory('glpk')

    instance = model.create_instance()
    results = opt.solve(instance)

    instance.solutions.load_from(results)

    return instance
