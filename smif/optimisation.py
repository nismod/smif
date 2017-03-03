"""Solve the optimal planning problem for a system of systems model

"""

from pyomo.environ import (AbstractModel, Binary, Constraint,
                           NonNegativeIntegers, Objective, Param, RangeSet,
                           Set, Var, minimize, summation)
from pyomo.opt import SolverFactory


def define_basic_model(assets, availability_constraint, asset_costs):
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

    Notes
    =====

    :math:`\hat{v}_t^n = \min_{a_t \in A_t^n} C_t^{INV}(S^n_t, a^n_t) + \
    {V}^n_t(S^n_t, a^n_t)`

    """
    model = define_basic_model(assets, availability_constraint, asset_costs)

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

    model = define_basic_model(assets, availability_constraint, asset_costs)

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


def solve_model(model, state=None):
    """Solves the model using glpk

    Passing in a `state` as a list of asset names initialises the state
    of the model, fixing those decision variables.

    Arguments
    =========
    model : pyomo.environ.ConcreteModel
        A concrete instance of the model
    state : dict, optional, default=None
        A list of assets which were installed in a previous iteration or time
        period

    returns : pyomo.core.Model.Instance
        An instance of the model populated with results
    """

    opt = SolverFactory('glpk')

    instance = model.create_instance()

    if state:
        for asset in state:
            instance.x[asset] = 1
            instance.x[asset].fixed = True
        instance.preprocess()

    results = opt.solve(instance)

    instance.solutions.load_from(results)
    instance.display()

    return instance


def feature_vfa_model(assets, availability_constraint, asset_costs,
                      feature_coefficients, asset_features):
    """Define the value function approximation

    Here we assume that the value function approximation is a function
    of the features of the state, rather than individual assets, or enumeration
    of all possible states

    Arguments
    =========
    assets : list
        The list of assets
    availability_constraint : dict
        A dictionary of binary constraints on whether you can build
        each asset in `assets`
    asset_costs : dict
        The investment cost of each asset
    features : list
        The set of features
    feature_coefficients : dict
        The regression coefficients for each feature
    asset_features : dict
        The mapping of features to assets

    Returns
    =======
    model : pyomo.environ.ConcreteModel
        A concrete instance of the model

    Notes
    =====
    The use of features decomposes the optimisation problem into several
    sub-components.  The first is to find the clusters of 'bits' in the state,
    which correctly predict minimal cost investments
    and operation of infrastructure assets.

    These clusters can be used to define a feature, adding an entry to the
    `feature` set of features and a column to the `asset_features` matrix,
    where 1 indicates that the feature includes that asset.

    Initially, feature selection could begin by regressing the results from
    a random sample of investment policies. This could highlight which
    patterns of investments seem to result in least cost systems. In addition,
    the attributes of the assets could provide a set of features which at least
    help categorise the assets (e.g. according to sector, size, location,
    and asset type).

    The binary integer problem is posed as follows:

    :math:`\min \sum_i^I c_i x_i + x_i \sum_f^{F}( e_{if} b_{f})`

    where

    - :math:`i` is an element in the set of assets :math:`I`
    - :math:`f` is an element in the set of features :math:`F`
    - :math:`c_i` is the cost of asset :math:`i`
    - :math:`x_i` is the decision to invest in asset :math:`i`
    - :math:`e_{if}` is the mapping of feature :math:`f` to asset :math:`i`
    - :math:`b_{f}` is the basis coefficient of asset :math:`i` and
      feature :math:`f`

    """
    model = define_basic_model(assets, availability_constraint, asset_costs)

    features = list(feature_coefficients.keys())

    model.F = Set(initialize=features,
                  doc='The set of basis functions')

    model.coef = Param(model.F,
                       initialize=feature_coefficients,
                       doc='The basis values')

    model.basis = Param(model.F, model.I,
                        initialize=asset_features,
                        doc='The asset-feature mapping',
                        within=Binary)

    def weight_features(model, i):
        """Computes the blend of features associated with asset `i`

        Arguments
        =========
        model : pyomo.core.AbstractModel
            An instance of the model
        i : pyomo.core.Set
            An element in the set of assets

        Returns
        =======
        expr : pyomo.core.Expression
            The blend of features of asset `i`
        """
        expr = sum(model.coef[f] * model.basis[f, i] for f in model.F)
        return expr

    model.weighted = Param(model.I,
                           initialize=weight_features,
                           doc='The weighted basis values for each asset')

    def objective_function(model):
        return summation(model.c, model.x) + summation(model.weighted, model.x)

    model.OBJ = Objective(rule=objective_function,
                          sense=minimize)

    return model


def formulate_model(asset_register, availability_constraint,
                    feature_coefficients, asset_features):

    assets = [asset.name for asset in asset_register]
    costs = [asset.data['capital cost']['value'] for asset in asset_register]
    asset_costs = dict(zip(assets, costs))

    model = feature_vfa_model(assets, availability_constraint, asset_costs,
                              feature_coefficients, asset_features)

    return model
