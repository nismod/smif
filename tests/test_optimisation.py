"""Tests the definition and solution of the planning problem optimisation
"""

from smif.optimisation import (feature_vfa_model, linear_vfa_model,
                               solve_model, state_vfa_model)


def test_linear_vfa_model():

    assets = ['asset_one', 'asset_two']
    constraint = {'asset_one': 1, 'asset_two': 1}
    costs = {'asset_one': 1200, 'asset_two': 1000}
    value = {'asset_one': 1300, 'asset_two': 100}

    model = linear_vfa_model(assets, constraint, costs, value)
    results = solve_model(model)

    assert results.x['asset_one'].value == 0.0
    assert results.x['asset_two'].value == 0.0


def test_state_vfa_model():
    """Do everything, when that's cheaper than doing nothing

    States:

    #   a  b
    1 - 0  0
    2 - 0  1
    3 - 1  0
    4 - 1  1

    """

    assets = ['asset_one', 'asset_two']
    constraint = {'asset_one': 1, 'asset_two': 1}
    costs = {'asset_one': 1200, 'asset_two': 1000}
    states = {('asset_one', 1): 0,
              ('asset_one', 2): 0,
              ('asset_one', 3): 1,
              ('asset_one', 4): 1,
              ('asset_two', 1): 0,
              ('asset_two', 2): 1,
              ('asset_two', 3): 0,
              ('asset_two', 4): 1}

    value = {1: 2200, 2: 1200, 3: 1000, 4: -1}

    model = state_vfa_model(assets, constraint, costs, value, states)
    results = solve_model(model)

    assert results.x['asset_one'].value == 1.0
    assert results.x['asset_two'].value == 1.0


def test_state_vfa_model_one():
    """Take no action when inaction is cheaper than action

    """

    assets = ['asset_one', 'asset_two']
    constraint = {'asset_one': 1, 'asset_two': 1}
    costs = {'asset_one': 1200, 'asset_two': 1000}
    states = {('asset_one', 1): 0,
              ('asset_one', 2): 0,
              ('asset_one', 3): 1,
              ('asset_one', 4): 1,
              ('asset_two', 1): 0,
              ('asset_two', 2): 1,
              ('asset_two', 3): 0,
              ('asset_two', 4): 1}

    value = {1: 0, 2: 0, 3: 0, 4: 0}

    model = state_vfa_model(assets, constraint, costs, value, states)
    results = solve_model(model)

    assert results.x['asset_one'].value == 0.0
    assert results.x['asset_two'].value == 0.0


def test_features_vfa_model():
    """Use basis functions as the value function approximation

    Passes in a state as a list of asset names

    """
    assets = ['asset_one', 'asset_two']
    constraint = {'asset_one': 1, 'asset_two': 1}
    costs = {'asset_one': 40000, 'asset_two': 500000}

    features = ['bigness', 'colourfulness', 'powerfulness']
    feature_coefficients = {'bigness': 10,
                            'colourfulness': 200,
                            'powerfulness': 3000}
    asset_features = {('bigness', 'asset_one'): 1,
                      ('colourfulness', 'asset_one'): 1,
                      ('powerfulness', 'asset_one'): 0,
                      ('bigness', 'asset_two'): 0,
                      ('colourfulness', 'asset_two'): 1,
                      ('powerfulness', 'asset_two'): 1}

    state = ['asset_one']
    model = feature_vfa_model(assets, constraint, costs,
                              features, feature_coefficients, asset_features)
    results = solve_model(model, state)
    print(results)
    assert results.x['asset_one'].value == 1.0
    assert results.x['asset_two'].value == 0.0
    assert results.OBJ() == 40210

    state = ['asset_two']
    model = feature_vfa_model(assets, constraint, costs,
                              features, feature_coefficients, asset_features)
    results = solve_model(model, state)
    print(results)
    assert results.x['asset_one'].value == 0.0
    assert results.x['asset_two'].value == 1.0
    assert results.OBJ() == 503200
