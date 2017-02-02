"""Tests the ModelInputs class
"""
import numpy as np
from numpy.testing import assert_equal
from smif.inputs import ModelInputs


class TestModelInputs:

    def test_one_input_decision_variables(self, one_input):

        inputs = ModelInputs(one_input)
        act_names = inputs.decision_variables.names
        act_initial = inputs.decision_variables.values
        act_bounds = inputs.decision_variables.bounds
        act_indices = inputs.decision_variables.indices

        exp_names = np.array(['water treatment capacity'], dtype=str)
        exp_initial = np.array([10], dtype=float)
        exp_bounds = np.array([(0, 20)], dtype=(float, 2))
        exp_indices = {'water treatment capacity': 0}

        assert_equal(act_names, exp_names)
        assert_equal(act_initial, exp_initial)
        assert_equal(act_bounds, exp_bounds)
        assert act_indices == exp_indices

    def test_two_inputs_decision_variables(self, two_inputs):

        inputs = ModelInputs(two_inputs)
        act_names = inputs.decision_variables.names
        act_initial = inputs.decision_variables.values
        act_bounds = inputs.decision_variables.bounds

        exp_names = np.array(['reservoir pumpiness',
                              'water treatment capacity'], dtype='U30')
        exp_initial = np.array([24.583, 10], dtype=float)
        exp_bounds = np.array([(0, 100), (0, 20)], dtype=(float, 2))

        assert_equal(act_names, exp_names)
        assert_equal(act_initial, exp_initial)
        assert_equal(act_bounds, exp_bounds)

    def test_one_input_parameters(self, one_input):

        inputs = ModelInputs(one_input)
        act_names = inputs.parameters.names
        act_values = inputs.parameters.values
        act_bounds = inputs.parameters.bounds

        exp_names = np.array(['raininess'], dtype='U30')
        exp_values = np.array([3], dtype=float)
        exp_bounds = np.array([(0, 5)], dtype=(float, 2))

        assert_equal(act_names, exp_names)
        assert_equal(act_values, exp_values)
        assert_equal(act_bounds, exp_bounds)

    def test_one_dependency(self, one_dependency):
        inputs = ModelInputs(one_dependency)

        actual = inputs.dependencies.names
        expected = np.array(['macguffins produced'], dtype='U30')
        assert_equal(actual, expected)

        actual = inputs.dependencies.from_models
        expected = np.array(['macguffins_model'], dtype='U30')
        assert_equal(actual, expected)

        actual = inputs.dependencies.spatial_resolutions
        expected = np.array(['LSOA'], dtype='U30')
        assert_equal(actual, expected)

        actual = inputs.dependencies.temporal_resolutions
        expected = np.array(['annual'], dtype='U30')
        assert_equal(actual, expected)
