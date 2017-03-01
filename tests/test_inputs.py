"""Tests the ModelInputs class

"""
import numpy as np
from smif.inputs import ModelInputs


class TestModelInputs:

    def test_one_dependency(self, one_dependency):
        inputs = ModelInputs(one_dependency)

        actual = inputs.dependencies.names
        expected = ['macguffins produced']
        assert actual == expected

        actual = inputs.dependencies.from_models
        expected = ['macguffins_model']
        assert actual == expected
        actual = inputs.dependencies.spatial_resolutions
        expected = ['LSOA']
        assert actual == expected
        actual = inputs.dependencies.temporal_resolutions
        expected = ['annual']
        assert actual == expected
