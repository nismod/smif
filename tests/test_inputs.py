"""Tests the ModelInputs class

"""
import numpy as np
from pytest import fixture
from smif.inputs import ModelInputs

@fixture(scope='function')
def one_dependency():
    """Returns a model input dictionary with a single (unlikely to be met)
    dependency
    """
    inputs = {
        'dependencies': [
            {
                'name': 'macguffins produced',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'from_model': 'macguffins_model'
            }
        ]
    }
    return inputs

class TestDependencyList:

    pass

class TestModelInputs:
    """Given a dict of the format::

    {
        'dependencies': [
            {
                'name': 'macguffins produced',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'from_model': 'macguffins_model'
            }
        ]
    }

    Return a name tuple of dependencies

    """

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

        actual = len(inputs)
        expected = 1
        assert actual == expected
