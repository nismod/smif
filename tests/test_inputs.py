"""Tests the ModelInputs class

"""
from smif.inputs import ModelInputs


class TestModelInputs:
    """Given a dict of the format::

    {
        'dependencies': [
            {
                'name': 'macguffins produced',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual'
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

        actual = inputs.dependencies.spatial_resolutions
        expected = ['LSOA']
        assert actual == expected
        actual = inputs.dependencies.temporal_resolutions
        expected = ['annual']
        assert actual == expected

        actual = len(inputs)
        expected = 1
        assert actual == expected
