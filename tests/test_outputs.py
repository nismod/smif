"""Tests the ModelOutputs class

"""
from pytest import fixture
from smif.outputs import ModelOutputs


@fixture(scope='function')
def two_output_metrics():
    """Returns a model output dictionary with two metrics
    """
    outputs = {
        'metrics': [
            {'name': 'total_cost',
             'spatial_resolution': 'LSOA',
             'temporal_resolution': 'annual'},
            {'name': 'water_demand',
             'spatial_resolution': 'watershed',
             'temporal_resolution': 'daily'}
        ]
    }
    return outputs


class TestModelOutputs:
    """Given a list of dicts from ``outputs.yaml``, return a list of names::

        metrics:
        - name: total_cost
        - name: water_demand

    """

    def test_model_outputs(self, two_output_metrics):

        outputs = ModelOutputs(two_output_metrics)
        actual = [x for x in outputs.metrics]
        names = ['total_cost', 'water_demand']
        areas = ['LSOA', 'watershed']
        intervals = ['annual', 'daily']

        for actual, name, area, inter in zip(outputs.metrics,
                                             names,
                                             areas,
                                             intervals):
            assert actual.name == name
            assert actual.spatial_resolution == area
            assert actual.temporal_resolution == inter
