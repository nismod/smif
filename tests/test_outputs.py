"""Tests the ModelOutputs class

"""
from pytest import fixture
from smif.outputs import ModelOutputs


@fixture(scope='function')
def one_output_metric():
    """Returns a model input dictionary with a single (unlikely to be met)
    dependency
    """
    outputs = {
        'metrics': [
            {'name': 'total_cost'},
            {'name': 'water_demand'}
        ]
    }
    return outputs


class TestModelOutputs:
    """Given a list of dicts from ``outputs.yaml``, return a list of names::

        metrics:
        - name: total_cost
        - name: water_demand

    """

    def test_model_outputs(self, one_output_metric):

        outputs = ModelOutputs(one_output_metric)
        actual = [x for x in outputs.metrics]
        expected = ['total_cost', 'water_demand']
        assert actual == expected
