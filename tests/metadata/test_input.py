"""Test metadata specification
"""
from pytest import raises

# from smif.metadata import ExchangeItem, ElementSet

class TestExchangeItem():
    """A DataSpec serves as metadata for a model input/parameter/output
    (DataArray)

    Describes an Exchange item (OpenMI)

    dtype - see numpy.dtypes
    Quality (OpenMI) categorical
    Quantity (OpenMI) numeric
    """
    def test_construct_with_values(self):
        """A DataSpec has:
        - name
        - dimensions
        - default value
        - dtype
        - absolute range
        - expected range
        - unit

        The DataArray it describes may be sparse
        """
        DataSpec(
            name='population',
            dims=[
                ElementSet('regions', ["England", "Wales"])
            ],
            dtype='int',
            default=0,
            abs_range=(0, float('inf')),
            exp_range=(10e6, 10e9),
            unit='people'
        )
