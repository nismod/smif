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


    def test_data_example(self):
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
        da = DataArray.from_spec(spec)
        assert da.shape == (2, )

        # Expect defaults
        assert da[0] == 0
        assert da[1] == 0

        # xarray: da.data[0] or da[0].item()

        # Set slices
        da[:] = [1, 2]

        # Access by label
        assert da["England"] == 1
        assert da["Scotland"] == 2

        # xarray: da.loc["England"]
