"""Test metadata specification
"""
from pytest import raises
from smif.metadata import Coordinates, Spec


class TestSpec():
    """A Spec serves as metadata for a model input/parameter/output

    Describes an Exchange item (OpenMI)

    dtype - see numpy.dtypes
    Quality (OpenMI) categorical
    Quantity (OpenMI) numeric
    """
    def test_construct(self):
        """A Spec has:
        - coords: coordinates that label each point - list of Coordinates, one for each dim
        - name
        - default value
        - dtype
        - absolute range: (optional) for numerical types, to raise error if exceeded
        - expected range: (optional) for numerical types, to raise warning if exceeded
        - unit
        - acceptable values

        The DataArray it describes may be sparse
        """
        spec = Spec(
            name='population',
            coords=[Coordinates('countries', ["England", "Wales"])],
            dtype='int',
            default=0,
            abs_range=(0, float('inf')),
            exp_range=(10e6, 10e9),
            unit='people'
        )
        assert spec.name == 'population'
        assert spec.dtype == 'int'
        assert spec.shape == (2,)
        assert spec.ndim == 1

    def test_empty_dtype_error(self):
        """A Spec must be constructed with a dtype
        """
        with raises(ValueError) as ex:
            Spec(
                name='test',
                coords=[Coordinates('countries', ["England", "Wales"])]
            )
        assert "dtype must be provided" in str(ex)

    def test_empty_coords_error(self):
        """A Spec must be constructed with a list of Coordinates
        """
        with raises(ValueError) as ex:
            Spec(
                name='test',
                dtype='int'
            )
        assert "coords must be provided" in str(ex)

    def test_coords_type_error(self):
        """A Spec must be constructed with a list of Coordinates
        """
        with raises(ValueError) as ex:
            Spec(
                name='test',
                dtype='int',
                coords=["England", "Wales"]
            )
        assert "coords may be a dict of {dim: elements} or a list of Coordinate" in str(ex)

    def test_coords_from_dict(self):
        """A Spec must be constructed with a dtype
        """
        spec = Spec(
            name='test',
            dtype='int',
            coords={'countries': ["England", "Wales"]}
        )
        assert spec.shape == (2,)
        assert spec._coords[0].name == 'countries'
        assert spec._coords[0].ids == ["England", "Wales"]
