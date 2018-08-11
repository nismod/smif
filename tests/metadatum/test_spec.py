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
            coords=[
                Coordinates('countries', ["England", "Wales"]),
                Coordinates('age', [">30", "<30"])
            ],
            dtype='int',
            default=0,
            abs_range=(0, float('inf')),
            exp_range=(10e6, 10e9),
            unit='people'
        )
        assert spec.name == 'population'
        assert spec.unit == 'people'
        assert spec.abs_range == (0, float('inf'))
        assert spec.exp_range == (10e6, 10e9)
        assert spec.dtype == 'int'
        assert spec.shape == (2, 2)
        assert spec.ndim == 2
        assert spec.dims == ['countries', 'age']
        assert spec.coords == [
            Coordinates('countries', ["England", "Wales"]),
            Coordinates('age', [">30", "<30"])
        ]

    def test_from_dict(self):
        """classmethod to construct from serialisation
        """
        spec = Spec.from_dict({
            'name': 'population',
            'dims': ['countries', 'age'],
            'coords': {
                'age': [">30", "<30"],
                'countries': ["England", "Wales"]
            },
            'dtype': 'int',
            'default': 0,
            'abs_range': (0, float('inf')),
            'exp_range': (10e6, 10e9),
            'unit': 'people'
        })
        assert spec.name == 'population'
        assert spec.unit == 'people'
        assert spec.abs_range == (0, float('inf'))
        assert spec.exp_range == (10e6, 10e9)
        assert spec.dtype == 'int'
        assert spec.shape == (2, 2)
        assert spec.ndim == 2
        assert spec.dims == ['countries', 'age']
        assert spec.coords == [
            Coordinates('countries', ["England", "Wales"]),
            Coordinates('age', [">30", "<30"])
        ]

    def test_from_dict_defaults(self):
        """classmethod to construct from serialisation
        """
        spec = Spec.from_dict({
            'dims': ['countries'],
            'coords': {
                'countries': ["England", "Wales"]
            },
            'dtype': 'int'
        })
        assert spec.name is None
        assert spec.unit is None
        assert spec.abs_range is None
        assert spec.exp_range is None
        assert spec.dtype == 'int'
        assert spec.shape == (2,)
        assert spec.ndim == 1
        assert spec.dims == ['countries']
        assert spec.coords == [
            Coordinates('countries', ["England", "Wales"])
        ]

    def test_to_dict(self):
        actual = Spec(
            name='population',
            coords=[Coordinates('countries', ["England", "Wales"])],
            dtype='int',
            default=0,
            abs_range=(0, float('inf')),
            exp_range=(10e6, 10e9),
            unit='people'
        ).as_dict()
        expected = {
            'name': 'population',
            'dims': ['countries'],
            'coords': {'countries': ["England", "Wales"]},
            'dtype': 'int',
            'default': 0,
            'abs_range': (0, float('inf')),
            'exp_range': (10e6, 10e9),
            'unit': 'people'
        }
        assert actual == expected

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
        """A Spec may be constructed with a dict
        """
        spec = Spec(
            name='test',
            dtype='int',
            dims=['countries'],
            coords={'countries': ["England", "Wales"]}
        )
        assert spec.shape == (2,)
        assert spec._coords[0].name == 'countries'
        assert spec._coords[0].ids == ["England", "Wales"]

    def test_coords_from_dict_error(self):
        """A Spec constructed with a dict must have dims
        """
        with raises(ValueError) as ex:
            Spec(
                name='test',
                dtype='int',
                coords={'countries': ["England", "Wales"]}
            )
        assert "dims must be specified" in str(ex)

        with raises(ValueError) as ex:
            Spec(
                name='test',
                dtype='int',
                dims=['countries', 'age'],
                coords={'countries': ["England", "Wales"]}
            )
        assert "dims must match the keys in coords" in str(ex)

        with raises(ValueError) as ex:
            Spec(
                name='test',
                dtype='int',
                dims=['countries'],
                coords={
                    'countries': ["England", "Wales"],
                    'age': [">30", "<30"]
                }
            )
        assert "dims must match the keys in coords" in str(ex)

    def test_coords_from_list_error(self):
        """A Spec constructed with a dict must have dims
        """
        with raises(ValueError) as ex:
            Spec(
                name='test',
                dtype='int',
                coords=[Coordinates('countries', ["England", "Wales"])],
                dims=['countries']
            )
        assert "dims are derived" in str(ex)

    def test_eq(self):
        """Equality based on equivalent dtype, dims, coords, unit
        """
        a = Spec(
            name='population',
            coords=[Coordinates('countries', ["England", "Wales"])],
            dtype='int',
            unit='people'
        )
        b = Spec(
            name='pop',
            coords=[Coordinates('countries', ["England", "Wales"])],
            dtype='int',
            unit='people'
        )
        c = Spec(
            name='population',
            coords=[Coordinates('countries', ["England", "Scotland", "Wales"])],
            dtype='int',
            unit='people'
        )
        d = Spec(
            name='population',
            coords=[Coordinates('countries', ["England", "Wales"])],
            dtype='float',
            unit='people'
        )
        e = Spec(
            name='population',
            coords=[Coordinates('countries', ["England", "Wales"])],
            dtype='int',
            unit='thousand people'
        )
        assert a == b
        assert a != c
        assert a != d
        assert a != e
