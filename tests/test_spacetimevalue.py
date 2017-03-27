from pytest import raises
from smif import SpaceTimeValue


def test_create():
    """Object creation and accessors
    """
    val = SpaceTimeValue("UK", "year", 180, "units")

    assert val.region == "UK"
    assert val.interval == "year"
    assert val.value == 180
    assert val.units == "units"
    expected = "SpaceTimeValue('UK', 'year', 180, 'units')"
    assert repr(val) == expected


def test_equality():
    """Expect different objects with the same values to compare equal
    """
    val = SpaceTimeValue("UK", "year", 180, "units")
    same_val = SpaceTimeValue("UK", "year", 180, "units")

    assert val == same_val


def test_addition():
    """Add values, all else being equal
    """
    val = SpaceTimeValue("UK", "year", 180, "units")
    other_val = SpaceTimeValue("UK", "year", 20, "units")

    new_val = val + other_val
    # add values
    assert new_val.value == 200
    # otherwise the same
    assert val.region == "UK"
    assert val.interval == "year"
    assert val.units == "units"


def test_addition_invalid():
    """Fail to add values, if anything else is not eqal
    """
    val = SpaceTimeValue("UK", "year", 180, "units")
    other_val = SpaceTimeValue("France", "year", 20, "units")

    with raises(ValueError) as excinfo:
        val + other_val
    assert "Cannot add SpaceTimeValue of differing region" in str(excinfo)
