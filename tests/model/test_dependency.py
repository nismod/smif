"""Test dependency representation
"""
from copy import copy
from unittest.mock import Mock

from pytest import fixture, raises
from smif.metadata import Spec
from smif.model.dependency import Dependency


@fixture(scope='function')
def dep():
    """Dependency with mocked models
    """
    source_model = Mock()
    source_spec = Spec(
        name='source',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
    )
    sink_model = Mock()
    sink_spec = Spec(
        name='sink',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
    )
    return Dependency(source_model, source_spec, sink_model, sink_spec)


def test_create():
    """Create with models and specs, access properties
    """
    source_model = Mock()
    source_spec = Spec(
        name='source',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
    )
    sink_model = Mock()
    sink_spec = Spec(
        name='sink',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
    )
    dep = Dependency(source_model, source_spec, sink_model, sink_spec)
    assert dep.source_model == source_model
    assert dep.source == source_spec
    assert dep.sink_model == sink_model
    assert dep.sink == sink_spec


def test_create_with_identical_meta():
    source = Spec(
        name='source',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
        unit='m'
    )
    a_sink = Spec(
        name='sink',  # differs but that's okay
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
        unit='m'
    )
    Dependency(Mock(), source, Mock(), a_sink)

    b_sink = Spec(
        name='sink',
        dtype='int',  # differs
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
        unit='m'
    )
    with raises(ValueError) as ex:
        Dependency(Mock(), source, Mock(), b_sink)
    assert 'mismatched dtype' in str(ex)

    c_sink = Spec(
        name='sink',
        dtype='float',
        dims=['x', 'z'],  # differs
        coords={'x': [0, 1], 'z': [0, 1]},
        unit='m'
    )
    with raises(ValueError) as ex:
        Dependency(Mock(), source, Mock(), c_sink)
    assert 'mismatched dims' in str(ex)

    d_sink = Spec(
        name='sink',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [1, 2]},  # differs
        unit='m'
    )
    with raises(ValueError) as ex:
        Dependency(Mock(), source, Mock(), d_sink)
    assert 'mismatched coords' in str(ex)

    e_sink = Spec(
        name='sink',
        dtype='float',
        dims=['x', 'y'],
        coords={'x': [0, 1], 'y': [0, 1]},
        unit='km'  # differs
    )
    with raises(ValueError) as ex:
        Dependency(Mock(), source, Mock(), e_sink)
    assert 'mismatched unit' in str(ex)


def test_repr(dep):
    actual = repr(dep)
    expected = "<Dependency({}, {}, {}, {})>".format(
        dep.source_model, dep.source, dep.sink_model, dep.sink)
    assert actual == expected


def test_equality(dep):
    a = dep
    b = copy(dep)
    assert a == b

    c = copy(dep)
    c.source_model = Mock()
    assert a != c

    another_spec = Spec(
        name='another',
        dtype='float',
        dims=['z'],
        coords={'z': [0, 1]}
    )
    d = copy(dep)
    d.source = another_spec
    assert a != d

    e = copy(dep)
    e.sink = another_spec
    assert a != e

    f = copy(dep)
    f.sink_model = Mock
    assert a != f
