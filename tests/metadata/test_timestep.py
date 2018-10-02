"""Test relative timesteps
"""
from pytest import raises
from smif.exception import SmifTimestepResolutionError
from smif.metadata import RelativeTimestep


class TestRelativeTimestep(object):
    def test_create_from_name(self):
        """Should be able to create enum from string parameter
        """
        assert RelativeTimestep.ALL == RelativeTimestep.from_name("ALL")
        assert RelativeTimestep.BASE == RelativeTimestep.from_name("BASE")
        assert RelativeTimestep.CURRENT == RelativeTimestep.from_name("CURRENT")
        assert RelativeTimestep.PREVIOUS == RelativeTimestep.from_name("PREVIOUS")
        with raises(ValueError) as ex:
            RelativeTimestep.from_name("UNKNOWN")
        assert "Relative timestep 'UNKNOWN' is not recognised" in str(ex)

    def test_resolve(self):
        """Should be able to resolve current/previous/base timesteps
        """
        timesteps = list(range(5))
        rel = RelativeTimestep.CURRENT
        assert rel.resolve_relative_to(3, timesteps) == 3
        rel = RelativeTimestep.PREVIOUS
        assert rel.resolve_relative_to(3, timesteps) == 2
        rel = RelativeTimestep.BASE
        assert rel.resolve_relative_to(3, timesteps) == 0
        rel = RelativeTimestep.ALL
        assert rel.resolve_relative_to(3, timesteps) is None

        with raises(SmifTimestepResolutionError) as ex:
            rel.resolve_relative_to(99, timesteps)
        assert "is not present" in str(ex)

        rel = RelativeTimestep.PREVIOUS
        with raises(SmifTimestepResolutionError) as ex:
            rel.resolve_relative_to(0, timesteps)
        assert "has no previous timestep" in str(ex)
