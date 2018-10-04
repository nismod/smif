"""smif runs models over a series of planning timesteps.

:class:`~smif.metadata.timestep.RelativeTimestep` is used to specify one timestep relative to
another in the context of the set of planning timesteps.

For example, if a model needs to know its own outputs from a previous year, we can specify the
self-dependency using a relative timestep.
"""
from enum import Enum

from smif.exception import SmifTimestepResolutionError


class RelativeTimestep(Enum):
    """Specify current, previous or base year timestep
    """
    CURRENT = 'CURRENT'  # current planning timestep
    PREVIOUS = 'PREVIOUS'  # previous planning timestep
    BASE = 'BASE'  # base year planning timestep
    ALL = 'ALL'  # all planning timesteps

    @classmethod
    def from_name(cls, name):
        if name == 'CURRENT':
            return cls.CURRENT
        elif name == 'PREVIOUS':
            return cls.PREVIOUS
        elif name == 'BASE':
            return cls.BASE
        elif name == 'ALL':
            return cls.ALL
        raise ValueError("Relative timestep '%s' is not recognised" % name)

    def resolve_relative_to(self, timestep, timesteps):
        """Resolve a relative timestep with respect to a given timestep and
        sequence of timesteps.

        Parameters
        ----------
        timestep : int
        timesteps : list of int
        """
        if timestep not in timesteps:
            raise SmifTimestepResolutionError(
                "Timestep {} is not present in {}".format(timestep, timesteps))

        # default None, e.g. for ALL
        relative_timestep = None

        if self.name == 'CURRENT':
            relative_timestep = timestep
        elif self.name == 'PREVIOUS':
            try:
                relative_timestep = element_before(timestep, timesteps)
            except ValueError:
                raise SmifTimestepResolutionError(
                    "{} has no previous timestep in {}".format(timestep, timesteps))
        elif self.name == 'BASE':
            relative_timestep = timesteps[0]

        return relative_timestep


def element_before(element, list_):
    """Return the element before a given element in a list, or None if the
    given element is first or not in the list.
    """
    if element not in list_ or element == list_[0]:
        raise ValueError("No element before {} in {}".format(element, list_))
    else:
        index = list_.index(element)
        return list_[index - 1]
