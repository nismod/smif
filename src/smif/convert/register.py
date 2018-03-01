"""Register and ResolutionSet abstract classes to contain area and interval
metadata.
"""
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import numpy as np


class Register(metaclass=ABCMeta):

    @property
    @abstractmethod
    def names(self):
        raise NotImplementedError

    @abstractmethod
    def register(self, resolution_set):
        raise NotImplementedError

    @abstractmethod
    def get_coefficients(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, name):
        """Implement to return the smif.convert.ResolutionSet associated with the `name`

        Arguments
        ---------
        name : str
            The unique identifier of the ResolutionSet
        """
        raise NotImplementedError


class NDimensionalRegister(Register):

    def __init__(self):
        self._register = OrderedDict()
        self._conversions = defaultdict(dict)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_bounds(self, entry):
        """Implement this helper method to return bounds from an entry in the register

        Arguments
        ---------
        entry
            An entry from a ResolutionSet
        """
        raise NotImplementedError

    def get_proportion(self, entry_a, entry_b):
        raise NotImplementedError

    def get_coefficients(self, source, destination):
        """Get coefficients representing intersection of sets

        Arguments
        ---------
        source : string
            The name of the source set
        destination : string
            The name of the destination set

        Returns
        -------
        numpy.ndarray
        """

        from_set = self._register[source]
        to_set = self._register[destination]

        from_names = from_set.get_entry_names()

        coefficients = np.zeros((len(from_set), len(to_set)), dtype=np.float)

        for to_idx, to_entry in enumerate(to_set):
            bounds = self.get_bounds(to_entry)
            intersecting_from_entries = from_set.intersection(bounds)
            for from_entry in intersecting_from_entries:
                proportion = self.get_proportion(from_entry, to_entry)

                from_idx = from_names.index(from_entry.name)

                coefficients[from_idx, to_idx] = proportion

        return coefficients


class ResolutionSet(metaclass=ABCMeta):

    def __init__(self):

        self.name = ''
        self.description = ''
        self.data = []

    def as_dict(self):
        """
        """
        return {'name': self.name,
                'description': self.description}

    @abstractmethod
    def get_entry_names(self):
        """Get the names of the entries in the ResolutionSet

        Returns
        -------
        set
            The set of names which identify each entry in the ResolutionSet
        """
        raise NotImplementedError

    @abstractmethod
    def intersection(self, bounds):
        """Return the subset of entries intersecting with the bounds
        """
        raise NotImplementedError
