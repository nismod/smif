"""Register and ResolutionSet abstract classes to contain area and interval
metadata.
"""
from abc import ABCMeta, abstractmethod


class Register(metaclass=ABCMeta):

    @abstractmethod
    def register(self, resolution_set):
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
