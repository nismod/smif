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

    def as_dict(self):
        return {'name': self.name,
                'filename': self.filename,
                'description': self.description}

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @name.setter
    @abstractmethod
    def name(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self):
        raise NotImplementedError

    @description.setter
    @abstractmethod
    def description(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def filename(self):
        raise NotImplementedError

    @filename.setter
    @abstractmethod
    def filename(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self):
        raise NotImplementedError

    @data.setter
    @abstractmethod
    def data(self, value):
        raise NotImplementedError

    @abstractmethod
    def get_entry_names(self):
        """Get the names of the entries in the ResolutionSet

        Returns
        -------
        set
            The set of names which identify each entry in the ResolutionSet
        """
        raise NotImplementedError
