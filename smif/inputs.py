# -*- coding: utf-8 -*-
"""Encapsulates the inputs to a sector model


        - name: petrol_price
          spatial_resolution: GB
          temporal_resolution: annual
        - name: diesel_price
          spatial_resolution: GB
          temporal_resolution: annual
        - name: LPG_price
          spatial_resolution: GB
          temporal_resolution: annual

"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell, University of Oxford 2017"
__license__ = "mit"


Dependency = namedtuple(
    "Dependency",
    [
        "name",
        "spatial_resolution",
        "temporal_resolution"
    ]
)


class DependencyList(object):
    """Holds the dependencies defined in a file called ``inputs.yaml``.

    Dependencies have several attributes: ``name``, ``spatial_resolution``
    and ``temporal_resolution``.

    The ``name`` entry denotes the unique identifier of a model or scenario output.
    The ``spatial_resolution`` and ``temporal_resolution`` are references to the
    catalogue held by the :class:`~smif.sector_model.SosModel` which define the
    available conversion formats.

    An example yaml file::

            dependencies:
            - name: eletricity_price
            spatial_resolution: GB
            temporal_resolution: annual

    Parameters
    ----------
    dependencies: dict
        A dictionary of dependencies

    """

    def __init__(self, dependencies):

        self.logger = logging.getLogger(__name__)

        names = []
        spatial_resolutions = []
        temporal_resolutions = []

        for dependency in dependencies:
            names.append(dependency['name'])
            spatial_resolutions.append(dependency['spatial_resolution'])
            temporal_resolutions.append(dependency['temporal_resolution'])

        self.names = names
        self.spatial_resolutions = spatial_resolutions
        self.temporal_resolutions = temporal_resolutions

    def __repr__(self):
        """Return literal string representation of this instance
        """
        template = "{{'name': {}, 'spatial_resolution': {}, " + \
                   "'temporal_resolution': {}}}"

        return template.format(
            self.names,
            self.spatial_resolutions,
            self.temporal_resolutions,
        )

    def __getitem__(self, key):
        """Implement __getitem__ to make this class iterable

        Example
        =======
        >>> dependency_list = DependencyList()
        >>> for dep in dependency_list:
        >>>     # do something with each dependency

        - uses Dependency (defined as a namedtuple) to wrap the data
        - lets the np.arrays raise TypeError or IndexError for incorrect
          or out-of-bounds accesses
        """
        dependency = Dependency(
            self.names[key],
            self.spatial_resolutions[key],
            self.temporal_resolutions[key]
        )
        return dependency

    def __len__(self):
        return len(self.names)


class ModelInputs(object):
    """A container for all the model inputs

    Arguments
    =========
    inputs : dict
        A dictionary of key: val pairs including a list of input types and
        names, followed by nested dictionaries of input attributes
    """
    def __init__(self, inputs):
        if 'dependencies' not in inputs:
            inputs['dependencies'] = []

        self._dependencies = DependencyList(inputs['dependencies'])

    @property
    def dependencies(self):
        """A list of the model dependencies

        Returns
        =======
        :class:`smif.inputs.DependencyList`
        """
        return self._dependencies

    def __len__(self):
        return len(self.dependencies)
