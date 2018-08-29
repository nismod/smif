"""Metadata describes the details of data structures to be exchanged between models.

- :class:`~smif.metadata.spec.Spec` provides metadata to describe multi-dimensional variables,
  with an API modelled on `xarray <http://xarray.pydata.org/en/stable/>`_
- :class:`~smif.metadata.coordinates.Coordinates` label each dimension in a
  :class:`~smif.metadata.spec.Spec`. The elements of a set of
  :class:`~smif.metadata.coordinates.Coordinates` correspond to ElementSets under the
  `OGC® Open Modelling Interface (OpenMI) Interface Standard
  <http://www.opengeospatial.org/standards/openmi>`_
"""

# import classes here if they should be accessed at the subpackage level, for example ::
#         from smif.metadata import Spec
from smif.metadata.coordinates import Coordinates
from smif.metadata.spec import Spec

# Define what should be imported as * ::
#         from smif.metadata import *
__all__ = ['Coordinates', 'Spec']
