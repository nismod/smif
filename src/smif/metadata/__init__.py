"""Metadata describes the details of data structures to be exchanged between models and
adapters.
"""

# import classes for access like ::
#         from smif.metadata import Coordinates
from smif.metadata.coordinates import Coordinates
from smif.metadata.spec import Spec

# Define what should be imported as * ::
#         from smif.metadata import *
__all__ = ['Coordinates', 'Spec']