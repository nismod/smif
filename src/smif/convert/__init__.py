"""This module contains implementations of :class:`~smif.adaptor.adaptor.Adaptor` to perform
conversions between units and between different spatial or temporal dimensions where sufficient
metadata is provided in the :class:`~smif.metadata.spec.Spec` definition.

These should be useful to link models in simple cases, where it may be reasonable to rely on
strong assumptions about the underlying distributions of the variables to be converted.
"""
from smif.convert.adaptor import Adaptor
from smif.convert.interval import IntervalAdaptor
from smif.convert.region import RegionAdaptor
from smif.convert.unit import UnitAdaptor

__all__ = ["Adaptor", "IntervalAdaptor", "UnitAdaptor", "RegionAdaptor"]

__author__ = "Will Usher, Tom Russell, Roald Schoenmakers"
__copyright__ = "Will Usher, Tom Russell, Roald Schoenmakers"
__license__ = "mit"
