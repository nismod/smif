#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function, absolute_import
from collections import namedtuple

import pkg_resources

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'


SpaceTimeValue = namedtuple('SpaceTimeValue', ['region', 'interval', 'value', 'units'])
docs = """A tuple of scenario data

Parameters
----------
region: str
    A valid (unique) region name which is registered in the region register
interval: str
    A valid (unique) interval name which is registered in the interval
    register
value: float
    The value
units: str
    The units associated with the `value`
"""
SpaceTimeValue.__doc__ = docs
