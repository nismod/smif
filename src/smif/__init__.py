#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""smif
"""
from __future__ import division, print_function, absolute_import

import pkg_resources

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'


class StateData(object):
    """A piece of state data

    Parameters
    ----------
    target
        The id or name of the object described by this state
    data
        The state attribute/data to apply - could typically be a dict of
        attributes
    """
    def __init__(self, target, data):
        self.target = target
        self.data = data

    def __repr__(self):
        return "StateData({}, {})".format(
            repr(self.target),
            repr(self.data)
        )
