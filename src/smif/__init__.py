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
