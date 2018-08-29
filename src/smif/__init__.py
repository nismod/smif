#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""smif
"""
from __future__ import division, print_function, absolute_import

import pkg_resources
import warnings

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

# Filter out warnings arising from some installed combinations of scipy/numpy
# - problem and fix discussed in [numpy/numpy#432](https://github.com/numpy/numpy/pull/432)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
