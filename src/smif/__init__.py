#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""smif"""
from __future__ import absolute_import, division, print_function

import warnings
from importlib.metadata import version

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
