#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function, absolute_import

import argparse
import logging
import sys

import pkg_resources

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

LOGGER = logging.getLogger(__name__)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'
