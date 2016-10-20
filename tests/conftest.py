#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for smif.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
from __future__ import absolute_import, division, print_function

import logging

import pytest

_log_format = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format=_log_format,
                    filemode='w')
