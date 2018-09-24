# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""

# import classes for access like ::
#         from smif.data_layer import DatabaseInterface`
from smif.data_layer.database_interface import DatabaseInterface
from smif.data_layer.datafile_interface import DatafileInterface

from smif.data_layer.data_handle import DataHandle, TimestepResolutionError
from smif.data_layer.memory_interface import MemoryInterface

# Define what should be imported as * ::
#         from smif.data_layer import *
__all__ = ['DatabaseInterface', 'DatafileInterface', 'DataHandle', 'MemoryInterface',
           'TimestepResolutionError']
